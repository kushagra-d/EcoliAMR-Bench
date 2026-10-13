"""SQLite-backed replacement for `streaming_hashed_matrix.DiskBackedRowStore`.

Motivation (measured on the real 100-genome ciprofloxacin pilot at
k=15/n_buckets=262144): the JSONL store wrote each row as a JSON line
containing explicit decimal-text `cols`/`values` arrays - ~2.9 MB/row as
ASCII text for a fully-dense row, 291 MB total for 100 genomes. Beyond
raw size, EVERY duplicate-genome check
(`DiskBackedRowStore.existing_genome_ids`) parsed the ENTIRE file -
cost tied to total row *payload* size, not row *count*.

This module fixes both problems with one file format: a single SQLite
database (stdlib `sqlite3`, no new dependency) with one `rows` table:

    CREATE TABLE rows (
        row_index INTEGER PRIMARY KEY,
        genome_id TEXT NOT NULL UNIQUE,
        n_entries INTEGER NOT NULL,
        cols BLOB NOT NULL,   -- packed np.uint32 array, little-endian
        vals BLOB NOT NULL    -- packed np.uint32 array, same length as cols
    )

Binary packing alone roughly halves storage versus JSON-as-text (4
raw bytes/int vs. ~7+ ASCII characters/int). The bigger win is
structural: `genome_id` is UNIQUE (SQLite builds an index for this
automatically), so duplicate-detection becomes one indexed point lookup
(`row_exists`) instead of a full-file scan to build an in-memory set.
`row_index` is the PRIMARY KEY, so ordered iteration and random access
by row position are both indexed, not linear scans.

CRASH SAFETY - what is and is not claimed:

Each `append_row` call is one SQLite transaction (`with self._conn:
...`, which commits on success and rolls back on any exception). SQLite
transactions are atomic: a process killed between the `INSERT` and the
commit leaves NO trace of that row once a new connection reopens the
database - the engine's own rollback-journal replay (the default
journal_mode, not WAL - see note below) discards the uncommitted write.
There is no partial/half-written row state to detect, because the
column constraints (`NOT NULL`) mean a row is either entirely present
(all five columns written, transaction committed) or entirely absent.

This is tested two ways (`tests/test_sqlite_row_store.py`):
  (1) an in-process rollback (explicit `conn.rollback()` after an
      uncommitted INSERT) - fast, no subprocess, confirms the row is
      simply invisible.
  (2) a true cross-process simulation: a helper script
      (`tests/_sqlite_interrupt_subprocess.py`) commits N rows normally,
      begins an N+1th INSERT, and calls `os._exit(1)` - a hard process
      kill that skips Python/SQLite cleanup entirely, the closest
      reproducible approximation of a real crash (SIGKILL) available in
      a test. The parent process then reopens the SAME database file
      and verifies exactly N rows are present, in order, with no
      corruption, and that resuming a build safely retries genome N+1
      without duplicating 0..N-1.

NOT tested or claimed: corruption of the SQLite file itself (disk
hardware failure, truncated file from a `dd`-style partial copy),
multi-process concurrent writers, or crash safety under WAL mode (WAL
is not used here - the default rollback-journal mode is kept
deliberately, specifically so the crash story stays this simple: one
journal file, replayed automatically by the next connection that opens
the database for writing).

The old `DiskBackedRowStore`/JSONL path in `streaming_hashed_matrix.py`
is UNCHANGED and UNTOUCHED by this module - both exist side by side so
the original pilot artifacts remain available for comparison.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from .streaming_hashed_matrix import (
    FailureLog,
    ResourceLimits,
    StreamingRunStatus,
    check_disk_space,
    hash_genome_to_bucket_counts,
)
from .streaming_kmer_build import peak_memory_mb

__all__ = [
    "FailureLog",
    "ResourceLimits",
    "StreamingRunStatus",
    "SqliteRowStore",
    "build_sqlite_backed_matrix",
]


class SqliteRowStore:
    """One SQLite file holding every genome's sparse hashed row.

    Holds a single long-lived connection (the common sqlite3 usage
    pattern) rather than reopening per call. `n_buckets` is stored only
    in Python (not persisted in the schema) - it is supplied by the
    caller consistently across a run, exactly as `DiskBackedRowStore`
    already requires."""

    def __init__(self, path: Path, n_buckets: int):
        self.path = Path(path)
        self.n_buckets = n_buckets
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS rows ("
            "row_index INTEGER PRIMARY KEY, "
            "genome_id TEXT NOT NULL UNIQUE, "
            "n_entries INTEGER NOT NULL, "
            "cols BLOB NOT NULL, "
            "vals BLOB NOT NULL)"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def row_exists(self, genome_id: str) -> bool:
        """Indexed point lookup (hits the UNIQUE index on genome_id) -
        never scans the table, unlike `DiskBackedRowStore.existing_genome_ids`."""
        cur = self._conn.execute("SELECT 1 FROM rows WHERE genome_id = ? LIMIT 1", (genome_id,))
        return cur.fetchone() is not None

    def existing_genome_ids(self) -> set[str]:
        """Kept for API parity / tests. Still a full scan, but of tiny
        genome_id strings only, not of dense sparse payloads - this is
        the "smaller constant" half of the fix; `row_exists` (used by
        `build_sqlite_backed_matrix`) is the "no scan at all" half."""
        cur = self._conn.execute("SELECT genome_id FROM rows")
        return {row[0] for row in cur.fetchall()}

    def n_rows_written(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM rows")
        return cur.fetchone()[0]

    def append_row(self, genome_id: str, row_index: int, bucket_counts: dict[int, int]) -> None:
        """One atomic transaction: the row is fully present after this
        returns, or (if the process dies first) not present at all."""
        cols_arr = np.array(list(bucket_counts.keys()), dtype=np.uint32)
        vals_arr = np.array(list(bucket_counts.values()), dtype=np.uint32)
        with self._conn:
            self._conn.execute(
                "INSERT INTO rows (row_index, genome_id, n_entries, cols, vals) VALUES (?, ?, ?, ?, ?)",
                (row_index, genome_id, len(cols_arr), cols_arr.tobytes(), vals_arr.tobytes()),
            )

    def get_row(self, row_index: int) -> tuple[str, np.ndarray, np.ndarray] | None:
        """Single indexed row lookup by primary key - for exact
        row-level verification against another matrix, not for bulk access."""
        cur = self._conn.execute(
            "SELECT genome_id, cols, vals FROM rows WHERE row_index = ?", (row_index,)
        )
        result = cur.fetchone()
        if result is None:
            return None
        genome_id, cols_blob, vals_blob = result
        return genome_id, np.frombuffer(cols_blob, dtype=np.uint32), np.frombuffer(vals_blob, dtype=np.uint32)

    def iter_batches(self, batch_size: int):
        """Yields (genome_ids, row_indices, csr_batch) in row_index order,
        `batch_size` rows at a time. `cursor.fetchmany` pulls only that
        many rows' blobs into memory per iteration - the full table is
        never materialized at once, regardless of total row count."""
        cur = self._conn.execute("SELECT row_index, genome_id, n_entries, cols, vals FROM rows ORDER BY row_index")
        while True:
            chunk = cur.fetchmany(batch_size)
            if not chunk:
                return
            yield self._rows_to_csr(chunk)

    def read_rows_for_indices(self, wanted_row_indices: set[int], batch_size: int = 64):
        """Yields only the rows whose `row_index` is in
        `wanted_row_indices`, `batch_size` at a time, via an indexed
        `WHERE row_index IN (...)` lookup per chunk - never scans rows
        outside the requested set.

        Indices are coerced to plain `int`: sqlite3 silently binds a
        `numpy.int64` as something that matches no `INTEGER` row (no
        exception - the query just returns zero rows), so a caller
        passing indices from e.g. `train_test_split` would otherwise
        get an empty result with no warning."""
        wanted_sorted = sorted(int(i) for i in wanted_row_indices)
        for i in range(0, len(wanted_sorted), batch_size):
            chunk_indices = wanted_sorted[i : i + batch_size]
            placeholders = ",".join("?" * len(chunk_indices))
            cur = self._conn.execute(
                f"SELECT row_index, genome_id, n_entries, cols, vals FROM rows "
                f"WHERE row_index IN ({placeholders}) ORDER BY row_index",
                chunk_indices,
            )
            rows = cur.fetchall()
            if rows:
                yield self._rows_to_csr(rows)

    def _rows_to_csr(self, rows: list[tuple]):
        """Vectorized CSR reconstruction (the default, as of the
        out-of-core read-path optimization task). Builds COO arrays via
        `np.concatenate`/`np.repeat` - never a Python-level int list.
        This matters because this pilot's rows are 100% dense (262,144
        entries each): `_rows_to_csr_naive`'s `.tolist()` + `list.extend`
        materializes ~4.2M individual Python `int` objects per 16-row
        batch; this path never leaves numpy array operations. Output is
        byte-for-byte identical to `_rows_to_csr_naive` - verified in
        `tests/test_sqlite_row_store.py`."""
        n_rows = len(rows)
        if n_rows == 0:
            return [], [], sparse.csr_matrix((0, self.n_buckets), dtype=np.int64)
        genome_ids = [r[1] for r in rows]
        row_indices = [r[0] for r in rows]
        cols_arrays = [np.frombuffer(r[3], dtype=np.uint32) for r in rows]
        vals_arrays = [np.frombuffer(r[4], dtype=np.uint32) for r in rows]
        counts = np.array([len(c) for c in cols_arrays], dtype=np.int64)
        all_cols = np.concatenate(cols_arrays)
        all_vals = np.concatenate(vals_arrays)
        all_rows = np.repeat(np.arange(n_rows, dtype=np.int64), counts)
        matrix = sparse.coo_matrix(
            (all_vals, (all_rows, all_cols)), shape=(n_rows, self.n_buckets), dtype=np.int64
        ).tocsr()
        return genome_ids, row_indices, matrix

    def _rows_to_csr_naive(self, rows: list[tuple]):
        """The ORIGINAL implementation, kept deliberately (not deleted)
        as: (1) the equivalence oracle tests compare the fast path
        against, and (2) the "before" side of the benchmark comparing
        reconstruction cost. Not called by `iter_batches`/
        `read_rows_for_indices` anymore - see `_rows_to_csr`."""
        genome_ids: list[str] = []
        row_indices: list[int] = []
        coo_rows: list[int] = []
        coo_cols: list[int] = []
        coo_vals: list[int] = []
        for local_idx, (row_index, genome_id, n_entries, cols_blob, vals_blob) in enumerate(rows):
            genome_ids.append(genome_id)
            row_indices.append(row_index)
            cols = np.frombuffer(cols_blob, dtype=np.uint32)
            vals = np.frombuffer(vals_blob, dtype=np.uint32)
            coo_rows.extend([local_idx] * len(cols))
            coo_cols.extend(cols.tolist())
            coo_vals.extend(vals.tolist())
        matrix = sparse.coo_matrix(
            (coo_vals, (coo_rows, coo_cols)), shape=(len(rows), self.n_buckets), dtype=np.int64
        ).tocsr()
        return genome_ids, row_indices, matrix


def build_sqlite_backed_matrix(
    contigs_loader,
    genome_order: list[str],
    k: int,
    n_buckets: int,
    row_store: SqliteRowStore,
    failure_log: FailureLog,
    limits: ResourceLimits = ResourceLimits(),
    progress_callback=None,
) -> StreamingRunStatus:
    """Same contract as `streaming_hashed_matrix.build_streaming_hashed_matrix`
    (one genome's contigs loaded/hashed/discarded at a time, never a
    prerequisite raw matrix), but duplicate-checks each genome via
    `row_store.row_exists` - an indexed point lookup - instead of
    pre-scanning the store into an in-memory set."""
    status = StreamingRunStatus(n_genomes_requested=len(genome_order))
    t0 = time.perf_counter()

    already_failed = failure_log.genome_ids()

    if not check_disk_space(row_store.path, limits.min_free_disk_bytes):
        status.aborted_reason = "insufficient disk space before starting"
        status.elapsed_seconds = time.perf_counter() - t0
        return status

    for row_index, genome_id in enumerate(genome_order):
        if row_store.row_exists(genome_id):
            status.n_genomes_skipped_already_present += 1
            continue
        if genome_id in already_failed:
            continue

        if not check_disk_space(row_store.path, limits.min_free_disk_bytes):
            status.aborted_reason = f"insufficient disk space after {row_index} genomes"
            break

        mem_mb = peak_memory_mb()
        if mem_mb > limits.memory_limit_mb:
            status.aborted_reason = f"peak RSS {mem_mb:.0f}MB exceeded limit after {row_index} genomes"
            break

        try:
            contigs = contigs_loader(genome_id)
            bucket_counts = hash_genome_to_bucket_counts(contigs, k, n_buckets)
            del contigs
        except Exception as exc:  # noqa: BLE001 - any failure is logged, never silently zeroed
            failure_log.append(genome_id, f"{type(exc).__name__}: {exc}")
            status.n_failures_this_run += 1
            if failure_log.count() > limits.max_failed_rows:
                status.aborted_reason = f"exceeded max_failed_rows ({limits.max_failed_rows})"
                break
            continue

        row_store.append_row(genome_id, row_index, bucket_counts)
        status.n_rows_written_this_run += 1
        del bucket_counts

        if progress_callback is not None:
            progress_callback(row_index, genome_id, status.n_rows_written_this_run, peak_memory_mb())

    status.n_failures_total = failure_log.count()
    status.n_rows_total = row_store.n_rows_written()
    status.peak_rss_mb = peak_memory_mb()
    status.elapsed_seconds = time.perf_counter() - t0
    status.complete = (
        status.aborted_reason is None
        and status.n_rows_total + status.n_failures_total == status.n_genomes_requested
    )
    return status
