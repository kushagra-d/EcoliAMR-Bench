"""Hybrid disk-backed, memory-bounded reader for repeated-scan
incremental training.

Three tiers, each falling back to the next:

  1. MEMORY - a small, non-evicting in-memory batch cache (the policy
     the prior task measured as best for this project's exact access
     pattern: the same fixed batches scanned every epoch). Bounded by
     `max_memory_cache_bytes` / `max_reader_memory_bytes` (same two-
     budget semantics as `cached_row_reader.py` - not conflated).
  2. DISK - a compact, compressed `.npz` artifact per batch (same
     serialization as `epoch_batch_cache.py`, reused because it
     measured a genuine ~3.7x disk-footprint win on this project's
     100%-dense data). A memory miss is served from here with NO
     SQLite query - just one `load_npz` decompression.
  3. SOURCE - the SQLite store, queried only to BUILD or REPAIR the
     disk tier, never on an ordinary read once the disk tier is
     complete.

WHY A SEPARATE MODULE, NOT A MODIFICATION OF `epoch_batch_cache.py`:
this task's partial-build recovery requirement needed a more granular,
per-batch-resumable build process than `PersistentBatchCache` has today
(which rebuilds its ENTIRE partition from scratch on any invalidation).
Rather than risk `PersistentBatchCache`'s already-tested behavior, this
module reimplements the disk tier with that finer-grained recovery,
self-contained.

PARTIAL-BUILD RECOVERY (mechanically enforced, not just documented):
two separate marker files distinguish "a build is in progress for this
exact configuration" from "the artifact is fully committed and
trustworthy":

  - `build_fingerprint.json`: written BEFORE any batch is written,
    recording the configuration (source DB path/size/mtime, row
    indices, batch size, n_buckets, normalize_rows) this build is FOR.
  - `fingerprint.json`: written ONLY after every expected batch is
    confirmed present and loadable - the "commit" signal.

If the process is killed mid-build, `fingerprint.json` is never
written, so the artifact is correctly seen as incomplete on the next
run. If `build_fingerprint.json` matches the CURRENT configuration
(same DB, same partition, same settings), already-written batches are
individually re-validated (file exists AND loads without error) and
REUSED - only the missing or corrupt ones are re-fetched from SQLite
and rewritten. If the configuration differs at all, everything is
wiped and rebuilt from scratch - partial batches from a DIFFERENT
configuration are never trusted or mixed in.

Each batch's `.npz` write is itself atomic: written to a staging name
that still ends in `.npz` (scipy's `save_npz` silently appends `.npz`
to any name that doesn't already end with it, so a `.tmp` suffix would
be mangled) under a `.staging/` subdirectory, then `Path.rename`d into
place - a rename on the same filesystem is atomic, so a batch file at
its final name is always fully written.

Row ordering, feature columns, genome IDs, and the hash mapping are
untouched - this module only changes WHERE a batch's bytes come from,
never what they contain.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from scipy import sparse
from sklearn.preprocessing import normalize as sk_normalize

from .sqlite_row_store import SqliteRowStore
from .streaming_hashed_matrix import check_disk_space

DEFAULT_MAX_MEMORY_CACHE_BYTES = 64_000_000


@dataclass
class HybridCacheStats:
    n_memory_hits: int = 0
    n_disk_hits: int = 0  # served from the npz artifact - a decompression, not a SQLite query
    n_decompressions: int = 0  # == n_disk_hits; named separately per this task's explicit counter list
    n_source_reads: int = 0  # batches fetched from SQLite (build or repair of the disk tier)
    n_evictions: int = 0  # memory-tier evictions (0 under the non-evicting memory policy used here)
    n_fallbacks: int = 0  # batches that didn't fit the memory budget - served from disk every time
    n_batches_rebuilt_this_session: int = 0
    memory_bytes_cached: int = 0
    peak_memory_bytes_cached: int = 0
    disk_artifact_bytes: int = 0
    warnings: list[str] = field(default_factory=list)


class HybridTrainingReader:
    """Scoped to ONE fixed (row_indices, batch_size, normalize_rows)
    combination, exactly like `PersistentBatchCache` - a different
    partition/batch_size requires a separate instance (see
    `hybrid_training_reader.MultiPartitionHybridReader` below for the
    train+test adapter used by real end-to-end runs)."""

    def __init__(
        self,
        cache_dir: Path,
        row_store: SqliteRowStore,
        row_indices: set[int],
        batch_size: int,
        normalize_rows: bool = False,
        max_memory_cache_bytes: int = DEFAULT_MAX_MEMORY_CACHE_BYTES,
        max_reader_memory_bytes: int | None = None,
        min_free_disk_bytes: int = 1_000_000_000,
    ):
        self.cache_dir = Path(cache_dir)
        self._store = row_store
        self.row_indices = tuple(sorted(int(i) for i in row_indices))
        self.batch_size = batch_size
        self.normalize_rows = normalize_rows
        self.max_memory_cache_bytes = max_memory_cache_bytes
        self.max_reader_memory_bytes = (
            max_reader_memory_bytes if max_reader_memory_bytes is not None else max_memory_cache_bytes
        )
        self.min_free_disk_bytes = min_free_disk_bytes

        self.build_fingerprint_path = self.cache_dir / "build_fingerprint.json"
        self.fingerprint_path = self.cache_dir / "fingerprint.json"
        self.meta_path = self.cache_dir / "meta.json"

        self._memory_cache: dict[tuple[int, ...], tuple[list[str], list[int], sparse.csr_matrix]] = {}
        self.stats = HybridCacheStats()
        self._verified_complete = False  # memoized for this instance's lifetime only - see ensure_disk_materialized

    # --- fingerprinting / disk-tier state ---

    def _compute_fingerprint(self) -> dict:
        db_path = Path(self._store.path)
        db_stat = db_path.stat()
        return {
            "db_path": str(db_path),
            "db_size_bytes": db_stat.st_size,
            "db_mtime_ns": db_stat.st_mtime_ns,
            "row_indices": list(self.row_indices),
            "batch_size": self.batch_size,
            "n_buckets": self._store.n_buckets,
            "normalize_rows": self.normalize_rows,
        }

    def _expected_chunks(self) -> list[tuple[int, ...]]:
        return [
            self.row_indices[i : i + self.batch_size] for i in range(0, len(self.row_indices), self.batch_size)
        ]

    def _read_json(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _write_json_atomic(self, path: Path, data: dict) -> None:
        staging = path.with_suffix(path.suffix + ".staging")
        staging.write_text(json.dumps(data))
        staging.rename(path)

    def is_complete(self) -> bool:
        """The strict, final check: fully committed AND matching the
        current configuration AND every expected batch file present AND
        loadable. Re-validates loadability on every call (an O(batches)
        real disk read, not just an existence check) - a deliberate
        choice: silently trusting a corrupted file would defeat the
        point of this check."""
        stored = self._read_json(self.fingerprint_path)
        if stored != self._compute_fingerprint():
            return False
        meta = self._read_json(self.meta_path)
        if meta is None:
            return False
        completed = {b["batch_index"] for b in meta["batches"]}
        if completed != set(range(len(self._expected_chunks()))):
            return False
        return all(self._batch_is_valid_on_disk(i) for i in completed)

    def _batch_is_valid_on_disk(self, batch_index: int) -> bool:
        """A batch counts as present only if its file exists AND loads
        without error - catches corruption, not just absence."""
        npz_path = self.cache_dir / f"batch_{batch_index:05d}.npz"
        if not npz_path.exists():
            return False
        try:
            sparse.load_npz(npz_path)
            return True
        except Exception:  # noqa: BLE001 - any load failure means "rebuild this one"
            return False

    def ensure_disk_materialized(self) -> None:
        """Builds or repairs the disk tier. `is_complete()` re-validates
        every batch by loading it (real disk I/O) - correct for
        catching corruption on a FRESH instance (e.g. a new process
        after a crash), but far too expensive to repeat on every one of
        50 training epochs using the SAME instance. `_verified_complete`
        memoizes a true result for this instance's lifetime: once this
        instance has confirmed the disk tier is complete, it trusts that
        for the rest of its life (nothing in this codebase mutates a
        committed artifact in place) - a NEW instance (new process) does
        not inherit this memo and re-validates properly."""
        if self._verified_complete:
            return
        if self.is_complete():
            self._verified_complete = True
            return

        current_fp = self._compute_fingerprint()
        build_fp = self._read_json(self.build_fingerprint_path)

        if build_fp != current_fp:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._write_json_atomic(self.build_fingerprint_path, current_fp)
            self._write_json_atomic(self.meta_path, {"batches": []})
        else:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if self.fingerprint_path.exists():
                self.fingerprint_path.unlink()  # not complete yet if we're here - don't let a stale commit marker linger

        if not check_disk_space(self.cache_dir / "_probe", self.min_free_disk_bytes):
            self.stats.warnings.append("insufficient disk space to materialize hybrid disk tier")
            raise RuntimeError("insufficient disk space to materialize hybrid disk tier")

        staging_dir = self.cache_dir / ".staging"
        staging_dir.mkdir(parents=True, exist_ok=True)

        meta = self._read_json(self.meta_path) or {"batches": []}
        batches_by_index = {b["batch_index"]: b for b in meta["batches"]}
        expected = self._expected_chunks()

        for idx, chunk in enumerate(expected):
            if idx in batches_by_index and self._batch_is_valid_on_disk(idx):
                continue  # already built and verified loadable - reused, not rebuilt

            batch = None
            for genome_ids, row_indices, matrix in self._store.read_rows_for_indices(
                set(chunk), batch_size=len(chunk)
            ):
                batch = (genome_ids, row_indices, matrix)
            self.stats.n_source_reads += 1

            payload = sk_normalize(batch[2], norm="l2") if self.normalize_rows else batch[2]
            final_path = self.cache_dir / f"batch_{idx:05d}.npz"
            staging_path = staging_dir / f"batch_{idx:05d}.npz"
            sparse.save_npz(staging_path, payload)
            staging_path.rename(final_path)

            batches_by_index[idx] = {
                "batch_index": idx,
                "npz_file": final_path.name,
                "genome_ids": batch[0],
                "row_indices": batch[1],
            }
            meta["batches"] = sorted(batches_by_index.values(), key=lambda b: b["batch_index"])
            self._write_json_atomic(self.meta_path, meta)  # record progress after EVERY batch
            self.stats.n_batches_rebuilt_this_session += 1

        if staging_dir.exists() and not any(staging_dir.iterdir()):
            staging_dir.rmdir()

        self._write_json_atomic(self.fingerprint_path, current_fp)  # commit - only now is this "complete"
        self.stats.disk_artifact_bytes = sum(
            (self.cache_dir / f"batch_{i:05d}.npz").stat().st_size for i in range(len(expected))
        )
        self._verified_complete = True

    # --- reading ---

    @staticmethod
    def _csr_bytes(matrix: sparse.csr_matrix) -> int:
        return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)

    def read_rows_for_indices(self, wanted_row_indices: set[int], batch_size: int = 64):
        wanted_sorted = tuple(sorted(int(i) for i in wanted_row_indices))
        if wanted_sorted != self.row_indices or batch_size != self.batch_size:
            raise ValueError(
                "HybridTrainingReader is scoped to the exact (row_indices, batch_size) it was "
                "constructed with - got a different request. Use MultiPartitionHybridReader for "
                "train+test partitions in the same training run, or construct a new instance."
            )
        self.ensure_disk_materialized()
        meta = json.loads(self.meta_path.read_text())

        for entry in meta["batches"]:
            chunk_key = tuple(entry["row_indices"])
            cached = self._memory_cache.get(chunk_key)
            if cached is not None:
                self.stats.n_memory_hits += 1
                yield cached
                continue

            matrix = sparse.load_npz(self.cache_dir / entry["npz_file"])
            self.stats.n_disk_hits += 1
            self.stats.n_decompressions += 1
            batch = (entry["genome_ids"], entry["row_indices"], matrix)

            size = self._csr_bytes(matrix)
            fits_reader = self.stats.memory_bytes_cached + size <= self.max_reader_memory_bytes
            fits_memory = self.stats.memory_bytes_cached + size <= self.max_memory_cache_bytes
            if fits_reader and fits_memory:
                self._memory_cache[chunk_key] = batch
                self.stats.memory_bytes_cached += size
                self.stats.peak_memory_bytes_cached = max(self.stats.peak_memory_bytes_cached, self.stats.memory_bytes_cached)
            else:
                self.stats.n_fallbacks += 1
            yield batch


class MultiPartitionHybridReader:
    """Routes `read_rows_for_indices` to the matching
    `HybridTrainingReader` for that exact (row_indices, batch_size) -
    the same adapter pattern as `epoch_batch_cache.MultiPartitionPersistentReader`,
    needed because `evaluate_streaming_hashed_matrix` makes two distinct
    calls per run (train partition every epoch, test partition once)."""

    def __init__(self, readers: list[HybridTrainingReader]):
        self._by_key = {(r.row_indices, r.batch_size): r for r in readers}

    def read_rows_for_indices(self, wanted_row_indices: set[int], batch_size: int = 64):
        key = (tuple(sorted(int(i) for i in wanted_row_indices)), batch_size)
        reader = self._by_key.get(key)
        if reader is None:
            raise ValueError("No HybridTrainingReader configured for this (row_indices, batch_size) combination")
        yield from reader.read_rows_for_indices(wanted_row_indices, batch_size)
