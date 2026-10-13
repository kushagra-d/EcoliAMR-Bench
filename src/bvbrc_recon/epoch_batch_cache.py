"""Strategy #4 (this task): a scan-aware, disk-persisted batch artifact
for ONE fixed, repeatedly-scanned partition (e.g. one split's training
rows, read identically every epoch).

Different mechanism from `cached_row_reader.py`'s in-memory caches: this
materializes the partition's batches to a small set of files ONCE (via
`scipy.sparse.save_npz`/`load_npz` - reusing this project's existing,
already-tested sparse-matrix serialization rather than inventing a new
binary format), then serves every subsequent read directly from those
files - no SQL query, no raw-count-to-CSR reconstruction, on any read
after the first. This targets exactly the cost `reports/
memory_budgeted_reader_assessment.md` identified as dominant (SQL +
reconstruction), independent of how much RAM is available for an
in-memory cache: the artifact lives on disk, not in the process.

SCOPE: this cache is built for ONE specific (row_indices, batch_size,
normalize_rows) combination, decided at construction - it is not a
general-purpose reader replacement. Calling `read_rows_for_indices` with
different `wanted_row_indices` or `batch_size` than it was built for
raises `ValueError` rather than silently reading wrong data.

REPRODUCIBILITY AND INVALIDATION (required by this task - not merely
documented, mechanically enforced): every artifact is paired with a
`fingerprint.json` recording:
  - the source SQLite database's path, file size, and mtime,
  - the exact sorted `row_indices` tuple,
  - `batch_size`, `n_buckets`, and `normalize_rows`.
Before serving ANY read, `ensure_materialized()` recomputes the current
fingerprint from the live SQLite store and compares it to the stored
one. On any mismatch - the database file changed, a different row set
or batch size or normalization flag is requested - the artifact is
discarded and rebuilt from the SQLite source of truth. The cache can
never go stale silently, because it is never trusted without this check.

NORMALIZATION: if `normalize_rows=True`, each batch is L2-normalized
(via `sklearn.preprocessing.normalize`, the same stateless, parameter-
free transform `incremental_model.py` already applies) BEFORE being
written to disk, so repeated reads skip re-normalizing too - this is
the "reusing normalized sparse batches" option from this task's Step 3.
This never changes the feature mapping or model behavior: a caller that
asks for `normalize_rows=True` gets exactly what
`evaluate_streaming_hashed_matrix(..., normalize_rows=True)` would have
computed on the fly.
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


@dataclass
class PersistentCacheStats:
    n_batches_read_from_artifact: int = 0  # served from disk, no SQL query
    n_rebuilds: int = 0  # times the SQLite source had to be re-scanned (initial build or invalidation)
    artifact_bytes: int = 0
    last_validation_passed: bool | None = None
    warnings: list[str] = field(default_factory=list)


class PersistentBatchCache:
    def __init__(
        self,
        cache_dir: Path,
        row_store: SqliteRowStore,
        row_indices: set[int],
        batch_size: int,
        normalize_rows: bool = False,
        min_free_disk_bytes: int = 1_000_000_000,
    ):
        self.cache_dir = Path(cache_dir)
        self._store = row_store
        self.row_indices = tuple(sorted(int(i) for i in row_indices))
        self.batch_size = batch_size
        self.normalize_rows = normalize_rows
        self.min_free_disk_bytes = min_free_disk_bytes
        self.fingerprint_path = self.cache_dir / "fingerprint.json"
        self.meta_path = self.cache_dir / "meta.json"
        self.stats = PersistentCacheStats()

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

    def _stored_fingerprint(self) -> dict | None:
        if not self.fingerprint_path.exists():
            return None
        try:
            return json.loads(self.fingerprint_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def is_valid(self) -> bool:
        stored = self._stored_fingerprint()
        valid = stored is not None and stored == self._compute_fingerprint() and self.meta_path.exists()
        self.stats.last_validation_passed = valid
        return valid

    def ensure_materialized(self) -> None:
        """Rebuilds the artifact from the SQLite source IFF the current
        fingerprint doesn't match the stored one (or no artifact exists
        yet). A no-op, cheap check on every call after the first valid
        build."""
        if self.is_valid():
            return

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not check_disk_space(self.cache_dir / "placeholder", self.min_free_disk_bytes):
            self.stats.warnings.append("insufficient disk space to materialize persistent batch cache")
            raise RuntimeError("insufficient disk space to materialize persistent batch cache")

        batches_meta = []
        total_bytes = 0
        for batch_idx, (genome_ids, row_indices, matrix) in enumerate(
            self._store.read_rows_for_indices(set(self.row_indices), batch_size=self.batch_size)
        ):
            if self.normalize_rows:
                matrix = sk_normalize(matrix, norm="l2")
            npz_name = f"batch_{batch_idx:05d}.npz"
            sparse.save_npz(self.cache_dir / npz_name, matrix)
            file_bytes = (self.cache_dir / npz_name).stat().st_size
            total_bytes += file_bytes
            batches_meta.append({"npz_file": npz_name, "genome_ids": genome_ids, "row_indices": row_indices})

        self.meta_path.write_text(json.dumps({"batches": batches_meta}))
        self.fingerprint_path.write_text(json.dumps(self._compute_fingerprint()))
        self.stats.n_rebuilds += 1
        self.stats.artifact_bytes = total_bytes + self.meta_path.stat().st_size + self.fingerprint_path.stat().st_size
        self.stats.last_validation_passed = True

    def read_rows_for_indices(self, wanted_row_indices: set[int], batch_size: int = 64):
        wanted_sorted = tuple(sorted(int(i) for i in wanted_row_indices))
        if wanted_sorted != self.row_indices or batch_size != self.batch_size:
            raise ValueError(
                "PersistentBatchCache is scoped to the exact (row_indices, batch_size) it was "
                "constructed with - got a different request. Construct a new instance for a "
                "different partition or batch size."
            )
        self.ensure_materialized()
        meta = json.loads(self.meta_path.read_text())
        for entry in meta["batches"]:
            matrix = sparse.load_npz(self.cache_dir / entry["npz_file"])
            self.stats.n_batches_read_from_artifact += 1
            yield entry["genome_ids"], entry["row_indices"], matrix


class MultiPartitionPersistentReader:
    """Thin adapter so `evaluate_streaming_hashed_matrix`'s two distinct
    calls per run (train partition, then test partition) each get
    routed to their own scoped `PersistentBatchCache`.
    `PersistentBatchCache` itself intentionally stays strict and
    single-partition (clearer to test); this adapter is what makes it
    usable as a genuine drop-in `row_store` for a real end-to-end
    training/evaluation call."""

    def __init__(self, caches: list[PersistentBatchCache]):
        self._by_key = {(c.row_indices, c.batch_size): c for c in caches}

    def read_rows_for_indices(self, wanted_row_indices: set[int], batch_size: int = 64):
        key = (tuple(sorted(int(i) for i in wanted_row_indices)), batch_size)
        cache = self._by_key.get(key)
        if cache is None:
            raise ValueError("No PersistentBatchCache configured for this (row_indices, batch_size) combination")
        yield from cache.read_rows_for_indices(wanted_row_indices, batch_size)
