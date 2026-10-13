"""Genuinely out-of-core, disk-backed, fixed-width hashed feature matrix.

Architecturally distinct from `feature_hashing.py` (which transforms an
ALREADY-BUILT raw k-mer matrix) and from `streaming_kmer_build.py`
(whose vocabulary grows unboundedly with genome count): this module
computes `hash_kmer_to_bucket(canonical_kmer(window), n_buckets)`
DIRECTLY during the per-genome sequence walk, accumulating counts in a
dict bounded by `n_buckets` (never a string-keyed, ever-growing raw
vocabulary). Each genome's row is written to disk immediately and
discarded from memory before the next genome starts - peak memory is
therefore bounded by ONE genome's own sequence + its (<=n_buckets)
bucket-count dict, not by genome count or total raw k-mer diversity.

Reuses, unmodified: `kmer_features.canonical_kmer` (canonicalization),
`feature_hashing.hash_kmer_to_bucket` (deterministic hashlib-based
hashing - the SAME hash used by the in-memory hashing pipeline, so
collision behavior is identical and already characterized there).

DISK FORMAT (documented, not a standard): one JSON line per genome in
an append-only `.jsonl` file:
    {"genome_id": "...", "row_index": N, "cols": [...], "values": [...]}
`row_index` is the genome's FIXED position in the overall genome_order
(not file-append order) - this is what makes resumption safe: a row's
place in the final matrix never depends on when it was written, and a
reader reconstructs matrices from `row_index`, not line order.

Resumption: before processing a genome, its id is checked against every
`genome_id` already present in the row file; if found, it is skipped
(never reprocessed or re-appended) - this prevents duplicate rows
without requiring a separate journal, at the cost of an O(rows written
so far) scan per resume (acceptable at pilot scale; documented as a
chosen simplicity-over-throughput tradeoff, not hidden).

Failures: a genome whose extraction raises is recorded in a separate
failure log with its error, and is NEVER given an all-zero row - an
all-zero row is indistinguishable from "no signal detected" and would
silently corrupt the dataset. A run aborts once `max_failed_rows` is
exceeded.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import sparse

from .feature_hashing import hash_kmer_to_bucket
from .kmer_features import canonical_kmer
from .streaming_kmer_build import peak_memory_mb

VALID_BASES = set("ACGT")
DEFAULT_N_BUCKETS = 2**18


def hash_genome_to_bucket_counts(contigs: list[str], k: int, n_buckets: int) -> dict[int, int]:
    """Direct hash-during-extraction: canonicalizes each valid k-mer
    window and immediately hashes it to a bucket - no raw vocabulary is
    ever built. Contigs are processed independently (never spanning a
    boundary); windows containing an ambiguous base are skipped
    entirely, exactly as everywhere else in this project."""
    counts: dict[int, int] = {}
    for contig in contigs:
        seq = contig.upper()
        n = len(seq)
        for i in range(n - k + 1):
            window = seq[i : i + k]
            if not VALID_BASES.issuperset(window):
                continue
            bucket = hash_kmer_to_bucket(canonical_kmer(window), n_buckets)
            counts[bucket] = counts.get(bucket, 0) + 1
    return counts


@dataclass
class ResourceLimits:
    memory_limit_mb: float = 11_000.0
    min_free_disk_bytes: int = 1_000_000_000  # 1 GB headroom, checked before and during the run
    max_failed_rows: int = 5
    batch_size: int = 16


class FailureLog:
    """Append-only JSONL of {genome_id, reason} - never an all-zero row."""

    def __init__(self, path: Path):
        self.path = path

    def append(self, genome_id: str, reason: str) -> None:
        with open(self.path, "a") as fh:
            fh.write(json.dumps({"genome_id": genome_id, "reason": reason}) + "\n")

    def count(self) -> int:
        if not self.path.exists():
            return 0
        with open(self.path) as fh:
            return sum(1 for _ in fh)

    def genome_ids(self) -> set[str]:
        if not self.path.exists():
            return set()
        with open(self.path) as fh:
            return {json.loads(line)["genome_id"] for line in fh if line.strip()}


class DiskBackedRowStore:
    """Append-only JSONL sparse-row store. Never loads the full file
    into one in-memory matrix - `iter_batches` yields one small CSR
    chunk at a time."""

    def __init__(self, path: Path, n_buckets: int):
        self.path = path
        self.n_buckets = n_buckets

    def existing_genome_ids(self) -> set[str]:
        if not self.path.exists():
            return set()
        ids: set[str] = set()
        with open(self.path) as fh:
            for line in fh:
                if line.strip():
                    ids.add(json.loads(line)["genome_id"])
        return ids

    def n_rows_written(self) -> int:
        if not self.path.exists():
            return 0
        with open(self.path) as fh:
            return sum(1 for line in fh if line.strip())

    def append_row(self, genome_id: str, row_index: int, bucket_counts: dict[int, int]) -> None:
        record = {
            "genome_id": genome_id,
            "row_index": row_index,
            "cols": list(bucket_counts.keys()),
            "values": list(bucket_counts.values()),
        }
        with open(self.path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    def iter_batches(self, batch_size: int):
        """Yields (genome_ids: list[str], row_indices: list[int], csr_batch)
        one bounded-size chunk at a time - the file is read line-by-line,
        never loaded whole, and each yielded batch is discarded by the
        caller before the next is produced."""
        if not self.path.exists():
            return
        buffer: list[dict] = []
        with open(self.path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                buffer.append(json.loads(line))
                if len(buffer) >= batch_size:
                    yield self._batch_to_csr(buffer)
                    buffer = []
        if buffer:
            yield self._batch_to_csr(buffer)

    def read_rows_for_indices(self, wanted_row_indices: set[int], batch_size: int = 64):
        """Reads only the rows whose `row_index` is in `wanted_row_indices`
        (e.g. one split partition), still batch-by-batch - used so
        evaluating a specific split's rows never requires loading every
        row in the file, only scanning it."""
        if not self.path.exists():
            return
        buffer: list[dict] = []
        with open(self.path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["row_index"] not in wanted_row_indices:
                    continue
                buffer.append(record)
                if len(buffer) >= batch_size:
                    yield self._batch_to_csr(buffer)
                    buffer = []
        if buffer:
            yield self._batch_to_csr(buffer)

    def _batch_to_csr(self, records: list[dict]):
        genome_ids = [r["genome_id"] for r in records]
        row_indices = [r["row_index"] for r in records]
        rows, cols, data = [], [], []
        for local_row, record in enumerate(records):
            for col, val in zip(record["cols"], record["values"]):
                rows.append(local_row)
                cols.append(col)
                data.append(val)
        matrix = sparse.coo_matrix(
            (data, (rows, cols)), shape=(len(records), self.n_buckets), dtype=np.int64
        ).tocsr()
        return genome_ids, row_indices, matrix


def check_disk_space(path: Path, min_free_bytes: int) -> bool:
    usage = shutil.disk_usage(path.parent if path.parent.exists() else Path("."))
    return usage.free >= min_free_bytes


@dataclass
class StreamingRunStatus:
    n_genomes_requested: int = 0
    n_rows_written_this_run: int = 0
    n_rows_total: int = 0
    n_genomes_skipped_already_present: int = 0
    n_failures_this_run: int = 0
    n_failures_total: int = 0
    aborted_reason: str | None = None
    peak_rss_mb: float = 0.0
    elapsed_seconds: float = 0.0
    complete: bool = False


def build_streaming_hashed_matrix(
    contigs_loader,
    genome_order: list[str],
    k: int,
    n_buckets: int,
    row_store: DiskBackedRowStore,
    failure_log: FailureLog,
    limits: ResourceLimits = ResourceLimits(),
    progress_callback=None,
) -> StreamingRunStatus:
    """`contigs_loader(genome_id) -> list[str]` - a genome's contigs are
    loaded, hashed, written, and discarded ONE AT A TIME; the full set
    of genomes' sequences is never held in memory simultaneously (unlike
    every prior pipeline in this project, which loaded all genomes'
    contigs into one dict upfront)."""
    status = StreamingRunStatus(n_genomes_requested=len(genome_order))
    t0 = time.perf_counter()

    already_present = row_store.existing_genome_ids()
    already_failed = failure_log.genome_ids()

    if not check_disk_space(row_store.path, limits.min_free_disk_bytes):
        status.aborted_reason = "insufficient disk space before starting"
        status.elapsed_seconds = time.perf_counter() - t0
        return status

    for row_index, genome_id in enumerate(genome_order):
        if genome_id in already_present:
            status.n_genomes_skipped_already_present += 1
            continue
        if genome_id in already_failed:
            # already recorded as a failure in a prior run - do not retry
            # silently or duplicate the failure entry; surfaced via the count.
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
