"""Memory-bounded, one-genome-at-a-time canonical k-mer feature-matrix
builder.

Exists because of an empirically discovered failure mode (not an assumed
one): a naive `{genome_id: count_kmers_in_assembly(...) for ...}` dict
comprehension holds every genome's full k-mer count table in memory
simultaneously before merging - fine at small k (where the shared
vocabulary is small and bounded), but at large k (e.g. k=31, where
genomes share little and each contributes close to its own full window
count) this OOM-killed a 100-genome run on a 17GB machine. This module
instead folds one genome's counts into the shared accumulator at a time,
discarding that genome's raw count table before moving to the next, and
aborts cleanly above a memory threshold rather than risking an
uncontrolled OS OOM-kill.

Reuses the same canonical/ambiguous-base/contig-boundary rules as
`kmer_features.py` (`canonical_kmer`, the ACGT validity check) - never a
different extraction rule for the "large k" path.
"""

from __future__ import annotations

import resource
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from .kmer_features import canonical_kmer

VALID_BASES = set("ACGT")
DEFAULT_MEMORY_LIMIT_MB = 11_000.0


def peak_memory_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


@dataclass
class StreamingBuildResult:
    matrix: sparse.csr_matrix
    vocabulary: dict[str, int]
    processed_genome_ids: list[str] = field(default_factory=list)
    n_requested_genomes: int = 0
    complete: bool = False
    aborted_for_memory: bool = False


def build_matrix_streaming(
    contigs_by_genome: dict[str, list[str]],
    genome_order: list[str],
    k: int,
    memory_limit_mb: float = DEFAULT_MEMORY_LIMIT_MB,
    memory_check_fn=peak_memory_mb,
    progress_callback=None,
) -> StreamingBuildResult:
    """Processes `genome_order` in sequence; at most ONE genome's raw
    per-contig k-mer count table exists at a time. Stops early (and
    reports `aborted_for_memory=True`, `complete=False`) if
    `memory_check_fn()` exceeds `memory_limit_mb` after any genome -
    never silently returns a partial matrix labeled as complete.
    `progress_callback(row_idx, genome_id, vocab_size, nnz_so_far, mem_mb)`
    is called after every genome, if provided (for logging).
    """
    vocabulary: dict[str, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []
    processed_genome_ids: list[str] = []
    aborted_for_memory = False

    for row_idx, genome_id in enumerate(genome_order):
        sequence_counts: dict[str, int] = {}
        for contig in contigs_by_genome[genome_id]:
            seq = contig.upper()
            n = len(seq)
            for i in range(n - k + 1):
                window = seq[i : i + k]
                if not VALID_BASES.issuperset(window):
                    continue
                key = canonical_kmer(window)
                sequence_counts[key] = sequence_counts.get(key, 0) + 1

        for kmer, count in sequence_counts.items():
            col_idx = vocabulary.setdefault(kmer, len(vocabulary))
            rows.append(row_idx)
            cols.append(col_idx)
            data.append(count)
        processed_genome_ids.append(genome_id)
        del sequence_counts  # this genome's raw counts are gone before the next one starts

        mem_mb = memory_check_fn()
        if progress_callback is not None:
            progress_callback(row_idx, genome_id, len(vocabulary), len(data), mem_mb)
        if mem_mb > memory_limit_mb:
            aborted_for_memory = True
            break

    n_processed = len(processed_genome_ids)
    matrix = sparse.coo_matrix(
        (data, (rows, cols)), shape=(n_processed, len(vocabulary)), dtype=np.int64
    ).tocsr()

    return StreamingBuildResult(
        matrix=matrix,
        vocabulary=vocabulary,
        processed_genome_ids=processed_genome_ids,
        n_requested_genomes=len(genome_order),
        complete=(not aborted_for_memory) and n_processed == len(genome_order),
        aborted_for_memory=aborted_for_memory,
    )
