"""Canonical k-mer feature extraction and sparse matrix construction.

k-mer size note (not resolved here, by design): the submitted proposal
mentions k is approximately 10, while instructor feedback separately
raised a concern about k=31. This module makes k a configurable
parameter rather than silently picking one value and hiding the
disagreement; the pilot's chosen value and the open question are
recorded explicitly in the pilot report, not decided by this code.

Design choices, stated rather than assumed:
  - Canonical k-mers: a k-mer and its reverse complement are the same
    feature (the lexicographically smaller of the two strings is used
    as the canonical form).
  - A k-mer window containing any non-ACGT character (ambiguous base)
    is skipped entirely - never counted under either orientation.
  - Contigs are processed independently; a k-mer is never built across
    a contig boundary (this falls out naturally from calling
    `count_kmers_in_contig` once per contig and never concatenating
    contigs before counting).
  - Output is always a SciPy sparse matrix (CSR). Nothing here ever
    calls `.toarray()`/`.todense()` on the full matrix.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

COMPLEMENT = str.maketrans("ACGT", "TGCA")
VALID_BASES = set("ACGT")


def reverse_complement(kmer: str) -> str:
    return kmer.translate(COMPLEMENT)[::-1]


def canonical_kmer(kmer: str) -> str:
    rc = reverse_complement(kmer)
    return kmer if kmer <= rc else rc


def count_kmers_in_contig(sequence: str, k: int) -> Counter:
    """Counts canonical k-mers within ONE contig only. A window containing
    any character outside {A,C,G,T} (case-insensitive) is skipped - it is
    never counted, under either orientation."""
    counts: Counter = Counter()
    sequence = sequence.upper()
    n = len(sequence)
    if n < k:
        return counts
    for i in range(n - k + 1):
        window = sequence[i : i + k]
        if not VALID_BASES.issuperset(window):
            continue
        counts[canonical_kmer(window)] += 1
    return counts


def count_kmers_in_assembly(contig_sequences: list[str], k: int) -> Counter:
    """Sums per-contig canonical k-mer counts. Each contig is counted
    independently via `count_kmers_in_contig` - no two contigs are ever
    concatenated before counting, so no k-mer spans a contig boundary."""
    total: Counter = Counter()
    for contig_seq in contig_sequences:
        total.update(count_kmers_in_contig(contig_seq, k))
    return total


@dataclass
class KmerMatrixResult:
    k: int
    genome_order: list[str] = field(default_factory=list)  # row index -> genome_id, in order
    vocabulary: dict[str, int] = field(default_factory=dict)  # canonical kmer -> column index
    matrix: sparse.csr_matrix = None
    n_genomes: int = 0
    n_features: int = 0
    nnz: int = 0
    estimated_dense_bytes: int = 0


def estimate_dense_bytes(n_genomes: int, n_features: int, itemsize: int = 8) -> int:
    return n_genomes * n_features * itemsize


def build_feature_matrix(
    genome_kmer_counts: dict[str, Counter],
    k: int,
    genome_order: list[str] | None = None,
) -> KmerMatrixResult:
    """Builds a genomes x canonical-k-mers sparse CSR matrix directly from
    COO-style (row, col, value) triples - never via a dense intermediate.
    `genome_order` fixes the row order explicitly (defaults to dict
    insertion order) so the genome_id<->row mapping is never ambiguous.
    """
    order = genome_order if genome_order is not None else list(genome_kmer_counts.keys())
    if set(order) != set(genome_kmer_counts.keys()):
        raise ValueError("genome_order must contain exactly the same genome_ids as genome_kmer_counts")

    vocabulary: dict[str, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []

    for row_idx, genome_id in enumerate(order):
        for kmer, count in genome_kmer_counts[genome_id].items():
            col_idx = vocabulary.setdefault(kmer, len(vocabulary))
            rows.append(row_idx)
            cols.append(col_idx)
            data.append(count)

    n_genomes = len(order)
    n_features = len(vocabulary)
    matrix = sparse.coo_matrix(
        (data, (rows, cols)), shape=(n_genomes, n_features), dtype=np.int64
    ).tocsr()

    return KmerMatrixResult(
        k=k,
        genome_order=order,
        vocabulary=vocabulary,
        matrix=matrix,
        n_genomes=n_genomes,
        n_features=n_features,
        nnz=matrix.nnz,
        estimated_dense_bytes=estimate_dense_bytes(n_genomes, n_features),
    )
