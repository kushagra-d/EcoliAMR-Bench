"""Deterministic feature hashing of an EXISTING canonical k-mer matrix
into a fixed, smaller number of columns ("the hashing trick").

Deliberately a separate, downstream transformation of an already-built
raw k-mer matrix (e.g. the existing k=15 pilot matrix), not a
reimplementation of k-mer counting: canonicalization, ambiguous-base
skipping, and contig-boundary handling are already correctly baked into
that raw matrix by `streaming_kmer_build.py` (already tested) - hashing
here only ever remaps already-canonical columns into buckets, by their
already-canonical string. This also means genome rows are never touched:
the genome-id-to-row mapping of the raw matrix carries over unchanged.

Hash function: `hashlib.blake2b` (NOT Python's built-in `hash()`, which
is randomized per-process by `PYTHONHASHSEED` and would make bucket
assignment non-reproducible across runs/processes). A k-mer's bucket is
`int.from_bytes(blake2b(kmer).digest(), "big") % n_buckets`.

Design choice, stated rather than assumed: bucket values are the SUM of
the raw counts of every k-mer that hashes into that bucket (no sign
flip). A signed variant (as in scikit-learn's `FeatureHasher`, which
multiplies each contribution by +1/-1 derived from the hash to reduce
collision bias for dot-product models) was NOT used here, to keep each
bucket's value a directly interpretable "total k-mer occurrences
collapsed into this bucket" count, consistent with every other k-mer
representation already used in this project. This is a judgment call,
not a measured fact about which is better for this task.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

DEFAULT_DIMENSIONS = [2**16, 2**18, 2**20]


def hash_kmer_to_bucket(kmer: str, n_buckets: int) -> int:
    digest = hashlib.blake2b(kmer.encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % n_buckets


def build_hash_mapping(vocabulary: dict[str, int], n_buckets: int) -> np.ndarray:
    """`bucket_of_column[original_column_index] = hashed_bucket_index`,
    one entry per column of the raw matrix this vocabulary describes."""
    bucket_of_column = np.empty(len(vocabulary), dtype=np.int64)
    for kmer, col in vocabulary.items():
        bucket_of_column[col] = hash_kmer_to_bucket(kmer, n_buckets)
    return bucket_of_column


@dataclass
class CollisionDiagnostics:
    n_buckets: int
    n_raw_features: int
    n_empty_buckets: int = 0
    n_lonely_buckets: int = 0  # exactly one distinct raw k-mer
    n_colliding_buckets: int = 0  # more than one distinct raw k-mer
    n_raw_features_in_a_collision: int = 0
    max_distinct_kmers_in_one_bucket: int = 0
    mean_distinct_kmers_per_occupied_bucket: float = 0.0


def compute_collision_diagnostics(bucket_of_column: np.ndarray, n_buckets: int) -> CollisionDiagnostics:
    """Measures, exactly (not estimated), how many DISTINCT raw canonical
    k-mers collapse into the same bucket - this is the hash-collision
    count. It is NOT the same thing as two genomes sharing a k-mer
    (ordinary, expected biological similarity, already represented
    correctly in the raw matrix before hashing) - collisions here refer
    only to distinct k-mer STRINGS losing their separate identity."""
    counts = np.bincount(bucket_of_column, minlength=n_buckets)
    occupied = counts[counts > 0]
    return CollisionDiagnostics(
        n_buckets=n_buckets,
        n_raw_features=len(bucket_of_column),
        n_empty_buckets=int((counts == 0).sum()),
        n_lonely_buckets=int((counts == 1).sum()),
        n_colliding_buckets=int((counts > 1).sum()),
        n_raw_features_in_a_collision=int(counts[counts > 1].sum()),
        max_distinct_kmers_in_one_bucket=int(counts.max()) if len(counts) else 0,
        mean_distinct_kmers_per_occupied_bucket=float(occupied.mean()) if len(occupied) else 0.0,
    )


def hash_matrix(raw_matrix: sparse.spmatrix, bucket_of_column: np.ndarray, n_buckets: int) -> sparse.csr_matrix:
    """Folds `raw_matrix`'s columns into `n_buckets` columns by summing
    the counts of every original column that hashes to the same bucket.
    Rows (genomes) are untouched - same count, same order. Never
    densifies: operates on the COO representation and relies on
    `tocsr()`'s documented behavior of summing duplicate (row, col)
    entries, rather than building an intermediate dense array."""
    if not sparse.issparse(raw_matrix):
        raise TypeError("raw_matrix must be a scipy sparse matrix - this never densifies")
    if raw_matrix.shape[1] != len(bucket_of_column):
        raise ValueError(
            f"raw_matrix has {raw_matrix.shape[1]} columns but bucket_of_column has "
            f"{len(bucket_of_column)} entries - these must correspond 1:1"
        )
    coo = raw_matrix.tocoo()
    new_cols = bucket_of_column[coo.col]
    hashed = sparse.coo_matrix(
        (coo.data, (coo.row, new_cols)), shape=(raw_matrix.shape[0], n_buckets)
    ).tocsr()
    return hashed


@dataclass
class HashedMatrixResult:
    matrix: sparse.csr_matrix
    n_buckets: int
    bucket_of_column: np.ndarray
    collisions: CollisionDiagnostics = field(default_factory=lambda: None)


def build_hashed_representation(
    raw_matrix: sparse.spmatrix, vocabulary: dict[str, int], n_buckets: int
) -> HashedMatrixResult:
    bucket_of_column = build_hash_mapping(vocabulary, n_buckets)
    collisions = compute_collision_diagnostics(bucket_of_column, n_buckets)
    hashed = hash_matrix(raw_matrix, bucket_of_column, n_buckets)
    return HashedMatrixResult(
        matrix=hashed, n_buckets=n_buckets, bucket_of_column=bucket_of_column, collisions=collisions
    )
