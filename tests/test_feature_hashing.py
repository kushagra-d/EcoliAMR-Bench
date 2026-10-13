import numpy as np
import pytest
from scipy import sparse

from bvbrc_recon.feature_hashing import (
    build_hash_mapping,
    build_hashed_representation,
    compute_collision_diagnostics,
    hash_kmer_to_bucket,
    hash_matrix,
)


# --- determinism -----------------------------------------------------------


def test_hash_kmer_to_bucket_is_deterministic_within_a_process():
    assert hash_kmer_to_bucket("ACGTACGTACGTACG", 1000) == hash_kmer_to_bucket("ACGTACGTACGTACG", 1000)


def test_hash_kmer_to_bucket_deterministic_across_subprocess():
    """Python's built-in hash() is randomized per process by PYTHONHASHSEED
    unless disabled - this must NOT rely on that. Verified by running the
    hash in a genuinely separate interpreter process (default, randomized
    PYTHONHASHSEED inherited) and comparing across two independent runs."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    code = (
        "from bvbrc_recon.feature_hashing import hash_kmer_to_bucket; "
        "print(hash_kmer_to_bucket('ACGTACGTACGTACG', 1000))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "src")
    env.pop("PYTHONHASHSEED", None)  # ensure each subprocess picks its own random seed

    results = set()
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stderr
        results.add(proc.stdout.strip())
    assert len(results) == 1, f"hash varied across subprocess runs: {results}"


def test_hash_kmer_to_bucket_is_always_within_range():
    for kmer in ["AAAAAAAAAAAAAAA", "TTTTTTTTTTTTTTT", "ACGTACGTACGTACG"]:
        for n_buckets in (16, 65536, 1_048_576):
            bucket = hash_kmer_to_bucket(kmer, n_buckets)
            assert 0 <= bucket < n_buckets


def test_different_kmers_usually_hash_differently():
    kmers = [f"ACGT{'A' * 11}"[:15].replace("A", "ACGT"[i % 4], 1) for i in range(20)]
    buckets = {hash_kmer_to_bucket(k, 1_048_576) for k in set(kmers)}
    assert len(buckets) > 1  # not a degenerate constant hash


# --- build_hash_mapping / fixed output dimensions --------------------------


def test_build_hash_mapping_covers_every_column():
    vocabulary = {"AAAAAAAAAAAAAAA": 0, "TTTTTTTTTTTTTTT": 1, "ACGTACGTACGTACG": 2}
    mapping = build_hash_mapping(vocabulary, n_buckets=1000)
    assert len(mapping) == 3
    assert all(0 <= b < 1000 for b in mapping)


@pytest.mark.parametrize("n_buckets", [2**16, 2**18, 2**20])
def test_build_hash_mapping_respects_fixed_dimension(n_buckets):
    vocabulary = {f"kmer{i}" * 2: i for i in range(50)}
    mapping = build_hash_mapping(vocabulary, n_buckets=n_buckets)
    assert mapping.max() < n_buckets


# --- collision diagnostics --------------------------------------------------


def test_collision_diagnostics_detects_forced_collision():
    # 4 raw columns hashing to only 2 buckets -> at least one collision guaranteed
    bucket_of_column = np.array([0, 0, 1, 1])
    diag = compute_collision_diagnostics(bucket_of_column, n_buckets=2)
    assert diag.n_colliding_buckets == 2
    assert diag.n_lonely_buckets == 0
    assert diag.n_raw_features_in_a_collision == 4
    assert diag.max_distinct_kmers_in_one_bucket == 2


def test_collision_diagnostics_no_collision_case():
    bucket_of_column = np.array([0, 1, 2])
    diag = compute_collision_diagnostics(bucket_of_column, n_buckets=5)
    assert diag.n_colliding_buckets == 0
    assert diag.n_lonely_buckets == 3
    assert diag.n_empty_buckets == 2
    assert diag.n_raw_features_in_a_collision == 0


# --- hash_matrix: sparse preservation, summing, row preservation ----------


def test_hash_matrix_rejects_dense_input():
    with pytest.raises(TypeError):
        hash_matrix(np.zeros((2, 3)), np.array([0, 1, 0]), n_buckets=2)


def test_hash_matrix_rejects_mismatched_vocabulary_length():
    raw = sparse.csr_matrix(np.eye(3))
    with pytest.raises(ValueError):
        hash_matrix(raw, np.array([0, 1]), n_buckets=2)


def test_hash_matrix_sums_colliding_columns():
    # 2 genomes x 3 raw columns; columns 0 and 2 collide into bucket 0
    raw = sparse.csr_matrix(np.array([[5, 1, 3], [0, 2, 4]]))
    bucket_of_column = np.array([0, 1, 0])
    hashed = hash_matrix(raw, bucket_of_column, n_buckets=2)
    assert hashed.shape == (2, 2)
    assert hashed[0, 0] == 8  # 5 + 3
    assert hashed[0, 1] == 1
    assert hashed[1, 0] == 4
    assert hashed[1, 1] == 2


def test_hash_matrix_preserves_row_count_and_order():
    raw = sparse.csr_matrix(np.array([[1, 0], [0, 1], [1, 1]]))
    bucket_of_column = np.array([0, 1])
    hashed = hash_matrix(raw, bucket_of_column, n_buckets=4)
    assert hashed.shape[0] == 3  # same genome count
    assert hashed[0].toarray().tolist() == [[1, 0, 0, 0]]
    assert hashed[2].toarray().tolist() == [[1, 1, 0, 0]]


def test_hash_matrix_output_stays_sparse():
    raw = sparse.csr_matrix(np.eye(5))
    bucket_of_column = np.array([0, 1, 0, 1, 2])
    hashed = hash_matrix(raw, bucket_of_column, n_buckets=3)
    assert sparse.issparse(hashed)


def test_hash_matrix_never_calls_toarray_internally(monkeypatch):
    calls = {"n": 0}
    original = sparse.csr_matrix.toarray

    def tracking(self, *a, **kw):
        calls["n"] += 1
        return original(self, *a, **kw)

    monkeypatch.setattr(sparse.csr_matrix, "toarray", tracking)
    raw = sparse.csr_matrix(np.eye(5))
    hash_matrix(raw, np.array([0, 1, 0, 1, 2]), n_buckets=3)
    assert calls["n"] == 0


# --- build_hashed_representation: end-to-end ------------------------------


def test_build_hashed_representation_end_to_end():
    raw = sparse.csr_matrix(np.array([[5, 1, 3], [0, 2, 4]]))
    vocabulary = {"AAAAAAAAAAAAAAA": 0, "TTTTTTTTTTTTTTT": 1, "ACGTACGTACGTACG": 2}
    result = build_hashed_representation(raw, vocabulary, n_buckets=1000)
    assert result.matrix.shape == (2, 1000)
    assert sparse.issparse(result.matrix)
    assert result.collisions.n_raw_features == 3
    # total count preserved (no data lost, only columns folded)
    assert result.matrix.sum() == raw.sum()
