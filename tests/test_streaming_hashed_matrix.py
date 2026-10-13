import json

import pytest
from scipy import sparse

from bvbrc_recon.feature_hashing import hash_kmer_to_bucket
from bvbrc_recon.kmer_features import canonical_kmer
from bvbrc_recon.streaming_hashed_matrix import (
    DiskBackedRowStore,
    FailureLog,
    ResourceLimits,
    StreamingRunStatus,
    build_streaming_hashed_matrix,
    hash_genome_to_bucket_counts,
)


# --- direct hashing: canonicalization, ambiguous bases, contig boundaries --


def test_hash_genome_to_bucket_counts_matches_manual_computation():
    contigs = ["ACGTACGT"]
    k, n_buckets = 4, 100
    result = hash_genome_to_bucket_counts(contigs, k, n_buckets)
    expected = {}
    for i in range(len(contigs[0]) - k + 1):
        bucket = hash_kmer_to_bucket(canonical_kmer(contigs[0][i : i + k]), n_buckets)
        expected[bucket] = expected.get(bucket, 0) + 1
    assert result == expected


def test_hash_genome_to_bucket_counts_is_deterministic():
    contigs = ["ACGTACGTACGT"]
    r1 = hash_genome_to_bucket_counts(contigs, k=4, n_buckets=1000)
    r2 = hash_genome_to_bucket_counts(contigs, k=4, n_buckets=1000)
    assert r1 == r2


def test_hash_genome_to_bucket_counts_skips_ambiguous_windows():
    # k=4: "ACGTNACGT" -> windows ACGT, CGTN(skip), GTNA(skip), TNAC(skip), NACG(skip), ACGT
    result = hash_genome_to_bucket_counts(["ACGTNACGT"], k=4, n_buckets=1000)
    assert sum(result.values()) == 2


def test_hash_genome_to_bucket_counts_never_spans_contig_boundary():
    # if "AAAA"+"TTTT" were concatenated, k=4 would see "AATT" at the join
    result_separate = hash_genome_to_bucket_counts(["AAAA", "TTTT"], k=4, n_buckets=10000)
    boundary_bucket = hash_kmer_to_bucket(canonical_kmer("AATT"), 10000)
    assert result_separate.get(boundary_bucket, 0) == 0


def test_hash_genome_to_bucket_counts_bounded_by_n_buckets():
    result = hash_genome_to_bucket_counts(["ACGT" * 50], k=4, n_buckets=10)
    assert all(0 <= b < 10 for b in result)
    assert len(result) <= 10


def test_hash_genome_to_bucket_counts_short_contig_returns_empty():
    assert hash_genome_to_bucket_counts(["AC"], k=4, n_buckets=100) == {}


# --- FailureLog -------------------------------------------------------


def test_failure_log_records_and_counts(tmp_path):
    log = FailureLog(tmp_path / "failures.jsonl")
    assert log.count() == 0
    log.append("g1", "ValueError: boom")
    log.append("g2", "KeyError: x")
    assert log.count() == 2
    assert log.genome_ids() == {"g1", "g2"}


def test_failure_log_empty_file_is_zero(tmp_path):
    log = FailureLog(tmp_path / "nonexistent.jsonl")
    assert log.count() == 0
    assert log.genome_ids() == set()


# --- DiskBackedRowStore -------------------------------------------------


def test_row_store_append_and_read_back(tmp_path):
    store = DiskBackedRowStore(tmp_path / "rows.jsonl", n_buckets=10)
    store.append_row("g1", 0, {2: 5, 7: 1})
    store.append_row("g2", 1, {3: 2})

    batches = list(store.iter_batches(batch_size=10))
    assert len(batches) == 1
    genome_ids, row_indices, matrix = batches[0]
    assert genome_ids == ["g1", "g2"]
    assert row_indices == [0, 1]
    assert sparse.issparse(matrix)
    assert matrix.shape == (2, 10)
    assert matrix[0, 2] == 5
    assert matrix[0, 7] == 1
    assert matrix[1, 3] == 2


def test_row_store_iter_batches_respects_batch_size(tmp_path):
    store = DiskBackedRowStore(tmp_path / "rows.jsonl", n_buckets=5)
    for i in range(7):
        store.append_row(f"g{i}", i, {0: 1})
    batches = list(store.iter_batches(batch_size=3))
    sizes = [m.shape[0] for _, _, m in batches]
    assert sizes == [3, 3, 1]


def test_row_store_existing_genome_ids_for_resumption(tmp_path):
    store = DiskBackedRowStore(tmp_path / "rows.jsonl", n_buckets=5)
    assert store.existing_genome_ids() == set()
    store.append_row("g1", 0, {0: 1})
    assert store.existing_genome_ids() == {"g1"}


def test_row_store_n_rows_written(tmp_path):
    store = DiskBackedRowStore(tmp_path / "rows.jsonl", n_buckets=5)
    store.append_row("g1", 0, {0: 1})
    store.append_row("g2", 1, {0: 1})
    assert store.n_rows_written() == 2


def test_row_store_read_rows_for_indices_filters(tmp_path):
    store = DiskBackedRowStore(tmp_path / "rows.jsonl", n_buckets=5)
    for i in range(5):
        store.append_row(f"g{i}", i, {0: i + 1})
    batches = list(store.read_rows_for_indices({1, 3}, batch_size=10))
    assert len(batches) == 1
    genome_ids, row_indices, matrix = batches[0]
    assert set(row_indices) == {1, 3}
    assert set(genome_ids) == {"g1", "g3"}


def test_row_store_placement_uses_row_index_not_append_order(tmp_path):
    """Rows appended OUT OF ORDER (as could happen across resumed runs)
    must still be attributable to their correct global row_index - the
    reader must never assume file order == row order."""
    store = DiskBackedRowStore(tmp_path / "rows.jsonl", n_buckets=5)
    store.append_row("g2", 2, {0: 99})  # appended first, but belongs at row_index=2
    store.append_row("g0", 0, {0: 11})
    store.append_row("g1", 1, {0: 22})

    genome_ids, row_indices, matrix = next(store.iter_batches(batch_size=10))
    row_index_to_value = {ri: matrix[local, 0] for local, ri in enumerate(row_indices)}
    assert row_index_to_value[0] == 11
    assert row_index_to_value[1] == 22
    assert row_index_to_value[2] == 99


# --- build_streaming_hashed_matrix: orchestration -------------------------


def _loader_factory(sequences: dict[str, list[str]]):
    def loader(genome_id):
        return sequences[genome_id]

    return loader


def test_build_processes_all_genomes(tmp_path):
    sequences = {"g1": ["ACGTACGT"], "g2": ["TTTTGGGG"], "g3": ["AAAACCCC"]}
    store = DiskBackedRowStore(tmp_path / "rows.jsonl", n_buckets=100)
    failures = FailureLog(tmp_path / "failures.jsonl")
    status = build_streaming_hashed_matrix(
        _loader_factory(sequences), ["g1", "g2", "g3"], k=4, n_buckets=100,
        row_store=store, failure_log=failures,
    )
    assert status.complete is True
    assert status.n_rows_written_this_run == 3
    assert store.existing_genome_ids() == {"g1", "g2", "g3"}


def test_build_resumption_skips_already_present_genomes(tmp_path):
    sequences = {"g1": ["ACGTACGT"], "g2": ["TTTTGGGG"]}
    store = DiskBackedRowStore(tmp_path / "rows.jsonl", n_buckets=100)
    failures = FailureLog(tmp_path / "failures.jsonl")
    store.append_row("g1", 0, {5: 1})  # simulate a prior partial run

    status = build_streaming_hashed_matrix(
        _loader_factory(sequences), ["g1", "g2"], k=4, n_buckets=100,
        row_store=store, failure_log=failures,
    )
    assert status.n_genomes_skipped_already_present == 1
    assert status.n_rows_written_this_run == 1  # only g2 newly written
    assert store.n_rows_written() == 2  # no duplicate for g1


def test_build_never_duplicates_a_row_on_rerun(tmp_path):
    sequences = {"g1": ["ACGTACGT"], "g2": ["TTTTGGGG"]}
    store = DiskBackedRowStore(tmp_path / "rows.jsonl", n_buckets=100)
    failures = FailureLog(tmp_path / "failures.jsonl")
    build_streaming_hashed_matrix(_loader_factory(sequences), ["g1", "g2"], k=4, n_buckets=100, row_store=store, failure_log=failures)
    build_streaming_hashed_matrix(_loader_factory(sequences), ["g1", "g2"], k=4, n_buckets=100, row_store=store, failure_log=failures)
    assert store.n_rows_written() == 2


def test_build_failure_is_logged_not_written_as_zero_row(tmp_path):
    def loader(genome_id):
        if genome_id == "g2":
            raise ValueError("simulated extraction failure")
        return ["ACGTACGT"]

    store = DiskBackedRowStore(tmp_path / "rows.jsonl", n_buckets=100)
    failures = FailureLog(tmp_path / "failures.jsonl")
    status = build_streaming_hashed_matrix(
        loader, ["g1", "g2", "g3"], k=4, n_buckets=100, row_store=store, failure_log=failures,
        limits=ResourceLimits(max_failed_rows=5),
    )
    assert "g2" not in store.existing_genome_ids()
    assert "g2" in failures.genome_ids()
    assert status.n_failures_this_run == 1
    assert status.n_rows_written_this_run == 2  # g1 and g3 still succeeded


def test_build_aborts_after_exceeding_max_failed_rows(tmp_path):
    def always_fails(genome_id):
        raise RuntimeError("fail")

    store = DiskBackedRowStore(tmp_path / "rows.jsonl", n_buckets=100)
    failures = FailureLog(tmp_path / "failures.jsonl")
    status = build_streaming_hashed_matrix(
        always_fails, ["g1", "g2", "g3", "g4"], k=4, n_buckets=100,
        row_store=store, failure_log=failures, limits=ResourceLimits(max_failed_rows=2),
    )
    assert status.aborted_reason is not None
    assert "max_failed_rows" in status.aborted_reason
    assert status.complete is False


def test_build_aborts_on_memory_limit(tmp_path, monkeypatch):
    sequences = {f"g{i}": ["ACGTACGT"] for i in range(5)}
    store = DiskBackedRowStore(tmp_path / "rows.jsonl", n_buckets=100)
    failures = FailureLog(tmp_path / "failures.jsonl")

    calls = {"n": 0}

    def fake_memory():
        calls["n"] += 1
        return 0.0 if calls["n"] < 2 else 999999.0

    monkeypatch.setattr("bvbrc_recon.streaming_hashed_matrix.peak_memory_mb", fake_memory)
    status = build_streaming_hashed_matrix(
        _loader_factory(sequences), list(sequences), k=4, n_buckets=100,
        row_store=store, failure_log=failures, limits=ResourceLimits(memory_limit_mb=1000.0),
    )
    assert status.aborted_reason is not None
    assert "peak RSS" in status.aborted_reason
    assert status.complete is False


def test_build_aborts_on_insufficient_disk_space(tmp_path, monkeypatch):
    sequences = {"g1": ["ACGTACGT"], "g2": ["TTTTGGGG"]}
    store = DiskBackedRowStore(tmp_path / "rows.jsonl", n_buckets=100)
    failures = FailureLog(tmp_path / "failures.jsonl")

    monkeypatch.setattr("bvbrc_recon.streaming_hashed_matrix.check_disk_space", lambda path, min_bytes: False)
    status = build_streaming_hashed_matrix(
        _loader_factory(sequences), ["g1", "g2"], k=4, n_buckets=100,
        row_store=store, failure_log=failures,
    )
    assert status.aborted_reason is not None
    assert "disk space" in status.aborted_reason
    assert status.n_rows_written_this_run == 0


def test_build_calls_loader_exactly_once_per_genome(tmp_path):
    sequences = {"g1": ["ACGTACGT"], "g2": ["TTTTGGGG"]}
    call_counts = {"g1": 0, "g2": 0}

    def counting_loader(genome_id):
        call_counts[genome_id] += 1
        return sequences[genome_id]

    store = DiskBackedRowStore(tmp_path / "rows.jsonl", n_buckets=100)
    failures = FailureLog(tmp_path / "failures.jsonl")
    build_streaming_hashed_matrix(counting_loader, ["g1", "g2"], k=4, n_buckets=100, row_store=store, failure_log=failures)
    assert call_counts == {"g1": 1, "g2": 1}


def test_build_progress_callback_invoked_per_genome(tmp_path):
    sequences = {"g1": ["ACGTACGT"], "g2": ["TTTTGGGG"]}
    calls = []
    store = DiskBackedRowStore(tmp_path / "rows.jsonl", n_buckets=100)
    failures = FailureLog(tmp_path / "failures.jsonl")
    build_streaming_hashed_matrix(
        _loader_factory(sequences), ["g1", "g2"], k=4, n_buckets=100,
        row_store=store, failure_log=failures,
        progress_callback=lambda *args: calls.append(args),
    )
    assert len(calls) == 2
