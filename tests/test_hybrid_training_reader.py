import os
import subprocess
import sys
from pathlib import Path

import pytest

from bvbrc_recon.hybrid_training_reader import HybridTrainingReader, MultiPartitionHybridReader
from bvbrc_recon.incremental_model import evaluate_streaming_hashed_matrix
from bvbrc_recon.pilot_model import NEGATIVE_CLASS, POSITIVE_CLASS
from bvbrc_recon.sqlite_row_store import SqliteRowStore

HELPER_SCRIPT = Path(__file__).parent / "_hybrid_interrupt_subprocess.py"


def _build_store(tmp_path, n_rows=10, n_buckets=20):
    store = SqliteRowStore(tmp_path / "rows.db", n_buckets=n_buckets)
    for i in range(n_rows):
        store.append_row(f"g{i}", i, {c: (i + 1) * (c + 1) for c in range(n_buckets)})
    return store


# --- exact sparse-value preservation ---


def test_hybrid_reader_matches_direct_store_exactly(tmp_path):
    store = _build_store(tmp_path)
    reader = HybridTrainingReader(tmp_path / "cache", store, row_indices=set(range(10)), batch_size=4)

    direct = list(store.read_rows_for_indices(set(range(10)), batch_size=4))
    via_hybrid = list(reader.read_rows_for_indices(set(range(10)), batch_size=4))

    assert len(direct) == len(via_hybrid)
    for (gd, rd, md), (gh, rh, mh) in zip(direct, via_hybrid):
        assert gd == gh
        assert rd == rh
        assert (md != mh).nnz == 0


# --- memory/disk/source tiers and counters ---


def test_three_tier_counters_behave_as_designed(tmp_path):
    store = _build_store(tmp_path, n_rows=20, n_buckets=20)
    reader = HybridTrainingReader(tmp_path / "cache", store, row_indices=set(range(20)), batch_size=4, max_memory_cache_bytes=10_000_000)

    list(reader.read_rows_for_indices(set(range(20)), batch_size=4))  # pass 1: builds + disk reads
    assert reader.stats.n_source_reads == 5  # 20/4 batches, all freshly built
    assert reader.stats.n_disk_hits == 5
    assert reader.stats.n_memory_hits == 0

    list(reader.read_rows_for_indices(set(range(20)), batch_size=4))  # pass 2: all from memory
    assert reader.stats.n_memory_hits == 5
    assert reader.stats.n_source_reads == 5  # unchanged - no re-fetch from SQLite
    assert reader.stats.n_disk_hits == 5  # unchanged - no re-decompression either


def test_memory_budget_enforcement_falls_back_to_disk_not_source(tmp_path):
    store = _build_store(tmp_path, n_rows=20, n_buckets=20)
    reader = HybridTrainingReader(tmp_path / "cache", store, row_indices=set(range(20)), batch_size=4, max_memory_cache_bytes=1)

    list(reader.read_rows_for_indices(set(range(20)), batch_size=4))
    assert reader.stats.memory_bytes_cached <= 1
    assert reader.stats.n_fallbacks > 0

    source_reads_after_pass1 = reader.stats.n_source_reads
    list(reader.read_rows_for_indices(set(range(20)), batch_size=4))  # pass 2
    assert reader.stats.n_source_reads == source_reads_after_pass1  # never re-touches SQLite
    assert reader.stats.n_disk_hits > 5  # pass 2 served entirely from disk, not memory


# --- deterministic batch ordering ---


def test_batch_ordering_is_stable_across_passes(tmp_path):
    store = _build_store(tmp_path, n_rows=20, n_buckets=20)
    reader = HybridTrainingReader(tmp_path / "cache", store, row_indices=set(range(20)), batch_size=4, max_memory_cache_bytes=10_000_000)

    order1 = [ri for _, ris, _ in reader.read_rows_for_indices(set(range(20)), batch_size=4) for ri in ris]
    order2 = [ri for _, ris, _ in reader.read_rows_for_indices(set(range(20)), batch_size=4) for ri in ris]
    assert order1 == order2 == list(range(20))


# --- strict scope enforcement ---


def test_wrong_partition_raises(tmp_path):
    store = _build_store(tmp_path, n_rows=10)
    reader = HybridTrainingReader(tmp_path / "cache", store, row_indices=set(range(5)), batch_size=4)
    with pytest.raises(ValueError):
        list(reader.read_rows_for_indices(set(range(5, 10)), batch_size=4))


# --- fingerprint validation / source invalidation ---


def test_disk_tier_invalidates_on_source_database_change(tmp_path):
    import time

    store = _build_store(tmp_path, n_rows=10)
    reader = HybridTrainingReader(tmp_path / "cache", store, row_indices=set(range(10)), batch_size=4)
    list(reader.read_rows_for_indices(set(range(10)), batch_size=4))
    assert reader.is_complete() is True

    time.sleep(0.01)
    store.append_row("g10", 10, {0: 1})
    assert reader.is_complete() is False

    reader2 = HybridTrainingReader(tmp_path / "cache", store, row_indices=set(range(10)), batch_size=4)
    list(reader2.read_rows_for_indices(set(range(10)), batch_size=4))
    assert reader2.stats.n_source_reads == 3  # rebuilt from scratch (config-level change)


def test_disk_tier_invalidates_on_different_row_indices(tmp_path):
    store = _build_store(tmp_path, n_rows=10)
    cache_dir = tmp_path / "cache"
    reader_a = HybridTrainingReader(cache_dir, store, row_indices=set(range(0, 5)), batch_size=4)
    list(reader_a.read_rows_for_indices(set(range(0, 5)), batch_size=4))

    reader_b = HybridTrainingReader(cache_dir, store, row_indices=set(range(5, 10)), batch_size=4)
    assert reader_b.is_complete() is False
    batches = list(reader_b.read_rows_for_indices(set(range(5, 10)), batch_size=4))
    seen = sorted(ri for _, ris, _ in batches for ri in ris)
    assert seen == list(range(5, 10))


# --- partial-build detection and recovery ---


def test_missing_batch_file_is_detected_and_selectively_rebuilt(tmp_path):
    store = _build_store(tmp_path, n_rows=20, n_buckets=20)
    reader = HybridTrainingReader(tmp_path / "cache", store, row_indices=set(range(20)), batch_size=4)
    list(reader.read_rows_for_indices(set(range(20)), batch_size=4))
    assert reader.is_complete() is True

    os.remove(tmp_path / "cache" / "batch_00002.npz")

    reader2 = HybridTrainingReader(tmp_path / "cache", store, row_indices=set(range(20)), batch_size=4)
    assert reader2.is_complete() is False
    batches = list(reader2.read_rows_for_indices(set(range(20)), batch_size=4))
    assert reader2.stats.n_source_reads == 1  # only the missing batch was rebuilt, not all 5
    direct = list(store.read_rows_for_indices(set(range(20)), batch_size=4))
    for (gd, rd, md), (gh, rh, mh) in zip(direct, batches):
        assert gd == gh and rd == rh
        assert (md != mh).nnz == 0


def test_corrupt_batch_file_is_detected_and_rebuilt(tmp_path):
    store = _build_store(tmp_path, n_rows=20, n_buckets=20)
    reader = HybridTrainingReader(tmp_path / "cache", store, row_indices=set(range(20)), batch_size=4)
    list(reader.read_rows_for_indices(set(range(20)), batch_size=4))

    (tmp_path / "cache" / "batch_00001.npz").write_bytes(b"not a real npz file")

    reader2 = HybridTrainingReader(tmp_path / "cache", store, row_indices=set(range(20)), batch_size=4)
    assert reader2.is_complete() is False
    list(reader2.read_rows_for_indices(set(range(20)), batch_size=4))
    assert reader2.stats.n_source_reads == 1
    assert reader2.is_complete() is True


def test_never_marks_incomplete_build_as_complete(tmp_path):
    """Directly exercises the build loop's own invariant: if the
    process is interrupted after `build_fingerprint.json` is written
    but before the final commit, `is_complete()` must stay False."""
    store = _build_store(tmp_path, n_rows=20, n_buckets=20)
    reader = HybridTrainingReader(tmp_path / "cache", store, row_indices=set(range(20)), batch_size=4)

    reader.cache_dir.mkdir(parents=True, exist_ok=True)
    reader._write_json_atomic(reader.build_fingerprint_path, reader._compute_fingerprint())
    reader._write_json_atomic(reader.meta_path, {"batches": []})
    # fingerprint.json (the commit marker) deliberately never written
    assert reader.is_complete() is False


def test_true_process_kill_mid_build_is_recovered(tmp_path):
    db_path = tmp_path / "rows.db"
    cache_dir = tmp_path / "cache"
    result = subprocess.run(
        [sys.executable, str(HELPER_SCRIPT), str(db_path), str(cache_dir), "20", "4", "2"],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 1  # the helper's deliberate os._exit(1)

    store = SqliteRowStore(db_path, n_buckets=20)
    reader = HybridTrainingReader(cache_dir, store, row_indices=set(range(20)), batch_size=4)
    assert reader.is_complete() is False  # 2 of 5 batches committed, build never finished

    batches = list(reader.read_rows_for_indices(set(range(20)), batch_size=4))
    assert reader.stats.n_source_reads == 3  # only the 3 NOT already committed were rebuilt
    assert reader.is_complete() is True

    direct = list(store.read_rows_for_indices(set(range(20)), batch_size=4))
    for (gd, rd, md), (gh, rh, mh) in zip(direct, batches):
        assert gd == gh and rd == rh
        assert (md != mh).nnz == 0


# --- stable model inputs and predictions ---


def test_hybrid_reader_gives_identical_training_results_to_direct_store(tmp_path):
    store = SqliteRowStore(tmp_path / "rows.db", n_buckets=10)
    labels = {}
    for i in range(10):
        pattern = {0: 10, 1: 1} if i % 2 == 0 else {2: 1, 3: 10}
        store.append_row(f"g{i}", i, pattern)
        labels[i] = POSITIVE_CLASS if i % 2 == 0 else NEGATIVE_CLASS

    train_idx = set(range(0, 8))
    test_idx = set(range(8, 10))

    direct_result = evaluate_streaming_hashed_matrix(store, train_idx, test_idx, labels, seed=1, batch_size=4, n_epochs=5)

    train_hybrid = HybridTrainingReader(tmp_path / "train_cache", store, train_idx, batch_size=4)
    test_hybrid = HybridTrainingReader(tmp_path / "test_cache", store, test_idx, batch_size=4)
    reader = MultiPartitionHybridReader([train_hybrid, test_hybrid])
    hybrid_result = evaluate_streaming_hashed_matrix(reader, train_idx, test_idx, labels, seed=1, batch_size=4, n_epochs=5)

    assert direct_result.accuracy == hybrid_result.accuracy
    assert direct_result.roc_auc == hybrid_result.roc_auc
    assert direct_result.confusion_matrix == hybrid_result.confusion_matrix
    assert train_hybrid.stats.n_source_reads == 2  # built once
    assert train_hybrid.stats.n_memory_hits == 2 * 4  # epochs 2-5, 2 batches each


def test_multi_partition_hybrid_reader_raises_for_unconfigured_partition(tmp_path):
    store = _build_store(tmp_path, n_rows=10)
    reader = HybridTrainingReader(tmp_path / "cache", store, row_indices=set(range(5)), batch_size=4)
    multi = MultiPartitionHybridReader([reader])
    with pytest.raises(ValueError):
        list(multi.read_rows_for_indices(set(range(5, 10)), batch_size=4))


# --- artifact isolation ---


def test_hybrid_cache_paths_do_not_collide_with_other_strategies(tmp_path):
    store = _build_store(tmp_path, n_rows=5)
    cache_a = HybridTrainingReader(tmp_path / "hybrid_a", store, row_indices=set(range(5)), batch_size=4)
    cache_b = HybridTrainingReader(tmp_path / "hybrid_b", store, row_indices=set(range(5)), batch_size=4)
    list(cache_a.read_rows_for_indices(set(range(5)), batch_size=4))
    list(cache_b.read_rows_for_indices(set(range(5)), batch_size=4))
    assert cache_a.cache_dir != cache_b.cache_dir
    assert (tmp_path / "hybrid_a").exists() and (tmp_path / "hybrid_b").exists()
