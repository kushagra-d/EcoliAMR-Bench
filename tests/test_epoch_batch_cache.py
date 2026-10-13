import time

import pytest

from bvbrc_recon.epoch_batch_cache import MultiPartitionPersistentReader, PersistentBatchCache
from bvbrc_recon.incremental_model import evaluate_streaming_hashed_matrix
from bvbrc_recon.pilot_model import NEGATIVE_CLASS, POSITIVE_CLASS
from bvbrc_recon.sqlite_row_store import SqliteRowStore


def _build_store(tmp_path, n_rows=10, n_buckets=20):
    store = SqliteRowStore(tmp_path / "rows.db", n_buckets=n_buckets)
    for i in range(n_rows):
        store.append_row(f"g{i}", i, {c: (i + 1) * (c + 1) for c in range(n_buckets)})
    return store


# --- exact row and feature preservation ---


def test_persistent_cache_matches_direct_store_reads_exactly(tmp_path):
    store = _build_store(tmp_path)
    cache = PersistentBatchCache(tmp_path / "cache", store, row_indices=set(range(10)), batch_size=4)

    direct = list(store.read_rows_for_indices(set(range(10)), batch_size=4))
    via_cache = list(cache.read_rows_for_indices(set(range(10)), batch_size=4))

    assert len(direct) == len(via_cache)
    for (gd, rd, md), (gc, rc, mc) in zip(direct, via_cache):
        assert gd == gc
        assert rd == rc
        assert (md != mc).nnz == 0


# --- deterministic / repeated-scan reuse behavior ---


def test_repeated_reads_reuse_artifact_without_rebuilding(tmp_path):
    store = _build_store(tmp_path)
    cache = PersistentBatchCache(tmp_path / "cache", store, row_indices=set(range(10)), batch_size=4)

    list(cache.read_rows_for_indices(set(range(10)), batch_size=4))
    assert cache.stats.n_rebuilds == 1
    n_batches = cache.stats.n_batches_read_from_artifact

    for _ in range(4):  # simulate 4 more epochs
        list(cache.read_rows_for_indices(set(range(10)), batch_size=4))

    assert cache.stats.n_rebuilds == 1  # never rebuilt again
    assert cache.stats.n_batches_read_from_artifact == n_batches * 5


def test_stable_row_ordering_across_repeated_reads(tmp_path):
    store = _build_store(tmp_path)
    cache = PersistentBatchCache(tmp_path / "cache", store, row_indices={1, 3, 5, 7, 9}, batch_size=2)

    pass1 = [ri for _, ris, _ in cache.read_rows_for_indices({1, 3, 5, 7, 9}, batch_size=2) for ri in ris]
    pass2 = [ri for _, ris, _ in cache.read_rows_for_indices({1, 3, 5, 7, 9}, batch_size=2) for ri in ris]
    assert pass1 == pass2 == sorted({1, 3, 5, 7, 9})


# --- scope enforcement: wrong partition/batch_size must not silently read wrong data ---


def test_wrong_row_indices_raises_instead_of_silently_misreading(tmp_path):
    store = _build_store(tmp_path)
    cache = PersistentBatchCache(tmp_path / "cache", store, row_indices=set(range(5)), batch_size=4)
    with pytest.raises(ValueError):
        list(cache.read_rows_for_indices(set(range(5, 10)), batch_size=4))


def test_wrong_batch_size_raises_instead_of_silently_misreading(tmp_path):
    store = _build_store(tmp_path)
    cache = PersistentBatchCache(tmp_path / "cache", store, row_indices=set(range(5)), batch_size=4)
    with pytest.raises(ValueError):
        list(cache.read_rows_for_indices(set(range(5)), batch_size=2))


# --- source invalidation: mechanically enforced, not just documented ---


def test_cache_invalidates_when_source_database_changes(tmp_path):
    store = _build_store(tmp_path, n_rows=10)
    cache = PersistentBatchCache(tmp_path / "cache", store, row_indices=set(range(10)), batch_size=4)
    list(cache.read_rows_for_indices(set(range(10)), batch_size=4))
    assert cache.is_valid() is True

    time.sleep(0.01)  # ensure a distinguishable mtime
    store.append_row("g10", 10, {0: 1})  # changes the db file's size/mtime

    assert cache.is_valid() is False
    list(cache.read_rows_for_indices(set(range(10)), batch_size=4))
    assert cache.stats.n_rebuilds == 2


def test_cache_invalidates_when_requested_partition_changes_construction(tmp_path):
    """A NEW PersistentBatchCache instance scoped to a different
    row_indices set must not reuse another instance's stale artifact
    directory - validated via a fresh, non-matching fingerprint."""
    store = _build_store(tmp_path, n_rows=10)
    cache_dir = tmp_path / "cache"
    cache_a = PersistentBatchCache(cache_dir, store, row_indices=set(range(0, 5)), batch_size=4)
    list(cache_a.read_rows_for_indices(set(range(0, 5)), batch_size=4))

    cache_b = PersistentBatchCache(cache_dir, store, row_indices=set(range(5, 10)), batch_size=4)
    assert cache_b.is_valid() is False  # same dir, different scope - must not be trusted
    batches = list(cache_b.read_rows_for_indices(set(range(5, 10)), batch_size=4))
    assert cache_b.stats.n_rebuilds == 1
    row_indices_seen = sorted(ri for _, ris, _ in batches for ri in ris)
    assert row_indices_seen == list(range(5, 10))


def test_cache_invalidates_when_normalize_rows_flag_changes(tmp_path):
    store = _build_store(tmp_path, n_rows=5, n_buckets=5)
    cache_dir = tmp_path / "cache"
    cache_raw = PersistentBatchCache(cache_dir, store, row_indices=set(range(5)), batch_size=5, normalize_rows=False)
    list(cache_raw.read_rows_for_indices(set(range(5)), batch_size=5))

    cache_norm = PersistentBatchCache(cache_dir, store, row_indices=set(range(5)), batch_size=5, normalize_rows=True)
    assert cache_norm.is_valid() is False
    list(cache_norm.read_rows_for_indices(set(range(5)), batch_size=5))
    assert cache_norm.stats.n_rebuilds == 1


# --- normalization reuse correctness ---


def test_prenormalized_artifact_matches_on_the_fly_normalization(tmp_path):
    import numpy as np
    from sklearn.preprocessing import normalize as sk_normalize

    store = _build_store(tmp_path, n_rows=5, n_buckets=5)
    cache = PersistentBatchCache(tmp_path / "cache", store, row_indices=set(range(5)), batch_size=5, normalize_rows=True)
    _, _, m_cached = next(cache.read_rows_for_indices(set(range(5)), batch_size=5))

    _, _, m_raw = next(store.read_rows_for_indices(set(range(5)), batch_size=5))
    m_expected = sk_normalize(m_raw, norm="l2")
    assert np.allclose(m_cached.toarray(), m_expected.toarray())


# --- stable model inputs and predictions (drop-in via the adapter) ---


def test_multi_partition_reader_gives_identical_training_results(tmp_path):
    store = SqliteRowStore(tmp_path / "rows.db", n_buckets=10)
    labels = {}
    for i in range(10):
        pattern = {0: 10, 1: 1} if i % 2 == 0 else {2: 1, 3: 10}
        store.append_row(f"g{i}", i, pattern)
        labels[i] = POSITIVE_CLASS if i % 2 == 0 else NEGATIVE_CLASS

    train_idx = set(range(0, 8))
    test_idx = set(range(8, 10))

    direct_result = evaluate_streaming_hashed_matrix(store, train_idx, test_idx, labels, seed=1, batch_size=4, n_epochs=5)

    train_cache = PersistentBatchCache(tmp_path / "train_cache", store, train_idx, batch_size=4)
    test_cache = PersistentBatchCache(tmp_path / "test_cache", store, test_idx, batch_size=4)
    reader = MultiPartitionPersistentReader([train_cache, test_cache])
    cached_result = evaluate_streaming_hashed_matrix(reader, train_idx, test_idx, labels, seed=1, batch_size=4, n_epochs=5)

    assert direct_result.accuracy == cached_result.accuracy
    assert direct_result.roc_auc == cached_result.roc_auc
    assert direct_result.confusion_matrix == cached_result.confusion_matrix
    assert train_cache.stats.n_rebuilds == 1
    assert train_cache.stats.n_batches_read_from_artifact == 5 * 2  # 5 epochs x 2 batches/epoch


def test_multi_partition_reader_raises_for_unconfigured_partition(tmp_path):
    store = _build_store(tmp_path, n_rows=10)
    cache = PersistentBatchCache(tmp_path / "cache", store, row_indices=set(range(5)), batch_size=4)
    reader = MultiPartitionPersistentReader([cache])
    with pytest.raises(ValueError):
        list(reader.read_rows_for_indices(set(range(5, 10)), batch_size=4))


# --- artifact isolation ---


def test_cache_artifact_paths_do_not_collide_with_other_task_paths(tmp_path):
    store = _build_store(tmp_path, n_rows=5)
    cache_a = PersistentBatchCache(tmp_path / "cache_a", store, row_indices=set(range(5)), batch_size=4)
    cache_b = PersistentBatchCache(tmp_path / "cache_b", store, row_indices=set(range(5)), batch_size=4)
    list(cache_a.read_rows_for_indices(set(range(5)), batch_size=4))
    list(cache_b.read_rows_for_indices(set(range(5)), batch_size=4))
    assert cache_a.cache_dir != cache_b.cache_dir
    assert (tmp_path / "cache_a").exists() and (tmp_path / "cache_b").exists()
