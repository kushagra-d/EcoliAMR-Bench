from bvbrc_recon.cached_row_reader import CachedRowReader
from bvbrc_recon.sqlite_row_store import SqliteRowStore


def _build_store(tmp_path, n_rows=10, n_buckets=20):
    store = SqliteRowStore(tmp_path / "rows.db", n_buckets=n_buckets)
    for i in range(n_rows):
        store.append_row(f"g{i}", i, {c: (i + 1) * (c + 1) for c in range(n_buckets)})
    return store


# --- exact batch equivalence against the direct SqliteRowStore reader ---


def test_cached_reader_matches_direct_store_reads_exactly(tmp_path):
    store = _build_store(tmp_path)
    reader = CachedRowReader(store, max_cache_bytes=10_000_000)

    direct_batches = list(store.read_rows_for_indices(set(range(10)), batch_size=4))
    cached_batches = list(reader.read_rows_for_indices(set(range(10)), batch_size=4))

    assert len(direct_batches) == len(cached_batches)
    for (gid_d, ri_d, m_d), (gid_c, ri_c, m_c) in zip(direct_batches, cached_batches):
        assert gid_d == gid_c
        assert ri_d == ri_c
        assert (m_d != m_c).nnz == 0


def test_cached_reader_second_pass_is_served_from_cache_not_sqlite(tmp_path):
    store = _build_store(tmp_path)
    reader = CachedRowReader(store, max_cache_bytes=10_000_000)

    first_pass = list(reader.read_rows_for_indices(set(range(10)), batch_size=4))
    assert reader.stats.n_hits == 0
    assert reader.stats.n_misses == len(first_pass)

    second_pass = list(reader.read_rows_for_indices(set(range(10)), batch_size=4))
    assert reader.stats.n_hits == len(second_pass)  # every chunk now a cache hit

    for (gid_1, ri_1, m_1), (gid_2, ri_2, m_2) in zip(first_pass, second_pass):
        assert gid_1 == gid_2
        assert ri_1 == ri_2
        assert (m_1 != m_2).nnz == 0


# --- stable row and feature ordering across repeated / cached reads ---


def test_row_and_feature_ordering_stable_across_repeated_reads(tmp_path):
    store = _build_store(tmp_path)
    reader = CachedRowReader(store, max_cache_bytes=10_000_000)
    wanted = {1, 3, 5, 7, 9}

    pass1 = list(reader.read_rows_for_indices(wanted, batch_size=2))
    pass2 = list(reader.read_rows_for_indices(wanted, batch_size=2))

    row_order_1 = [ri for _, ris, _ in pass1 for ri in ris]
    row_order_2 = [ri for _, ris, _ in pass2 for ri in ris]
    assert row_order_1 == row_order_2 == sorted(wanted)


# --- bounded cache behavior ---


def test_cache_respects_byte_budget_and_stops_caching_over_budget(tmp_path):
    store = _build_store(tmp_path, n_rows=20, n_buckets=20)
    reader = CachedRowReader(store, max_cache_bytes=1)  # effectively zero headroom
    list(reader.read_rows_for_indices(set(range(20)), batch_size=4))

    assert reader.stats.n_batches_cached == 0
    assert reader.stats.n_fallbacks > 0
    assert reader.stats.n_bytes_cached <= 1


def test_cache_never_exceeds_configured_budget(tmp_path):
    store = _build_store(tmp_path, n_rows=20, n_buckets=20)
    budget = 2_000
    reader = CachedRowReader(store, max_cache_bytes=budget)
    list(reader.read_rows_for_indices(set(range(20)), batch_size=4))
    assert reader.stats.n_bytes_cached <= budget


# --- cache invalidation and consistency ---


def test_clear_cache_resets_stats_and_forces_fresh_sqlite_reads(tmp_path):
    store = _build_store(tmp_path)
    reader = CachedRowReader(store, max_cache_bytes=10_000_000)
    list(reader.read_rows_for_indices(set(range(10)), batch_size=4))
    assert reader.stats.n_batches_cached > 0

    reader.clear_cache()
    assert reader.stats.n_hits == 0
    assert reader.stats.n_misses == 0
    assert reader.stats.n_bytes_cached == 0

    list(reader.read_rows_for_indices(set(range(10)), batch_size=4))
    assert reader.stats.n_misses > 0  # re-fetched from SQLite, not a stale hit


def test_cache_remains_valid_after_unrelated_rows_are_appended(tmp_path):
    """Rows are append-only and immutable once written (enforced by the
    schema's PRIMARY KEY/UNIQUE constraints - no UPDATE path exists), so
    a cached batch for a fixed set of row_index values stays correct even
    after MORE rows are appended to the same store."""
    store = _build_store(tmp_path, n_rows=10)
    reader = CachedRowReader(store, max_cache_bytes=10_000_000)
    first = list(reader.read_rows_for_indices({0, 1, 2, 3}, batch_size=4))

    store.append_row("g10", 10, {0: 1})  # unrelated new row

    second = list(reader.read_rows_for_indices({0, 1, 2, 3}, batch_size=4))
    assert reader.stats.n_hits >= 1  # served from cache, not re-queried
    for (gid_1, ri_1, m_1), (gid_2, ri_2, m_2) in zip(first, second):
        assert gid_1 == gid_2 and ri_1 == ri_2
        assert (m_1 != m_2).nnz == 0


# --- SQLite fallback behavior: cache is never required for correctness ---


def test_zero_budget_cache_still_produces_correct_results():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        store = _build_store(Path(d), n_rows=10)
        reader = CachedRowReader(store, max_cache_bytes=0)

        batches = list(reader.read_rows_for_indices(set(range(10)), batch_size=4))
        assert reader.stats.n_batches_cached == 0
        assert reader.stats.n_misses == len(batches)  # always a fresh SQLite read

        direct = list(store.read_rows_for_indices(set(range(10)), batch_size=4))
        assert len(batches) == len(direct)
        for (gid_a, ri_a, m_a), (gid_b, ri_b, m_b) in zip(batches, direct):
            assert gid_a == gid_b and ri_a == ri_b
            assert (m_a != m_b).nnz == 0


def test_iter_batches_is_uncached_passthrough(tmp_path):
    store = _build_store(tmp_path)
    reader = CachedRowReader(store, max_cache_bytes=10_000_000)
    list(reader.iter_batches(batch_size=4))
    assert reader.stats.n_hits == 0
    assert reader.stats.n_misses == 0  # iter_batches never touches cache bookkeeping


# --- deterministic outputs ---


def test_cached_reader_is_deterministic_across_fresh_instances(tmp_path):
    store = _build_store(tmp_path)
    reader_a = CachedRowReader(SqliteRowStore(tmp_path / "rows.db", n_buckets=20), max_cache_bytes=10_000_000)
    reader_b = CachedRowReader(SqliteRowStore(tmp_path / "rows.db", n_buckets=20), max_cache_bytes=10_000_000)

    batches_a = list(reader_a.read_rows_for_indices(set(range(10)), batch_size=3))
    batches_b = list(reader_b.read_rows_for_indices(set(range(10)), batch_size=3))
    for (gid_a, ri_a, m_a), (gid_b, ri_b, m_b) in zip(batches_a, batches_b):
        assert gid_a == gid_b and ri_a == ri_b
        assert (m_a != m_b).nnz == 0


# --- compatibility with the model-evaluation path (drop-in, no model code changes) ---


def test_cached_reader_works_as_drop_in_for_evaluate_streaming_hashed_matrix(tmp_path):
    from bvbrc_recon.incremental_model import evaluate_streaming_hashed_matrix
    from bvbrc_recon.pilot_model import NEGATIVE_CLASS, POSITIVE_CLASS

    store = SqliteRowStore(tmp_path / "rows.db", n_buckets=10)
    labels = {}
    for i in range(10):
        pattern = {0: 10, 1: 1} if i % 2 == 0 else {2: 1, 3: 10}
        store.append_row(f"g{i}", i, pattern)
        labels[i] = POSITIVE_CLASS if i % 2 == 0 else NEGATIVE_CLASS

    reader = CachedRowReader(store, max_cache_bytes=10_000_000)
    train_idx = set(range(0, 8))
    test_idx = set(range(8, 10))

    result_direct = evaluate_streaming_hashed_matrix(store, train_idx, test_idx, labels, seed=1, batch_size=4, n_epochs=5)
    result_cached = evaluate_streaming_hashed_matrix(reader, train_idx, test_idx, labels, seed=1, batch_size=4, n_epochs=5)

    assert result_direct.accuracy == result_cached.accuracy
    assert result_direct.roc_auc == result_cached.roc_auc
    assert result_direct.confusion_matrix == result_cached.confusion_matrix
    assert reader.stats.n_hits > 0  # repeated epochs actually hit the cache


# --- deterministic LRU eviction ---


def test_eviction_is_strict_lru_and_deterministic(tmp_path):
    """Each row's batch is its own chunk (batch_size=1) so cache
    occupancy is easy to reason about exactly. Budget holds exactly 2
    single-row batches; a third distinct read must evict the
    LEAST-recently-used one, not an arbitrary one."""
    store = _build_store(tmp_path, n_rows=5, n_buckets=20)
    one_row_bytes = None
    probe = CachedRowReader(store, max_cache_bytes=10_000_000)
    for _, _, m in probe.read_rows_for_indices({0}, batch_size=1):
        one_row_bytes = CachedRowReader._csr_bytes(m)
    budget = one_row_bytes * 2 + 10  # room for exactly two 1-row batches

    reader = CachedRowReader(store, max_cache_bytes=budget, max_reader_memory_bytes=budget)
    list(reader.read_rows_for_indices({0}, batch_size=1))  # cache: [0]
    list(reader.read_rows_for_indices({1}, batch_size=1))  # cache: [0, 1]
    list(reader.read_rows_for_indices({0}, batch_size=1))  # hit; cache order: [1, 0] (0 now MRU)
    list(reader.read_rows_for_indices({2}, batch_size=1))  # must evict 1 (LRU), not 0

    assert reader.stats.n_evictions == 1
    assert (0,) in reader._cache
    assert (1,) not in reader._cache
    assert (2,) in reader._cache


def test_eviction_order_is_reproducible_across_runs(tmp_path):
    store = _build_store(tmp_path, n_rows=5, n_buckets=20)
    one_row_bytes = None
    probe = CachedRowReader(store, max_cache_bytes=10_000_000)
    for _, _, m in probe.read_rows_for_indices({0}, batch_size=1):
        one_row_bytes = CachedRowReader._csr_bytes(m)
    budget = one_row_bytes + 10  # room for exactly ONE 1-row batch

    def run_sequence():
        reader = CachedRowReader(store, max_cache_bytes=budget, max_reader_memory_bytes=budget)
        for idx in [0, 1, 2, 0, 3, 1]:
            list(reader.read_rows_for_indices({idx}, batch_size=1))
        return reader.stats.n_evictions, reader.stats.n_hits

    assert run_sequence() == run_sequence()


# --- oversized rows/batches ---


def test_oversized_batch_is_still_returned_correctly_but_not_cached(tmp_path):
    store = _build_store(tmp_path, n_rows=5, n_buckets=20)
    reader = CachedRowReader(store, max_cache_bytes=100, max_reader_memory_bytes=100)  # smaller than any real batch

    direct = list(store.read_rows_for_indices(set(range(5)), batch_size=5))
    via_reader = list(reader.read_rows_for_indices(set(range(5)), batch_size=5))

    assert len(direct) == len(via_reader) == 1
    gid_d, ri_d, m_d = direct[0]
    gid_r, ri_r, m_r = via_reader[0]
    assert gid_d == gid_r and ri_d == ri_r
    assert (m_d != m_r).nnz == 0  # full data correctly returned despite the tiny budget
    assert reader.stats.n_fallbacks == 1
    assert reader.stats.n_batches_cached == 0


# --- max_cache_bytes vs max_reader_memory_bytes: distinct ceilings ---


def test_max_reader_memory_bytes_distinct_from_max_cache_bytes(tmp_path):
    """A batch bigger than max_cache_bytes but smaller than
    max_reader_memory_bytes must be served (fits the reader-memory
    ceiling) but NOT retained in the cache (fails the stricter
    cache-only ceiling) - the two budgets are checked independently."""
    store = _build_store(tmp_path, n_rows=5, n_buckets=20)
    one_row_bytes = None
    probe = CachedRowReader(store, max_cache_bytes=10_000_000)
    for _, _, m in probe.read_rows_for_indices({0}, batch_size=1):
        one_row_bytes = CachedRowReader._csr_bytes(m)

    reader = CachedRowReader(
        store, max_cache_bytes=one_row_bytes // 2, max_reader_memory_bytes=one_row_bytes * 10
    )
    list(reader.read_rows_for_indices({0}, batch_size=1))
    assert reader.stats.n_fallbacks == 1
    assert reader.stats.n_batches_cached == 0
    assert reader.max_reader_memory_bytes > reader.max_cache_bytes


def test_default_reader_memory_equals_cache_bytes_when_not_specified(tmp_path):
    store = _build_store(tmp_path, n_rows=5)
    reader = CachedRowReader(store, max_cache_bytes=12_345)
    assert reader.max_reader_memory_bytes == 12_345


# --- reader behavior after reopening the database ---


def test_reader_cold_starts_correctly_after_reopening_the_database(tmp_path):
    db_path = tmp_path / "rows.db"
    store1 = SqliteRowStore(db_path, n_buckets=20)
    for i in range(5):
        store1.append_row(f"g{i}", i, {c: i + c + 1 for c in range(20)})
    reader1 = CachedRowReader(store1, max_cache_bytes=10_000_000)
    list(reader1.read_rows_for_indices(set(range(5)), batch_size=5))
    assert reader1.stats.n_batches_cached > 0
    store1.close()

    store2 = SqliteRowStore(db_path, n_buckets=20)
    reader2 = CachedRowReader(store2, max_cache_bytes=10_000_000)
    assert reader2.stats.n_batches_cached == 0  # cold cache, independent of reader1's

    batches1 = list(reader1.read_rows_for_indices(set(range(5)), batch_size=5))
    batches2 = list(reader2.read_rows_for_indices(set(range(5)), batch_size=5))
    for (g1, r1, m1), (g2, r2, m2) in zip(batches1, batches2):
        assert g1 == g2 and r1 == r2
        assert (m1 != m2).nnz == 0


# --- artifact isolation ---


def test_memory_budget_task_artifact_paths_do_not_collide_with_prior_task_paths():
    new_paths = {
        "data/processed/memory_budget_benchmark_ciprofloxacin_k15_d262144.json",
    }
    prior_paths = {
        "data/processed/out_of_core_ciprofloxacin_rows_k15_d262144.jsonl",
        "data/processed/out_of_core_v2_ciprofloxacin_rows_k15_d262144.db",
        "data/processed/read_optimization_benchmark_ciprofloxacin_k15_d262144.json",
        "data/processed/incremental_model_validation_ciprofloxacin_k15_d262144.json",
    }
    assert new_paths.isdisjoint(prior_paths)


# --- NonEvictingCachedRowReader (strategy #3, this task) ---


from bvbrc_recon.cached_row_reader import NonEvictingCachedRowReader  # noqa: E402


def test_non_evicting_cache_never_evicts_once_cached(tmp_path):
    from bvbrc_recon.cached_row_reader import CachedRowReader

    store = _build_store(tmp_path, n_rows=5, n_buckets=20)
    one_row_bytes = None
    for _, _, m in store.read_rows_for_indices({0}, batch_size=1):
        one_row_bytes = CachedRowReader._csr_bytes(m)
    budget = one_row_bytes + 10  # room for exactly ONE 1-row batch

    reader = NonEvictingCachedRowReader(store, max_cache_bytes=budget, max_reader_memory_bytes=budget)
    list(reader.read_rows_for_indices({0}, batch_size=1))  # cached
    list(reader.read_rows_for_indices({1}, batch_size=1))  # over budget - permanent fallback
    list(reader.read_rows_for_indices({1}, batch_size=1))  # still a fallback, NOT evicting {0} to try again

    assert (0,) in reader._cache
    assert (1,) not in reader._cache
    assert reader.stats.n_evictions == 0
    assert reader.stats.n_fallbacks == 2  # both reads of row 1 fell back


def test_non_evicting_cache_repeated_scan_gets_stable_partial_hits(tmp_path):
    """The exact scenario that defeated LRU: a budget smaller than the
    full working set, scanned repeatedly in the same order. Unlike LRU
    (0 hits, see test_cached_row_reader tests above), whatever fits on
    pass 1 must stay cached and deliver hits on every later pass."""
    store = _build_store(tmp_path, n_rows=20, n_buckets=20)
    one_batch_bytes = None
    for _, _, m in store.read_rows_for_indices(set(range(4)), batch_size=4):
        from bvbrc_recon.cached_row_reader import CachedRowReader

        one_batch_bytes = CachedRowReader._csr_bytes(m)
    budget = one_batch_bytes + 10  # room for exactly one of the five 4-row batches

    reader = NonEvictingCachedRowReader(store, max_cache_bytes=budget, max_reader_memory_bytes=budget)
    for _ in range(5):  # simulate 5 repeated passes/epochs
        list(reader.read_rows_for_indices(set(range(20)), batch_size=4))

    assert reader.stats.n_hits > 0  # LRU gave exactly 0 here; non-evicting must give > 0
    assert reader.stats.n_bytes_cached <= budget


def test_non_evicting_cache_respects_budget_and_matches_direct_reads(tmp_path):
    store = _build_store(tmp_path, n_rows=10, n_buckets=20)
    reader = NonEvictingCachedRowReader(store, max_cache_bytes=30_000_000)
    direct = list(store.read_rows_for_indices(set(range(10)), batch_size=4))
    via_reader = list(reader.read_rows_for_indices(set(range(10)), batch_size=4))
    for (gd, rd, md), (gr, rr, mr) in zip(direct, via_reader):
        assert gd == gr and rd == rr
        assert (md != mr).nnz == 0
    assert reader.stats.n_bytes_cached <= reader.max_cache_bytes
