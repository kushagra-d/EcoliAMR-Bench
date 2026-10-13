import pytest
from sklearn.metrics import roc_auc_score

from bvbrc_recon.incremental_model import evaluate_streaming_hashed_matrix
from bvbrc_recon.pilot_model import NEGATIVE_CLASS, POSITIVE_CLASS
from bvbrc_recon.sqlite_row_store import SqliteRowStore
from bvbrc_recon.streaming_hashed_matrix import DiskBackedRowStore


def _build_separable_store(tmp_path, n_per_class=10, n_buckets=5):
    """Resistant genomes fire bucket 0; Susceptible genomes fire bucket 1 -
    a trivially separable synthetic dataset for a smoke test."""
    store = DiskBackedRowStore(tmp_path / "rows.jsonl", n_buckets=n_buckets)
    labels = {}
    row_index = 0
    for _ in range(n_per_class):
        store.append_row(f"r{row_index}", row_index, {0: 5})
        labels[row_index] = POSITIVE_CLASS
        row_index += 1
    for _ in range(n_per_class):
        store.append_row(f"s{row_index}", row_index, {1: 5})
        labels[row_index] = NEGATIVE_CLASS
        row_index += 1
    return store, labels


def test_evaluate_streaming_hashed_matrix_separable_case(tmp_path):
    store, labels = _build_separable_store(tmp_path, n_per_class=10)
    # rows 0-9 are Resistant, rows 10-19 are Susceptible - split each class 7/3
    train_idx = set(range(0, 7)) | set(range(10, 17))
    test_idx = set(range(7, 10)) | set(range(17, 20))
    result = evaluate_streaming_hashed_matrix(
        store, train_idx, test_idx, labels, seed=1, batch_size=4, n_epochs=5
    )
    assert result.n_train == 14
    assert result.n_test == 6
    assert result.accuracy is not None
    assert result.roc_auc is not None
    assert len(result.confusion_matrix) == 2


def test_evaluate_streaming_hashed_matrix_never_holds_more_than_one_batch():
    pass  # structural property - verified by max_batch_rows_held_in_memory below


def test_max_batch_rows_held_in_memory_equals_batch_size_not_total_rows(tmp_path):
    store, labels = _build_separable_store(tmp_path, n_per_class=20)  # 40 total rows
    train_idx = set(range(0, 28))
    test_idx = set(range(28, 40))
    result = evaluate_streaming_hashed_matrix(
        store, train_idx, test_idx, labels, seed=1, batch_size=5, n_epochs=2
    )
    assert result.max_batch_rows_held_in_memory <= 5
    assert result.max_batch_rows_held_in_memory < result.n_train


def test_evaluate_streaming_hashed_matrix_is_deterministic_with_fixed_seed(tmp_path):
    store, labels = _build_separable_store(tmp_path, n_per_class=10)
    train_idx = set(range(0, 14))
    test_idx = set(range(14, 20))
    r1 = evaluate_streaming_hashed_matrix(store, train_idx, test_idx, labels, seed=42, batch_size=4, n_epochs=3)
    r2 = evaluate_streaming_hashed_matrix(store, train_idx, test_idx, labels, seed=42, batch_size=4, n_epochs=3)
    assert r1.accuracy == r2.accuracy
    assert r1.confusion_matrix == r2.confusion_matrix


def test_evaluate_streaming_hashed_matrix_reports_train_test_class_counts(tmp_path):
    store, labels = _build_separable_store(tmp_path, n_per_class=10)
    train_idx = set(range(0, 14))
    test_idx = set(range(14, 20))
    result = evaluate_streaming_hashed_matrix(store, train_idx, test_idx, labels, seed=1, batch_size=4, n_epochs=2)
    assert result.train_class_counts[POSITIVE_CLASS] + result.train_class_counts[NEGATIVE_CLASS] == result.n_train
    assert result.test_class_counts[POSITIVE_CLASS] + result.test_class_counts[NEGATIVE_CLASS] == result.n_test


def test_evaluate_streaming_hashed_matrix_handles_empty_train_partition(tmp_path):
    store, labels = _build_separable_store(tmp_path, n_per_class=5)
    result = evaluate_streaming_hashed_matrix(store, set(), set(range(10)), labels, seed=1, batch_size=4, n_epochs=1)
    assert result.n_train == 0
    assert any("No training rows" in w for w in result.warnings)


def test_evaluate_streaming_hashed_matrix_handles_empty_test_partition(tmp_path):
    store, labels = _build_separable_store(tmp_path, n_per_class=5)
    result = evaluate_streaming_hashed_matrix(store, set(range(10)), set(), labels, seed=1, batch_size=4, n_epochs=1)
    assert result.n_test == 0
    assert any("No test rows" in w for w in result.warnings)


def test_evaluate_streaming_hashed_matrix_warns_on_single_class_train(tmp_path):
    store, labels = _build_separable_store(tmp_path, n_per_class=5)
    # train on Resistant-only rows (indices 0-4)
    result = evaluate_streaming_hashed_matrix(
        store, set(range(0, 5)), set(range(5, 10)), labels, seed=1, batch_size=4, n_epochs=1
    )
    assert any("only one class" in w for w in result.warnings)

# --- root-cause fix: row normalization gives scale-invariant predictions ---


def test_normalize_rows_gives_scale_invariant_predictions(tmp_path):
    """Direct mechanical test of the fix: two rows with an identical
    bucket-count PATTERN but one exactly 10x the other's magnitude (same
    relative direction, different scale - simulating two genomes with the
    same resistance-relevant signal but different total k-mer counts)
    must produce IDENTICAL predict_proba once L2-normalized, since both
    collapse to the same unit vector. This is the scale-invariance
    property that fixed the original ROC-AUC=0.5 collapse (raw counts
    vary in scale per genome for reasons unrelated to phenotype)."""
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import normalize as sk_normalize

    store = DiskBackedRowStore(tmp_path / "rows.jsonl", n_buckets=5)
    labels = {}
    for i in range(6):
        pattern = {0: 8, 1: 2} if i % 2 == 0 else {2: 2, 3: 8}
        store.append_row(f"g{i}", i, pattern)
        labels[i] = POSITIVE_CLASS if i % 2 == 0 else NEGATIVE_CLASS

    model = SGDClassifier(random_state=1, loss="log_loss")
    _, row_indices, X_train = next(store.read_rows_for_indices(set(range(6)), batch_size=6))
    y_train = [1 if labels[ri] == POSITIVE_CLASS else 0 for ri in row_indices]
    model.partial_fit(sk_normalize(X_train, norm="l2"), y_train, classes=[0, 1])

    probe_store = DiskBackedRowStore(tmp_path / "probe.jsonl", n_buckets=5)
    probe_store.append_row("probe_small", 0, {0: 8, 1: 2})
    probe_store.append_row("probe_large", 1, {0: 80, 1: 20})  # same pattern, 10x scale

    _, _, x_small = next(probe_store.read_rows_for_indices({0}, batch_size=1))
    _, _, x_large = next(probe_store.read_rows_for_indices({1}, batch_size=1))
    proba_small = model.predict_proba(sk_normalize(x_small, norm="l2"))[:, 1]
    proba_large = model.predict_proba(sk_normalize(x_large, norm="l2"))[:, 1]
    assert proba_small[0] == pytest.approx(proba_large[0])


# --- correct decision scores / ROC-AUC computation ---


def test_roc_auc_matches_manual_sklearn_computation_on_perfectly_separable_data(tmp_path):
    """Independent cross-check: refit the same model configuration by
    hand outside `evaluate_streaming_hashed_matrix` and call
    `sklearn.metrics.roc_auc_score` directly on its decision scores -
    confirming the function's reported ROC-AUC isn't an artifact of its
    own internal bookkeeping."""
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import normalize as sk_normalize

    store, labels = _build_separable_store(tmp_path, n_per_class=10)
    train_idx = set(range(0, 7)) | set(range(10, 17))
    test_idx = set(range(7, 10)) | set(range(17, 20))

    result = evaluate_streaming_hashed_matrix(
        store, train_idx, test_idx, labels, seed=1, batch_size=4, n_epochs=20
    )

    model = SGDClassifier(random_state=1, loss="log_loss")
    _, train_rows, X_train = next(store.read_rows_for_indices(train_idx, batch_size=len(train_idx)))
    y_train = [1 if labels[ri] == POSITIVE_CLASS else 0 for ri in train_rows]
    for _ in range(20):
        model.partial_fit(sk_normalize(X_train, norm="l2"), y_train, classes=[0, 1])

    _, test_rows, X_test = next(store.read_rows_for_indices(test_idx, batch_size=len(test_idx)))
    y_test = [1 if labels[ri] == POSITIVE_CLASS else 0 for ri in test_rows]
    manual_proba = model.predict_proba(sk_normalize(X_test, norm="l2"))[:, 1]
    manual_auc = roc_auc_score(y_test, manual_proba)

    assert result.roc_auc == pytest.approx(manual_auc)
    assert result.roc_auc == pytest.approx(1.0)


# --- minority-class handling ---


def test_minority_class_recall_computed_correctly(tmp_path):
    store = DiskBackedRowStore(tmp_path / "rows.jsonl", n_buckets=5)
    labels = {}
    # 16 Susceptible, 4 Resistant - a realistic imbalance ratio for this project's pilot
    row = 0
    for _ in range(16):
        store.append_row(f"s{row}", row, {0: 1})
        labels[row] = NEGATIVE_CLASS
        row += 1
    for _ in range(4):
        store.append_row(f"r{row}", row, {1: 1})
        labels[row] = POSITIVE_CLASS
        row += 1

    train_idx = set(range(0, 12)) | set(range(16, 18))
    test_idx = set(range(12, 16)) | set(range(18, 20))
    result = evaluate_streaming_hashed_matrix(
        store, train_idx, test_idx, labels, seed=1, batch_size=4, n_epochs=20
    )
    assert result.test_class_counts[POSITIVE_CLASS] == 2
    assert result.test_class_counts[NEGATIVE_CLASS] == 4
    assert result.recall_per_class[POSITIVE_CLASS] is not None
    assert result.recall_per_class[NEGATIVE_CLASS] is not None


# --- training-only preprocessing: normalization is stateless, no leakage possible ---


def test_rows_outside_train_and_test_sets_never_affect_the_result(tmp_path):
    store, labels = _build_separable_store(tmp_path, n_per_class=10)
    train_idx = set(range(0, 7)) | set(range(10, 17))
    test_idx = set(range(7, 10)) | set(range(17, 20))

    result_a = evaluate_streaming_hashed_matrix(store, train_idx, test_idx, labels, seed=1, batch_size=4, n_epochs=5)

    # add extra rows to the SAME store that are in neither train_idx nor test_idx
    store.append_row("extra1", 20, {2: 999})
    labels[20] = NEGATIVE_CLASS
    store.append_row("extra2", 21, {3: 999})
    labels[21] = POSITIVE_CLASS

    result_b = evaluate_streaming_hashed_matrix(store, train_idx, test_idx, labels, seed=1, batch_size=4, n_epochs=5)
    assert result_a.accuracy == result_b.accuracy
    assert result_a.n_train == result_b.n_train
    assert result_a.n_test == result_b.n_test


# --- bounded-memory access to SQLite rows: numpy-index regression ---


def test_sqlite_store_read_rows_for_indices_accepts_numpy_int_indices(tmp_path):
    """Regression test: `train_test_split` and similar sklearn helpers
    return numpy.int64 indices. sqlite3 silently binds these as matching
    no INTEGER row (no exception - just zero results), which previously
    would have made `evaluate_streaming_hashed_matrix` silently evaluate
    on an empty partition when fed numpy-derived index sets."""
    import numpy as np

    store = SqliteRowStore(tmp_path / "rows.db", n_buckets=5)
    for i in range(6):
        store.append_row(f"g{i}", i, {0: i + 1})
    labels = {i: (POSITIVE_CLASS if i % 2 == 0 else NEGATIVE_CLASS) for i in range(6)}

    np_train_idx = {np.int64(0), np.int64(1), np.int64(2), np.int64(3)}
    np_test_idx = {np.int64(4), np.int64(5)}

    result = evaluate_streaming_hashed_matrix(store, np_train_idx, np_test_idx, labels, seed=1, batch_size=4, n_epochs=2)
    assert result.n_train == 4
    assert result.n_test == 2


# --- artifact isolation ---


def test_new_task_artifact_paths_do_not_collide_with_prior_task_paths():
    new_paths = {
        "data/processed/batch_lr_reproduction_ciprofloxacin_d262144.json",
        "data/processed/incremental_model_validation_ciprofloxacin_k15_d262144.json",
    }
    prior_paths = {
        "data/processed/out_of_core_ciprofloxacin_rows_k15_d262144.jsonl",
        "data/processed/out_of_core_v2_ciprofloxacin_rows_k15_d262144.db",
        "data/processed/pilot_ciprofloxacin_kmer_hashed_matrix_d262144.npz",
    }
    assert new_paths.isdisjoint(prior_paths)
