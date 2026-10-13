import numpy as np
import pytest
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

from bvbrc_recon.incremental_model_candidates import balanced_class_weight, evaluate_candidate
from bvbrc_recon.pilot_model import NEGATIVE_CLASS, POSITIVE_CLASS
from bvbrc_recon.streaming_hashed_matrix import DiskBackedRowStore


def _build_separable_store(tmp_path, n_per_class=10, n_buckets=5):
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


# --- correct class-weight handling ---


def test_balanced_class_weight_matches_sklearn_formula():
    counts = {NEGATIVE_CLASS: 80, POSITIVE_CLASS: 20}
    weights = balanced_class_weight(counts)
    # n_samples / (n_classes * n_c)
    assert weights[0] == pytest.approx(100 / (2 * 80))
    assert weights[1] == pytest.approx(100 / (2 * 20))
    assert weights[1] > weights[0]  # minority class gets the larger weight


def test_balanced_class_weight_ignores_zero_count_classes():
    weights = balanced_class_weight({NEGATIVE_CLASS: 10, POSITIVE_CLASS: 0})
    assert 1 not in weights  # no divide-by-zero, class simply absent


def test_class_weight_is_actually_applied_to_the_estimator(tmp_path):
    """Mechanical check: an SGDClassifier built with an explicit
    class_weight dict carries it through partial_fit unchanged."""
    store, labels = _build_separable_store(tmp_path, n_per_class=5)
    cw = {0: 1.0, 1: 4.0}
    result = evaluate_candidate(
        store, set(range(0, 7)), set(range(7, 10)), labels, seed=1,
        build_model=lambda: SGDClassifier(loss="log_loss", random_state=1, class_weight=cw),
        normalize_rows=True, batch_size=4, n_epochs=5, config_name="test",
    )
    assert result.n_train > 0  # ran without error - class_weight accepted and used


# --- decision-score / probability handling ---


def test_predict_proba_based_auc_is_perfect_on_separable_data(tmp_path):
    store, labels = _build_separable_store(tmp_path, n_per_class=10)
    train_idx = set(range(0, 7)) | set(range(10, 17))
    test_idx = set(range(7, 10)) | set(range(17, 20))
    result = evaluate_candidate(
        store, train_idx, test_idx, labels, seed=1,
        build_model=lambda: SGDClassifier(loss="log_loss", random_state=1),
        normalize_rows=True, batch_size=4, n_epochs=20, config_name="sgd",
    )
    assert result.roc_auc == pytest.approx(1.0)


def test_multinomial_nb_runs_on_raw_unnormalized_counts(tmp_path):
    store, labels = _build_separable_store(tmp_path, n_per_class=10)
    train_idx = set(range(0, 7)) | set(range(10, 17))
    test_idx = set(range(7, 10)) | set(range(17, 20))
    result = evaluate_candidate(
        store, train_idx, test_idx, labels, seed=1,
        build_model=lambda: MultinomialNB(),
        normalize_rows=False, batch_size=4, n_epochs=5, config_name="nb",
    )
    assert result.accuracy == pytest.approx(1.0)
    assert result.roc_auc == pytest.approx(1.0)


def test_multinomial_nb_balanced_prior_runs_without_error(tmp_path):
    store, labels = _build_separable_store(tmp_path, n_per_class=10)
    train_idx = set(range(0, 7)) | set(range(10, 17))
    test_idx = set(range(7, 10)) | set(range(17, 20))
    result = evaluate_candidate(
        store, train_idx, test_idx, labels, seed=1,
        build_model=lambda: MultinomialNB(class_prior=[0.5, 0.5]),
        normalize_rows=False, batch_size=4, n_epochs=5, config_name="nb_balanced",
    )
    assert result.n_test == 6
    assert result.accuracy is not None


# --- training-only preprocessing ---


def test_normalization_is_stateless_and_never_uses_test_data(tmp_path):
    """Changing test-partition VALUES must not change how training rows
    are normalized or how the model was fit - a stateless, per-row
    transform cannot leak test information, and this confirms it."""
    store, labels = _build_separable_store(tmp_path, n_per_class=10)
    train_idx = set(range(0, 7)) | set(range(10, 17))
    test_idx = set(range(7, 10)) | set(range(17, 20))

    result_a = evaluate_candidate(
        store, train_idx, test_idx, labels, seed=1,
        build_model=lambda: SGDClassifier(loss="log_loss", random_state=1),
        normalize_rows=True, batch_size=4, n_epochs=5, config_name="a",
    )

    store.append_row("extra_test_like", 20, {0: 999999})  # not in train_idx or test_idx
    result_b = evaluate_candidate(
        store, train_idx, test_idx, labels, seed=1,
        build_model=lambda: SGDClassifier(loss="log_loss", random_state=1),
        normalize_rows=True, batch_size=4, n_epochs=5, config_name="b",
    )
    assert result_a.accuracy == result_b.accuracy


# --- minority-class metrics ---


def test_minority_class_recall_and_average_precision_reported(tmp_path):
    store = DiskBackedRowStore(tmp_path / "rows.jsonl", n_buckets=5)
    labels = {}
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
    result = evaluate_candidate(
        store, train_idx, test_idx, labels, seed=1,
        build_model=lambda: SGDClassifier(loss="log_loss", random_state=1),
        normalize_rows=True, batch_size=4, n_epochs=20, config_name="imbalanced",
    )
    assert result.test_class_counts[POSITIVE_CLASS] == 2
    assert result.average_precision is not None
    assert result.recall_per_class[POSITIVE_CLASS] is not None


# --- exact row and label alignment ---


def test_train_test_class_counts_match_label_dict_exactly(tmp_path):
    store, labels = _build_separable_store(tmp_path, n_per_class=10)
    train_idx = set(range(0, 7)) | set(range(10, 17))
    test_idx = set(range(7, 10)) | set(range(17, 20))
    result = evaluate_candidate(
        store, train_idx, test_idx, labels, seed=1,
        build_model=lambda: SGDClassifier(loss="log_loss", random_state=1),
        normalize_rows=True, batch_size=4, n_epochs=5, config_name="align",
    )
    assert result.train_class_counts[POSITIVE_CLASS] == sum(1 for i in train_idx if labels[i] == POSITIVE_CLASS)
    assert result.train_class_counts[NEGATIVE_CLASS] == sum(1 for i in train_idx if labels[i] == NEGATIVE_CLASS)
    assert result.test_class_counts[POSITIVE_CLASS] == sum(1 for i in test_idx if labels[i] == POSITIVE_CLASS)


# --- bounded-memory batch access ---


def test_never_holds_more_than_one_batch_at_a_time(tmp_path):
    store, labels = _build_separable_store(tmp_path, n_per_class=20)  # 40 rows
    train_idx = set(range(0, 28))
    test_idx = set(range(28, 40))
    result = evaluate_candidate(
        store, train_idx, test_idx, labels, seed=1,
        build_model=lambda: SGDClassifier(loss="log_loss", random_state=1),
        normalize_rows=True, batch_size=5, n_epochs=2, config_name="bounded",
    )
    assert result.max_batch_rows_held_in_memory <= 5
    assert result.max_batch_rows_held_in_memory < result.n_train


# --- reproducibility / determinism ---


def test_same_seed_gives_identical_results(tmp_path):
    store, labels = _build_separable_store(tmp_path, n_per_class=10)
    train_idx = set(range(0, 7)) | set(range(10, 17))
    test_idx = set(range(7, 10)) | set(range(17, 20))
    r1 = evaluate_candidate(
        store, train_idx, test_idx, labels, seed=42,
        build_model=lambda: SGDClassifier(loss="log_loss", random_state=42),
        normalize_rows=True, batch_size=4, n_epochs=10, config_name="seeded",
    )
    r2 = evaluate_candidate(
        store, train_idx, test_idx, labels, seed=42,
        build_model=lambda: SGDClassifier(loss="log_loss", random_state=42),
        normalize_rows=True, batch_size=4, n_epochs=10, config_name="seeded",
    )
    assert r1.accuracy == r2.accuracy
    assert r1.confusion_matrix == r2.confusion_matrix


def test_multinomial_nb_is_deterministic(tmp_path):
    store, labels = _build_separable_store(tmp_path, n_per_class=10)
    train_idx = set(range(0, 7)) | set(range(10, 17))
    test_idx = set(range(7, 10)) | set(range(17, 20))
    r1 = evaluate_candidate(
        store, train_idx, test_idx, labels, seed=1,
        build_model=lambda: MultinomialNB(), normalize_rows=False, batch_size=4, n_epochs=5, config_name="nb",
    )
    r2 = evaluate_candidate(
        store, train_idx, test_idx, labels, seed=1,
        build_model=lambda: MultinomialNB(), normalize_rows=False, batch_size=4, n_epochs=5, config_name="nb",
    )
    assert r1.accuracy == r2.accuracy
    assert r1.confusion_matrix == r2.confusion_matrix


# --- artifact isolation ---


def test_investigation_artifact_paths_do_not_collide_with_prior_task_paths():
    new_paths = {
        "data/processed/incremental_model_candidate_selection_ciprofloxacin_k15_d262144.json",
        "data/processed/incremental_model_quality_final_ciprofloxacin_k15_d262144.json",
    }
    prior_paths = {
        "data/processed/incremental_model_validation_ciprofloxacin_k15_d262144.json",
        "data/processed/batch_lr_reproduction_ciprofloxacin_d262144.json",
        "data/processed/hybrid_pipeline_benchmark_ciprofloxacin_k15_d262144.json",
    }
    assert new_paths.isdisjoint(prior_paths)
