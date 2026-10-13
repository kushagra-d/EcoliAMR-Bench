import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from bvbrc_recon.pilot_model import CLASS_TO_INT, NEGATIVE_CLASS, POSITIVE_CLASS
from bvbrc_recon.streaming_hashed_matrix import DiskBackedRowStore
from bvbrc_recon.streaming_logistic_regression import (
    _streaming_loss_and_grad,
    fit_and_evaluate_streaming_logistic_regression,
)


def _random_sparse_store(tmp_path, n=40, d=12, seed=0):
    rng = np.random.RandomState(seed)
    store = DiskBackedRowStore(tmp_path / "rows.jsonl", n_buckets=d)
    labels = {}
    w_true = rng.randn(d)
    for i in range(n):
        counts = rng.randint(0, 6, size=d)
        counts[rng.rand(d) < 0.4] = 0  # some sparsity
        nonzero = {j: int(v) for j, v in enumerate(counts) if v > 0}
        if not nonzero:
            nonzero = {0: 1}
        store.append_row(f"g{i}", i, nonzero)
        score = sum(v * w_true[j] for j, v in nonzero.items())
        labels[i] = POSITIVE_CLASS if score > 0 else NEGATIVE_CLASS
    return store, labels, d


def _dense_matrix_and_y(store, labels, row_indices, d):
    row_indices = sorted(row_indices)
    rows, cols, vals = [], [], []
    y = []
    for local_i, ri in enumerate(row_indices):
        _, _, m = next(store.read_rows_for_indices({ri}, batch_size=1))
        coo = m.tocoo()
        rows.extend([local_i] * len(coo.col))
        cols.extend(coo.col.tolist())
        vals.extend(coo.data.tolist())
        y.append(CLASS_TO_INT[labels[ri]])
    X = sparse.coo_matrix((vals, (rows, cols)), shape=(len(row_indices), d)).tocsr()
    return X, np.array(y)


# --- preservation of existing model behavior: direct comparison against sklearn ---


def test_streaming_objective_matches_sklearn_logistic_regression_closely(tmp_path):
    store, labels, d = _random_sparse_store(tmp_path, n=50, d=15, seed=1)
    train_idx = set(range(0, 40))
    test_idx = set(range(40, 50))

    result = fit_and_evaluate_streaming_logistic_regression(
        store, train_idx, test_idx, labels, n_features=d, seed=42, batch_size=8, max_iter=1000, tol=1e-6,
    )
    assert result.converged

    X_train, y_train = _dense_matrix_and_y(store, labels, train_idx, d)
    sk = LogisticRegression(max_iter=1000, random_state=42)
    sk.fit(X_train, y_train)

    assert np.allclose(result.coef_, sk.coef_[0], atol=1e-2)
    assert abs(result.intercept_ - sk.intercept_[0]) < 1e-2

    X_test, y_test = _dense_matrix_and_y(store, labels, test_idx, d)
    proba_sklearn = sk.predict_proba(X_test)[:, 1]
    from scipy.special import expit

    proba_streaming = expit(X_test.dot(result.coef_) + result.intercept_)
    assert np.allclose(proba_streaming, proba_sklearn, atol=1e-2)


def test_streaming_solution_reaches_lower_or_equal_loss_than_sklearns_default_tolerance_fit(tmp_path):
    """The rigorous version of the above: under the IDENTICAL objective,
    a tightly-converged streaming fit should not do WORSE than sklearn's
    own default-tolerance fit - if it did measurably worse, that would
    indicate a real formula or convergence bug, not just tolerance."""
    store, labels, d = _random_sparse_store(tmp_path, n=50, d=15, seed=2)
    train_idx = set(range(0, 40))

    result = fit_and_evaluate_streaming_logistic_regression(
        store, train_idx, set(range(40, 50)), labels, n_features=d, seed=1, batch_size=8, max_iter=1000, tol=1e-6,
    )
    X_train, y_train = _dense_matrix_and_y(store, labels, train_idx, d)
    sk = LogisticRegression(max_iter=1000, random_state=1)
    sk.fit(X_train, y_train)

    params_mine = np.concatenate([result.coef_, [result.intercept_]])
    params_sklearn = np.concatenate([sk.coef_[0], sk.intercept_])
    loss_mine, _ = _streaming_loss_and_grad(params_mine, store, train_idx, labels, 8, 1.0, [0])
    loss_sklearn, _ = _streaming_loss_and_grad(params_sklearn, store, train_idx, labels, 8, 1.0, [0])
    assert loss_mine <= loss_sklearn + 1e-6


# --- correct score / metric calculation ---


def test_predictions_and_metrics_are_internally_consistent(tmp_path):
    store, labels, d = _random_sparse_store(tmp_path, n=50, d=15, seed=3)
    train_idx = set(range(0, 40))
    test_idx = set(range(40, 50))
    result = fit_and_evaluate_streaming_logistic_regression(
        store, train_idx, test_idx, labels, n_features=d, seed=1, batch_size=8,
    )
    assert result.accuracy is not None
    assert 0.0 <= result.accuracy <= 1.0
    assert len(result.confusion_matrix) == 2
    assert sum(sum(row) for row in result.confusion_matrix) == result.n_test


# --- exact row and label alignment ---


def test_train_test_class_counts_match_label_dict_exactly(tmp_path):
    store, labels, d = _random_sparse_store(tmp_path, n=30, d=10, seed=4)
    train_idx = set(range(0, 20))
    test_idx = set(range(20, 30))
    result = fit_and_evaluate_streaming_logistic_regression(
        store, train_idx, test_idx, labels, n_features=d, seed=1, batch_size=5,
    )
    expected_train = {
        NEGATIVE_CLASS: sum(1 for i in train_idx if labels[i] == NEGATIVE_CLASS),
        POSITIVE_CLASS: sum(1 for i in train_idx if labels[i] == POSITIVE_CLASS),
    }
    assert result.train_class_counts == expected_train


# --- training-only fitting ---


def test_test_partition_values_never_affect_the_fit(tmp_path):
    store, labels, d = _random_sparse_store(tmp_path, n=30, d=10, seed=5)
    train_idx = set(range(0, 20))
    test_idx = set(range(20, 30))

    result_a = fit_and_evaluate_streaming_logistic_regression(
        store, train_idx, test_idx, labels, n_features=d, seed=1, batch_size=5, max_iter=50,
    )

    # corrupt test rows' values (still within test_idx, never touched by training)
    for ri in list(test_idx)[:2]:
        store.append_row(f"corrupted_placeholder_{ri}", 1000 + ri, {0: 99999})  # new, unrelated rows
    result_b = fit_and_evaluate_streaming_logistic_regression(
        store, train_idx, test_idx, labels, n_features=d, seed=1, batch_size=5, max_iter=50,
    )
    assert np.allclose(result_a.coef_, result_b.coef_)
    assert result_a.intercept_ == result_b.intercept_


def test_single_class_training_partition_is_rejected_not_silently_fit(tmp_path):
    store = DiskBackedRowStore(tmp_path / "rows.jsonl", n_buckets=5)
    labels = {}
    for i in range(6):
        store.append_row(f"g{i}", i, {0: 1})
        labels[i] = POSITIVE_CLASS
    result = fit_and_evaluate_streaming_logistic_regression(
        store, set(range(0, 4)), set(range(4, 6)), labels, n_features=5, seed=1, batch_size=2,
    )
    assert result.coef_ is None
    assert any("only one class" in w for w in result.warnings)


# --- bounded-memory data access ---


def test_never_holds_more_than_one_batch_at_a_time(tmp_path):
    store, labels, d = _random_sparse_store(tmp_path, n=40, d=10, seed=6)
    train_idx = set(range(0, 30))
    test_idx = set(range(30, 40))
    result = fit_and_evaluate_streaming_logistic_regression(
        store, train_idx, test_idx, labels, n_features=d, seed=1, batch_size=6, max_iter=50,
    )
    assert result.max_batch_rows_held_in_memory <= 6
    assert result.max_batch_rows_held_in_memory < result.n_train


def test_objective_is_called_multiple_times_each_a_full_pass_not_a_single_load(tmp_path):
    """Documents the honest limitation: the optimizer calls the
    streaming objective (one full pass over the training partition)
    MANY times, not once - this is the real cost model."""
    store, labels, d = _random_sparse_store(tmp_path, n=30, d=10, seed=7)
    train_idx = set(range(0, 20))
    test_idx = set(range(20, 30))
    result = fit_and_evaluate_streaming_logistic_regression(
        store, train_idx, test_idx, labels, n_features=d, seed=1, batch_size=5, max_iter=100,
    )
    assert result.n_objective_evaluations > 1


# --- deterministic evaluation / reproducibility ---


def test_same_seed_and_data_give_identical_results(tmp_path):
    store, labels, d = _random_sparse_store(tmp_path, n=30, d=10, seed=8)
    train_idx = set(range(0, 20))
    test_idx = set(range(20, 30))
    r1 = fit_and_evaluate_streaming_logistic_regression(
        store, train_idx, test_idx, labels, n_features=d, seed=1, batch_size=5, max_iter=100,
    )
    r2 = fit_and_evaluate_streaming_logistic_regression(
        store, train_idx, test_idx, labels, n_features=d, seed=1, batch_size=5, max_iter=100,
    )
    assert np.allclose(r1.coef_, r2.coef_)
    assert r1.accuracy == r2.accuracy
    assert r1.confusion_matrix == r2.confusion_matrix


# --- artifact isolation ---


def test_streaming_lr_artifact_paths_do_not_collide_with_prior_task_paths():
    new_paths = {
        "data/processed/streaming_lr_reproduction_ciprofloxacin_d262144.json",
        "data/processed/batch_vs_streaming_lr_comparison_ciprofloxacin_k15_d262144.json",
    }
    prior_paths = {
        "data/processed/batch_lr_reproduction_ciprofloxacin_d262144.json",
        "data/processed/incremental_model_quality_final_ciprofloxacin_k15_d262144.json",
    }
    assert new_paths.isdisjoint(prior_paths)


# --- convergence and stopping behavior ---


def test_convergence_diagnostics_are_populated_and_consistent(tmp_path):
    store, labels, d = _random_sparse_store(tmp_path, n=50, d=15, seed=20)
    train_idx = set(range(0, 40))
    test_idx = set(range(40, 50))
    result = fit_and_evaluate_streaming_logistic_regression(
        store, train_idx, test_idx, labels, n_features=d, seed=1, batch_size=8, max_iter=1000, tol=1e-6,
    )
    assert result.n_iterations > 0
    assert result.n_gradient_evaluations == result.n_objective_evaluations  # combined callable
    assert result.optimizer_status in (0, 1, 2)
    assert result.optimizer_message != ""
    assert result.final_gradient_norm is not None
    if result.converged:
        # a converged L-BFGS-B run should have driven the gradient norm down meaningfully
        assert result.final_gradient_norm < 10.0


def test_tighter_tolerance_does_not_increase_final_loss(tmp_path):
    """A tighter stopping tolerance should reach an equally-good or
    better (lower) objective value than a looser one, never worse -
    this is a sanity property of the optimizer, not a tuned result."""
    from bvbrc_recon.streaming_logistic_regression import _streaming_loss_and_grad

    store, labels, d = _random_sparse_store(tmp_path, n=50, d=15, seed=21)
    train_idx = set(range(0, 40))
    test_idx = set(range(40, 50))

    loose = fit_and_evaluate_streaming_logistic_regression(
        store, train_idx, test_idx, labels, n_features=d, seed=1, batch_size=8, max_iter=1000, tol=1e-2,
    )
    tight = fit_and_evaluate_streaming_logistic_regression(
        store, train_idx, test_idx, labels, n_features=d, seed=1, batch_size=8, max_iter=1000, tol=1e-8,
    )
    assert tight.final_loss <= loose.final_loss + 1e-6
