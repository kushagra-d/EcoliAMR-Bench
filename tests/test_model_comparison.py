import sys

import numpy as np
import pytest
from scipy import sparse

from bvbrc_recon.model_comparison import (
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
    evaluate_random_forest,
    evaluate_xgboost,
)
from bvbrc_recon.pilot_model import POSITIVE_CLASS, NEGATIVE_CLASS, compute_random_split_indices


def _separable_dataset(n_per_class=15, n_extra_features=200):
    rng = np.random.RandomState(0)
    rows, cols, data, labels = [], [], [], []
    row = 0
    for _ in range(n_per_class):
        rows += [row, row]
        cols += [0, 2]
        data += [5, int(rng.randint(0, 2))]
        labels.append(POSITIVE_CLASS)
        row += 1
    for _ in range(n_per_class):
        rows += [row, row]
        cols += [1, 2]
        data += [5, int(rng.randint(0, 2))]
        labels.append(NEGATIVE_CLASS)
        row += 1
    X = sparse.coo_matrix((data, (rows, cols)), shape=(row, n_extra_features)).tocsr()
    return X, labels


@pytest.fixture
def dataset_and_split():
    X, labels = _separable_dataset(15)
    train_idx, test_idx = compute_random_split_indices(labels, seed=1, test_size=0.3)
    return X, labels, train_idx, test_idx


# --- Random Forest ---------------------------------------------------


def test_random_forest_is_reproducible_with_fixed_seed(dataset_and_split):
    X, labels, train_idx, test_idx = dataset_and_split
    r1 = evaluate_random_forest(X, labels, train_idx, test_idx, seed=7)
    r2 = evaluate_random_forest(X, labels, train_idx, test_idx, seed=7)
    assert r1.accuracy == r2.accuracy
    assert r1.confusion_matrix == r2.confusion_matrix


def test_random_forest_accepts_sparse_input_without_densifying(dataset_and_split, monkeypatch):
    X, labels, train_idx, test_idx = dataset_and_split
    calls = {"n": 0}
    original = sparse.csr_matrix.toarray

    def tracking(self, *a, **kw):
        calls["n"] += 1
        return original(self, *a, **kw)

    monkeypatch.setattr(sparse.csr_matrix, "toarray", tracking)
    evaluate_random_forest(X, labels, train_idx, test_idx, seed=1)
    assert calls["n"] == 0


def test_random_forest_records_parameters(dataset_and_split):
    X, labels, train_idx, test_idx = dataset_and_split
    result = evaluate_random_forest(X, labels, train_idx, test_idx, seed=3)
    assert result.params["random_state"] == 3
    assert result.params["n_estimators"] == RANDOM_FOREST_PARAMS["n_estimators"]


def test_random_forest_uses_the_exact_given_split_not_a_new_one(dataset_and_split):
    X, labels, train_idx, test_idx = dataset_and_split
    result = evaluate_random_forest(X, labels, train_idx, test_idx, seed=1)
    assert result.n_train == len(train_idx)
    assert result.n_test == len(test_idx)


def test_random_forest_reports_timing():
    X, labels = _separable_dataset(15)
    train_idx, test_idx = compute_random_split_indices(labels, seed=1, test_size=0.3)
    result = evaluate_random_forest(X, labels, train_idx, test_idx, seed=1)
    assert result.fit_seconds is not None and result.fit_seconds >= 0
    assert result.predict_seconds is not None and result.predict_seconds >= 0


def test_random_forest_handles_single_class_test_partition(dataset_and_split):
    X, labels, train_idx, test_idx = dataset_and_split
    # force an all-negative test partition
    y = np.array([0 if lbl == NEGATIVE_CLASS else 1 for lbl in labels])
    forced_test_idx = np.where(y == 0)[0][:3]
    forced_train_idx = np.setdiff1d(np.arange(len(labels)), forced_test_idx)
    result = evaluate_random_forest(X, labels, forced_train_idx, forced_test_idx, seed=1)
    assert result.roc_auc is None
    assert any("undefined" in w for w in result.warnings)
    assert result.recall_per_class[POSITIVE_CLASS] is None


def test_random_forest_reports_error_on_fit_failure(dataset_and_split, monkeypatch):
    X, labels, train_idx, test_idx = dataset_and_split

    def boom(self, *a, **kw):
        raise MemoryError("simulated allocation failure")

    monkeypatch.setattr("sklearn.ensemble.RandomForestClassifier.fit", boom)
    result = evaluate_random_forest(X, labels, train_idx, test_idx, seed=1)
    assert result.error is not None
    assert "MemoryError" in result.error
    assert result.accuracy is None  # no fabricated metrics after a failure


def test_random_forest_rejects_dense_input():
    X_dense = np.zeros((10, 5))
    with pytest.raises(TypeError):
        evaluate_random_forest(X_dense, [POSITIVE_CLASS] * 10, np.arange(5), np.arange(5, 10), seed=1)


# --- XGBoost -----------------------------------------------------------


def test_xgboost_is_reproducible_with_fixed_seed(dataset_and_split):
    X, labels, train_idx, test_idx = dataset_and_split
    r1 = evaluate_xgboost(X, labels, train_idx, test_idx, seed=7)
    r2 = evaluate_xgboost(X, labels, train_idx, test_idx, seed=7)
    assert r1.accuracy == r2.accuracy
    assert r1.confusion_matrix == r2.confusion_matrix


def test_xgboost_accepts_sparse_input_without_densifying(dataset_and_split, monkeypatch):
    X, labels, train_idx, test_idx = dataset_and_split
    calls = {"n": 0}
    original = sparse.csr_matrix.toarray

    def tracking(self, *a, **kw):
        calls["n"] += 1
        return original(self, *a, **kw)

    monkeypatch.setattr(sparse.csr_matrix, "toarray", tracking)
    evaluate_xgboost(X, labels, train_idx, test_idx, seed=1)
    assert calls["n"] == 0


def test_xgboost_records_parameters(dataset_and_split):
    X, labels, train_idx, test_idx = dataset_and_split
    result = evaluate_xgboost(X, labels, train_idx, test_idx, seed=3)
    assert result.params["random_state"] == 3
    assert result.params["n_estimators"] == XGBOOST_PARAMS["n_estimators"]


def test_xgboost_uses_the_exact_given_split(dataset_and_split):
    X, labels, train_idx, test_idx = dataset_and_split
    result = evaluate_xgboost(X, labels, train_idx, test_idx, seed=1)
    assert result.n_train == len(train_idx)
    assert result.n_test == len(test_idx)


def test_xgboost_reports_timing():
    X, labels = _separable_dataset(15)
    train_idx, test_idx = compute_random_split_indices(labels, seed=1, test_size=0.3)
    result = evaluate_xgboost(X, labels, train_idx, test_idx, seed=1)
    assert result.fit_seconds is not None and result.fit_seconds >= 0


def test_xgboost_handles_single_class_test_partition(dataset_and_split):
    X, labels, train_idx, test_idx = dataset_and_split
    y = np.array([0 if lbl == NEGATIVE_CLASS else 1 for lbl in labels])
    forced_test_idx = np.where(y == 0)[0][:3]
    forced_train_idx = np.setdiff1d(np.arange(len(labels)), forced_test_idx)
    result = evaluate_xgboost(X, labels, forced_train_idx, forced_test_idx, seed=1)
    assert result.roc_auc is None
    assert any("undefined" in w for w in result.warnings)


def test_xgboost_reports_error_on_fit_failure(dataset_and_split, monkeypatch):
    X, labels, train_idx, test_idx = dataset_and_split

    def boom(self, *a, **kw):
        raise ValueError("simulated estimator failure")

    monkeypatch.setattr("xgboost.XGBClassifier.fit", boom)
    result = evaluate_xgboost(X, labels, train_idx, test_idx, seed=1)
    assert result.error is not None
    assert result.accuracy is None


def test_xgboost_reports_import_error_without_crashing(dataset_and_split, monkeypatch):
    X, labels, train_idx, test_idx = dataset_and_split
    monkeypatch.setitem(sys.modules, "xgboost", None)
    result = evaluate_xgboost(X, labels, train_idx, test_idx, seed=1)
    assert result.error is not None
    assert "xgboost" in result.error.lower()


def test_xgboost_rejects_dense_input():
    X_dense = np.zeros((10, 5))
    with pytest.raises(TypeError):
        evaluate_xgboost(X_dense, [POSITIVE_CLASS] * 10, np.arange(5), np.arange(5, 10), seed=1)
