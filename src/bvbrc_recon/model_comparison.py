"""Random Forest and XGBoost, evaluated on the EXACT same pilot feature
matrix, labels, and train/test row assignments already used for Logistic
Regression - no new split is ever generated here.

Both estimators are configured once, documented below, and never tuned
against test-set results. Metrics are computed via
`pilot_model.compute_classification_metrics`, the same function used for
Logistic Regression, so all three models' results are directly comparable
and share identical edge-case handling (missing class in a partition,
etc).

PRELIMINARY PILOT RESULTS ONLY - see reports/pilot_model_comparison.md.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier

from .pilot_model import CLASS_TO_INT, NEGATIVE_CLASS, POSITIVE_CLASS, compute_classification_metrics

# Library defaults plus only what's needed for reproducibility/performance -
# no grid search, no tuning against test results. n_estimators=100 is
# scikit-learn's own RandomForestClassifier default, n_jobs=-1 just uses
# available cores (does not change results), random_state fixes
# reproducibility. max_features stays at sklearn's own classifier default
# ("sqrt") - deliberately NOT raised to use all 524k features per split,
# since that default is exactly the standard mitigation for the p >> n
# (524,346 features vs 100 genomes) regime this pilot's k=10 matrix is in.
RANDOM_FOREST_PARAMS = {"n_estimators": 100, "n_jobs": -1}

# XGBoost's own library defaults, plus only n_jobs/random_state for the same
# reasons as above. tree_method is left unset (XGBoost 3.x auto-selects a
# histogram-based method suited to the input, including sparse input).
XGBOOST_PARAMS = {"n_estimators": 100, "n_jobs": -1}


@dataclass
class ModelEvalResult:
    model_name: str
    params: dict
    seed: int
    n_train: int = 0
    n_test: int = 0
    train_class_counts: dict[str, int] = field(default_factory=dict)
    test_class_counts: dict[str, int] = field(default_factory=dict)
    accuracy: float | None = None
    roc_auc: float | None = None
    average_precision: float | None = None
    balanced_accuracy: float | None = None
    macro_f1: float | None = None
    recall_per_class: dict[str, float | None] = field(default_factory=dict)
    precision_per_class: dict[str, float | None] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fit_seconds: float | None = None
    predict_seconds: float | None = None
    error: str | None = None


def _labels_to_int(labels: list[str]) -> np.ndarray:
    return np.array([CLASS_TO_INT[label] for label in labels])


def _evaluate_estimator(
    model_name: str,
    estimator,
    params: dict,
    X: sparse.spmatrix,
    labels: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> ModelEvalResult:
    if not sparse.issparse(X):
        raise TypeError("X must be a scipy sparse matrix - this comparison never densifies features")
    if len(labels) != X.shape[0]:
        raise ValueError("labels length must match the number of matrix rows")

    result = ModelEvalResult(model_name=model_name, params=dict(params), seed=seed)
    y = _labels_to_int(labels)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    result.n_train, result.n_test = X_train.shape[0], X_test.shape[0]
    result.train_class_counts = {
        NEGATIVE_CLASS: int((y_train == 0).sum()),
        POSITIVE_CLASS: int((y_train == 1).sum()),
    }
    result.test_class_counts = {
        NEGATIVE_CLASS: int((y_test == 0).sum()),
        POSITIVE_CLASS: int((y_test == 1).sum()),
    }

    if len(set(y_train)) < 2:
        result.warnings.append(
            "Cannot fit: the training partition contains only one class."
        )
        return result
    if result.n_test == 0:
        result.warnings.append("Test partition is empty - no metrics can be computed.")
        return result

    try:
        t0 = time.perf_counter()
        estimator.fit(X_train, y_train)  # fit on TRAINING data only
        result.fit_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = estimator.predict(X_test)
        y_proba = estimator.predict_proba(X_test)[:, 1]
        result.predict_seconds = time.perf_counter() - t0
    except (MemoryError, ValueError, RuntimeError) as exc:
        # Report the exact failure rather than silently falling back to a
        # different estimator or skipping the model.
        result.error = f"{type(exc).__name__}: {exc}"
        result.warnings.append(f"{model_name} failed to fit/predict: {result.error}")
        return result

    metrics = compute_classification_metrics(y_test, y_pred, y_proba)
    result.accuracy = metrics["accuracy"]
    result.roc_auc = metrics["roc_auc"]
    result.average_precision = metrics["average_precision"]
    result.balanced_accuracy = metrics["balanced_accuracy"]
    result.macro_f1 = metrics["macro_f1"]
    result.confusion_matrix = metrics["confusion_matrix"]
    result.recall_per_class = metrics["recall_per_class"]
    result.precision_per_class = metrics["precision_per_class"]
    result.warnings.extend(metrics["warnings"])
    return result


def evaluate_random_forest(
    X: sparse.spmatrix,
    labels: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> ModelEvalResult:
    params = {**RANDOM_FOREST_PARAMS, "random_state": seed}
    estimator = RandomForestClassifier(**params)
    return _evaluate_estimator("RandomForest", estimator, params, X, labels, train_idx, test_idx, seed)


def evaluate_xgboost(
    X: sparse.spmatrix,
    labels: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> ModelEvalResult:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        result = ModelEvalResult(model_name="XGBoost", params=dict(XGBOOST_PARAMS), seed=seed)
        result.error = f"xgboost is not importable: {exc}"
        result.warnings.append(result.error)
        return result

    params = {**XGBOOST_PARAMS, "random_state": seed}
    estimator = XGBClassifier(**params)
    return _evaluate_estimator("XGBoost", estimator, params, X, labels, train_idx, test_idx, seed)
