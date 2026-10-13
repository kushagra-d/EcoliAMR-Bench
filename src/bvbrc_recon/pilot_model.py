"""One preliminary Logistic Regression pilot model.

Explicitly a PRELIMINARY PILOT result, not a benchmark: trained on a
~100-genome, non-lineage-aware, randomly split pilot cohort for one
antibiotic. Nothing here should be read as predicting the full
three-antibiotic benchmark's performance, and the train/test split here
is an ordinary random split - it is NOT lineage-independent and must not
be described as such.
"""

from __future__ import annotations

import time
import warnings as warnings_module
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

POSITIVE_CLASS = "Resistant"
NEGATIVE_CLASS = "Susceptible"
CLASS_TO_INT = {NEGATIVE_CLASS: 0, POSITIVE_CLASS: 1}


@dataclass
class PilotModelResult:
    seed: int
    test_size: float
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
    fitted_on_test_data: bool = False  # sanity flag, always verified False by the caller


def _validate_inputs(X: sparse.spmatrix, labels: list[str]) -> None:
    if not sparse.issparse(X):
        raise TypeError("X must be a scipy sparse matrix - this pilot never densifies features")
    if len(labels) != X.shape[0]:
        raise ValueError("labels length must match the number of matrix rows")
    if not set(labels) <= {POSITIVE_CLASS, NEGATIVE_CLASS}:
        raise ValueError(f"labels must be exactly {{'{POSITIVE_CLASS}', '{NEGATIVE_CLASS}'}}")


def compute_classification_metrics(y_test: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Shared metric computation, with explicit missing-class handling.

    Used by every model in this project's comparison (Logistic Regression,
    Random Forest, XGBoost - see `model_comparison.py`) so results are
    computed identically and comparably across models, not reimplemented
    per model with subtly different edge-case handling.
    """
    metrics: dict = {"warnings": []}
    metrics["accuracy"] = float((y_pred == y_test).mean()) if len(y_test) else None

    if len(set(y_test)) < 2:
        metrics["warnings"].append(
            "ROC-AUC, average precision, balanced accuracy, and macro-F1 are undefined: the test "
            "set contains only one class (this can happen with a small pilot and a highly "
            "imbalanced minority class, or with a lineage-aware split where entire MLST groups "
            "fall on one side)."
        )
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
        metrics["balanced_accuracy"] = None
        metrics["macro_f1"] = None
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        metrics["average_precision"] = float(average_precision_score(y_test, y_proba))
        # balanced accuracy = mean of per-class recall; every class is guaranteed
        # present here (len(set(y_test)) >= 2 checked above), so neither sklearn
        # function needs to guess at an undefined per-class recall.
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_test, y_pred))
        metrics["macro_f1"] = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

    metrics["recall_per_class"] = {}
    for class_name, class_int in CLASS_TO_INT.items():
        actual_positive = (y_test == class_int).sum()
        if actual_positive == 0:
            metrics["recall_per_class"][class_name] = None
            metrics["warnings"].append(
                f"Recall for class {class_name!r} is undefined: no test examples of that class."
            )
        else:
            true_positive = ((y_test == class_int) & (y_pred == class_int)).sum()
            metrics["recall_per_class"][class_name] = float(true_positive / actual_positive)

    metrics["precision_per_class"] = {}
    for class_name, class_int in CLASS_TO_INT.items():
        predicted_positive = (y_pred == class_int).sum()
        if predicted_positive == 0:
            metrics["precision_per_class"][class_name] = None
            metrics["warnings"].append(
                f"Precision for class {class_name!r} is undefined: the model predicted no examples "
                f"of that class in this test partition."
            )
        else:
            true_positive = ((y_test == class_int) & (y_pred == class_int)).sum()
            metrics["precision_per_class"][class_name] = float(true_positive / predicted_positive)

    return metrics


def compute_random_split_indices(
    labels: list[str], seed: int, test_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the exact (train_idx, test_idx) row positions that
    `train_and_evaluate_pilot_model` uses internally, WITHOUT needing the
    feature matrix - `train_test_split`'s stratified splitting logic
    depends only on `y`, `random_state`, and `test_size`, never on the
    feature values, so splitting `np.arange(n)` alongside `y` reproduces
    the identical partition as splitting `X` alongside `y`. Exists so
    other models (Random Forest, XGBoost) can reuse the EXACT same
    random-split row assignment as Logistic Regression without
    regenerating a new split, and so that assignment can be persisted
    for inspection. Equivalence is verified by a dedicated test."""
    y = np.array([CLASS_TO_INT[label] for label in labels])
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed, stratify=y)
    return train_idx, test_idx


def _fit_and_score(X_train, y_train: np.ndarray, X_test, y_test: np.ndarray, seed: int) -> PilotModelResult:
    """Fits one LogisticRegression on (X_train, y_train) ONLY, scores it on
    (X_test, y_test), and reports metrics with explicit handling for a
    missing class in the test partition. Shared by both the random-split
    and the lineage-group-split evaluation paths below, so both reuse
    exactly the same model configuration and metric logic."""
    result = PilotModelResult(seed=seed, test_size=float("nan"))  # test_size filled in by caller
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
            "Cannot fit: the training partition contains only one class - this LogisticRegression "
            "configuration requires both classes to be present in training data."
        )
        return result

    # sklearn's LogisticRegression (lbfgs, the default solver) accepts sparse
    # input natively - X_train/X_test are never densified. Fit on training
    # data ONLY; X_test/y_test are not touched until after fit() returns.
    model = LogisticRegression(max_iter=1000, random_state=seed)
    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always", ConvergenceWarning)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        result.fit_seconds = time.perf_counter() - t0
    for w in caught:
        if issubclass(w.category, ConvergenceWarning):
            result.warnings.append(f"ConvergenceWarning: {w.message}")

    if result.n_test == 0:
        result.warnings.append("Test partition is empty - no metrics can be computed.")
        return result

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    result.predict_seconds = time.perf_counter() - t0

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


def train_and_evaluate_pilot_model(
    X: sparse.spmatrix,
    labels: list[str],
    seed: int,
    test_size: float = 0.3,
) -> PilotModelResult:
    """Ordinary random, stratified train/test split - NOT lineage-aware."""
    _validate_inputs(X, labels)
    y = np.array([CLASS_TO_INT[label] for label in labels])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    result = _fit_and_score(X_train, y_train, X_test, y_test, seed)
    result.test_size = test_size
    return result


def evaluate_with_precomputed_split(
    X: sparse.spmatrix,
    labels: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> PilotModelResult:
    """Same model configuration and metrics as `train_and_evaluate_pilot_model`,
    but the train/test partition is supplied by the caller (e.g. a lineage-
    group-aware split) instead of being computed here - this function never
    looks at `labels` to decide the split, only to label the rows it's given."""
    _validate_inputs(X, labels)
    y = np.array([CLASS_TO_INT[label] for label in labels])

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    result = _fit_and_score(X_train, y_train, X_test, y_test, seed)
    result.test_size = len(test_idx) / (len(train_idx) + len(test_idx)) if (len(train_idx) + len(test_idx)) else float("nan")
    return result
