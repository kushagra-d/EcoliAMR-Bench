"""Generalized incremental training/evaluation, parameterized by
estimator and preprocessing - built for this task's model-quality
investigation. `incremental_model.py` (the established,
already-tested SGDClassifier path) is left completely unmodified;
this is a SEPARATE, clearly scoped module, reusing the same
batch-at-a-time, bounded-memory contract against any `row_store`-like
object (`read_rows_for_indices(indices, batch_size) -> (genome_ids,
row_indices, csr_matrix)`), including `HybridTrainingReader`/
`MultiPartitionHybridReader`.

Supports any sklearn estimator with `partial_fit` and `predict_proba`
(both required: ROC-AUC/average-precision need real probability
scores, not just a hard label). Two estimator families are evaluated
in `reports/incremental_model_quality_assessment.md`:

  - `SGDClassifier` (loss in {"log_loss", "modified_huber"}) - the
    established discriminative linear path, now also tried with
    scikit-learn's "balanced" `class_weight` (computed ONCE from the
    real training partition's label counts, passed as an explicit
    dict - not the string "balanced", which scikit-learn recomputes
    per `partial_fit` call from only THAT batch's class distribution,
    an earlier task's finding).
  - `MultinomialNB` - a fundamentally different, generative model
    natively suited to non-negative COUNT features (exactly what this
    project's hashed k-mer representation is), with no learning-rate
    or regularization knobs to tune. Must be fed RAW counts, never
    L2-normalized - normalizing would turn integer counts into
    continuous fractions, breaking the multinomial-count assumption
    the model is built on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.preprocessing import normalize as sk_normalize

from .pilot_model import CLASS_TO_INT, NEGATIVE_CLASS, POSITIVE_CLASS, compute_classification_metrics


@dataclass
class CandidateModelResult:
    config_name: str
    seed: int
    batch_size: int
    n_epochs: int
    normalize_rows: bool
    n_train: int = 0
    n_test: int = 0
    n_train_batches: int = 0
    n_test_batches: int = 0
    train_class_counts: dict[str, int] = field(default_factory=dict)
    test_class_counts: dict[str, int] = field(default_factory=dict)
    accuracy: float | None = None
    roc_auc: float | None = None
    average_precision: float | None = None
    recall_per_class: dict[str, float | None] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    max_batch_rows_held_in_memory: int = 0
    fit_seconds: float = 0.0


def balanced_class_weight(label_counts: dict[str, int]) -> dict[int, float]:
    """sklearn's own "balanced" formula (`n_samples / (n_classes *
    n_c)`), computed ONCE from real label counts - not the string
    "balanced" passed to `SGDClassifier`, which is recomputed from only
    the CURRENT `partial_fit` call's batch, a noisy proxy for the true
    training-partition imbalance at small batch sizes."""
    n = sum(label_counts.values())
    n_classes = len(label_counts)
    return {CLASS_TO_INT[label]: n / (n_classes * count) for label, count in label_counts.items() if count > 0}


def evaluate_candidate(
    row_store,
    train_row_indices: set[int],
    test_row_indices: set[int],
    label_by_row_index: dict[int, str],
    seed: int,
    build_model,
    normalize_rows: bool,
    batch_size: int = 16,
    n_epochs: int = 50,
    config_name: str = "",
) -> CandidateModelResult:
    """`build_model()` must return a fresh, unfitted sklearn estimator
    with `partial_fit` and `predict_proba`. Training reads
    `train_row_indices` batch-by-batch, `n_epochs` times, via
    `row_store.read_rows_for_indices` - never materializing more than
    one batch at a time, regardless of estimator."""
    result = CandidateModelResult(
        config_name=config_name, seed=seed, batch_size=batch_size, n_epochs=n_epochs, normalize_rows=normalize_rows
    )
    model = build_model()
    max_batch_rows = 0
    fit_t0 = time.perf_counter()

    for epoch in range(n_epochs):
        for genome_ids, row_indices, X_batch in row_store.read_rows_for_indices(train_row_indices, batch_size):
            if normalize_rows:
                X_batch = sk_normalize(X_batch, norm="l2")
            y_batch = np.array([CLASS_TO_INT[label_by_row_index[ri]] for ri in row_indices])
            max_batch_rows = max(max_batch_rows, X_batch.shape[0])
            if epoch == 0 and result.n_train_batches == 0:
                model.partial_fit(X_batch, y_batch, classes=[0, 1])
            else:
                model.partial_fit(X_batch, y_batch)
            if epoch == 0:
                result.n_train += X_batch.shape[0]
                result.n_train_batches += 1
                for label in (NEGATIVE_CLASS, POSITIVE_CLASS):
                    result.train_class_counts[label] = result.train_class_counts.get(label, 0) + int(
                        (y_batch == CLASS_TO_INT[label]).sum()
                    )

    result.fit_seconds = time.perf_counter() - fit_t0

    if result.n_train == 0:
        result.warnings.append("No training rows were available for this split.")
        result.max_batch_rows_held_in_memory = max_batch_rows
        return result
    if result.train_class_counts.get(POSITIVE_CLASS, 0) == 0 or result.train_class_counts.get(NEGATIVE_CLASS, 0) == 0:
        result.warnings.append("Training partition contains only one class - model may be degenerate.")

    y_test_parts, y_pred_parts, y_proba_parts = [], [], []
    for genome_ids, row_indices, X_batch in row_store.read_rows_for_indices(test_row_indices, batch_size):
        if normalize_rows:
            X_batch = sk_normalize(X_batch, norm="l2")
        max_batch_rows = max(max_batch_rows, X_batch.shape[0])
        y_batch = np.array([CLASS_TO_INT[label_by_row_index[ri]] for ri in row_indices])
        y_test_parts.append(y_batch)
        y_pred_parts.append(model.predict(X_batch))
        y_proba_parts.append(model.predict_proba(X_batch)[:, 1])
        result.n_test += X_batch.shape[0]
        result.n_test_batches += 1
        for label in (NEGATIVE_CLASS, POSITIVE_CLASS):
            result.test_class_counts[label] = result.test_class_counts.get(label, 0) + int(
                (y_batch == CLASS_TO_INT[label]).sum()
            )

    result.max_batch_rows_held_in_memory = max_batch_rows

    if result.n_test == 0:
        result.warnings.append("No test rows were available for this split.")
        return result

    y_test = np.concatenate(y_test_parts)
    y_pred = np.concatenate(y_pred_parts)
    y_proba = np.concatenate(y_proba_parts)

    metrics = compute_classification_metrics(y_test, y_pred, y_proba)
    result.accuracy = metrics["accuracy"]
    result.roc_auc = metrics["roc_auc"]
    result.average_precision = metrics["average_precision"]
    result.confusion_matrix = metrics["confusion_matrix"]
    result.recall_per_class = metrics["recall_per_class"]
    result.warnings.extend(metrics["warnings"])
    return result
