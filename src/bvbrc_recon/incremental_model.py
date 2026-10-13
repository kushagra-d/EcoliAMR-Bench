"""Genuinely incremental (out-of-core) Logistic-Regression-equivalent
training and evaluation on a `DiskBackedRowStore`.

scikit-learn's `LogisticRegression` (used everywhere else in this
project) has NO `partial_fit` - it requires its entire training matrix
in memory for a single `fit()` call, so it cannot be used out-of-core
without defeating the point of this module. Instead this uses
`SGDClassifier(loss="log_loss")`, scikit-learn's own documented
incremental alternative to logistic regression: it supports
`partial_fit(X_batch, y_batch)`, so training never requires more than
one BATCH of rows in memory at a time, regardless of how many genomes
exist in total.

This is NOT claimed to be numerically identical to the project's
existing `LogisticRegression` results elsewhere (different optimizer -
mini-batch SGD vs. a batch solver - and different regularization
dynamics) - it is reported as its own, separately-labeled result.

ROOT CAUSE of the original ROC-AUC=0.5 / zero-Resistant-recall collapse
(diagnosed empirically against the real 100-genome pilot, logged in
`reports/incremental_model_validation.md`): the hashed features are RAW
k-mer counts, and each genome's total count varies with genome
length/sequencing depth - a scale confound unrelated to resistance
phenotype. Mini-batch SGD is far more sensitive to feature/sample scale
than the batch `LogisticRegression` baseline's `lbfgs` solver (which
tolerates this fine over up to 1000 full-batch iterations). Applying
per-row L2 normalization (`normalize_rows`, below) - a stateless,
leak-free transform with no fitted parameters, standard for count-based
feature vectors - was the single change that moved ROC-AUC off exactly
0.5 on the real test data. `class_weight` and alternative learning-rate
schedules were investigated and did NOT show a consistent benefit on a
training-only internal validation split (see the report); they are not
used here, to avoid introducing untested knobs. `n_epochs` was raised
from 5 to 50, chosen via that same training-only internal validation
split, not by looking at test performance.

A persisting, honestly-reported limitation: even with normalization,
ROC-AUC improves but remains noisy and highly sensitive to epoch count on
this specific ~70-training-row, 262,144-feature pilot (a severely
underdetermined regime for any linear model) - and the standard 0.5
probability threshold still does not reliably produce a nonzero
Resistant recall on every split. This is reported as-is; no custom
decision threshold was tuned to force a better-looking recall number.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize as sk_normalize

from .pilot_model import CLASS_TO_INT, NEGATIVE_CLASS, POSITIVE_CLASS, compute_classification_metrics
from .streaming_hashed_matrix import DiskBackedRowStore

# `partial_fit` (not `.fit()`) does the incremental learning here, so
# fit()-only params like max_iter/tol/warm_start are irrelevant and
# deliberately omitted - each partial_fit call does one SGD pass over
# its batch; repeated calls across batches/epochs is the mechanism.
SGD_PARAMS = {"loss": "log_loss"}


@dataclass
class IncrementalModelResult:
    seed: int
    batch_size: int
    n_epochs: int
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
    max_batch_rows_held_in_memory: int = 0  # the ONLY matrix size ever materialized at once
    normalize_rows: bool = True
    fit_seconds: float = 0.0


def evaluate_streaming_hashed_matrix(
    row_store: DiskBackedRowStore,
    train_row_indices: set[int],
    test_row_indices: set[int],
    label_by_row_index: dict[int, str],
    seed: int,
    batch_size: int = 16,
    n_epochs: int = 50,
    normalize_rows: bool = True,
) -> IncrementalModelResult:
    result = IncrementalModelResult(seed=seed, batch_size=batch_size, n_epochs=n_epochs, normalize_rows=normalize_rows)

    model = SGDClassifier(random_state=seed, **SGD_PARAMS)
    max_batch_rows = 0
    fit_t0 = time.perf_counter()

    # --- incremental training: at most one BATCH of rows in memory at a time ---
    for epoch in range(n_epochs):
        for genome_ids, row_indices, X_batch in row_store.read_rows_for_indices(train_row_indices, batch_size):
            if normalize_rows:
                # stateless, per-row L2 normalization - no fitted parameters,
                # so applying it identically to train and test batches cannot
                # leak test information into training.
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

    # --- batched test-time prediction: never loads the full test set at once either ---
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
