"""Bounded-memory-per-batch logistic regression that minimizes the
EXACT SAME mathematical objective as `pilot_model.py`'s batch
`sklearn.linear_model.LogisticRegression(max_iter=1000, random_state=seed)`
(solver="lbfgs", the sklearn default; default `penalty="l2"`, `C=1.0`,
`fit_intercept=True`) - not an approximation of it.

WHY THE ORIGINAL ESTIMATOR CANNOT BE USED OUT-OF-CORE DIRECTLY:
`LogisticRegression(solver="lbfgs")` has no `partial_fit`. Internally,
scipy's L-BFGS-B optimizer itself only needs O(d * history_size) memory
(the "limited memory" in L-BFGS - sklearn does not expose the history
size directly, but it is small and independent of the number of rows,
n) - the optimizer's OWN memory footprint was never the obstacle.
The obstacle is that sklearn's objective/gradient function
(`sklearn.linear_model._logistic._logistic_loss_and_grad`, called many
times by the optimizer) is handed the WHOLE matrix `X` as one array and
computes `X @ w` and `X.T @ residual` as single dense/sparse matrix
operations - this requires `X` to already exist as one in-memory object
before the first call.

THE ALTERNATIVE IMPLEMENTED HERE: the identical loss and gradient are
mathematically EXPRESSIBLE as a sum over rows -
`loss = sum_i log(1+exp(-y_i*(w.x_i+b))) + 0.5*||w||^2/C`,
`grad_w = sum_i s_i*x_i + w/C` where `s_i = -y_i*sigmoid(-y_i*(w.x_i+b))`,
`grad_b = sum_i s_i` - so they can be accumulated one batch at a time
via `row_store.read_rows_for_indices`, discarding each batch before the
next, with the running accumulators (`grad_w`: one d-length float64
array; `loss`/`grad_b`: scalars) as the only persistent per-pass state.
These accumulators are handed to `scipy.optimize.minimize(method="L-BFGS-B")`
as a standard `(loss, grad)`-returning callable - the SAME optimizer
family sklearn uses internally, just fed a streaming objective instead
of an in-memory one.

VALIDATED, NOT ASSUMED: compared directly against `sklearn.linear_model.LogisticRegression`
on a small synthetic dataset (`tests/test_streaming_logistic_regression.py`):
with a tight convergence tolerance, this implementation reaches a
LOWER value of the identical objective than sklearn's own default-
tolerance fit does - i.e. it is not merely "close," it more precisely
minimizes the same function sklearn's solver is also minimizing, with
the residual gap entirely attributable to sklearn's own default
stopping tolerance, not a difference in objective.

THE HONEST LIMITATION, stated per this task's explicit instruction:
each loss/gradient evaluation requires a FULL PASS over the training
partition (one `read_rows_for_indices` call per L-BFGS-B evaluation),
and L-BFGS-B's line search may call the objective MANY times per outer
iteration. This is "bounded peak memory per read" (never more than one
batch's rows materialized at once), NOT "fewer total reads than the
in-memory version" - with a bounded, non-evicting or hybrid cache
reader (as used here), repeat passes after the first are fast, but the
mechanism is still "read the whole partition again," not something
genuinely cheaper than that for i>1 unless the reader's cache already
holds everything.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .pilot_model import POSITIVE_CLASS, compute_classification_metrics

DEFAULT_C = 1.0
DEFAULT_MAX_ITER = 1000
DEFAULT_TOL = 1e-4


@dataclass
class StreamingLogisticRegressionResult:
    seed: int
    C: float
    max_iter: int
    tol: float
    n_train: int = 0
    n_test: int = 0
    n_objective_evaluations: int = 0  # how many FULL passes over the training partition this run performed
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
    converged: bool = False
    final_loss: float | None = None
    coef_: np.ndarray | None = None
    intercept_: float | None = None
    n_iterations: int = 0  # scipy's `nit` - outer L-BFGS-B iterations
    n_gradient_evaluations: int = 0  # scipy's `njev` - equals n_objective_evaluations here (combined callable)
    optimizer_status: int = -1  # scipy's `status` code
    optimizer_message: str = ""
    final_gradient_norm: float | None = None  # ||grad|| at the reported solution, from scipy's own `jac`


def _streaming_loss_and_grad(
    params: np.ndarray,
    row_store,
    train_row_indices: set[int],
    label_by_row_index: dict[int, str],
    batch_size: int,
    C: float,
    max_batch_rows: list[int],
) -> tuple[float, np.ndarray]:
    """One FULL streaming pass over `train_row_indices`'s batches.
    Mathematically identical to evaluating sklearn's L2-regularized
    logistic loss/gradient on the complete in-memory matrix at once -
    see module docstring for the derivation and its empirical
    validation. `max_batch_rows` is a 1-element list used as an
    out-parameter to record the largest batch actually materialized."""
    w = params[:-1]
    b = params[-1]
    n_features = w.shape[0]
    grad_w = np.zeros(n_features, dtype=np.float64)
    grad_b = 0.0
    loss = 0.0

    for genome_ids, row_indices, X_batch in row_store.read_rows_for_indices(train_row_indices, batch_size):
        max_batch_rows[0] = max(max_batch_rows[0], X_batch.shape[0])
        y_batch = np.array(
            [1.0 if label_by_row_index[ri] == POSITIVE_CLASS else -1.0 for ri in row_indices], dtype=np.float64
        )
        z = X_batch.dot(w) + b
        margin = y_batch * z
        # stable log(1+exp(-margin)): naively computing -log(expit(margin))
        # underflows to log(0)=-inf whenever margin is very negative (which
        # happens routinely here - 262,144 features vs ~70 rows means the
        # data is almost certainly linearly separable, so early L-BFGS-B
        # steps can push margins to extreme values before regularization
        # reins them in). This was measured, not assumed: an early version
        # hit exactly this and silently "converged" after 3 evaluations to
        # a degenerate point (AUC=0.5) - the fix below is required, not
        # cosmetic.
        neg_margin = -margin
        loss += np.sum(np.maximum(neg_margin, 0) + np.log1p(np.exp(-np.abs(neg_margin))))
        s = -y_batch * expit(-margin)
        grad_w += X_batch.T.dot(s)
        grad_b += np.sum(s)

    loss += 0.5 * np.dot(w, w) / C
    grad_w += w / C
    return loss, np.concatenate([grad_w, [grad_b]])


def fit_and_evaluate_streaming_logistic_regression(
    row_store,
    train_row_indices: set[int],
    test_row_indices: set[int],
    label_by_row_index: dict[int, str],
    n_features: int,
    seed: int,
    batch_size: int = 16,
    C: float = DEFAULT_C,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
) -> StreamingLogisticRegressionResult:
    """Fits by minimizing the streaming objective via L-BFGS-B (never
    materializing the full training matrix as one object - only one
    batch at a time, plus the O(n_features) parameter/gradient
    vectors), then evaluates on the test partition the same way."""
    result = StreamingLogisticRegressionResult(seed=seed, C=C, max_iter=max_iter, tol=tol)

    train_label_counts: dict[str, int] = {}
    for ri in train_row_indices:
        label = label_by_row_index[ri]
        train_label_counts[label] = train_label_counts.get(label, 0) + 1
    result.train_class_counts = train_label_counts
    result.n_train = len(train_row_indices)

    if len(train_label_counts) < 2:
        result.warnings.append("Cannot fit: the training partition contains only one class.")
        return result

    x0 = np.zeros(n_features + 1, dtype=np.float64)
    max_batch_rows = [0]
    n_evals = [0]

    def objective(params):
        n_evals[0] += 1
        return _streaming_loss_and_grad(params, row_store, train_row_indices, label_by_row_index, batch_size, C, max_batch_rows)

    fit_t0 = time.perf_counter()
    opt_result = minimize(
        objective, x0, jac=True, method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": tol * np.finfo(float).eps, "gtol": tol},
    )
    result.fit_seconds = time.perf_counter() - fit_t0
    result.n_objective_evaluations = n_evals[0]
    result.converged = bool(opt_result.success)
    result.n_iterations = int(opt_result.nit)
    result.n_gradient_evaluations = int(opt_result.njev)
    result.optimizer_status = int(opt_result.status)
    result.optimizer_message = str(opt_result.message)
    result.final_gradient_norm = float(np.linalg.norm(opt_result.jac))
    result.final_loss = float(opt_result.fun)
    if not opt_result.success:
        result.warnings.append(f"L-BFGS-B did not report success: {opt_result.message}")

    w, b = opt_result.x[:-1], opt_result.x[-1]
    result.coef_ = w
    result.intercept_ = float(b)

    test_label_counts: dict[str, int] = {}
    y_test_parts, y_pred_parts, y_proba_parts = [], [], []
    for genome_ids, row_indices, X_batch in row_store.read_rows_for_indices(test_row_indices, batch_size):
        max_batch_rows[0] = max(max_batch_rows[0], X_batch.shape[0])
        for ri in row_indices:
            label = label_by_row_index[ri]
            test_label_counts[label] = test_label_counts.get(label, 0) + 1
        proba_pos = expit(X_batch.dot(w) + b)
        y_batch_01 = np.array([1 if label_by_row_index[ri] == POSITIVE_CLASS else 0 for ri in row_indices])
        y_test_parts.append(y_batch_01)
        y_pred_parts.append((proba_pos > 0.5).astype(int))
        y_proba_parts.append(proba_pos)

    result.test_class_counts = test_label_counts
    result.max_batch_rows_held_in_memory = max_batch_rows[0]
    result.n_test = sum(test_label_counts.values())

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
