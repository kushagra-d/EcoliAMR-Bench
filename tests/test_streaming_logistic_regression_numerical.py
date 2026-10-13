import numpy as np
import pytest
from scipy import sparse

from bvbrc_recon.pilot_model import POSITIVE_CLASS, NEGATIVE_CLASS
from bvbrc_recon.sqlite_row_store import SqliteRowStore
from bvbrc_recon.streaming_logistic_regression import _streaming_loss_and_grad


def _build_store(tmp_path, n=20, d=12, seed=0, density=0.5):
    rng = np.random.RandomState(seed)
    store = SqliteRowStore(tmp_path / "rows.db", n_buckets=d)
    labels = {}
    for i in range(n):
        n_nonzero = max(1, int(round(d * density)))
        cols = rng.choice(d, size=n_nonzero, replace=False)
        vals = rng.randint(1, 10, size=n_nonzero)
        store.append_row(f"g{i}", i, {int(c): int(v) for c, v in zip(cols, vals)})
        labels[i] = POSITIVE_CLASS if rng.rand() > 0.5 else NEGATIVE_CLASS
    # guarantee both classes present
    labels[0] = POSITIVE_CLASS
    labels[1] = NEGATIVE_CLASS
    return store, labels, d


def _independent_dense_loss_grad(store, row_indices, labels, w, b, C):
    """A SEPARATE, independently-written dense computation of the same
    objective - deliberately not sharing any code path with
    `_streaming_loss_and_grad`, used as the ground truth to compare
    against."""
    row_indices = sorted(row_indices)
    d = w.shape[0]
    X = np.zeros((len(row_indices), d))
    y = np.zeros(len(row_indices))
    for local_i, ri in enumerate(row_indices):
        _, _, m = next(store.read_rows_for_indices({ri}, batch_size=1))
        X[local_i, :] = m.toarray()[0]
        y[local_i] = 1.0 if labels[ri] == POSITIVE_CLASS else -1.0

    z = X @ w + b
    margin = y * z
    loss = np.sum(np.log1p(np.exp(-margin))) + 0.5 * np.dot(w, w) / C
    sig = 1.0 / (1.0 + np.exp(margin))
    s = -y * sig
    grad_w = X.T @ s + w / C
    grad_b = np.sum(s)
    return loss, np.concatenate([grad_w, [grad_b]])


# --- independent full-matrix loss/gradient comparison ---


def test_streamed_loss_matches_independent_dense_computation(tmp_path):
    store, labels, d = _build_store(tmp_path, n=20, d=12, seed=1)
    row_indices = set(range(20))
    rng = np.random.RandomState(99)
    w = rng.randn(d) * 0.1
    b = 0.3

    loss_streamed, grad_streamed = _streaming_loss_and_grad(
        np.concatenate([w, [b]]), store, row_indices, labels, batch_size=5, C=1.0, max_batch_rows=[0]
    )
    loss_dense, grad_dense = _independent_dense_loss_grad(store, row_indices, labels, w, b, C=1.0)

    assert loss_streamed == pytest.approx(loss_dense, rel=1e-10)
    assert np.allclose(grad_streamed, grad_dense, rtol=1e-10, atol=1e-10)


def test_streamed_loss_matches_independent_dense_computation_different_batch_sizes(tmp_path):
    """The result must not depend on how rows happen to be chunked."""
    store, labels, d = _build_store(tmp_path, n=23, d=10, seed=2)  # 23 doesn't divide evenly
    row_indices = set(range(23))
    rng = np.random.RandomState(1)
    w = rng.randn(d) * 0.1
    b = -0.2

    results = []
    for batch_size in (1, 3, 7, 100):
        loss, grad = _streaming_loss_and_grad(
            np.concatenate([w, [b]]), store, row_indices, labels, batch_size=batch_size, C=1.0, max_batch_rows=[0]
        )
        results.append((loss, grad))

    base_loss, base_grad = results[0]
    for loss, grad in results[1:]:
        assert loss == pytest.approx(base_loss, rel=1e-10)
        assert np.allclose(grad, base_grad, rtol=1e-10, atol=1e-10)


# --- finite-difference gradient check ---


def test_gradient_matches_finite_differences(tmp_path):
    store, labels, d = _build_store(tmp_path, n=25, d=20, seed=3)
    row_indices = set(range(25))
    rng = np.random.RandomState(7)
    params = rng.randn(d + 1) * 0.05

    _, analytic_grad = _streaming_loss_and_grad(params, store, row_indices, labels, batch_size=6, C=1.0, max_batch_rows=[0])

    eps = 1e-5
    # check a random subset of dimensions (full d+1 would be slow and is unnecessary - the
    # same formula is applied uniformly to every coordinate, so a sample is representative)
    check_dims = sorted(rng.choice(d + 1, size=8, replace=False))
    for dim in check_dims:
        params_plus = params.copy()
        params_plus[dim] += eps
        params_minus = params.copy()
        params_minus[dim] -= eps
        loss_plus, _ = _streaming_loss_and_grad(params_plus, store, row_indices, labels, batch_size=6, C=1.0, max_batch_rows=[0])
        loss_minus, _ = _streaming_loss_and_grad(params_minus, store, row_indices, labels, batch_size=6, C=1.0, max_batch_rows=[0])
        numerical_grad = (loss_plus - loss_minus) / (2 * eps)
        assert numerical_grad == pytest.approx(analytic_grad[dim], abs=1e-3), f"dim {dim}"


# --- L2 regularization convention ---


def test_regularization_term_matches_expected_convention_exactly(tmp_path):
    """Isolates the regularization term: the DATA term does not depend
    on C, so the loss difference between two C values at a FIXED (w, b)
    must equal exactly 0.5*w.w*(1/C1 - 1/C2) - a clean, unambiguous
    check of the exact penalty convention (0.5*||w||^2/C, matching
    sklearn's documented "C = inverse regularization strength")."""
    store, labels, d = _build_store(tmp_path, n=15, d=10, seed=4)
    row_indices = set(range(15))
    rng = np.random.RandomState(5)
    w = rng.randn(d) * 0.3
    b = 0.1

    loss_c1, _ = _streaming_loss_and_grad(np.concatenate([w, [b]]), store, row_indices, labels, batch_size=4, C=1.0, max_batch_rows=[0])
    loss_c2, _ = _streaming_loss_and_grad(np.concatenate([w, [b]]), store, row_indices, labels, batch_size=4, C=4.0, max_batch_rows=[0])

    expected_diff = 0.5 * np.dot(w, w) * (1.0 / 1.0 - 1.0 / 4.0)
    assert (loss_c1 - loss_c2) == pytest.approx(expected_diff, rel=1e-10)


def test_intercept_is_not_regularized(tmp_path):
    """The gradient's intercept component must have NO 1/C dependence -
    only the data term contributes to grad_b."""
    store, labels, d = _build_store(tmp_path, n=15, d=10, seed=6)
    row_indices = set(range(15))
    rng = np.random.RandomState(8)
    w = rng.randn(d) * 0.3
    b = 0.5

    _, grad_c1 = _streaming_loss_and_grad(np.concatenate([w, [b]]), store, row_indices, labels, batch_size=4, C=1.0, max_batch_rows=[0])
    _, grad_c2 = _streaming_loss_and_grad(np.concatenate([w, [b]]), store, row_indices, labels, batch_size=4, C=9.0, max_batch_rows=[0])

    assert grad_c1[-1] == pytest.approx(grad_c2[-1], rel=1e-10)  # grad_b identical regardless of C
    assert not np.allclose(grad_c1[:-1], grad_c2[:-1])  # grad_w DOES change (it has w/C)


# --- numerical stability ---


def test_no_nan_or_inf_for_extreme_logits(tmp_path):
    store, labels, d = _build_store(tmp_path, n=15, d=10, seed=9)
    row_indices = set(range(15))
    huge_w = np.full(d, 1e4)
    params = np.concatenate([huge_w, [1e4]])

    loss, grad = _streaming_loss_and_grad(params, store, row_indices, labels, batch_size=4, C=1.0, max_batch_rows=[0])
    assert np.isfinite(loss)
    assert np.all(np.isfinite(grad))


def test_no_nan_or_inf_for_highly_imbalanced_labels(tmp_path):
    store = SqliteRowStore(tmp_path / "rows.db", n_buckets=10)
    labels = {}
    rng = np.random.RandomState(10)
    for i in range(40):
        cols = rng.choice(10, size=5, replace=False)
        vals = rng.randint(1, 10, size=5)
        store.append_row(f"g{i}", i, {int(c): int(v) for c, v in zip(cols, vals)})
        labels[i] = POSITIVE_CLASS if i < 2 else NEGATIVE_CLASS  # 2/40 = 5% positive

    params = np.zeros(11)
    loss, grad = _streaming_loss_and_grad(params, store, set(range(40)), labels, batch_size=8, C=1.0, max_batch_rows=[0])
    assert np.isfinite(loss)
    assert np.all(np.isfinite(grad))


# --- determinism ---


def test_streamed_loss_and_grad_is_bit_for_bit_deterministic(tmp_path):
    store, labels, d = _build_store(tmp_path, n=20, d=12, seed=11)
    row_indices = set(range(20))
    rng = np.random.RandomState(12)
    params = rng.randn(d + 1) * 0.1

    loss1, grad1 = _streaming_loss_and_grad(params, store, row_indices, labels, batch_size=5, C=1.0, max_batch_rows=[0])
    loss2, grad2 = _streaming_loss_and_grad(params, store, row_indices, labels, batch_size=5, C=1.0, max_batch_rows=[0])

    assert loss1 == loss2
    assert np.array_equal(grad1, grad2)
