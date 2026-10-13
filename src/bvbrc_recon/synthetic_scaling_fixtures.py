"""Deterministic synthetic sparse datasets for scaling experiments on
`streaming_logistic_regression.py`, independent of any real genome
data. Every dataset here is clearly a SYNTHETIC BENCHMARK FIXTURE, not
a sample of real biological data, and must never be read as evidence
of real-world predictive performance - only of the optimizer's
resource/convergence scaling behavior as row count (n) and/or feature
count (d) grow.

Design choices, stated explicitly:
  - Rows are generated with a fixed density (default 20%, a deliberate
    middle ground - not the real pilot's ~100% density at k=15/d=262144,
    and not a typical sparse-text-style density either, chosen simply
    to keep synthetic generation and storage tractable across the
    larger (n, d) combinations this task's scaling grid requires).
  - Class balance is fixed at ~74%/26% (Susceptible/Resistant) across
    every scaling scenario, matching the real ciprofloxacin pilot's
    actual imbalance - this isolates the effect of n/d scaling from the
    effect of class balance, which was separately stress-tested (at a
    deliberately extreme 95/5 ratio) in the numerical-stability tests.
  - A weight vector `w_true` and a per-row score threshold define labels
    deterministically from a fixed seed, so every fixture is fully
    reproducible.
"""

from __future__ import annotations

import numpy as np

from .sqlite_row_store import SqliteRowStore

POSITIVE_CLASS = "Resistant"
NEGATIVE_CLASS = "Susceptible"
DEFAULT_DENSITY = 0.20
DEFAULT_POSITIVE_FRACTION = 0.26  # matches the real ciprofloxacin pilot's ~26% Resistant rate


def build_synthetic_store(path, n_rows: int, n_features: int, seed: int, density: float = DEFAULT_DENSITY):
    """Writes `n_rows` deterministic sparse rows (each ~`density` of
    `n_features` nonzero, non-negative integer "counts" - same shape
    as this project's real hashed k-mer representation) to a fresh
    `SqliteRowStore` at `path`, and returns (store, label_by_row_index).
    Labels are assigned by a fixed linear function of the row's own
    features plus noise, thresholded to hit `DEFAULT_POSITIVE_FRACTION`
    exactly - deterministic given `seed`."""
    rng = np.random.RandomState(seed)
    store = SqliteRowStore(path, n_buckets=n_features)

    n_nonzero_per_row = max(1, int(round(n_features * density)))
    w_true = rng.randn(n_features)

    scores = np.empty(n_rows)
    row_specs = []
    for i in range(n_rows):
        cols = rng.choice(n_features, size=n_nonzero_per_row, replace=False)
        vals = rng.randint(1, 20, size=n_nonzero_per_row)
        row_specs.append((cols, vals))
        scores[i] = float(np.dot(vals, w_true[cols]))

    # threshold to hit the target positive fraction exactly, deterministically
    threshold = np.quantile(scores, 1.0 - DEFAULT_POSITIVE_FRACTION)
    label_by_row_index = {}
    for i, (cols, vals) in enumerate(row_specs):
        bucket_counts = {int(c): int(v) for c, v in zip(cols, vals)}
        store.append_row(f"synthetic_{i}", i, bucket_counts)
        label_by_row_index[i] = POSITIVE_CLASS if scores[i] > threshold else NEGATIVE_CLASS

    return store, label_by_row_index


def train_test_split_indices(n_rows: int, seed: int, test_fraction: float = 0.3):
    """Simple deterministic shuffle-based split - synthetic fixtures
    don't need stratification machinery; rows are IID by construction."""
    rng = np.random.RandomState(seed)
    indices = np.arange(n_rows)
    rng.shuffle(indices)
    n_test = max(1, int(round(n_rows * test_fraction)))
    test_idx = set(int(i) for i in indices[:n_test])
    train_idx = set(int(i) for i in indices[n_test:])
    return train_idx, test_idx
