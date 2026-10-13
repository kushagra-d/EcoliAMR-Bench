"""Selects candidate predictive k-mers for exploratory biological
interpretation, from a refit restricted to the TRAINING partition only.

No fitted model was persisted by the earlier pilot/model-comparison work
(confirmed by inspection - model_comparison.py and pilot_model.py return
metrics, never the fitted estimator), so this module refits the exact
same Logistic Regression and Random Forest configurations already used
elsewhere in this project, on the training rows of the EXISTING saved
random split only. Test-partition rows and labels are never read by
this module - selection cannot be influenced by them.

This is explicitly exploratory: a feature's rank here reflects a model
fit to only ~70 genomes with ~524,000 raw features, and is reported as
such, never as a validated biomarker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .pilot_model import CLASS_TO_INT

# Same configurations already used and reported elsewhere in this project
# (pilot_model.py, model_comparison.py) - reused verbatim, not retuned.
LOGISTIC_REGRESSION_PARAMS = {"max_iter": 1000}
RANDOM_FOREST_PARAMS = {"n_estimators": 100, "n_jobs": -1}


@dataclass
class ModelCandidateRanking:
    model_name: str
    params: dict
    top_kmers: list[str] = field(default_factory=list)
    top_scores: list[float] = field(default_factory=list)  # signed coef for LR, importance for RF
    top_ranks: dict[str, int] = field(default_factory=dict)  # kmer -> 1-based rank within this model


@dataclass
class CandidateSelection:
    seed: int
    n_train: int
    top_n_per_model: int
    rankings: dict[str, ModelCandidateRanking] = field(default_factory=dict)
    combined_candidate_kmers: list[str] = field(default_factory=list)


def select_candidate_kmers(
    X: sparse.spmatrix,
    labels: list[str],
    vocabulary: dict[str, int],
    train_idx: np.ndarray,
    seed: int,
    top_n_per_model: int = 20,
) -> CandidateSelection:
    if not sparse.issparse(X):
        raise TypeError("X must be a scipy sparse matrix")

    column_to_kmer = {col: kmer for kmer, col in vocabulary.items()}

    y_train = np.array([CLASS_TO_INT[label] for label in labels])[train_idx]
    X_train = X[train_idx]

    if len(set(y_train)) < 2:
        raise ValueError(
            "Cannot select candidates: the training partition contains only one class."
        )

    selection = CandidateSelection(seed=seed, n_train=X_train.shape[0], top_n_per_model=top_n_per_model)

    lr_params = {**LOGISTIC_REGRESSION_PARAMS, "random_state": seed}
    lr = LogisticRegression(**lr_params)
    lr.fit(X_train, y_train)
    lr_scores = lr.coef_[0]  # signed: positive -> pushes toward class 1 (Resistant)
    lr_ranking = _rank_by_absolute_score(lr_scores, column_to_kmer, top_n_per_model)
    selection.rankings["LogisticRegression"] = ModelCandidateRanking(
        model_name="LogisticRegression", params=lr_params, **lr_ranking
    )

    rf_params = {**RANDOM_FOREST_PARAMS, "random_state": seed}
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train, y_train)
    rf_scores = rf.feature_importances_  # unsigned
    rf_ranking = _rank_by_absolute_score(rf_scores, column_to_kmer, top_n_per_model)
    selection.rankings["RandomForest"] = ModelCandidateRanking(
        model_name="RandomForest", params=rf_params, **rf_ranking
    )

    combined = []
    seen = set()
    for ranking in selection.rankings.values():
        for kmer in ranking.top_kmers:
            if kmer not in seen:
                seen.add(kmer)
                combined.append(kmer)
    selection.combined_candidate_kmers = combined

    return selection


def _rank_by_absolute_score(scores: np.ndarray, column_to_kmer: dict[int, str], top_n: int) -> dict:
    order = np.argsort(-np.abs(scores))[:top_n]
    top_kmers = [column_to_kmer[int(col)] for col in order]
    top_scores = [float(scores[col]) for col in order]
    top_ranks = {kmer: rank + 1 for rank, kmer in enumerate(top_kmers)}
    return {"top_kmers": top_kmers, "top_scores": top_scores, "top_ranks": top_ranks}
