import numpy as np
import pytest
from scipy import sparse

from bvbrc_recon.kmer_candidates import select_candidate_kmers
from bvbrc_recon.pilot_model import NEGATIVE_CLASS, POSITIVE_CLASS

VOCAB = {"TRAINSIGNAL": 0, "TESTSIGNAL": 1, "NOISE1": 2, "NOISE2": 3, "NOISE3": 4}


def _dataset_with_train_only_and_test_only_signals():
    """20 genomes (10 R / 10 S). Column 0 ("TRAINSIGNAL") perfectly
    separates the TRAINING rows (first 14) by class. Column 1
    ("TESTSIGNAL") perfectly separates the remaining TEST-only rows (last
    6) by class but is pure noise among the training rows. If candidate
    selection ever looked at test rows, TESTSIGNAL would rank at least as
    high as TRAINSIGNAL; it must not."""
    rng = np.random.RandomState(0)
    labels = [POSITIVE_CLASS] * 10 + [NEGATIVE_CLASS] * 10
    train_idx = np.array(list(range(7)) + list(range(10, 17)))  # 7 R + 7 S
    test_idx = np.array(list(range(7, 10)) + list(range(17, 20)))  # 3 R + 3 S

    rows, cols, data = [], [], []
    for i in range(20):
        is_resistant = i < 10
        is_train = i in train_idx
        # TRAINSIGNAL: perfectly separates class among TRAIN rows only
        rows.append(i)
        cols.append(0)
        if is_train:
            data.append(10 if is_resistant else 0)
        else:
            data.append(int(rng.randint(0, 10)))  # noise for test rows
        # TESTSIGNAL: perfectly separates class among TEST rows only
        rows.append(i)
        cols.append(1)
        if not is_train:
            data.append(10 if is_resistant else 0)
        else:
            data.append(int(rng.randint(0, 10)))  # noise for train rows
        # a few pure-noise columns
        for noise_col in (2, 3, 4):
            rows.append(i)
            cols.append(noise_col)
            data.append(int(rng.randint(0, 3)))

    X = sparse.coo_matrix((data, (rows, cols)), shape=(20, 5)).tocsr()
    return X, labels, train_idx, test_idx


def test_select_candidate_kmers_never_selects_a_test_only_signal():
    X, labels, train_idx, test_idx = _dataset_with_train_only_and_test_only_signals()
    selection = select_candidate_kmers(X, labels, VOCAB, train_idx, seed=1, top_n_per_model=2)

    lr_top = set(selection.rankings["LogisticRegression"].top_kmers)
    rf_top = set(selection.rankings["RandomForest"].top_kmers)

    assert "TRAINSIGNAL" in lr_top or "TRAINSIGNAL" in rf_top
    assert "TESTSIGNAL" not in lr_top
    assert "TESTSIGNAL" not in rf_top


def test_select_candidate_kmers_requires_two_classes_in_train():
    X, labels, train_idx, test_idx = _dataset_with_train_only_and_test_only_signals()
    single_class_train_idx = np.arange(10)  # all Resistant
    with pytest.raises(ValueError):
        select_candidate_kmers(X, labels, VOCAB, single_class_train_idx, seed=1)


def test_select_candidate_kmers_is_reproducible():
    X, labels, train_idx, test_idx = _dataset_with_train_only_and_test_only_signals()
    s1 = select_candidate_kmers(X, labels, VOCAB, train_idx, seed=5, top_n_per_model=3)
    s2 = select_candidate_kmers(X, labels, VOCAB, train_idx, seed=5, top_n_per_model=3)
    assert s1.combined_candidate_kmers == s2.combined_candidate_kmers
    assert s1.rankings["LogisticRegression"].top_scores == s2.rankings["LogisticRegression"].top_scores


def test_select_candidate_kmers_combined_list_is_deduplicated():
    X, labels, train_idx, test_idx = _dataset_with_train_only_and_test_only_signals()
    selection = select_candidate_kmers(X, labels, VOCAB, train_idx, seed=1, top_n_per_model=5)
    assert len(selection.combined_candidate_kmers) == len(set(selection.combined_candidate_kmers))


def test_select_candidate_kmers_records_n_train():
    X, labels, train_idx, test_idx = _dataset_with_train_only_and_test_only_signals()
    selection = select_candidate_kmers(X, labels, VOCAB, train_idx, seed=1, top_n_per_model=2)
    assert selection.n_train == len(train_idx)


def test_select_candidate_kmers_rejects_dense_input():
    X_dense = np.zeros((20, 5))
    labels = [POSITIVE_CLASS] * 10 + [NEGATIVE_CLASS] * 10
    with pytest.raises(TypeError):
        select_candidate_kmers(X_dense, labels, VOCAB, np.arange(14), seed=1)


def test_model_candidate_ranking_records_1_based_ranks():
    X, labels, train_idx, test_idx = _dataset_with_train_only_and_test_only_signals()
    selection = select_candidate_kmers(X, labels, VOCAB, train_idx, seed=1, top_n_per_model=3)
    lr_ranking = selection.rankings["LogisticRegression"]
    assert set(lr_ranking.top_ranks.values()) == set(range(1, len(lr_ranking.top_kmers) + 1))
