"""Investigates incremental-model quality (root-caused in an earlier
task to a mix of unscaled counts - already fixed via row normalization
- and a possibly ill-suited estimator/imbalance handling, still
unresolved). Tests a small, predefined, justified set of candidate
configurations via a TRAINING-ONLY internal validation split (never the
real test sets), then applies the winner once to both real saved
splits.

Reuses the existing SQLite store and `HybridTrainingReader` disk-tier
artifacts built in the prior task - does not rebuild them if already
valid, does not re-extract anything, does not download genomes.

Usage (from the project root, with the project venv active, AFTER the
SQLite store and hybrid cache artifacts from the prior tasks exist):

    python src/run_incremental_model_quality_investigation.py

Writes:
    data/processed/incremental_model_candidate_selection_ciprofloxacin_k15_d262144.json
    data/processed/incremental_model_quality_final_ciprofloxacin_k15_d262144.json
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

from bvbrc_recon.hybrid_training_reader import HybridTrainingReader, MultiPartitionHybridReader
from bvbrc_recon.incremental_model_candidates import balanced_class_weight, evaluate_candidate
from bvbrc_recon.pilot_model import CLASS_TO_INT, NEGATIVE_CLASS, POSITIVE_CLASS
from bvbrc_recon.sqlite_row_store import SqliteRowStore
from bvbrc_recon.streaming_kmer_build import peak_memory_mb

ANTIBIOTIC = "ciprofloxacin"
SEED = 20261009
INTERNAL_VAL_SEED = 7
K = 15
N_BUCKETS = 2**18
BATCH_SIZE = 16
N_EPOCHS = 50

DATA_PROCESSED = Path("data/processed")
DB_PATH = DATA_PROCESSED / f"out_of_core_v2_{ANTIBIOTIC}_rows_k{K}_d{N_BUCKETS}.db"
HYBRID_CACHE_DIR = DATA_PROCESSED / f"hybrid_cache_{ANTIBIOTIC}_k{K}_d{N_BUCKETS}"  # existing, from the prior task
INTERNAL_VAL_CACHE_DIR = DATA_PROCESSED / f"model_quality_internal_val_cache_{ANTIBIOTIC}_k{K}_d{N_BUCKETS}"
SELECTION_OUTPUT_PATH = DATA_PROCESSED / f"incremental_model_candidate_selection_{ANTIBIOTIC}_k{K}_d{N_BUCKETS}.json"
FINAL_OUTPUT_PATH = DATA_PROCESSED / f"incremental_model_quality_final_{ANTIBIOTIC}_k{K}_d{N_BUCKETS}.json"
BATCH_BASELINE_PATH = DATA_PROCESSED / f"batch_lr_reproduction_{ANTIBIOTIC}_d{N_BUCKETS}.json"


def load_pilot():
    genome_order = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_genome_order.csv", dtype={"genome_id": str}
    )["genome_id"].tolist()
    selected = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_selected_cohort.csv", dtype={"genome_id": str}
    )
    label_by_genome = dict(zip(selected["genome_id"], selected["label"]))
    random_split = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_random_split.csv", dtype={"genome_id": str}
    )
    lineage_split = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_lineage_split.csv", dtype={"genome_id": str}
    )
    label_by_row_index = {i: label_by_genome[gid] for i, gid in enumerate(genome_order) if gid in label_by_genome}
    return genome_order, label_by_row_index, random_split, lineage_split


def split_indices(genome_order, split_df):
    genome_to_partition = dict(zip(split_df["genome_id"], split_df["partition"]))
    train_idx = {i for i, gid in enumerate(genome_order) if genome_to_partition.get(gid) == "train"}
    test_idx = {i for i, gid in enumerate(genome_order) if genome_to_partition.get(gid) == "test"}
    return train_idx, test_idx


def make_candidates(train_label_counts: dict[str, int]):
    """The predefined, small, justified set - decided BEFORE looking at
    any validation or test metric. Each entry: (name, build_model,
    normalize_rows, justification)."""
    cw = balanced_class_weight(train_label_counts)
    return [
        (
            "sgd_log_loss_baseline",
            lambda: SGDClassifier(loss="log_loss", random_state=SEED),
            True,
            "The established path from the prior task (row-normalized, no "
            "class weighting) - the baseline this task tries to beat.",
        ),
        (
            "sgd_log_loss_balanced",
            lambda: SGDClassifier(loss="log_loss", random_state=SEED, class_weight=cw),
            True,
            "Same as baseline plus explicit class_weight computed ONCE from "
            "real training-label counts (not sklearn's per-batch 'balanced' "
            "string) - standard practice for imbalanced linear classification.",
        ),
        (
            "sgd_modified_huber_baseline",
            lambda: SGDClassifier(loss="modified_huber", random_state=SEED),
            True,
            "modified_huber is documented by scikit-learn as more robust to "
            "outliers/noise than log_loss, while still supporting predict_proba "
            "(required for ROC-AUC/average precision) - untried in prior tasks.",
        ),
        (
            "sgd_modified_huber_balanced",
            lambda: SGDClassifier(loss="modified_huber", random_state=SEED, class_weight=cw),
            True,
            "Combines the robust-loss and class-imbalance candidates above.",
        ),
        (
            "multinomial_nb_default_prior",
            lambda: MultinomialNB(),
            False,  # MUST use raw counts, never normalized - see module docstring
            "A fundamentally different, generative estimator natively suited "
            "to non-negative COUNT features (exactly this project's hashed "
            "k-mer representation) - no learning-rate/regularization tuning, "
            "supports partial_fit, never tried in this project before.",
        ),
        (
            "multinomial_nb_balanced_prior",
            lambda: MultinomialNB(class_prior=[0.5, 0.5]),
            False,
            "Same generative model, with an explicit uniform class prior so "
            "the decision does not default to the majority class's base rate.",
        ),
    ]


def select_best(results: list[dict]) -> dict:
    """Ranked primarily by average_precision (this task's explicit "pay "
    particular attention to... average precision" instruction), secondarily "
    by Resistant recall - never by accuracy alone, and never using any real "
    test-set metric (these are all internal-validation results)."""
    def sort_key(r):
        ap = r["average_precision"] if r["average_precision"] is not None else -1.0
        rr = r["recall_per_class"].get(POSITIVE_CLASS)
        rr = rr if rr is not None else -1.0
        return (ap, rr)

    ranked = sorted(results, key=sort_key, reverse=True)
    return ranked[0]


def main() -> None:
    genome_order, label_by_row_index, random_split, lineage_split = load_pilot()
    train_idx, test_idx = split_indices(genome_order, random_split)
    train_idx_lineage, test_idx_lineage = split_indices(genome_order, lineage_split)

    # --- Step 2 reference: already reproduced and verified bit-for-bit
    # identical to the original saved batch LR result in an earlier task;
    # reused here read-only, not recomputed. ---
    batch_baseline = json.loads(BATCH_BASELINE_PATH.read_text())
    print(f"batch LR baseline (reused, matches_existing_saved_result_exactly="
          f"{batch_baseline.get('matches_existing_saved_result_exactly')})")

    # --- Step 3/4: train-only internal validation carve from the RANDOM
    # split's train partition only - mirrors the precedent set in the
    # incremental-model-fix task. Never touches either real test set. ---
    train_idx_sorted = sorted(train_idx)
    y_train_labels = [label_by_row_index[i] for i in train_idx_sorted]
    y_train_int = np.array([CLASS_TO_INT[label] for label in y_train_labels])
    sub_train_idx, val_idx = train_test_split(
        train_idx_sorted, test_size=0.2, random_state=INTERNAL_VAL_SEED, stratify=y_train_int
    )
    sub_train_idx, val_idx = set(sub_train_idx), set(val_idx)

    sub_train_label_counts = {
        NEGATIVE_CLASS: sum(1 for i in sub_train_idx if label_by_row_index[i] == NEGATIVE_CLASS),
        POSITIVE_CLASS: sum(1 for i in sub_train_idx if label_by_row_index[i] == POSITIVE_CLASS),
    }
    print(f"internal validation carve: sub_train={len(sub_train_idx)} ({sub_train_label_counts}), val={len(val_idx)}")

    if INTERNAL_VAL_CACHE_DIR.exists():
        shutil.rmtree(INTERNAL_VAL_CACHE_DIR)
    store = SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS)
    sub_train_reader = HybridTrainingReader(
        INTERNAL_VAL_CACHE_DIR / "sub_train", store, sub_train_idx, batch_size=BATCH_SIZE, max_memory_cache_bytes=256_000_000
    )
    val_reader = HybridTrainingReader(
        INTERNAL_VAL_CACHE_DIR / "val", store, val_idx, batch_size=BATCH_SIZE, max_memory_cache_bytes=256_000_000
    )
    internal_reader = MultiPartitionHybridReader([sub_train_reader, val_reader])

    candidates = make_candidates(sub_train_label_counts)
    selection_results = []
    for name, build_model, normalize_rows, justification in candidates:
        result = evaluate_candidate(
            internal_reader, sub_train_idx, val_idx, label_by_row_index,
            seed=SEED, build_model=build_model, normalize_rows=normalize_rows,
            batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, config_name=name,
        )
        entry = {
            "config_name": name,
            "justification": justification,
            "normalize_rows": normalize_rows,
            "accuracy": result.accuracy,
            "roc_auc": result.roc_auc,
            "average_precision": result.average_precision,
            "recall_per_class": result.recall_per_class,
            "confusion_matrix": result.confusion_matrix,
            "warnings": result.warnings,
            "fit_seconds": result.fit_seconds,
        }
        selection_results.append(entry)
        print(f"[internal val] {name}: acc={result.accuracy} auc={result.roc_auc} "
              f"avg_prec={result.average_precision} recall={result.recall_per_class}")

    best = select_best(selection_results)
    print(f"SELECTED (by internal validation average_precision, then Resistant recall): {best['config_name']}")

    SELECTION_OUTPUT_PATH.write_text(json.dumps(
        {
            "internal_validation_carve": {
                "method": "stratified 80/20 of the random split's TRAIN rows only, seed="
                f"{INTERNAL_VAL_SEED}; never touches either real test partition",
                "sub_train_label_counts": sub_train_label_counts,
                "n_val": len(val_idx),
            },
            "candidates_evaluated": selection_results,
            "selected_config": best["config_name"],
            "selection_criterion": "average_precision (primary), Resistant recall (tie-break) - "
            "never accuracy alone, never a real test-set metric",
        },
        indent=2,
    ))
    print(f"wrote {SELECTION_OUTPUT_PATH}")

    # --- Step 5: apply the SAME selected config ONCE to both real splits'
    # real test sets, using the already-built real hybrid cache artifacts. ---
    candidates_by_name = {name: (build_model, normalize_rows) for name, build_model, normalize_rows, _ in candidates}

    real_train_label_counts_random = {
        NEGATIVE_CLASS: sum(1 for i in train_idx if label_by_row_index[i] == NEGATIVE_CLASS),
        POSITIVE_CLASS: sum(1 for i in train_idx if label_by_row_index[i] == POSITIVE_CLASS),
    }
    real_train_label_counts_lineage = {
        NEGATIVE_CLASS: sum(1 for i in train_idx_lineage if label_by_row_index[i] == NEGATIVE_CLASS),
        POSITIVE_CLASS: sum(1 for i in train_idx_lineage if label_by_row_index[i] == POSITIVE_CLASS),
    }
    # rebuild the chosen build_model callable using the REAL training
    # partition's label counts for each split (class_weight must come from
    # that split's own real training data, never the internal-validation
    # sub-train's counts, and never test data).
    candidates_random = make_candidates(real_train_label_counts_random)
    candidates_lineage = make_candidates(real_train_label_counts_lineage)
    build_model_random, normalize_rows_sel = next((b, n) for nm, b, n, _ in candidates_random if nm == best["config_name"])
    build_model_lineage, _ = next((b, n) for nm, b, n, _ in candidates_lineage if nm == best["config_name"])

    final_results = {}
    for split_name, t_idx, te_idx, reader_train_dir, reader_test_dir, build_model in [
        ("random", train_idx, test_idx, HYBRID_CACHE_DIR / "train", HYBRID_CACHE_DIR / "test", build_model_random),
        ("lineage", train_idx_lineage, test_idx_lineage, HYBRID_CACHE_DIR / "lineage" / "train", HYBRID_CACHE_DIR / "lineage" / "test", build_model_lineage),
    ]:
        store_fresh = SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS)
        train_reader = HybridTrainingReader(reader_train_dir, store_fresh, t_idx, batch_size=BATCH_SIZE, max_memory_cache_bytes=256_000_000)
        test_reader = HybridTrainingReader(reader_test_dir, store_fresh, te_idx, batch_size=BATCH_SIZE, max_memory_cache_bytes=256_000_000)
        reader = MultiPartitionHybridReader([train_reader, test_reader])

        mem_before = peak_memory_mb()
        result = evaluate_candidate(
            reader, t_idx, te_idx, label_by_row_index,
            seed=SEED, build_model=build_model, normalize_rows=normalize_rows_sel,
            batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, config_name=best["config_name"],
        )
        mem_after = peak_memory_mb()

        final_results[split_name] = {
            "config_name": best["config_name"],
            "n_train": result.n_train, "n_test": result.n_test,
            "train_class_counts": result.train_class_counts, "test_class_counts": result.test_class_counts,
            "accuracy": result.accuracy, "roc_auc": result.roc_auc, "average_precision": result.average_precision,
            "recall_per_class": result.recall_per_class, "confusion_matrix": result.confusion_matrix,
            "warnings": result.warnings, "fit_seconds": result.fit_seconds,
            "peak_rss_mb_before": mem_before, "peak_rss_mb_after": mem_after,
            "max_batch_rows_held_in_memory": result.max_batch_rows_held_in_memory,
        }
        print(f"[FINAL {split_name}] {best['config_name']}: acc={result.accuracy} auc={result.roc_auc} "
              f"avg_prec={result.average_precision} recall={result.recall_per_class} "
              f"fit_seconds={result.fit_seconds:.2f}")

    output = {
        "k": K, "n_buckets": N_BUCKETS, "antibiotic": ANTIBIOTIC,
        "selected_config": best["config_name"],
        "batch_lr_baseline": batch_baseline["model_results"],
        "incremental_final_results": final_results,
    }
    FINAL_OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"wrote {FINAL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
