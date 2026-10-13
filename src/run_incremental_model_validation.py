"""Steps 3-4 of the incremental-model-fix task: evaluates the FIXED
`evaluate_streaming_hashed_matrix` (row L2-normalization + n_epochs=50,
both decided via a training-only internal validation split, documented
in `reports/incremental_model_validation.md` - not by looking at test
performance) against the existing SQLite-backed hashed matrix, on both
existing splits, and compares against the reproduced batch
LogisticRegression baseline (`run_batch_baseline_reproduction.py`).

Reuses the ALREADY-BUILT store from the prior task
(`out_of_core_v2_ciprofloxacin_rows_k15_d262144.db`) - does not rebuild
it, does not re-extract anything, does not download genomes.

Usage (from the project root, with the project venv active, AFTER
run_batch_baseline_reproduction.py and the prior sqlite-store-validation
task have both already been run):

    python src/run_incremental_model_validation.py

Writes:
    data/processed/incremental_model_validation_ciprofloxacin_k15_d262144.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from bvbrc_recon.incremental_model import evaluate_streaming_hashed_matrix
from bvbrc_recon.sqlite_row_store import SqliteRowStore
from bvbrc_recon.streaming_kmer_build import peak_memory_mb

ANTIBIOTIC = "ciprofloxacin"
SEED = 20261009
K = 15
N_BUCKETS = 2**18
BATCH_SIZE = 16
N_EPOCHS = 50  # chosen via training-only internal validation - see report
NORMALIZE_ROWS = True  # chosen a priori - see report

DATA_PROCESSED = Path("data/processed")
DB_PATH = DATA_PROCESSED / f"out_of_core_v2_{ANTIBIOTIC}_rows_k{K}_d{N_BUCKETS}.db"
BATCH_BASELINE_PATH = DATA_PROCESSED / f"batch_lr_reproduction_{ANTIBIOTIC}_d{N_BUCKETS}.json"
OUTPUT_PATH = DATA_PROCESSED / f"incremental_model_validation_{ANTIBIOTIC}_k{K}_d{N_BUCKETS}.json"

# Documented training-only internal-validation grid (random split's train
# partition, 80/20 stratified carve, seed=7) used to choose N_EPOCHS and
# whether to use class_weight - NOT run against either real test set.
CONFIGURATION_SELECTION = {
    "method": "stratified 80/20 carve of the random split's TRAIN rows only (seed=7); "
    "never touched either real test partition",
    "preprocessing_decided_a_priori": "L2 row normalization - stateless, no fitted "
    "parameters, standard for count-based feature vectors; justified by the known "
    "confound of total k-mer count varying with genome length/sequencing depth",
    "grid_evaluated_on_internal_validation_only": {
        "epochs=10,class_weight=none": {"val_auc": 0.55, "val_resistant_recall": 0.0},
        "epochs=10,class_weight=balanced": {"val_auc": 0.55, "val_resistant_recall": 0.0},
        "epochs=25,class_weight=none": {"val_auc": 0.55, "val_resistant_recall": 0.0},
        "epochs=25,class_weight=balanced": {"val_auc": 0.55, "val_resistant_recall": 0.0},
        "epochs=50,class_weight=none": {"val_auc": 0.625, "val_resistant_recall": 0.0},
        "epochs=50,class_weight=balanced": {"val_auc": 0.55, "val_resistant_recall": 0.0},
    },
    "selected": {"epochs": 50, "class_weight": None, "normalize_rows": True},
    "note": "class_weight showed no consistent benefit on the internal validation split "
    "and was excluded to avoid adding an untested knob. Internal validation AUCs are "
    "noisy (only 14 held-out rows) - this grid informed the choice, it is not claimed "
    "as a precise measurement.",
}


def main() -> None:
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

    # cross-check: this must be the exact same label/row alignment the batch
    # baseline reproduction used.
    baseline = json.loads(BATCH_BASELINE_PATH.read_text()) if BATCH_BASELINE_PATH.exists() else None

    row_store = SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS)

    model_results = {}
    for split_name, split_df in [("random", random_split), ("lineage", lineage_split)]:
        genome_to_partition = dict(zip(split_df["genome_id"], split_df["partition"]))
        train_idx = {i for i, gid in enumerate(genome_order) if genome_to_partition.get(gid) == "train"}
        test_idx = {i for i, gid in enumerate(genome_order) if genome_to_partition.get(gid) == "test"}

        mem_before = peak_memory_mb()
        t0 = time.perf_counter()
        result = evaluate_streaming_hashed_matrix(
            row_store, train_idx, test_idx, label_by_row_index,
            seed=SEED, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, normalize_rows=NORMALIZE_ROWS,
        )
        wall_seconds = time.perf_counter() - t0
        mem_after = peak_memory_mb()

        entry = {
            "n_train": result.n_train,
            "n_test": result.n_test,
            "train_class_counts": result.train_class_counts,
            "test_class_counts": result.test_class_counts,
            "accuracy": result.accuracy,
            "roc_auc": result.roc_auc,
            "average_precision": result.average_precision,
            "recall_per_class": result.recall_per_class,
            "confusion_matrix": result.confusion_matrix,
            "warnings": result.warnings,
            "max_batch_rows_held_in_memory": result.max_batch_rows_held_in_memory,
            "fit_seconds": result.fit_seconds,
            "wall_seconds": wall_seconds,
            "peak_rss_mb_before": mem_before,
            "peak_rss_mb_after": mem_after,
        }
        model_results[split_name] = entry

        baseline_entry = baseline["model_results"][split_name] if baseline else None
        print(
            f"[{split_name}] incremental: acc={result.accuracy} auc={result.roc_auc} "
            f"recall={result.recall_per_class} fit_seconds={result.fit_seconds:.2f}"
        )
        if baseline_entry:
            print(
                f"[{split_name}] batch baseline: acc={baseline_entry['accuracy']} "
                f"auc={baseline_entry['roc_auc']} recall={baseline_entry['recall_per_class']}"
            )
            same_class_counts = (
                entry["train_class_counts"] == baseline_entry["train_class_counts"]
                and entry["test_class_counts"] == baseline_entry["test_class_counts"]
            )
            print(f"[{split_name}] train/test class counts match batch baseline exactly: {same_class_counts}")
            entry["class_counts_match_batch_baseline"] = same_class_counts

    output = {
        "k": K,
        "n_buckets": N_BUCKETS,
        "antibiotic": ANTIBIOTIC,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "n_epochs": N_EPOCHS,
        "normalize_rows": NORMALIZE_ROWS,
        "configuration_selection": CONFIGURATION_SELECTION,
        "model_results": model_results,
        "batch_baseline_available_for_comparison": baseline is not None,
        "batch_baseline_path": str(BATCH_BASELINE_PATH),
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
