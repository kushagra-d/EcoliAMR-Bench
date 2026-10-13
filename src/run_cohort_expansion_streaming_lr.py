"""Step 5 of the cohort-expansion task: fits the streaming Logistic
Regression implementation (`streaming_logistic_regression.py`,
validated in an earlier task) on the COMBINED 200-genome cohort's
hashed-feature SQLite store, using a NEW, deterministic train/validation
partition created SEPARATELY from the existing saved pilot splits
(random_split.csv / lineage_split.csv - both untouched).

Explicitly exploratory: 200 genomes is still far below the full cohort,
and this partition is NOT the saved pilot splits - results here must
not be read as generalizing beyond this specific expanded pilot.

Usage (from the project root, with the project venv active, AFTER
run_cohort_expansion_hashed_store.py has built the combined store):

    python src/run_cohort_expansion_streaming_lr.py

Writes:
    data/processed/cohort_expansion_ciprofloxacin_train_val_partition.csv
    data/processed/cohort_expansion_ciprofloxacin_streaming_lr_results_k15_d262144.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from bvbrc_recon.pilot_model import CLASS_TO_INT
from bvbrc_recon.sqlite_row_store import SqliteRowStore
from bvbrc_recon.streaming_kmer_build import peak_memory_mb
from bvbrc_recon.streaming_logistic_regression import fit_and_evaluate_streaming_logistic_regression

ANTIBIOTIC = "ciprofloxacin"
SEED = 20261009
K = 15
N_BUCKETS = 2**18
BATCH_SIZE = 16
MAX_ITER = 1000
TOL = 1e-4
TEST_SIZE = 0.3  # matches the pilot's own random/lineage split convention

DATA_PROCESSED = Path("data/processed")
DB_PATH = DATA_PROCESSED / f"cohort_expansion_{ANTIBIOTIC}_rows_k{K}_d{N_BUCKETS}.db"
GENOME_ORDER_PATH = DATA_PROCESSED / f"cohort_expansion_{ANTIBIOTIC}_combined_genome_order.csv"
PARTITION_PATH = DATA_PROCESSED / f"cohort_expansion_{ANTIBIOTIC}_train_val_partition.csv"
RESULTS_PATH = DATA_PROCESSED / f"cohort_expansion_{ANTIBIOTIC}_streaming_lr_results_k{K}_d{N_BUCKETS}.json"


def main() -> None:
    genome_order = pd.read_csv(GENOME_ORDER_PATH, dtype={"genome_id": str})["genome_id"].tolist()
    pilot_selected = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_selected_cohort.csv", dtype={"genome_id": str}
    )
    expansion_selected = pd.read_csv(
        DATA_PROCESSED / f"cohort_expansion_{ANTIBIOTIC}_selected.csv", dtype={"genome_id": str}
    )
    label_by_genome = dict(zip(pilot_selected["genome_id"], pilot_selected["label"]))
    label_by_genome.update(dict(zip(expansion_selected["genome_id"], expansion_selected["label"])))
    label_by_row_index = {i: label_by_genome[gid] for i, gid in enumerate(genome_order)}

    class_counts = {"Resistant": 0, "Susceptible": 0}
    for label in label_by_row_index.values():
        class_counts[label] += 1
    print(f"combined cohort: {len(genome_order)} genomes, class counts: {class_counts}")
    if min(class_counts.values()) < 2:
        print("BLOCKER: fewer than 2 genomes in the minority class - cannot create a stratified "
              "train/validation partition or fit meaningfully. Stopping here (computational "
              "validation only, per this task's documented fallback).")
        RESULTS_PATH.write_text(json.dumps({"blocked": True, "reason": "insufficient minority-class representation", "class_counts": class_counts}, indent=2))
        return

    # NEW, deterministic partition - separate from the saved pilot random/lineage splits
    indices = np.arange(len(genome_order))
    y_int = np.array([CLASS_TO_INT[label_by_row_index[i]] for i in indices])
    train_idx_arr, val_idx_arr = train_test_split(indices, test_size=TEST_SIZE, random_state=SEED, stratify=y_int)
    train_idx, val_idx = set(int(i) for i in train_idx_arr), set(int(i) for i in val_idx_arr)

    partition_df = pd.DataFrame({
        "genome_id": genome_order,
        "partition": ["train" if i in train_idx else "validation" for i in range(len(genome_order))],
    })
    partition_df.to_csv(PARTITION_PATH, index=False)
    print(f"new train/validation partition (seed={SEED}, test_size={TEST_SIZE}): "
          f"{len(train_idx)} train, {len(val_idx)} validation -> wrote {PARTITION_PATH}")

    store = SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS)
    mem_before = peak_memory_mb()
    result = fit_and_evaluate_streaming_logistic_regression(
        store, train_idx, val_idx, label_by_row_index, n_features=N_BUCKETS,
        seed=SEED, batch_size=BATCH_SIZE, max_iter=MAX_ITER, tol=TOL,
    )
    mem_after = peak_memory_mb()

    print(f"streaming LR (exploratory, new partition): converged={result.converged} "
          f"n_evals={result.n_objective_evaluations} n_iterations={result.n_iterations} "
          f"final_loss={result.final_loss} grad_norm={result.final_gradient_norm} "
          f"fit_seconds={result.fit_seconds:.2f} acc={result.accuracy} auc={result.roc_auc} "
          f"avg_prec={result.average_precision} recall={result.recall_per_class}")

    output = {
        "EXPLORATORY_ONLY": "200-genome expanded pilot, NOT the saved pilot splits, NOT a claim of "
        "generalization to the full cohort.",
        "k": K, "n_buckets": N_BUCKETS, "antibiotic": ANTIBIOTIC, "seed": SEED,
        "C": 1.0, "max_iter": MAX_ITER, "tol": TOL, "batch_size": BATCH_SIZE, "test_size": TEST_SIZE,
        "n_train": result.n_train, "n_test": result.n_test,
        "train_class_counts": result.train_class_counts, "test_class_counts": result.test_class_counts,
        "converged": result.converged, "optimizer_status": result.optimizer_status,
        "optimizer_message": result.optimizer_message, "n_iterations": result.n_iterations,
        "n_objective_evaluations": result.n_objective_evaluations,
        "n_gradient_evaluations": result.n_gradient_evaluations,
        "final_loss": result.final_loss, "final_gradient_norm": result.final_gradient_norm,
        "fit_seconds": result.fit_seconds, "peak_rss_mb_before": mem_before, "peak_rss_mb_after": mem_after,
        "max_batch_rows_held_in_memory": result.max_batch_rows_held_in_memory,
        "accuracy": result.accuracy, "roc_auc": result.roc_auc, "average_precision": result.average_precision,
        "recall_per_class": result.recall_per_class, "confusion_matrix": result.confusion_matrix,
        "warnings": result.warnings,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print(f"wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
