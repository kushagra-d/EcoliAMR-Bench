"""Compares the batch `LogisticRegression` baseline against the
bounded-memory-per-batch streaming replica
(`streaming_logistic_regression.py`) on the real 100-genome
ciprofloxacin pilot, both saved splits, using the existing hybrid
disk-backed reader as the data source for the streaming path.

Reuses the ALREADY-BUILT SQLite store and hybrid cache artifacts - does
not rebuild them, does not re-extract anything, does not download
genomes. The hashed feature representation, labels, and splits are
read-only inputs, never modified.

Usage (from the project root, with the project venv active, AFTER the
SQLite store and hybrid cache artifacts from prior tasks exist):

    python src/run_batch_vs_streaming_lr_comparison.py

Writes:
    data/processed/streaming_lr_reproduction_ciprofloxacin_k15_d262144.json
    data/processed/batch_vs_streaming_lr_comparison_ciprofloxacin_k15_d262144.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bvbrc_recon.hybrid_training_reader import HybridTrainingReader, MultiPartitionHybridReader
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

DATA_PROCESSED = Path("data/processed")
DB_PATH = DATA_PROCESSED / f"out_of_core_v2_{ANTIBIOTIC}_rows_k{K}_d{N_BUCKETS}.db"
HYBRID_CACHE_DIR = DATA_PROCESSED / f"hybrid_cache_{ANTIBIOTIC}_k{K}_d{N_BUCKETS}"  # existing, from prior tasks
BATCH_BASELINE_PATH = DATA_PROCESSED / f"batch_lr_reproduction_{ANTIBIOTIC}_d{N_BUCKETS}.json"
OUTPUT_PATH = DATA_PROCESSED / f"streaming_lr_reproduction_{ANTIBIOTIC}_k{K}_d{N_BUCKETS}.json"
COMPARISON_PATH = DATA_PROCESSED / f"batch_vs_streaming_lr_comparison_{ANTIBIOTIC}_k{K}_d{N_BUCKETS}.json"


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


def main() -> None:
    genome_order, label_by_row_index, random_split, lineage_split = load_pilot()
    train_idx, test_idx = split_indices(genome_order, random_split)
    train_idx_lineage, test_idx_lineage = split_indices(genome_order, lineage_split)

    batch_baseline = json.loads(BATCH_BASELINE_PATH.read_text())
    print(f"batch LR baseline (reused, matches_existing_saved_result_exactly="
          f"{batch_baseline.get('matches_existing_saved_result_exactly')})")

    final_results = {}
    for split_name, t_idx, te_idx, train_dir, test_dir in [
        ("random", train_idx, test_idx, HYBRID_CACHE_DIR / "train", HYBRID_CACHE_DIR / "test"),
        ("lineage", train_idx_lineage, test_idx_lineage, HYBRID_CACHE_DIR / "lineage" / "train", HYBRID_CACHE_DIR / "lineage" / "test"),
    ]:
        store = SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS)
        train_reader = HybridTrainingReader(train_dir, store, t_idx, batch_size=BATCH_SIZE, max_memory_cache_bytes=256_000_000)
        test_reader = HybridTrainingReader(test_dir, store, te_idx, batch_size=BATCH_SIZE, max_memory_cache_bytes=256_000_000)
        reader = MultiPartitionHybridReader([train_reader, test_reader])

        mem_before = peak_memory_mb()
        result = fit_and_evaluate_streaming_logistic_regression(
            reader, t_idx, te_idx, label_by_row_index, n_features=N_BUCKETS,
            seed=SEED, batch_size=BATCH_SIZE, max_iter=MAX_ITER, tol=TOL,
        )
        mem_after = peak_memory_mb()

        final_results[split_name] = {
            "n_train": result.n_train, "n_test": result.n_test,
            "train_class_counts": result.train_class_counts, "test_class_counts": result.test_class_counts,
            "accuracy": result.accuracy, "roc_auc": result.roc_auc, "average_precision": result.average_precision,
            "recall_per_class": result.recall_per_class, "confusion_matrix": result.confusion_matrix,
            "warnings": result.warnings, "fit_seconds": result.fit_seconds,
            "n_objective_evaluations": result.n_objective_evaluations,
            "converged": result.converged, "final_loss": result.final_loss,
            "peak_rss_mb_before": mem_before, "peak_rss_mb_after": mem_after,
            "max_batch_rows_held_in_memory": result.max_batch_rows_held_in_memory,
        }
        print(f"[{split_name}] streaming LR: acc={result.accuracy} auc={result.roc_auc} "
              f"avg_prec={result.average_precision} recall={result.recall_per_class} "
              f"fit_seconds={result.fit_seconds:.2f} n_evals={result.n_objective_evaluations} "
              f"converged={result.converged}")

        baseline_entry = batch_baseline["model_results"][split_name]
        coef_diff_vs_baseline_class_counts = (
            final_results[split_name]["train_class_counts"] == baseline_entry["train_class_counts"]
            and final_results[split_name]["test_class_counts"] == baseline_entry["test_class_counts"]
        )
        final_results[split_name]["class_counts_match_batch_baseline"] = coef_diff_vs_baseline_class_counts
        print(f"[{split_name}] class counts match batch baseline exactly: {coef_diff_vs_baseline_class_counts}")

    OUTPUT_PATH.write_text(json.dumps({
        "k": K, "n_buckets": N_BUCKETS, "antibiotic": ANTIBIOTIC, "seed": SEED,
        "C": 1.0, "max_iter": MAX_ITER, "tol": TOL, "batch_size": BATCH_SIZE,
        "results": final_results,
    }, indent=2))
    print(f"wrote {OUTPUT_PATH}")

    comparison = {
        "batch_lr_baseline": batch_baseline["model_results"],
        "streaming_lr": final_results,
    }
    COMPARISON_PATH.write_text(json.dumps(comparison, indent=2))
    print(f"wrote {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
