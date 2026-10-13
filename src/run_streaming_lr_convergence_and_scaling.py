"""Step 3 (real-pilot, train-only-validated tolerance/iteration grid)
and Step 4 (synthetic, controlled n/d scaling grid) for validating
`streaming_logistic_regression.py`'s convergence behavior.

Step 3 uses the REAL 100-genome ciprofloxacin pilot's existing SQLite
store and random split, via a fresh train-only internal validation
carve (stratified 80/20 of the TRAIN partition only, seed=7 - same
precedent as two earlier tasks) - never touches either real test set.

Step 4 uses entirely SYNTHETIC data (`synthetic_scaling_fixtures.py`),
clearly labeled as such throughout - these measure the optimizer's
resource/convergence scaling, NOT real predictive performance.

Usage (from the project root, with the project venv active, AFTER the
SQLite store from prior tasks exists):

    python src/run_streaming_lr_convergence_and_scaling.py

Writes:
    data/processed/streaming_lr_convergence_grid_ciprofloxacin_k15_d262144.json
    data/processed/streaming_lr_synthetic_scaling_ciprofloxacin_k15_d262144.json
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from bvbrc_recon.hybrid_training_reader import HybridTrainingReader, MultiPartitionHybridReader
from bvbrc_recon.pilot_model import CLASS_TO_INT
from bvbrc_recon.sqlite_row_store import SqliteRowStore
from bvbrc_recon.streaming_kmer_build import peak_memory_mb
from bvbrc_recon.streaming_logistic_regression import fit_and_evaluate_streaming_logistic_regression
from bvbrc_recon.synthetic_scaling_fixtures import build_synthetic_store, train_test_split_indices

ANTIBIOTIC = "ciprofloxacin"
SEED = 20261009
INTERNAL_VAL_SEED = 7
K = 15
N_BUCKETS = 2**18
BATCH_SIZE = 16

DATA_PROCESSED = Path("data/processed")
DB_PATH = DATA_PROCESSED / f"out_of_core_v2_{ANTIBIOTIC}_rows_k{K}_d{N_BUCKETS}.db"
CONVERGENCE_VAL_CACHE_DIR = DATA_PROCESSED / f"streaming_lr_convergence_val_cache_{ANTIBIOTIC}_k{K}_d{N_BUCKETS}"
CONVERGENCE_OUTPUT_PATH = DATA_PROCESSED / f"streaming_lr_convergence_grid_{ANTIBIOTIC}_k{K}_d{N_BUCKETS}.json"
SCALING_OUTPUT_PATH = DATA_PROCESSED / f"streaming_lr_synthetic_scaling_{ANTIBIOTIC}_k{K}_d{N_BUCKETS}.json"

# --- Step 3: predefined (not searched) tolerance/iteration configurations ---
CONVERGENCE_CONFIGS = [
    {"name": "default", "tol": 1e-4, "max_iter": 1000, "justification": "sklearn's own default tol; current baseline"},
    {"name": "tighter_tol", "tol": 1e-6, "max_iter": 1000, "justification": "tests whether a stricter stopping rule changes the solution/metrics meaningfully"},
    {"name": "looser_tol", "tol": 1e-2, "max_iter": 1000, "justification": "tests the cost of accepting a cruder stopping rule for speed"},
    {"name": "iteration_capped", "tol": 1e-4, "max_iter": 50, "justification": "simulates a hard wall-clock/iteration budget constraint"},
]

# --- Step 4: synthetic scaling grid ---
SYNTHETIC_SCENARIOS = [
    {"name": "pilot_ratio", "n_rows": 100, "n_features": 262_144, "note": "same n:d ratio and absolute d as the real pilot"},
    {"name": "n_scale_100", "n_rows": 100, "n_features": 10_000, "note": "fixed d=10,000, increasing n baseline"},
    {"name": "n_scale_300", "n_rows": 300, "n_features": 10_000, "note": "fixed d=10,000, 3x rows"},
    {"name": "n_scale_1000", "n_rows": 1000, "n_features": 10_000, "note": "fixed d=10,000, 10x rows"},
    {"name": "d_scale_10000", "n_rows": 100, "n_features": 10_000, "note": "fixed n=100, increasing d baseline (same as n_scale_100)"},
    {"name": "d_scale_50000", "n_rows": 100, "n_features": 50_000, "note": "fixed n=100, 5x features"},
    {"name": "d_scale_262144", "n_rows": 100, "n_features": 262_144, "note": "fixed n=100, ~26x features (same as pilot_ratio)"},
    {"name": "both_scale_small", "n_rows": 100, "n_features": 10_000, "note": "both increasing together, step 1 (same as n_scale_100/d_scale_10000)"},
    {"name": "both_scale_medium", "n_rows": 300, "n_features": 50_000, "note": "both increasing together, step 2"},
    {"name": "both_scale_large", "n_rows": 1000, "n_features": 262_144, "note": "both increasing together, step 3"},
]
FIXED_SCALING_CONFIG = {"tol": 1e-4, "max_iter": 30}
# Deliberately capped at 30 iterations (not 1000, unlike the real-pilot Step 3 grid): the synthetic
# scenarios here use a BARE SqliteRowStore with no caching layer (to isolate pure data-access cost
# from caching infrastructure), and the largest scenario measured ~2.1s/evaluation - running every
# scenario to FULL convergence (likely 100-200+ evaluations, per the real pilot's own experience)
# would make this benchmark impractically slow. Measured directly before choosing this cap (not
# guessed): 10 iterations/17 evaluations on the n=1000/d=262144 scenario took ~36s. This still
# captures the real scaling signal this task asks for - per-pass cost and evaluation-count trends -
# at the cost of NOT reporting full convergence for the larger scenarios (reported honestly as
# `converged: false` where applicable, not hidden).


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
    label_by_row_index = {i: label_by_genome[gid] for i, gid in enumerate(genome_order) if gid in label_by_genome}
    return genome_order, label_by_row_index, random_split


def run_step3_convergence_grid():
    genome_order, label_by_row_index, random_split = load_pilot()
    genome_to_partition = dict(zip(random_split["genome_id"], random_split["partition"]))
    train_idx = {i for i, gid in enumerate(genome_order) if genome_to_partition.get(gid) == "train"}

    train_idx_sorted = sorted(train_idx)
    y_train_int = np.array([CLASS_TO_INT[label_by_row_index[i]] for i in train_idx_sorted])
    sub_train_idx, val_idx = train_test_split(
        train_idx_sorted, test_size=0.2, random_state=INTERNAL_VAL_SEED, stratify=y_train_int
    )
    sub_train_idx, val_idx = set(sub_train_idx), set(val_idx)
    print(f"internal validation carve: sub_train={len(sub_train_idx)}, val={len(val_idx)} (random split's TRAIN rows only)")

    if CONVERGENCE_VAL_CACHE_DIR.exists():
        shutil.rmtree(CONVERGENCE_VAL_CACHE_DIR)
    store = SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS)
    sub_train_reader = HybridTrainingReader(
        CONVERGENCE_VAL_CACHE_DIR / "sub_train", store, sub_train_idx, batch_size=BATCH_SIZE, max_memory_cache_bytes=256_000_000
    )
    val_reader = HybridTrainingReader(
        CONVERGENCE_VAL_CACHE_DIR / "val", store, val_idx, batch_size=BATCH_SIZE, max_memory_cache_bytes=256_000_000
    )
    reader = MultiPartitionHybridReader([sub_train_reader, val_reader])

    results = []
    for cfg in CONVERGENCE_CONFIGS:
        mem_before = peak_memory_mb()
        result = fit_and_evaluate_streaming_logistic_regression(
            reader, sub_train_idx, val_idx, label_by_row_index, n_features=N_BUCKETS,
            seed=SEED, batch_size=BATCH_SIZE, max_iter=cfg["max_iter"], tol=cfg["tol"],
        )
        mem_after = peak_memory_mb()
        entry = {
            "config_name": cfg["name"], "tol": cfg["tol"], "max_iter": cfg["max_iter"], "justification": cfg["justification"],
            "converged": result.converged, "optimizer_status": result.optimizer_status, "optimizer_message": result.optimizer_message,
            "n_iterations": result.n_iterations, "n_objective_evaluations": result.n_objective_evaluations,
            "n_gradient_evaluations": result.n_gradient_evaluations, "final_loss": result.final_loss,
            "final_gradient_norm": result.final_gradient_norm, "fit_seconds": result.fit_seconds,
            "peak_rss_mb_before": mem_before, "peak_rss_mb_after": mem_after,
            "accuracy": result.accuracy, "roc_auc": result.roc_auc, "average_precision": result.average_precision,
            "recall_per_class": result.recall_per_class, "confusion_matrix": result.confusion_matrix,
            "max_batch_rows_held_in_memory": result.max_batch_rows_held_in_memory,
        }
        results.append(entry)
        print(f"[{cfg['name']}] converged={result.converged} n_evals={result.n_objective_evaluations} "
              f"final_loss={result.final_loss:.6f} grad_norm={result.final_gradient_norm:.6f} "
              f"fit_seconds={result.fit_seconds:.2f} val_acc={result.accuracy} val_auc={result.roc_auc}")

    CONVERGENCE_OUTPUT_PATH.write_text(json.dumps(
        {
            "note": "internal train-only validation results (stratified 80/20 carve of the random "
            "split's TRAIN partition only, seed=7) - NEVER evaluated against either real test set. "
            "Used to characterize convergence/runtime/RSS/metric tradeoffs across a small predefined "
            "tolerance/iteration grid, not to select a new 'official' configuration for a different task.",
            "configs": results,
        },
        indent=2,
    ))
    print(f"wrote {CONVERGENCE_OUTPUT_PATH}")


def run_step4_synthetic_scaling():
    results = []
    for scenario in SYNTHETIC_SCENARIOS:
        n_rows, n_features = scenario["n_rows"], scenario["n_features"]
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            t0 = time.perf_counter()
            store, labels = build_synthetic_store(Path(tmp) / "rows.db", n_rows=n_rows, n_features=n_features, seed=SEED)
            build_seconds = time.perf_counter() - t0
            train_idx, test_idx = train_test_split_indices(n_rows, seed=SEED)

            mem_before = peak_memory_mb()
            result = fit_and_evaluate_streaming_logistic_regression(
                store, train_idx, test_idx, labels, n_features=n_features,
                seed=SEED, batch_size=BATCH_SIZE, max_iter=FIXED_SCALING_CONFIG["max_iter"], tol=FIXED_SCALING_CONFIG["tol"],
            )
            mem_after = peak_memory_mb()

        entry = {
            "scenario_name": scenario["name"], "note": scenario["note"], "SYNTHETIC": True,
            "n_rows": n_rows, "n_features": n_features, "build_seconds": build_seconds,
            "converged": result.converged, "n_iterations": result.n_iterations,
            "n_objective_evaluations": result.n_objective_evaluations, "n_gradient_evaluations": result.n_gradient_evaluations,
            "final_loss": result.final_loss, "final_gradient_norm": result.final_gradient_norm,
            "fit_seconds": result.fit_seconds, "peak_rss_mb_before": mem_before, "peak_rss_mb_after": mem_after,
            "max_batch_rows_held_in_memory": result.max_batch_rows_held_in_memory,
            "accuracy_on_synthetic_data_NOT_REAL_PERFORMANCE": result.accuracy,
        }
        results.append(entry)
        print(f"[SYNTHETIC {scenario['name']}] n={n_rows} d={n_features} converged={result.converged} "
              f"n_evals={result.n_objective_evaluations} fit_seconds={result.fit_seconds:.3f} "
              f"peak_rss_after={mem_after:.0f}MB")

    SCALING_OUTPUT_PATH.write_text(json.dumps(
        {
            "WARNING": "Every entry below is a SYNTHETIC benchmark fixture (synthetic_scaling_fixtures.py) - "
            "NOT real genome data. Accuracy/metrics here characterize the optimizer's behavior on "
            "controlled synthetic data ONLY and must never be read as evidence of real-world predictive "
            "performance.",
            "fixed_config": FIXED_SCALING_CONFIG,
            "scenarios": results,
        },
        indent=2,
    ))
    print(f"wrote {SCALING_OUTPUT_PATH}")


if __name__ == "__main__":
    print("=== Step 3: real-pilot convergence grid (train-only internal validation) ===")
    run_step3_convergence_grid()
    print("=== Step 4: synthetic scaling grid ===")
    run_step4_synthetic_scaling()
