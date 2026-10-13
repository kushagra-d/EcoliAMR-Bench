"""Primary three-antibiotic model benchmark for the course project.

Loads the frozen 956-genome hashed feature matrix (k=15, d=262144 -
`course_project_final_cohort_rows_k15_d262144.db`) ONCE into memory as
a single scipy sparse CSR matrix (never densified), then reproduces
the already-validated, deterministic random and lineage-aware splits
for each antibiotic (same seed/test_size as
`reports/course_project_final_cohort.md`) and asserts they match the
already-documented counts exactly before any model is fit - this task
reuses the frozen splits, it does not search for a new one.

Evaluates Logistic Regression, Random Forest, and XGBoost (the
project's existing model configurations, reused unchanged from
`pilot_model.py` / `model_comparison.py`) on each of the 3 antibiotics
x 2 split strategies = 18 evaluations. Each model is fit on the
TRAINING partition only; the test partition is touched only for
scoring.

A global memory-safety check (the same 11GB threshold already
established by `streaming_kmer_build.DEFAULT_MEMORY_LIMIT_MB`) is
checked before every fit; if ever exceeded, the benchmark stops
immediately, reports exactly which evaluations completed and which did
not, and does not continue "quietly" with a different sample or model.

Usage (from the project root, with the project venv active):

    python src/run_course_project_model_benchmark.py

Writes:
    data/processed/course_project_model_benchmark_results.json   (full detail, all 18 evaluations)
    data/processed/course_project_model_benchmark_summary.csv    (flat consolidated table)
"""

from __future__ import annotations

import json
import time
import warnings as warnings_module
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from bvbrc_recon.csv_loader import load_antibiotic_csv
from bvbrc_recon.lineage_split import assign_lineage_groups, split_by_lineage_group
from bvbrc_recon.model_comparison import RANDOM_FOREST_PARAMS, XGBOOST_PARAMS
from bvbrc_recon.pilot_model import CLASS_TO_INT, compute_classification_metrics, compute_random_split_indices
from bvbrc_recon.sqlite_row_store import SqliteRowStore
from bvbrc_recon.streaming_kmer_build import DEFAULT_MEMORY_LIMIT_MB, peak_memory_mb

ANTIBIOTICS = ["ciprofloxacin", "ampicillin", "gentamicin"]
SEED = 20261009
TEST_SIZE = 0.2
K = 15
N_BUCKETS = 2**18

DATA_PROCESSED = Path("data/processed")
DB_PATH = DATA_PROCESSED / f"course_project_final_cohort_rows_k{K}_d{N_BUCKETS}.db"
GENOME_ORDER_PATH = DATA_PROCESSED / "course_project_final_cohort_genome_order.csv"
RESULTS_JSON_PATH = DATA_PROCESSED / "course_project_model_benchmark_results.json"
SUMMARY_CSV_PATH = DATA_PROCESSED / "course_project_model_benchmark_summary.csv"

# Already-validated frozen-split counts from reports/course_project_final_cohort.md -
# reproducing the splits here must match these EXACTLY, or this script stops rather
# than silently proceeding on a different split than the one already frozen.
EXPECTED_RANDOM_SPLITS = {
    "ciprofloxacin": {"n_train": 668, "n_test": 168, "test_resistant": 38, "test_susceptible": 130},
    "ampicillin": {"n_train": 587, "n_test": 147, "test_resistant": 79, "test_susceptible": 68},
    "gentamicin": {"n_train": 661, "n_test": 166, "test_resistant": 19, "test_susceptible": 147},
}
EXPECTED_LINEAGE_SPLITS = {
    "ciprofloxacin": {"n_train": 631, "n_test": 205, "test_resistant": 91, "test_susceptible": 114},
    "ampicillin": {"n_train": 539, "n_test": 195, "test_resistant": 137, "test_susceptible": 58},
    "gentamicin": {"n_train": 741, "n_test": 86, "test_resistant": 6, "test_susceptible": 80},
}


class MemoryThresholdExceeded(RuntimeError):
    pass


def load_full_matrix() -> tuple[list[str], sparse.csr_matrix]:
    row_store = SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS)
    genome_order_df = pd.read_csv(GENOME_ORDER_PATH, dtype={"genome_id": str})
    expected_order = genome_order_df.sort_values("row_index")["genome_id"].tolist()

    all_genome_ids: list[str] = []
    chunks = []
    for genome_ids, row_indices, csr_chunk in row_store.iter_batches(batch_size=64):
        all_genome_ids.extend(genome_ids)
        chunks.append(csr_chunk)
    row_store.close()

    X = sparse.vstack(chunks, format="csr")
    if all_genome_ids != expected_order:
        raise RuntimeError("REFUSING TO PROCEED: row order read from the SQLite store does not match "
                            "course_project_final_cohort_genome_order.csv.")
    return all_genome_ids, X


def load_mlst_by_genome() -> dict[str, str | None]:
    mlst_by_genome: dict[str, str | None] = {}
    for antibiotic in ANTIBIOTICS:
        for r in load_antibiotic_csv(DATA_PROCESSED / f"{antibiotic}_amr_metadata.csv"):
            gid = r.get("genome_id")
            if gid is not None and gid not in mlst_by_genome:
                mlst_by_genome[gid] = r.get("genome_mlst")
    return mlst_by_genome


def reproduce_splits_for_antibiotic(antibiotic: str, mlst_by_genome: dict) -> dict:
    manifest = pd.read_csv(DATA_PROCESSED / f"course_project_final_cohort_{antibiotic}_manifest.csv", dtype={"genome_id": str})
    manifest = manifest.sort_values("genome_id").reset_index(drop=True)
    ids, labels = manifest["genome_id"].tolist(), manifest["label"].tolist()

    train_pos, test_pos = compute_random_split_indices(labels, seed=SEED, test_size=TEST_SIZE)
    random_split = {
        "train_genome_ids": [ids[i] for i in train_pos],
        "test_genome_ids": [ids[i] for i in test_pos],
        "train_labels": [labels[i] for i in train_pos],
        "test_labels": [labels[i] for i in test_pos],
    }

    group_by_genome = assign_lineage_groups(ids, mlst_by_genome).group_by_genome
    lineage_result = split_by_lineage_group(ids, labels, group_by_genome, seed=SEED, test_size=TEST_SIZE)
    if not lineage_result.feasible:
        raise RuntimeError(f"REFUSING TO PROCEED: lineage-aware split for {antibiotic} is not feasible: "
                            f"{lineage_result.blocker}")
    train_groups = {group_by_genome[g] for g in lineage_result.train_genome_ids}
    test_groups = {group_by_genome[g] for g in lineage_result.test_genome_ids}
    if train_groups & test_groups:
        raise RuntimeError(f"REFUSING TO PROCEED: lineage-aware split for {antibiotic} has overlapping "
                            f"MLST groups between train and test: {train_groups & test_groups}")
    label_by_id = dict(zip(ids, labels))
    lineage_split = {
        "train_genome_ids": lineage_result.train_genome_ids,
        "test_genome_ids": lineage_result.test_genome_ids,
        "train_labels": [label_by_id[g] for g in lineage_result.train_genome_ids],
        "test_labels": [label_by_id[g] for g in lineage_result.test_genome_ids],
    }

    # --- validate against the already-documented frozen-cohort numbers ---
    def _check(split: dict, expected: dict, label: str) -> None:
        n_train, n_test = len(split["train_genome_ids"]), len(split["test_genome_ids"])
        test_r = sum(1 for lbl in split["test_labels"] if lbl == "Resistant")
        test_s = sum(1 for lbl in split["test_labels"] if lbl == "Susceptible")
        actual = {"n_train": n_train, "n_test": n_test, "test_resistant": test_r, "test_susceptible": test_s}
        if actual != expected:
            raise RuntimeError(f"REFUSING TO PROCEED: reproduced {label} split for {antibiotic} ({actual}) "
                                f"does not match the frozen, already-documented split ({expected}).")

    _check(random_split, EXPECTED_RANDOM_SPLITS[antibiotic], "random")
    _check(lineage_split, EXPECTED_LINEAGE_SPLITS[antibiotic], "lineage-aware")

    return {"random": random_split, "lineage": lineage_split}


def check_memory_or_stop(context: str) -> float:
    mem_mb = peak_memory_mb()
    if mem_mb > DEFAULT_MEMORY_LIMIT_MB:
        raise MemoryThresholdExceeded(
            f"Peak RSS {mem_mb:.0f}MB exceeded the {DEFAULT_MEMORY_LIMIT_MB:.0f}MB safety threshold "
            f"at: {context}. Stopping safely - no further evaluations will be attempted."
        )
    return mem_mb


def _labels_to_int(labels: list[str]) -> np.ndarray:
    return np.array([CLASS_TO_INT[lbl] for lbl in labels])


def evaluate_one(model_name: str, antibiotic: str, split_name: str, X_train, y_train_labels, X_test, y_test_labels, seed: int) -> dict:
    result: dict = {
        "model": model_name, "antibiotic": antibiotic, "split": split_name, "seed": seed,
        "n_train": X_train.shape[0], "n_test": X_test.shape[0],
        "train_class_counts": {"Resistant": y_train_labels.count("Resistant"), "Susceptible": y_train_labels.count("Susceptible")},
        "test_class_counts": {"Resistant": y_test_labels.count("Resistant"), "Susceptible": y_test_labels.count("Susceptible")},
        "warnings": [], "error": None, "fit_seconds": None, "predict_seconds": None,
        "peak_rss_mb_after": None,
    }
    y_train, y_test = _labels_to_int(y_train_labels), _labels_to_int(y_test_labels)

    if len(set(y_train)) < 2:
        result["warnings"].append("Cannot fit: training partition contains only one class.")
        return result
    if len(y_test) == 0:
        result["warnings"].append("Test partition is empty.")
        return result

    try:
        if model_name == "LogisticRegression":
            params = {"max_iter": 1000, "random_state": seed}
            model = LogisticRegression(**params)
            with warnings_module.catch_warnings(record=True) as caught:
                warnings_module.simplefilter("always", ConvergenceWarning)
                t0 = time.perf_counter()
                model.fit(X_train, y_train)
                result["fit_seconds"] = time.perf_counter() - t0
            for w in caught:
                if issubclass(w.category, ConvergenceWarning):
                    result["warnings"].append(f"ConvergenceWarning: {w.message}")
        elif model_name == "RandomForest":
            params = {**RANDOM_FOREST_PARAMS, "random_state": seed}
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**params)
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            result["fit_seconds"] = time.perf_counter() - t0
        elif model_name == "XGBoost":
            from xgboost import XGBClassifier
            params = {**XGBOOST_PARAMS, "random_state": seed}
            model = XGBClassifier(**params)
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            result["fit_seconds"] = time.perf_counter() - t0
        else:
            raise ValueError(f"unknown model_name {model_name!r}")
        result["params"] = params

        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        result["predict_seconds"] = time.perf_counter() - t0
    except (MemoryError, ValueError, RuntimeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["warnings"].append(f"{model_name} failed to fit/predict on {antibiotic}/{split_name}: {result['error']}")
        return result

    metrics = compute_classification_metrics(y_test, y_pred, y_proba)
    result["accuracy"] = metrics["accuracy"]
    result["roc_auc"] = metrics["roc_auc"]
    result["average_precision"] = metrics["average_precision"]
    result["balanced_accuracy"] = metrics["balanced_accuracy"]
    result["macro_f1"] = metrics["macro_f1"]
    result["confusion_matrix"] = metrics["confusion_matrix"]
    result["recall_per_class"] = metrics["recall_per_class"]
    result["precision_per_class"] = metrics["precision_per_class"]
    result["warnings"].extend(metrics["warnings"])
    result["peak_rss_mb_after"] = peak_memory_mb()
    return result


def main() -> None:
    print("Loading the full 956-genome feature matrix from the SQLite row store...")
    t0 = time.perf_counter()
    genome_order, X_full = load_full_matrix()
    load_seconds = time.perf_counter() - t0
    load_peak_rss = check_memory_or_stop("after loading the full feature matrix")
    print(f"loaded {X_full.shape} in {load_seconds:.1f}s, peak_rss={load_peak_rss:.0f}MB, "
          f"in-memory CSR bytes={X_full.data.nbytes + X_full.indices.nbytes + X_full.indptr.nbytes:,}")
    row_index_by_genome = {gid: i for i, gid in enumerate(genome_order)}

    mlst_by_genome = load_mlst_by_genome()

    all_results = []
    stopped_early = None
    completed = set()
    planned = [(m, ab, sp) for ab in ANTIBIOTICS for sp in ["random", "lineage"] for m in ["LogisticRegression", "RandomForest", "XGBoost"]]

    try:
        for antibiotic in ANTIBIOTICS:
            print(f"\n=== {antibiotic} ===")
            splits = reproduce_splits_for_antibiotic(antibiotic, mlst_by_genome)
            print(f"  random split verified: train={len(splits['random']['train_genome_ids'])} "
                  f"test={len(splits['random']['test_genome_ids'])} (matches frozen cohort report)")
            print(f"  lineage split verified: train={len(splits['lineage']['train_genome_ids'])} "
                  f"test={len(splits['lineage']['test_genome_ids'])} (matches frozen cohort report, no group overlap)")

            for split_name, split in splits.items():
                train_rows = [row_index_by_genome[g] for g in split["train_genome_ids"]]
                test_rows = [row_index_by_genome[g] for g in split["test_genome_ids"]]
                X_train, X_test = X_full[train_rows], X_full[test_rows]

                for model_name in ["LogisticRegression", "RandomForest", "XGBoost"]:
                    check_memory_or_stop(f"before fitting {model_name} on {antibiotic}/{split_name}")
                    print(f"  [{model_name}] fitting on {split_name} split ({X_train.shape[0]} train / {X_test.shape[0]} test)...", flush=True)
                    result = evaluate_one(
                        model_name, antibiotic, split_name, X_train, split["train_labels"], X_test, split["test_labels"], seed=SEED
                    )
                    all_results.append(result)
                    completed.add((model_name, antibiotic, split_name))
                    status = "ERROR" if result.get("error") else "ok"
                    print(f"    -> {status} fit={result.get('fit_seconds')} "
                          f"balanced_acc={result.get('balanced_accuracy')} roc_auc={result.get('roc_auc')}")
    except MemoryThresholdExceeded as exc:
        stopped_early = str(exc)
        print(f"\nSTOPPED EARLY: {stopped_early}")

    missing = [c for c in planned if c not in completed]

    RESULTS_JSON_PATH.write_text(json.dumps({
        "seed": SEED, "test_size": TEST_SIZE, "feature_matrix_shape": list(X_full.shape),
        "feature_matrix_load_seconds": load_seconds, "feature_matrix_load_peak_rss_mb": load_peak_rss,
        "n_planned_evaluations": len(planned), "n_completed_evaluations": len(all_results),
        "stopped_early_reason": stopped_early,
        "missing_evaluations": [{"model": m, "antibiotic": a, "split": s} for m, a, s in missing],
        "results": all_results,
    }, indent=2, default=str))

    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "model": r["model"], "antibiotic": r["antibiotic"], "split": r["split"],
            "n_train": r["n_train"], "n_test": r["n_test"],
            "test_resistant": r["test_class_counts"]["Resistant"], "test_susceptible": r["test_class_counts"]["Susceptible"],
            "accuracy": r.get("accuracy"), "roc_auc": r.get("roc_auc"), "average_precision": r.get("average_precision"),
            "balanced_accuracy": r.get("balanced_accuracy"), "macro_f1": r.get("macro_f1"),
            "resistant_precision": (r.get("precision_per_class") or {}).get("Resistant"),
            "resistant_recall": (r.get("recall_per_class") or {}).get("Resistant"),
            "susceptible_precision": (r.get("precision_per_class") or {}).get("Susceptible"),
            "susceptible_recall": (r.get("recall_per_class") or {}).get("Susceptible"),
            "fit_seconds": r.get("fit_seconds"), "predict_seconds": r.get("predict_seconds"),
            "error": r.get("error"), "n_warnings": len(r.get("warnings", [])),
        })
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV_PATH, index=False)

    print(f"\n=== Benchmark finished: {len(all_results)}/{len(planned)} evaluations completed ===")
    if missing:
        print(f"MISSING: {missing}")
    print(f"wrote {RESULTS_JSON_PATH}, {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
