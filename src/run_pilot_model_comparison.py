"""Random Forest and XGBoost vs. the existing Logistic Regression pilot
model, evaluated on BOTH the existing random split and the existing
MLST-grouped lineage-aware split for the 100-genome ciprofloxacin pilot.

Reuses the EXISTING k-mer feature matrix, genome order, labels, and the
EXISTING saved lineage split (data/processed/pilot_ciprofloxacin_lineage_split.csv).
The random split is not separately on disk yet - it's regenerated here via
`compute_random_split_indices`, which is proven (by a dedicated test) to
reproduce the EXACT same row partition `train_and_evaluate_pilot_model`
already uses, not a new split - and is then persisted for the first time
as a new, additional file for auditability, matching the lineage split's
existing pattern.

Usage (from the project root, with the project venv active):

    python src/run_pilot_model_comparison.py

Writes reports/pilot_model_comparison.md and
data/processed/pilot_ciprofloxacin_random_split.csv (new file only).
Does not touch any existing pilot or lineage-evaluation output.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from bvbrc_recon.model_comparison import evaluate_random_forest, evaluate_xgboost
from bvbrc_recon.pilot_model import compute_random_split_indices, evaluate_with_precomputed_split

ANTIBIOTIC = "ciprofloxacin"
SEED = 20261009
TEST_SIZE = 0.3

DATA_PROCESSED = Path("data/processed")
REPORT_PATH = Path("reports/pilot_model_comparison.md")


def load_pilot_assets():
    genome_order = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_genome_order.csv", dtype={"genome_id": str}
    )["genome_id"].tolist()
    selected = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_selected_cohort.csv", dtype={"genome_id": str}
    )
    label_by_genome = dict(zip(selected["genome_id"], selected["label"]))
    missing = [g for g in genome_order if g not in label_by_genome]
    if missing:
        raise RuntimeError(f"ALIGNMENT BLOCKER: genome_ids missing a label: {missing}")
    labels = [label_by_genome[g] for g in genome_order]

    matrix = sparse.load_npz(DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_matrix.npz")
    if matrix.shape[0] != len(genome_order):
        raise RuntimeError(
            f"ALIGNMENT BLOCKER: matrix has {matrix.shape[0]} rows, genome_order has {len(genome_order)}"
        )
    return genome_order, labels, matrix


def load_saved_lineage_split(genome_order: list[str]):
    """Loads the EXISTING saved lineage split - does not regenerate it."""
    path = DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_lineage_split.csv"
    split_df = pd.read_csv(path, dtype={"genome_id": str})
    if sorted(split_df["genome_id"]) != sorted(genome_order):
        raise RuntimeError(
            "ALIGNMENT BLOCKER: saved lineage split's genome_ids do not exactly match the "
            "current genome_order - refusing to proceed rather than silently realigning."
        )
    genome_to_row = {gid: i for i, gid in enumerate(genome_order)}
    train_idx = np.array([genome_to_row[g] for g in split_df.loc[split_df["partition"] == "train", "genome_id"]])
    test_idx = np.array([genome_to_row[g] for g in split_df.loc[split_df["partition"] == "test", "genome_id"]])
    overlap = set(train_idx) & set(test_idx)
    if overlap:
        raise RuntimeError(f"ALIGNMENT BLOCKER: saved lineage split has overlapping rows: {overlap}")
    return train_idx, test_idx


def save_random_split(genome_order: list[str], train_idx, test_idx) -> None:
    out_path = DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_random_split.csv"
    rows = [{"genome_id": genome_order[i], "partition": "train"} for i in train_idx]
    rows += [{"genome_id": genome_order[i], "partition": "test"} for i in test_idx]
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Random split assignment saved -> {out_path}")


def main() -> None:
    genome_order, labels, matrix = load_pilot_assets()
    print(f"Loaded pilot: {len(genome_order)} genomes, matrix shape {matrix.shape}")

    random_train_idx, random_test_idx = compute_random_split_indices(labels, seed=SEED, test_size=TEST_SIZE)
    save_random_split(genome_order, random_train_idx, random_test_idx)

    lineage_train_idx, lineage_test_idx = load_saved_lineage_split(genome_order)
    print(
        f"Random split: train={len(random_train_idx)}, test={len(random_test_idx)} | "
        f"Lineage split: train={len(lineage_train_idx)}, test={len(lineage_test_idx)}"
    )

    results = {}
    for split_name, (train_idx, test_idx) in [
        ("random", (random_train_idx, random_test_idx)),
        ("lineage", (lineage_train_idx, lineage_test_idx)),
    ]:
        print(f"\n=== {split_name} split ===")

        lr_result = evaluate_with_precomputed_split(matrix, labels, train_idx, test_idx, seed=SEED)
        print(f"LogisticRegression: accuracy={lr_result.accuracy}, roc_auc={lr_result.roc_auc}")

        rf_result = evaluate_random_forest(matrix, labels, train_idx, test_idx, seed=SEED)
        print(
            f"RandomForest: accuracy={rf_result.accuracy}, roc_auc={rf_result.roc_auc}, "
            f"fit_seconds={rf_result.fit_seconds}"
        )

        xgb_result = evaluate_xgboost(matrix, labels, train_idx, test_idx, seed=SEED)
        print(
            f"XGBoost: accuracy={xgb_result.accuracy}, roc_auc={xgb_result.roc_auc}, "
            f"fit_seconds={xgb_result.fit_seconds}"
        )

        results[split_name] = {"LogisticRegression": lr_result, "RandomForest": rf_result, "XGBoost": xgb_result}

    from bvbrc_recon.model_comparison_report import render_model_comparison_report

    report_text = render_model_comparison_report(
        antibiotic=ANTIBIOTIC,
        seed=SEED,
        analysis_date=datetime.now(timezone.utc).isoformat(),
        n_genomes=len(genome_order),
        matrix_shape=matrix.shape,
        matrix_nnz=matrix.nnz,
        results=results,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"\nModel comparison report written -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
