"""Lineage-aware (MLST-group) evaluation of the existing 100-genome
ciprofloxacin pilot, compared against the existing random-split baseline.

Reuses the EXISTING pilot cohort, k-mer feature matrix, and Logistic
Regression configuration - does not download any genome, does not
recompute k-mers, and does not modify any existing pilot output.

Usage (from the project root, with the project venv active):

    python src/run_pilot_lineage_evaluation.py

Writes reports/pilot_lineage_evaluation.md and a new, additional CSV
(data/processed/pilot_ciprofloxacin_lineage_split.csv) recording the
group assignment and partition for transparency. Does not touch
reports/pilot_ciprofloxacin_report.md or any existing data/processed file.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy import sparse

from bvbrc_recon.csv_loader import load_antibiotic_csv
from bvbrc_recon.lineage_split import assign_lineage_groups, split_by_lineage_group
from bvbrc_recon.pilot_model import evaluate_with_precomputed_split, train_and_evaluate_pilot_model

ANTIBIOTIC = "ciprofloxacin"
SEED = 20261009
TEST_SIZE = 0.3

DATA_PROCESSED = Path("data/processed")
REPORT_PATH = Path("reports/pilot_lineage_evaluation.md")


def load_pilot_assets():
    genome_order = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_genome_order.csv", dtype={"genome_id": str}
    )["genome_id"].tolist()
    selected = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_selected_cohort.csv", dtype={"genome_id": str}
    )
    label_by_genome = dict(zip(selected["genome_id"], selected["label"]))

    missing_from_labels = [g for g in genome_order if g not in label_by_genome]
    if missing_from_labels:
        raise RuntimeError(
            f"ALIGNMENT BLOCKER: {len(missing_from_labels)} genome_ids in the k-mer matrix's row "
            f"order are missing from the selected-cohort label table: {missing_from_labels}"
        )
    labels = [label_by_genome[g] for g in genome_order]

    matrix = sparse.load_npz(DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_matrix.npz")
    if matrix.shape[0] != len(genome_order):
        raise RuntimeError(
            f"ALIGNMENT BLOCKER: matrix has {matrix.shape[0]} rows but genome_order has "
            f"{len(genome_order)} entries."
        )

    # Reuses the existing, already-tested CSV loader rather than a fresh
    # pandas.read_csv - that loader already converts NaN to None uniformly
    # (a raw pandas read leaves pandas's own NaN float, which is not `None`
    # and is truthy as a string, silently treating a missing MLST value as
    # present - confirmed live while building this script).
    raw_records = load_antibiotic_csv(DATA_PROCESSED / f"{ANTIBIOTIC}_amr_metadata.csv")
    mlst_by_genome: dict[str, str | None] = {}
    for record in raw_records:
        gid = record.get("genome_id")
        if gid is not None and gid not in mlst_by_genome:
            mlst_by_genome[gid] = record.get("genome_mlst")

    return genome_order, labels, matrix, mlst_by_genome


def main() -> None:
    genome_order, labels, matrix, mlst_by_genome = load_pilot_assets()
    print(f"Loaded pilot: {len(genome_order)} genomes, matrix shape {matrix.shape}")

    assignment = assign_lineage_groups(genome_order, mlst_by_genome)
    print(
        f"MLST coverage: {len(genome_order) - len(assignment.missing_mlst_genome_ids)}/"
        f"{len(genome_order)} have an MLST value; {assignment.n_groups} distinct groups"
    )
    if assignment.anomalous_scheme_genome_ids:
        print(f"Scheme anomalies vs majority ({assignment.majority_scheme}): {assignment.anomalous_scheme_genome_ids}")

    lineage_split = split_by_lineage_group(
        genome_order, labels, assignment.group_by_genome, seed=SEED, test_size=TEST_SIZE
    )
    if not lineage_split.feasible:
        print(f"LINEAGE SPLIT BLOCKED: {lineage_split.blocker}")
    else:
        print(
            f"Lineage split: train={len(lineage_split.train_genome_ids)} genomes/"
            f"{lineage_split.train_group_count} groups, "
            f"test={len(lineage_split.test_genome_ids)} genomes/{lineage_split.test_group_count} groups"
        )
        split_df = pd.DataFrame(
            {
                "genome_id": lineage_split.train_genome_ids + lineage_split.test_genome_ids,
                "partition": ["train"] * len(lineage_split.train_genome_ids)
                + ["test"] * len(lineage_split.test_genome_ids),
                "mlst_group": [
                    assignment.group_by_genome[g]
                    for g in lineage_split.train_genome_ids + lineage_split.test_genome_ids
                ],
            }
        )
        split_df.to_csv(DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_lineage_split.csv", index=False)

    print("\n=== Reproducing the random-split baseline (same seed/config as the original pilot) ===")
    random_result = train_and_evaluate_pilot_model(matrix, labels, seed=SEED, test_size=TEST_SIZE)
    print(f"Random split: train={random_result.n_train}, test={random_result.n_test}")
    print(f"  accuracy={random_result.accuracy}, roc_auc={random_result.roc_auc}")

    lineage_result = None
    if lineage_split.feasible:
        genome_to_row = {gid: i for i, gid in enumerate(genome_order)}
        train_idx = [genome_to_row[g] for g in lineage_split.train_genome_ids]
        test_idx = [genome_to_row[g] for g in lineage_split.test_genome_ids]
        print("\n=== Lineage-aware evaluation ===")
        lineage_result = evaluate_with_precomputed_split(matrix, labels, train_idx, test_idx, seed=SEED)
        print(f"  accuracy={lineage_result.accuracy}, roc_auc={lineage_result.roc_auc}")

    from bvbrc_recon.lineage_report import render_lineage_report

    report_text = render_lineage_report(
        antibiotic=ANTIBIOTIC,
        seed=SEED,
        test_size=TEST_SIZE,
        analysis_date=datetime.now(timezone.utc).isoformat(),
        n_genomes=len(genome_order),
        assignment=assignment,
        lineage_split=lineage_split,
        random_result=random_result,
        lineage_result=lineage_result,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"\nLineage evaluation report written -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
