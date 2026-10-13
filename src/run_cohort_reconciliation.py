"""Reconciles the original 100-genome pilot (Policy 2) and the 100-genome
expansion (Policy A) into a single Policy-A-consistent cohort, using
ONLY the 200 genomes already retrieved by the two prior tasks - never
retrieves anything new.

Usage (from the project root, with the project venv active, AFTER both
run_pilot.py and run_cohort_expansion.py have completed):

    python src/run_cohort_reconciliation.py

Writes:
    data/processed/policy_consistent_ciprofloxacin_reconciliation_table.csv
    data/processed/policy_consistent_ciprofloxacin_selected_cohort.csv
    data/processed/policy_consistent_ciprofloxacin_reconciliation_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bvbrc_recon.cohort_reconciliation import reconcile_cohort
from bvbrc_recon.csv_loader import load_antibiotic_csv

ANTIBIOTIC = "ciprofloxacin"
SEED = 20261009
TARGET_SIZE = 200

DATA_PROCESSED = Path("data/processed")
TABLE_PATH = DATA_PROCESSED / f"policy_consistent_{ANTIBIOTIC}_reconciliation_table.csv"
SELECTED_PATH = DATA_PROCESSED / f"policy_consistent_{ANTIBIOTIC}_selected_cohort.csv"
SUMMARY_PATH = DATA_PROCESSED / f"policy_consistent_{ANTIBIOTIC}_reconciliation_summary.json"


def main() -> None:
    pilot_selected = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_selected_cohort.csv", dtype={"genome_id": str}
    )
    expansion_selected = pd.read_csv(
        DATA_PROCESSED / f"cohort_expansion_{ANTIBIOTIC}_selected.csv", dtype={"genome_id": str}
    )
    pilot_genome_ids = pilot_selected["genome_id"].tolist()
    pilot_labels = dict(zip(pilot_selected["genome_id"], pilot_selected["label"]))
    expansion_genome_ids = expansion_selected["genome_id"].tolist()
    expansion_labels = dict(zip(expansion_selected["genome_id"], expansion_selected["label"]))

    print(f"pilot: {len(pilot_genome_ids)} genomes (Policy 2); expansion: {len(expansion_genome_ids)} genomes (Policy A)")

    records = load_antibiotic_csv(DATA_PROCESSED / f"{ANTIBIOTIC}_amr_metadata.csv")
    mlst_by_genome: dict[str, str | None] = {}
    for r in records:
        gid = r.get("genome_id")
        if gid is not None and gid not in mlst_by_genome:
            mlst_by_genome[gid] = r.get("genome_mlst")

    result = reconcile_cohort(
        records, mlst_by_genome, ANTIBIOTIC,
        pilot_genome_ids, pilot_labels, expansion_genome_ids, expansion_labels,
        target_size=TARGET_SIZE, seed=SEED,
    )

    for note in result.limitations:
        print(f"  NOTE: {note}")
    print(f"pilot eligible/ineligible: {result.n_pilot_eligible}/{result.n_pilot_ineligible}")
    print(f"expansion eligible/ineligible: {result.n_expansion_eligible}/{result.n_expansion_ineligible}")
    print(f"label mismatches (eligible genomes whose Policy A label differs from their original label): "
          f"{result.n_label_mismatches}")
    print(f"duplicate genome IDs across pilot+expansion: {result.n_duplicate_genome_ids}")
    print(f"FINAL reconciled cohort: {len(result.included_genome_ids)} genomes - "
          f"{result.n_included_resistant} Resistant, {result.n_included_susceptible} Susceptible")

    table_rows = [
        {
            "genome_id": e.genome_id, "original_cohort": e.original_cohort, "original_policy": e.original_policy,
            "original_label": e.original_label, "policy_a_eligible": e.policy_a_eligible,
            "policy_a_label": e.policy_a_label, "policy_a_outcome": e.policy_a_outcome,
            "label_matches_original": e.label_matches_original, "included": e.included,
            "exclusion_reason": e.exclusion_reason,
        }
        for e in result.entries
    ]
    pd.DataFrame(table_rows).to_csv(TABLE_PATH, index=False)

    selected_df = pd.DataFrame({
        "genome_id": result.included_genome_ids,
        "label": [result.included_labels[g] for g in result.included_genome_ids],
    })
    selected_df.to_csv(SELECTED_PATH, index=False)

    SUMMARY_PATH.write_text(json.dumps(
        {
            "antibiotic": ANTIBIOTIC, "seed": SEED, "target_size": TARGET_SIZE,
            "n_pilot_total": len(pilot_genome_ids), "n_expansion_total": len(expansion_genome_ids),
            "n_pilot_eligible": result.n_pilot_eligible, "n_pilot_ineligible": result.n_pilot_ineligible,
            "n_expansion_eligible": result.n_expansion_eligible, "n_expansion_ineligible": result.n_expansion_ineligible,
            "n_label_mismatches": result.n_label_mismatches, "n_duplicate_genome_ids": result.n_duplicate_genome_ids,
            "n_final_cohort": len(result.included_genome_ids),
            "n_included_resistant": result.n_included_resistant, "n_included_susceptible": result.n_included_susceptible,
            "limitations": result.limitations,
        },
        indent=2,
    ))
    print(f"wrote {TABLE_PATH}, {SELECTED_PATH}, {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
