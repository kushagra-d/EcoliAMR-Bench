"""Builds Policy A cohort manifests for all three antibiotics, a cross-
antibiotic overlap summary, an assembly-availability assessment, and a
deterministic batched retrieval PLAN - no network access, no genome
downloads, no model training. Safe to re-run (idempotent: recomputes
from the current saved metadata every time, never assumes a cached
number is final).

Usage (from the project root, with the project venv active):

    python src/run_full_cohort_planning.py

Writes:
    data/processed/full_cohort_ciprofloxacin_policy_a_manifest.csv
    data/processed/full_cohort_ampicillin_policy_a_manifest.csv
    data/processed/full_cohort_gentamicin_policy_a_manifest.csv
    data/processed/full_cohort_cross_antibiotic_overlap.json
    data/processed/full_cohort_assembly_availability.json
    data/processed/full_cohort_retrieval_plan.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bvbrc_recon.csv_loader import load_antibiotic_csv
from bvbrc_recon.full_cohort_planning import (
    assess_assembly_availability,
    build_policy_a_manifest,
    compute_cross_antibiotic_overlap,
    plan_retrieval_batches,
)

ANTIBIOTICS = ["ciprofloxacin", "ampicillin", "gentamicin"]
SEED = 20261009
RETRIEVAL_BATCH_SIZE = 100

DATA_PROCESSED = Path("data/processed")
DATA_RAW = Path("data/raw")

# A genome's ASSEMBLY is antibiotic-independent - the same FASTA file
# (named only by genome_id) serves every antibiotic a genome happens to
# be eligible for. The raw directories below are named after whichever
# task first retrieved them (historically ciprofloxacin-focused), but
# they must be checked for EVERY antibiotic's availability assessment,
# not just ciprofloxacin's - otherwise a genome already retrieved under
# ciprofloxacin would be wrongly reported "missing" for ampicillin or
# gentamicin just because THEIR directory lists were empty. This was a
# real bug caught by inspecting the first run's output (needed count
# 8,708 did not match the expected 8,710-200=8,510), not assumed.
ALL_RAW_DIRS = [DATA_RAW / "ciprofloxacin_pilot", DATA_RAW / "ciprofloxacin_expansion"]
ALL_RETRIEVAL_MANIFESTS = [
    DATA_PROCESSED / "pilot_ciprofloxacin_retrieval_manifest.csv",
    DATA_PROCESSED / "cohort_expansion_ciprofloxacin_retrieval_manifest.csv",
]
RAW_DIRS_BY_ANTIBIOTIC = {ab: ALL_RAW_DIRS for ab in ANTIBIOTICS}
MANIFESTS_BY_ANTIBIOTIC = {ab: ALL_RETRIEVAL_MANIFESTS for ab in ANTIBIOTICS}


def main() -> None:
    manifests: dict[str, pd.DataFrame] = {}
    eligible_ids_by_antibiotic: dict[str, set[str]] = {}
    labels_by_antibiotic: dict[str, dict[str, str]] = {}

    print("=== Step 2: Policy A cohort manifests (computed fresh from saved metadata) ===")
    for antibiotic in ANTIBIOTICS:
        records = load_antibiotic_csv(DATA_PROCESSED / f"{antibiotic}_amr_metadata.csv")
        mlst_by_genome: dict[str, str | None] = {}
        for r in records:
            gid = r.get("genome_id")
            if gid is not None and gid not in mlst_by_genome:
                mlst_by_genome[gid] = r.get("genome_mlst")

        manifest_df = build_policy_a_manifest(records, antibiotic, mlst_by_genome)
        manifests[antibiotic] = manifest_df
        manifest_path = DATA_PROCESSED / f"full_cohort_{antibiotic}_policy_a_manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)

        eligible = manifest_df[manifest_df["eligible"]]
        eligible_ids_by_antibiotic[antibiotic] = set(eligible["genome_id"])
        labels_by_antibiotic[antibiotic] = dict(zip(eligible["genome_id"], eligible["label"]))
        n_resistant = (eligible["label"] == "Resistant").sum()
        n_susceptible = (eligible["label"] == "Susceptible").sum()
        n_dupes = manifest_df["genome_id"].duplicated().sum()
        print(f"[{antibiotic}] total genomes seen: {len(manifest_df)}, eligible: {len(eligible)} "
              f"({n_resistant} Resistant, {n_susceptible} Susceptible), duplicates: {n_dupes} "
              f"-> wrote {manifest_path}")

    print("\n=== Step 3: cross-antibiotic overlap ===")
    overlap = compute_cross_antibiotic_overlap(eligible_ids_by_antibiotic, labels_by_antibiotic)
    print(f"total antibiotic-specific eligible records (sum across antibiotics): {overlap.total_antibiotic_specific_records}")
    print(f"unique genomes (union): {overlap.n_unique_genomes}")
    for key, count in overlap.pairwise_overlap.items():
        print(f"  pairwise overlap {key}: {count}")
    print(f"  three-way overlap: {overlap.three_way_overlap}")

    overlap_path = DATA_PROCESSED / "full_cohort_cross_antibiotic_overlap.json"
    overlap_path.write_text(json.dumps(
        {
            "antibiotics": overlap.antibiotics,
            "eligible_counts": overlap.eligible_counts,
            "pairwise_overlap": overlap.pairwise_overlap,
            "three_way_overlap": overlap.three_way_overlap,
            "total_antibiotic_specific_records": overlap.total_antibiotic_specific_records,
            "n_unique_genomes": overlap.n_unique_genomes,
            "n_genomes_eligible_for_multiple_antibiotics": sum(
                1 for abs_ in overlap.per_genome_antibiotics.values() if len(abs_) > 1
            ),
        },
        indent=2,
    ))
    # per-genome detail (antibiotic membership + per-antibiotic label) kept separate -
    # this can be large (thousands of rows) so it's its own file, not inlined above.
    per_genome_rows = [
        {
            "genome_id": gid,
            "antibiotics": ";".join(abs_),
            **{f"label_{ab}": overlap.per_genome_labels[gid].get(ab) for ab in overlap.antibiotics},
        }
        for gid, abs_ in overlap.per_genome_antibiotics.items()
    ]
    pd.DataFrame(per_genome_rows).sort_values("genome_id").to_csv(
        DATA_PROCESSED / "full_cohort_cross_antibiotic_per_genome_labels.csv", index=False
    )
    print(f"wrote {overlap_path} and full_cohort_cross_antibiotic_per_genome_labels.csv")

    print("\n=== Step 4: assembly availability ===")
    availability_by_antibiotic = {}
    needed_genome_antibiotics: dict[str, list[str]] = {}
    for antibiotic in ANTIBIOTICS:
        availability = assess_assembly_availability(
            eligible_ids_by_antibiotic[antibiotic],
            RAW_DIRS_BY_ANTIBIOTIC[antibiotic],
            MANIFESTS_BY_ANTIBIOTIC[antibiotic],
        )
        availability.antibiotic = antibiotic
        availability_by_antibiotic[antibiotic] = availability
        print(f"[{antibiotic}] eligible={availability.n_eligible} already_validated={len(availability.already_validated)} "
              f"previously_failed={len(availability.previously_failed)} missing_never_attempted={len(availability.missing_never_attempted)} "
              f"unresolved={len(availability.unresolved)}")

        for gid in availability.missing_never_attempted + availability.previously_failed + availability.unresolved:
            needed_genome_antibiotics.setdefault(gid, []).append(antibiotic)

    availability_path = DATA_PROCESSED / "full_cohort_assembly_availability.json"
    availability_path.write_text(json.dumps(
        {
            ab: {
                "n_eligible": a.n_eligible,
                "n_already_validated": len(a.already_validated),
                "n_previously_failed": len(a.previously_failed),
                "n_missing_never_attempted": len(a.missing_never_attempted),
                "n_unresolved": len(a.unresolved),
                "unresolved_genome_ids": a.unresolved,  # small enough to inline; the rest are large and skipped here
            }
            for ab, a in availability_by_antibiotic.items()
        },
        indent=2,
    ))
    print(f"wrote {availability_path}")
    print(f"genomes needing retrieval across ALL antibiotics (deduplicated): {len(needed_genome_antibiotics)}")

    print("\n=== Step 5: batched retrieval plan (planning only - no network access) ===")
    plan_rows = plan_retrieval_batches(needed_genome_antibiotics, batch_size=RETRIEVAL_BATCH_SIZE, seed=SEED)
    plan_path = DATA_PROCESSED / "full_cohort_retrieval_plan.csv"
    pd.DataFrame(plan_rows).to_csv(plan_path, index=False)
    n_batches = (len(plan_rows) + RETRIEVAL_BATCH_SIZE - 1) // RETRIEVAL_BATCH_SIZE if plan_rows else 0
    print(f"planned {len(plan_rows)} genome downloads across {n_batches} batches of up to {RETRIEVAL_BATCH_SIZE} "
          f"-> wrote {plan_path}")


if __name__ == "__main__":
    main()
