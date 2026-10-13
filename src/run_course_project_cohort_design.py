"""Read-only planning analysis: assesses whether a ~1,000-genome cohort
can support the course project's intended random-vs-lineage-aware,
three-antibiotic analysis. Downloads nothing, builds no feature matrix,
trains no model, and does not modify any existing manifest - it only
reads the existing Policy A manifests, metadata, and retrieval records,
and reports/simulates candidate cohort designs.

Usage (from the project root, with the project venv active):

    python src/run_course_project_cohort_design.py

Writes (new, read-only-analysis artifacts only):
    data/processed/course_project_cohort_design_analysis.json
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from bvbrc_recon.csv_loader import load_antibiotic_csv
from bvbrc_recon.lineage_split import assign_lineage_groups

ANTIBIOTICS = ["ciprofloxacin", "ampicillin", "gentamicin"]
SEED = 20261009
TARGET_SIZES = [300, 600, 1000, 1500]

DATA_PROCESSED = Path("data/processed")
DATA_RAW = Path("data/raw")
OUTPUT_PATH = DATA_PROCESSED / "course_project_cohort_design_analysis.json"


def load_manifests_and_mlst():
    manifests = {}
    mlst_by_genome: dict[str, str | None] = {}
    for antibiotic in ANTIBIOTICS:
        df = pd.read_csv(DATA_PROCESSED / f"full_cohort_{antibiotic}_policy_a_manifest.csv", dtype={"genome_id": str})
        manifests[antibiotic] = df
        records = load_antibiotic_csv(DATA_PROCESSED / f"{antibiotic}_amr_metadata.csv")
        for r in records:
            gid = r.get("genome_id")
            if gid is not None and gid not in mlst_by_genome:
                mlst_by_genome[gid] = r.get("genome_mlst")
    return manifests, mlst_by_genome


def already_validated_genome_ids() -> set[str]:
    ids: set[str] = set()
    for d in [DATA_RAW / "ciprofloxacin_pilot", DATA_RAW / "ciprofloxacin_expansion", DATA_RAW / "full_cohort_assemblies"]:
        if d.exists():
            ids |= {p.stem for p in d.glob("*.fna")}
    return ids


def per_antibiotic_counts(genome_ids: set[str], manifests: dict[str, pd.DataFrame]) -> dict:
    out = {}
    for antibiotic, df in manifests.items():
        eligible_df = df[df["eligible"]]
        eligible_in_set = eligible_df[eligible_df["genome_id"].isin(genome_ids)]
        out[antibiotic] = {
            "n_eligible_in_set": len(eligible_in_set),
            "n_resistant": int((eligible_in_set["label"] == "Resistant").sum()),
            "n_susceptible": int((eligible_in_set["label"] == "Susceptible").sum()),
        }
    return out


def lineage_summary(genome_ids: list[str], mlst_by_genome: dict[str, str | None]) -> dict:
    assignment = assign_lineage_groups(genome_ids, mlst_by_genome)
    sizes = sorted(assignment.group_sizes.values(), reverse=True)
    n_groups_size_ge_2 = sum(1 for s in sizes if s >= 2)
    return {
        "n_genomes": len(genome_ids),
        "n_groups": assignment.n_groups,
        "n_missing_mlst": len(assignment.missing_mlst_genome_ids),
        "n_groups_with_2_or_more_genomes": n_groups_size_ge_2,
        "largest_group_size": sizes[0] if sizes else 0,
        "majority_scheme": assignment.majority_scheme,
        "n_anomalous_scheme": len(assignment.anomalous_scheme_genome_ids),
    }


def main() -> None:
    manifests, mlst_by_genome = load_manifests_and_mlst()
    eligible_ids_by_antibiotic = {ab: set(df[df["eligible"]]["genome_id"]) for ab, df in manifests.items()}
    union_pool = set()
    for ids in eligible_ids_by_antibiotic.values():
        union_pool |= ids
    union_pool_sorted = sorted(union_pool)

    validated = already_validated_genome_ids()
    validated_in_union = validated & union_pool
    print(f"Already-validated genomes: {len(validated)} total, {len(validated_in_union)} within the Policy A union pool")

    print("\n=== Baseline: the 300 (ish) already-validated genomes ===")
    baseline_counts = per_antibiotic_counts(validated_in_union, manifests)
    for ab, c in baseline_counts.items():
        print(f"  [{ab}] eligible-in-set={c['n_eligible_in_set']} R={c['n_resistant']} S={c['n_susceptible']}")
    baseline_lineage = lineage_summary(sorted(validated_in_union), mlst_by_genome)
    print(f"  lineage groups: {baseline_lineage['n_groups']} total, "
          f"{baseline_lineage['n_groups_with_2_or_more_genomes']} with >=2 genomes, "
          f"largest={baseline_lineage['largest_group_size']}, missing_mlst={baseline_lineage['n_missing_mlst']}")

    print("\n=== Cross-antibiotic membership of the already-validated genomes ===")
    membership_counts = Counter(
        sum(1 for ab in ANTIBIOTICS if gid in eligible_ids_by_antibiotic[ab]) for gid in validated_in_union
    )
    for n_ab, count in sorted(membership_counts.items()):
        print(f"  eligible for exactly {n_ab} antibiotics: {count} genomes")

    print("\n=== Simulated deterministic uniform samples from the union pool, at several target sizes ===")
    rng = np.random.RandomState(SEED)
    remaining_pool = sorted(union_pool - validated_in_union)
    rng.shuffle(remaining_pool)  # one fixed shuffle, sliced for each target size below (nested/consistent)

    size_results = {}
    for target_size in TARGET_SIZES:
        n_additional_needed = max(0, target_size - len(validated_in_union))
        additional = remaining_pool[:n_additional_needed]
        sample_ids = sorted(validated_in_union | set(additional))

        counts = per_antibiotic_counts(set(sample_ids), manifests)
        lineage = lineage_summary(sample_ids, mlst_by_genome)
        membership = Counter(
            sum(1 for ab in ANTIBIOTICS if gid in eligible_ids_by_antibiotic[ab]) for gid in sample_ids
        )

        size_results[target_size] = {
            "n_unique_genomes": len(sample_ids),
            "n_already_validated_reused": len(validated_in_union),
            "n_new_assemblies_needed": n_additional_needed,
            "per_antibiotic": counts,
            "lineage": lineage,
            "eligible_for_n_antibiotics": {str(k): v for k, v in sorted(membership.items())},
        }
        print(f"\n--- target_size={target_size} (new assemblies needed: {n_additional_needed}) ---")
        for ab, c in counts.items():
            native_total = len(eligible_ids_by_antibiotic[ab])
            native_r_frac = sum(1 for g in eligible_ids_by_antibiotic[ab] if manifests[ab].set_index("genome_id").loc[g, "label"] == "Resistant") / native_total if native_total else 0
            sample_r_frac = c["n_resistant"] / c["n_eligible_in_set"] if c["n_eligible_in_set"] else 0
            print(f"  [{ab}] n={c['n_eligible_in_set']} R={c['n_resistant']} ({sample_r_frac:.1%}) S={c['n_susceptible']} "
                  f"(native R rate for comparison: {native_r_frac:.1%})")
        print(f"  lineage groups: {lineage['n_groups']} ({lineage['n_groups_with_2_or_more_genomes']} with >=2 genomes), "
              f"largest={lineage['largest_group_size']}, missing_mlst={lineage['n_missing_mlst']}")

    OUTPUT_PATH.write_text(json.dumps(
        {
            "seed": SEED,
            "union_pool_size": len(union_pool),
            "n_already_validated_total": len(validated),
            "n_already_validated_in_union": len(validated_in_union),
            "baseline_already_validated": {"per_antibiotic": baseline_counts, "lineage": baseline_lineage,
                                            "eligible_for_n_antibiotics": {str(k): v for k, v in sorted(membership_counts.items())}},
            "simulated_target_sizes": size_results,
        },
        indent=2,
    ))
    print(f"\nwrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
