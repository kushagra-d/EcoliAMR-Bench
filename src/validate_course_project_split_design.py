"""Read-only validation of the proposed ~1,000-genome cohort (300
already-validated + 700 deterministically sampled, seed 20261009) and
of the random / lineage-aware split design for each antibiotic.

Independently reconstructs the cohort from saved artifacts only (Policy
A manifests, on-disk assembly listings, antibiotic metadata CSVs) -
does not import or trust the prior planning script's in-memory state -
then checks every claim in the proposal (inclusion, eligibility, no
duplicates, reproducibility, per-antibiotic counts) before evaluating
both split strategies with the project's existing, tested
`compute_random_split_indices` and `split_by_lineage_group`.

Downloads nothing, builds no feature matrix, trains no model, and does
not modify any existing manifest.

Usage (from the project root, with the project venv active):

    python src/validate_course_project_split_design.py

Writes:
    data/processed/course_project_split_design_validation.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bvbrc_recon.csv_loader import load_antibiotic_csv
from bvbrc_recon.lineage_split import assign_lineage_groups, split_by_lineage_group
from bvbrc_recon.pilot_model import CLASS_TO_INT, compute_random_split_indices

ANTIBIOTICS = ["ciprofloxacin", "ampicillin", "gentamicin"]
SEED = 20261009
TARGET_SIZE = 1000
TEST_SIZE = 0.2

DATA_PROCESSED = Path("data/processed")
DATA_RAW = Path("data/raw")
OUTPUT_PATH = DATA_PROCESSED / "course_project_split_design_validation.json"

# Claims from the proposal under review, checked against freshly-computed values below.
CLAIMED_SUBSET_SIZES = {"ciprofloxacin": 876, "ampicillin": 776, "gentamicin": 860}
CLAIMED_ELIGIBLE_POOLS = {
    "ciprofloxacin": (7307, 1846, 5461),
    "ampicillin": (6490, 3659, 2831),
    "gentamicin": (7340, 862, 6478),
}
CLAIMED_LINEAGE = {"n_groups": 256, "n_groups_ge_2": 91, "largest_group": 129}


def load_manifests_and_mlst():
    manifests = {}
    mlst_by_genome: dict[str, str | None] = {}
    for antibiotic in ANTIBIOTICS:
        df = pd.read_csv(
            DATA_PROCESSED / f"full_cohort_{antibiotic}_policy_a_manifest.csv", dtype={"genome_id": str}
        )
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


def reproduce_sample(union_pool: set[str], validated_in_union: set[str]) -> list[str]:
    """The exact deterministic procedure documented in
    data/processed/course_project_cohort_design_spec.json: force-include
    the already-validated genomes, then draw the remainder uniformly at
    random (no label stratification) from the rest of the union pool,
    via one fixed seeded shuffle of the sorted remaining-pool list."""
    remaining_pool = sorted(union_pool - validated_in_union)
    rng = np.random.RandomState(SEED)
    rng.shuffle(remaining_pool)
    n_additional_needed = max(0, TARGET_SIZE - len(validated_in_union))
    additional = remaining_pool[:n_additional_needed]
    return sorted(validated_in_union | set(additional)), additional


def main() -> None:
    report: dict = {"seed": SEED, "target_size": TARGET_SIZE, "test_size": TEST_SIZE}

    manifests, mlst_by_genome = load_manifests_and_mlst()
    eligible_ids_by_antibiotic = {ab: set(df[df["eligible"]]["genome_id"]) for ab, df in manifests.items()}
    label_by_genome_by_antibiotic = {
        ab: dict(zip(df[df["eligible"]]["genome_id"], df[df["eligible"]]["label"])) for ab, df in manifests.items()
    }
    union_pool = set()
    for ids in eligible_ids_by_antibiotic.values():
        union_pool |= ids

    validated = already_validated_genome_ids()
    validated_in_union = validated & union_pool

    # --- Step 2: verify the sampling plan ---
    sample_ids_run1, additional_run1 = reproduce_sample(union_pool, validated_in_union)
    sample_ids_run2, additional_run2 = reproduce_sample(union_pool, validated_in_union)
    deterministic_reproducible = sample_ids_run1 == sample_ids_run2

    sample_ids = sample_ids_run1
    checks = {
        "n_validated_total_on_disk": len(validated),
        "n_validated_in_union_pool": len(validated_in_union),
        "all_300_included_in_sample": validated_in_union.issubset(set(sample_ids)),
        "n_additional_sampled": len(additional_run1),
        "all_additional_in_union_pool": set(additional_run1).issubset(union_pool),
        "all_additional_disjoint_from_validated": set(additional_run1).isdisjoint(validated_in_union),
        "n_duplicate_ids_in_final_sample": len(sample_ids) - len(set(sample_ids)),
        "n_unique_genomes_in_final_sample": len(set(sample_ids)),
        "two_independent_reruns_identical": deterministic_reproducible,
    }
    report["sampling_verification"] = checks

    # --- per-antibiotic eligible-pool claims check (full 8,710-genome pool) ---
    pool_checks = {}
    for ab in ANTIBIOTICS:
        eligible_df = manifests[ab][manifests[ab]["eligible"]]
        n_elig = len(eligible_df)
        n_r = int((eligible_df["label"] == "Resistant").sum())
        n_s = int((eligible_df["label"] == "Susceptible").sum())
        claimed_n, claimed_r, claimed_s = CLAIMED_ELIGIBLE_POOLS[ab]
        pool_checks[ab] = {
            "computed": {"n_eligible": n_elig, "n_resistant": n_r, "n_susceptible": n_s},
            "claimed": {"n_eligible": claimed_n, "n_resistant": claimed_r, "n_susceptible": claimed_s},
            "matches_claim": (n_elig, n_r, n_s) == (claimed_n, claimed_r, claimed_s),
        }
    report["eligible_pool_claim_check"] = pool_checks

    sample_set = set(sample_ids)
    per_antibiotic = {}
    for ab in ANTIBIOTICS:
        subset_ids = sorted(sample_set & eligible_ids_by_antibiotic[ab])
        labels = [label_by_genome_by_antibiotic[ab][gid] for gid in subset_ids]
        n_r = sum(1 for lbl in labels if lbl == "Resistant")
        n_s = sum(1 for lbl in labels if lbl == "Susceptible")
        per_antibiotic[ab] = {
            "subset_ids": subset_ids, "labels": labels,
            "n_total": len(subset_ids), "n_resistant": n_r, "n_susceptible": n_s,
            "resistant_fraction": n_r / len(subset_ids) if subset_ids else None,
            "native_resistant_fraction": pool_checks[ab]["computed"]["n_resistant"] / pool_checks[ab]["computed"]["n_eligible"],
            "matches_claimed_subset_size": len(subset_ids) == CLAIMED_SUBSET_SIZES[ab],
        }
    report["per_antibiotic_subset_check"] = {
        ab: {k: v for k, v in d.items() if k not in ("subset_ids", "labels")} for ab, d in per_antibiotic.items()
    }

    # --- lineage group claim check over the full ~1000-genome pool ---
    assignment = assign_lineage_groups(sample_ids, mlst_by_genome)
    sizes_sorted = sorted(assignment.group_sizes.values(), reverse=True)
    n_groups_ge_2 = sum(1 for s in sizes_sorted if s >= 2)
    lineage_claim_check = {
        "computed": {"n_groups": assignment.n_groups, "n_groups_ge_2": n_groups_ge_2, "largest_group": sizes_sorted[0] if sizes_sorted else 0},
        "claimed": CLAIMED_LINEAGE,
        "matches_claim": (assignment.n_groups, n_groups_ge_2, sizes_sorted[0] if sizes_sorted else 0)
        == (CLAIMED_LINEAGE["n_groups"], CLAIMED_LINEAGE["n_groups_ge_2"], CLAIMED_LINEAGE["largest_group"]),
    }
    report["lineage_claim_check"] = lineage_claim_check

    # --- Step 3: random splits, per antibiotic ---
    random_split_results = {}
    for ab in ANTIBIOTICS:
        d = per_antibiotic[ab]
        ids, labels = d["subset_ids"], d["labels"]
        train_idx, test_idx = compute_random_split_indices(labels, seed=SEED, test_size=TEST_SIZE)
        y = np.array([CLASS_TO_INT[lbl] for lbl in labels])
        train_counts = {"Resistant": int((y[train_idx] == CLASS_TO_INT["Resistant"]).sum()),
                        "Susceptible": int((y[train_idx] == CLASS_TO_INT["Susceptible"]).sum())}
        test_counts = {"Resistant": int((y[test_idx] == CLASS_TO_INT["Resistant"]).sum()),
                       "Susceptible": int((y[test_idx] == CLASS_TO_INT["Susceptible"]).sum())}
        random_split_results[ab] = {
            "n_train": len(train_idx), "n_test": len(test_idx),
            "train_class_counts": train_counts, "test_class_counts": test_counts,
            "both_classes_in_train": train_counts["Resistant"] > 0 and train_counts["Susceptible"] > 0,
            "both_classes_in_test": test_counts["Resistant"] > 0 and test_counts["Susceptible"] > 0,
            "actual_test_fraction": len(test_idx) / len(ids),
            "test_resistant_fraction": test_counts["Resistant"] / len(test_idx) if test_idx.size else None,
        }
    report["random_split_results"] = random_split_results

    # --- Step 4: lineage-aware splits, per antibiotic (own eligible subset) ---
    lineage_split_results = {}
    for ab in ANTIBIOTICS:
        d = per_antibiotic[ab]
        ids, labels = d["subset_ids"], d["labels"]
        sub_assignment = assign_lineage_groups(ids, mlst_by_genome)
        group_by_genome = sub_assignment.group_by_genome
        sub_sizes_sorted = sorted(sub_assignment.group_sizes.values(), reverse=True)

        split_result = split_by_lineage_group(ids, labels, group_by_genome, seed=SEED, test_size=TEST_SIZE)

        train_groups = {group_by_genome[g] for g in split_result.train_genome_ids}
        test_groups = {group_by_genome[g] for g in split_result.test_genome_ids}

        lineage_split_results[ab] = {
            "n_groups_in_subset": sub_assignment.n_groups,
            "n_groups_ge_2_in_subset": sum(1 for s in sub_sizes_sorted if s >= 2),
            "largest_groups_in_subset": sub_sizes_sorted[:5],
            "feasible": split_result.feasible,
            "blocker": split_result.blocker,
            "n_train": len(split_result.train_genome_ids),
            "n_test": len(split_result.test_genome_ids),
            "train_group_count": split_result.train_group_count,
            "test_group_count": split_result.test_group_count,
            "train_class_counts": split_result.train_class_counts,
            "test_class_counts": split_result.test_class_counts,
            "actual_test_fraction": split_result.actual_test_fraction,
            "warnings": split_result.warnings,
            "no_group_overlap_independently_verified": len(train_groups & test_groups) == 0,
        }
    report["lineage_split_results"] = lineage_split_results

    OUTPUT_PATH.write_text(json.dumps(report, indent=2, default=str))

    print("=== Sampling verification ===")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print("\n=== Eligible-pool claim check (full 8,710-genome pool) ===")
    for ab, d in pool_checks.items():
        print(f"  [{ab}] computed={d['computed']} claimed={d['claimed']} matches={d['matches_claim']}")
    print("\n=== Per-antibiotic subset of the ~1000-genome sample ===")
    for ab, d in report["per_antibiotic_subset_check"].items():
        print(f"  [{ab}] n={d['n_total']} R={d['n_resistant']} S={d['n_susceptible']} "
              f"R_frac={d['resistant_fraction']:.1%} native={d['native_resistant_fraction']:.1%} "
              f"matches_claimed_size={d['matches_claimed_subset_size']}")
    print("\n=== Lineage claim check (full ~1000-genome pool) ===")
    print(f"  {lineage_claim_check}")
    print("\n=== Random split results ===")
    for ab, d in random_split_results.items():
        print(f"  [{ab}] train={d['n_train']} test={d['n_test']} "
              f"train_classes={d['train_class_counts']} test_classes={d['test_class_counts']} "
              f"both_in_train={d['both_classes_in_train']} both_in_test={d['both_classes_in_test']}")
    print("\n=== Lineage-aware split results ===")
    for ab, d in lineage_split_results.items():
        print(f"  [{ab}] feasible={d['feasible']} train={d['n_train']} test={d['n_test']} "
              f"train_groups={d['train_group_count']} test_groups={d['test_group_count']} "
              f"train_classes={d['train_class_counts']} test_classes={d['test_class_counts']} "
              f"largest_groups={d['largest_groups_in_subset']} warnings={d['warnings']} "
              f"no_overlap={d['no_group_overlap_independently_verified']}")
    print(f"\nwrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
