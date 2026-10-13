"""Validates and freezes the ACTUAL course-project analysis cohort:
the intersection of the intended, deterministically-sampled 1,000-genome
pool (seed 20261009, see course_project_cohort_design_spec.json) with
the assembly index that is actually on disk after the 700-genome
retrieval run (656/700 succeeded - see
reports/course_project_cohort_design.md Section 8).

Verifies, rather than assumes, that the only genomes missing from the
intended 1,000 are the 44 already-known retrieval failures (no other
silent gap), then reassesses per-antibiotic class balance and MLST
lineage diversity on the real 956-genome cohort, and re-validates both
split strategies on it - reusing the same tested functions as every
prior planning/validation task
(`lineage_split.assign_lineage_groups`/`split_by_lineage_group`,
`pilot_model.compute_random_split_indices`).

Downloads nothing, builds no feature matrix, trains no model, and
does not modify any existing manifest.

Usage (from the project root, with the project venv active):

    python src/freeze_course_project_final_cohort.py

Writes:
    data/processed/course_project_final_cohort_manifest.csv
    data/processed/course_project_final_cohort_{antibiotic}_manifest.csv  (x3)
    data/processed/course_project_final_cohort_validation.json
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
REPORTS = Path("reports")

# The 44 genomes already known (and root-cause-documented) to have failed retrieval -
# loaded from the retrieval plan itself below, not hardcoded, so this script never
# drifts from the actual recorded outcome of the retrieval run.
ADDITIONAL_700_PLAN_PATH = DATA_PROCESSED / "course_project_additional_700_retrieval_plan.csv"

OUTPUT_MANIFEST = DATA_PROCESSED / "course_project_final_cohort_manifest.csv"
OUTPUT_VALIDATION = DATA_PROCESSED / "course_project_final_cohort_validation.json"


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


def reproduce_intended_sample(union_pool: set[str], originally_validated_300: set[str]) -> set[str]:
    """Same deterministic procedure as course_project_cohort_design_spec.json
    and validate_course_project_split_design.py: force-include the
    originally-already-validated 300, then draw 700 more via one fixed
    seeded shuffle of the sorted remaining union-pool IDs. Depends only
    on the union pool and the 300-genome set as they stood BEFORE the
    additional retrieval run, so it reproduces the same intended 1,000
    regardless of which of those 700 later succeeded or failed."""
    remaining_pool = sorted(union_pool - originally_validated_300)
    rng = np.random.RandomState(SEED)
    rng.shuffle(remaining_pool)
    additional = remaining_pool[:700]
    return originally_validated_300 | set(additional)


def lineage_summary(genome_ids: list[str], mlst_by_genome: dict[str, str | None]) -> dict:
    assignment = assign_lineage_groups(genome_ids, mlst_by_genome)
    sizes_sorted = sorted(assignment.group_sizes.values(), reverse=True)
    return {
        "n_genomes": len(genome_ids),
        "n_groups": assignment.n_groups,
        "n_groups_ge_2": sum(1 for s in sizes_sorted if s >= 2),
        "largest_groups": sizes_sorted[:5],
        "n_missing_mlst": len(assignment.missing_mlst_genome_ids),
        "missing_mlst_genome_ids": assignment.missing_mlst_genome_ids,
    }


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

    # the 300 "originally validated" genomes, as they stood BEFORE this retrieval run -
    # pilot (100) + expansion (100) + batch 0 of the full-cohort plan (100). batch 0's
    # genomes already live in the shared full_cohort_assemblies/ directory alongside the
    # new 656, so they are identified via the retrieval plan's own pre-existing record
    # (full_cohort_batch_0_retrieval_manifest.csv) rather than re-derived by guesswork.
    pilot_expansion_ids = set()
    for d in [DATA_RAW / "ciprofloxacin_pilot", DATA_RAW / "ciprofloxacin_expansion"]:
        pilot_expansion_ids |= {p.stem for p in d.glob("*.fna")}
    batch0_manifest = pd.read_csv(DATA_PROCESSED / "full_cohort_batch_0_retrieval_manifest.csv", dtype={"genome_id": str})
    batch0_ids = set(batch0_manifest["genome_id"])
    originally_validated_300 = pilot_expansion_ids | batch0_ids
    assert len(originally_validated_300) == 300, f"expected 300 originally-validated genomes, found {len(originally_validated_300)}"

    intended_1000 = reproduce_intended_sample(union_pool, originally_validated_300)
    assert len(intended_1000) == 1000, f"intended sample reconstruction gave {len(intended_1000)}, not 1000"

    actually_validated_now = already_validated_genome_ids()
    actual_cohort = intended_1000 & actually_validated_now

    missing_from_intended = intended_1000 - actually_validated_now
    extra_beyond_intended = actually_validated_now - intended_1000  # should be empty

    additional_plan = pd.read_csv(ADDITIONAL_700_PLAN_PATH, dtype={"genome_id": str, "status": str, "last_error": str})
    known_failed_44 = set(additional_plan[additional_plan["status"] == "failed"]["genome_id"])

    sampling_checks = {
        "intended_sample_size": len(intended_1000),
        "originally_validated_300_size": len(originally_validated_300),
        "actually_validated_now_size": len(actually_validated_now),
        "actual_cohort_size": len(actual_cohort),
        "n_missing_from_intended": len(missing_from_intended),
        "n_extra_beyond_intended_found_on_disk": len(extra_beyond_intended),
        "extra_beyond_intended_ids": sorted(extra_beyond_intended),
        "n_known_failed_in_additional_700_plan": len(known_failed_44),
        "missing_set_equals_known_44_failures": missing_from_intended == known_failed_44,
        "missing_minus_known_44 (should be empty - any OTHER silent gap)": sorted(missing_from_intended - known_failed_44),
        "known_44_minus_missing (should be empty)": sorted(known_failed_44 - missing_from_intended),
    }
    report["sampling_reconstruction_check"] = sampling_checks

    if sampling_checks["n_extra_beyond_intended_found_on_disk"] > 0:
        print(f"WARNING: {len(extra_beyond_intended)} genomes exist on disk but are NOT part of the intended "
              f"1,000-sample - these are excluded from the frozen cohort below, not silently included.")
    if not sampling_checks["missing_set_equals_known_44_failures"]:
        print("WARNING: the gap between the intended 1,000 and what's actually validated does NOT exactly "
              "match the 44 known failures - see 'missing_minus_known_44' / 'known_44_minus_missing' above.")

    # --- per-antibiotic usable cohort on the ACTUAL 956-genome cohort ---
    per_antibiotic = {}
    final_rows = []
    for gid in sorted(actual_cohort):
        final_rows.append({"genome_id": gid})
    pd.DataFrame(final_rows).to_csv(OUTPUT_MANIFEST, index=False)

    for ab in ANTIBIOTICS:
        usable_ids = sorted(actual_cohort & eligible_ids_by_antibiotic[ab])
        labels = [label_by_genome_by_antibiotic[ab][gid] for gid in usable_ids]
        n_r = sum(1 for lbl in labels if lbl == "Resistant")
        n_s = sum(1 for lbl in labels if lbl == "Susceptible")
        lineage = lineage_summary(usable_ids, mlst_by_genome)

        ab_manifest_path = DATA_PROCESSED / f"course_project_final_cohort_{ab}_manifest.csv"
        pd.DataFrame({"genome_id": usable_ids, "label": labels}).to_csv(ab_manifest_path, index=False)

        per_antibiotic[ab] = {
            "usable_ids": usable_ids, "labels": labels,
            "n_total": len(usable_ids), "n_resistant": n_r, "n_susceptible": n_s,
            "resistant_fraction": n_r / len(usable_ids) if usable_ids else None,
            "lineage": lineage,
        }
    report["per_antibiotic_actual_cohort"] = {
        ab: {k: v for k, v in d.items() if k not in ("usable_ids", "labels")} for ab, d in per_antibiotic.items()
    }

    # --- comparison against the original 1,000-genome design numbers ---
    original_design = {
        "ciprofloxacin": {"n": 876, "n_resistant": 211, "n_susceptible": 665, "n_groups": None},
        "ampicillin": {"n": 776, "n_resistant": 424, "n_susceptible": 352, "n_groups": None},
        "gentamicin": {"n": 860, "n_resistant": 97, "n_susceptible": 763, "n_groups": None},
    }
    comparison = {}
    for ab in ANTIBIOTICS:
        d = per_antibiotic[ab]
        orig = original_design[ab]
        comparison[ab] = {
            "original_design_n": orig["n"], "actual_n": d["n_total"], "shortfall": orig["n"] - d["n_total"],
            "original_design_r_frac": orig["n_resistant"] / orig["n"], "actual_r_frac": d["resistant_fraction"],
        }
    report["comparison_vs_original_1000_design"] = comparison

    # --- random splits on the actual cohort ---
    random_split_results = {}
    for ab in ANTIBIOTICS:
        d = per_antibiotic[ab]
        ids, labels = d["usable_ids"], d["labels"]
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
        }
    report["random_split_results"] = random_split_results

    # --- lineage-aware splits on the actual cohort ---
    lineage_split_results = {}
    for ab in ANTIBIOTICS:
        d = per_antibiotic[ab]
        ids, labels = d["usable_ids"], d["labels"]
        sub_assignment = assign_lineage_groups(ids, mlst_by_genome)
        group_by_genome = sub_assignment.group_by_genome

        split_result = split_by_lineage_group(ids, labels, group_by_genome, seed=SEED, test_size=TEST_SIZE)
        train_groups = {group_by_genome[g] for g in split_result.train_genome_ids}
        test_groups = {group_by_genome[g] for g in split_result.test_genome_ids}

        lineage_split_results[ab] = {
            "feasible": split_result.feasible, "blocker": split_result.blocker,
            "n_train": len(split_result.train_genome_ids), "n_test": len(split_result.test_genome_ids),
            "train_group_count": split_result.train_group_count, "test_group_count": split_result.test_group_count,
            "train_class_counts": split_result.train_class_counts, "test_class_counts": split_result.test_class_counts,
            "actual_test_fraction": split_result.actual_test_fraction,
            "warnings": split_result.warnings,
            "no_group_overlap_independently_verified": len(train_groups & test_groups) == 0,
            "both_classes_in_train": split_result.train_class_counts.get("Resistant", 0) > 0 and split_result.train_class_counts.get("Susceptible", 0) > 0,
            "both_classes_in_test": split_result.test_class_counts.get("Resistant", 0) > 0 and split_result.test_class_counts.get("Susceptible", 0) > 0,
        }
    report["lineage_split_results"] = lineage_split_results

    OUTPUT_VALIDATION.write_text(json.dumps(report, indent=2, default=str))

    print("=== Sampling reconstruction check ===")
    for k, v in sampling_checks.items():
        print(f"  {k}: {v}")
    print("\n=== Per-antibiotic actual cohort (956-genome intersection) ===")
    for ab, d in report["per_antibiotic_actual_cohort"].items():
        print(f"  [{ab}] n={d['n_total']} R={d['n_resistant']} S={d['n_susceptible']} "
              f"R_frac={d['resistant_fraction']:.1%} lineage_groups={d['lineage']['n_groups']} "
              f"(>=2: {d['lineage']['n_groups_ge_2']}) largest={d['lineage']['largest_groups'][:3]} "
              f"missing_mlst={d['lineage']['n_missing_mlst']}")
    print("\n=== Comparison vs original 1,000-genome design ===")
    for ab, d in comparison.items():
        print(f"  [{ab}] {d}")
    print("\n=== Random split results ===")
    for ab, d in random_split_results.items():
        print(f"  [{ab}] train={d['n_train']} test={d['n_test']} both_in_train={d['both_classes_in_train']} "
              f"both_in_test={d['both_classes_in_test']} test_classes={d['test_class_counts']}")
    print("\n=== Lineage-aware split results ===")
    for ab, d in lineage_split_results.items():
        print(f"  [{ab}] feasible={d['feasible']} train={d['n_train']} test={d['n_test']} "
              f"train_groups={d['train_group_count']} test_groups={d['test_group_count']} "
              f"both_in_train={d['both_classes_in_train']} both_in_test={d['both_classes_in_test']} "
              f"no_overlap={d['no_group_overlap_independently_verified']} warnings={d['warnings']}")
    print(f"\nwrote {OUTPUT_MANIFEST}, {OUTPUT_VALIDATION}, and 3 per-antibiotic manifests")


if __name__ == "__main__":
    main()
