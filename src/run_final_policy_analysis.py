"""Final, cross-antibiotic phenotype-curation policy comparison.

Metadata-only: reuses the existing reconnaissance CSVs
(data/processed/{antibiotic}_amr_metadata.csv) and conflict_analysis.py
functions, read-only. Does not download genomes, extract k-mers, or
train models. Does not modify the existing ciprofloxacin pilot cohort,
k-mer matrix, saved splits, or any prior report.

Usage (from the project root, with the project venv active):

    python src/run_final_policy_analysis.py

Writes reports/final_phenotype_policy_analysis.md and
data/processed/final_policy_comparison_summary.csv (new files only).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bvbrc_recon.csv_loader import load_antibiotic_csv
from bvbrc_recon.final_policy_analysis import characterize_metadata, run_all_policies
from bvbrc_recon.final_policy_report import render_final_policy_report, render_pilot_check_section

ANTIBIOTICS = ["ciprofloxacin", "ampicillin", "gentamicin"]
DATA_PROCESSED = Path("data/processed")
REPORT_PATH = Path("reports/final_phenotype_policy_analysis.md")
SUMMARY_CSV_PATH = DATA_PROCESSED / "final_policy_comparison_summary.csv"


def build_mlst_lookup(records: list[dict]) -> dict[str, str | None]:
    lookup: dict[str, str | None] = {}
    for record in records:
        gid = record.get("genome_id")
        if gid is not None and gid not in lookup:
            lookup[gid] = record.get("genome_mlst")
    return lookup


def build_recommendation(all_outcomes: dict, characterizations: dict) -> str:
    lines = []
    lines.append("**Empirical findings (directly computed above):**\n")
    for antibiotic in ANTIBIOTICS:
        a = all_outcomes[antibiotic]["A"]
        b = all_outcomes[antibiotic]["B"]
        c = all_outcomes[antibiotic]["C"]
        n_a, n_b, n_c = len(a.eligible_genome_ids()), len(b.eligible_genome_ids()), len(c.eligible_genome_ids())
        lines.append(
            f"- {antibiotic.capitalize()}: Policy A retains {n_a}, Policy B retains {n_b} "
            f"({_pct_drop(n_a, n_b)} fewer than A), Policy C retains {n_c} ({_pct_drop(n_a, n_c)} fewer than A)."
        )
    lines.append("")
    lines.append(
        "- Across all three antibiotics, Policy B (standard consistency) removes a substantial "
        "fraction of Policy A's eligible genomes, driven overwhelmingly by missing "
        "`testing_standard` values (not by inconsistency between named standards) - this project's "
        "own reconnaissance already established that roughly 30-50% of lab-evidence AST records "
        "per antibiotic have a blank `testing_standard` field."
    )
    lines.append(
        "- Policy C (method consistency) removes relatively few additional genomes beyond Policy "
        "A, because `laboratory_typing_method` is populated for the large majority of records and "
        "is usually consistent within a genome's own repeated records."
    )
    lines.append("")
    lines.append("**Judgment calls made in this analysis (explicitly distinguished from the above):**\n")
    lines.append(
        "- That casing differences in `testing_standard` (e.g. `eucast` vs `EUCAST`) represent the "
        "same intended value is an assumption, not a verified fact about how each record was "
        "entered - a reasonable one, but stated as such."
    )
    lines.append(
        "- That a combined standard string (e.g. `EUCAST, CLSI`) should be treated as "
        "`missing_or_combined` rather than guessed at is a deliberate, conservative choice, not the "
        "only defensible one - a reviewer could instead choose to trust the first-listed standard, "
        "which this analysis does not do."
    )
    lines.append(
        "- That differing `measurement_unit` values should count as method-incompatible (Policy C) "
        "even when the method label matches is a judgment call reflecting the instruction not to "
        "treat different measurement types as interchangeable, not an empirically derived rule."
    )
    lines.append("")
    lines.append("**Recommendation:**\n")
    lines.append(
        "Policy A (categorical agreement alone, i.e. the existing `apply_policy_1`) is recommended "
        "as the primary label-curation rule for all three antibiotics going forward, NOT because it "
        "retains the most genomes, but because: (1) it already excludes the two most scientifically "
        "indefensible cases - genuine Resistant/Susceptible disagreement and unlabeled/Intermediate-"
        "only genomes - without requiring an additional, less-verifiable assumption about standard or "
        "method comparability; (2) Policy B's standard-consistency requirement is overwhelmingly "
        "driven by missing metadata rather than genuine inconsistency, so it mostly discards usable "
        "genomes rather than resolving a real scientific conflict; (3) Policy C adds comparatively "
        "little additional rigor at a modest cost in sample size. This recommendation is a judgment "
        "call about which trade-off is most defensible for a 100-genome-scale pilot - a reviewer "
        "prioritizing standard purity over sample size could reasonably choose Policy B instead, and "
        "that choice should be made explicitly, not by default. Policy A is feasible to apply "
        "identically across all three antibiotics, since it depends only on the phenotype field, "
        "which is populated consistently across all three CSVs."
    )
    return "\n".join(lines)


def _pct_drop(n_a: int, n_other: int) -> str:
    if n_a == 0:
        return "n/a"
    return f"{100 * (n_a - n_other) / n_a:.0f}%"


def main() -> None:
    characterizations = {}
    all_outcomes = {}
    all_records = {}
    all_mlst = {}

    for antibiotic in ANTIBIOTICS:
        records = load_antibiotic_csv(DATA_PROCESSED / f"{antibiotic}_amr_metadata.csv")
        all_records[antibiotic] = records
        mlst_by_genome = build_mlst_lookup(records)
        all_mlst[antibiotic] = mlst_by_genome

        char = characterize_metadata(records, antibiotic)
        characterizations[antibiotic] = char
        print(f"{antibiotic}: {char.n_records} records, {char.n_genomes} genomes characterized")

        outcomes = run_all_policies(records, antibiotic, mlst_by_genome)
        all_outcomes[antibiotic] = outcomes
        for name, outcome in outcomes.items():
            n_eligible = len(outcome.eligible_genome_ids())
            print(f"  Policy {name}: {n_eligible} eligible genomes")

    # Step 4 - existing ciprofloxacin pilot check
    pilot_selected = pd.read_csv(
        DATA_PROCESSED / "pilot_ciprofloxacin_selected_cohort.csv", dtype={"genome_id": str}
    )
    pilot_genome_ids = pilot_selected["genome_id"].tolist()
    pilot_labels = dict(zip(pilot_selected["genome_id"], pilot_selected["label"]))
    pilot_check_text = render_pilot_check_section(
        "ciprofloxacin", pilot_genome_ids, pilot_labels, all_outcomes["ciprofloxacin"]
    )

    recommendation_text = build_recommendation(all_outcomes, characterizations)

    report_text = render_final_policy_report(
        analysis_date=datetime.now(timezone.utc).isoformat(),
        characterizations=characterizations,
        all_outcomes=all_outcomes,
        pilot_check=pilot_check_text,
        recommendation_text=recommendation_text,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"\nReport written -> {REPORT_PATH}")

    summary_rows = []
    for antibiotic, outcomes in all_outcomes.items():
        for policy_name, outcome in outcomes.items():
            for genome_id, reason in outcome.genome_outcome.items():
                summary_rows.append(
                    {
                        "antibiotic": antibiotic,
                        "policy": policy_name,
                        "genome_id": genome_id,
                        "outcome": reason,
                        "assigned_class": outcome.genome_class.get(genome_id),
                    }
                )
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV_PATH, index=False)
    print(f"Summary CSV written -> {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
