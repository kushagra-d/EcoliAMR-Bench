"""Renders reports/final_phenotype_policy_analysis.md."""

from __future__ import annotations

POLICY_LABELS = {
    "A": "Policy A - categorical agreement",
    "B": "Policy B - categorical agreement + standard consistency",
    "C": "Policy C - categorical agreement + method consistency",
}


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "n/a"


def _top_value_table(counter, label: str, top_n: int = 12) -> str:
    lines = [f"| {label} | Count |", "|---|---|"]
    for value, count in counter.most_common(top_n):
        lines.append(f"| `{value}` | {count} |")
    if len(counter) > top_n:
        lines.append(f"| ... | +{len(counter) - top_n} more distinct raw value(s) |")
    return "\n".join(lines)


def render_characterization_section(char) -> str:
    return f"""### {char.antibiotic.capitalize()}: raw metadata distributions

- Records: {char.n_records} | Unique genomes: {char.n_genomes}
- Genomes with repeated, categorically-CONCORDANT records (agree on phenotype, >1 record): **{char.repeated_concordant_genomes}**
  - of those, differing `testing_standard` across their own records: {char.repeated_concordant_heterogeneity.get('testing_standard', 0)}
  - of those, differing `testing_standard_year`: {char.repeated_concordant_heterogeneity.get('testing_standard_year', 0)}
  - of those, differing `laboratory_typing_method`: {char.repeated_concordant_heterogeneity.get('laboratory_typing_method', 0)}
  - of those, differing `laboratory_typing_platform`: {char.repeated_concordant_heterogeneity.get('laboratory_typing_platform', 0)}
  - of those, differing `measurement_unit`: {char.repeated_concordant_heterogeneity.get('measurement_unit', 0)}

**`testing_standard` (raw, exact strings - casing and combined values preserved):**

{_top_value_table(char.testing_standard_counts, "raw testing_standard")}

**`testing_standard_year`:**

{_top_value_table(char.testing_standard_year_counts, "raw testing_standard_year")}

**`laboratory_typing_method`:**

{_top_value_table(char.laboratory_typing_method_counts, "raw laboratory_typing_method")}

**`laboratory_typing_platform`:**

{_top_value_table(char.laboratory_typing_platform_counts, "raw laboratory_typing_platform")}

**`measurement_unit`:**

{_top_value_table(char.measurement_unit_counts, "raw measurement_unit")}
"""


def render_policy_section(outcome) -> str:
    counts = outcome.outcome_counts()
    class_counts = outcome.class_counts()
    eligible_total = sum(class_counts.values())
    lines = ["| Outcome | Genomes | % of genomes |", "|---|---|---|"]
    for reason, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{reason}` | {count} | {_pct(count, outcome.n_total_genomes)} |")
    lines.append(
        f"| **Total (partition check)** | **{sum(counts.values())}** | should equal {outcome.n_total_genomes} |"
    )
    outcome_table = "\n".join(lines)

    proportions = ""
    if eligible_total:
        r = class_counts.get("Resistant", 0)
        s = class_counts.get("Susceptible", 0)
        proportions = (
            f"Resistant {r}/{eligible_total} ({_pct(r, eligible_total)}), "
            f"Susceptible {s}/{eligible_total} ({_pct(s, eligible_total)})"
        )
    else:
        proportions = "No eligible genomes."

    het_lines = "\n".join(
        f"  - differing `{field_name}`: {count}"
        for field_name, count in outcome.eligible_with_heterogeneous_field.items()
    )

    return f"""#### {POLICY_LABELS.get(outcome.policy_name, outcome.policy_name)}

{outcome_table}

- Eligible: **{eligible_total}** genomes - {proportions}
- Eligible genomes with more than one AST record for this antibiotic: {outcome.eligible_with_repeated_records}
{het_lines}
- Eligible genomes with an MLST value: {outcome.eligible_with_mlst} / {eligible_total} ({outcome.eligible_distinct_mlst_groups} distinct groups among them)
"""


def render_pilot_check_section(antibiotic: str, pilot_genome_ids: list[str], pilot_labels: dict, outcomes: dict) -> str:
    lines = [
        f"The existing {len(pilot_genome_ids)}-genome {antibiotic} pilot cohort "
        f"(`data/processed/pilot_{antibiotic}_selected_cohort.csv`) was built using "
        f"`conflict_analysis.apply_policy_2` (a STRICTER rule than any policy A/B/C here - it "
        f"additionally forbids any Intermediate co-occurrence). The pilot cohort itself was NOT "
        f"changed by this task. Checking whether its 100 genomes and their ORIGINAL pilot labels "
        f"would be retained under each new policy:",
        "",
        "| Policy | Retained | Excluded | Label mismatches vs. pilot |",
        "|---|---|---|---|",
    ]
    for name, outcome in outcomes.items():
        retained = sum(1 for g in pilot_genome_ids if g in outcome.genome_class)
        excluded = len(pilot_genome_ids) - retained
        mismatches = sum(
            1
            for g in pilot_genome_ids
            if g in outcome.genome_class and outcome.genome_class[g] != pilot_labels[g]
        )
        lines.append(f"| {name} | {retained} | {excluded} | {mismatches} |")
    lines.append("")
    lines.append(
        "A policy's eligible-genome LABEL is never reinterpreted as a new pilot result here - "
        "the existing pilot's reported metrics remain exactly what they were computed from "
        "(`reports/pilot_ciprofloxacin_report.md`), regardless of what this task finds."
    )
    return "\n".join(lines)


def render_final_policy_report(
    analysis_date: str,
    characterizations: dict,
    all_outcomes: dict,
    pilot_check: dict,
    recommendation_text: str,
) -> str:
    char_sections = "\n".join(render_characterization_section(c) for c in characterizations.values())

    policy_sections = []
    for antibiotic, outcomes in all_outcomes.items():
        policy_sections.append(f"### {antibiotic.capitalize()}\n")
        for name in ("A", "B", "C"):
            policy_sections.append(render_policy_section(outcomes[name]))
    policy_sections_text = "\n".join(policy_sections)

    return f"""# Final Phenotype-Curation Policy Analysis (All Three Antibiotics)

- Analysis date (UTC): {analysis_date}
- Scope: metadata-only. No genome sequence was downloaded, no k-mer was extracted, no model was
  trained. Reuses the existing reconnaissance CSVs and `conflict_analysis.py` functions read-only;
  none of them were modified.

## Step 2 - Testing-standard and method heterogeneity, per antibiotic

{char_sections}

## Step 3 - Candidate policy comparison, per antibiotic

Policy A reuses `conflict_analysis.apply_policy_1` exactly (strict categorical agreement:
Resistant+Intermediate is retained as Resistant, Susceptible+Intermediate as Susceptible;
Resistant+Susceptible disagreement is excluded; Intermediate is never mapped to a binary class).
Policies B and C add, respectively, a testing-standard-consistency and a laboratory-method-
consistency requirement on top of Policy A, under the explicit rules stated above (see
`final_policy_analysis.py` docstrings) - casing is folded for a single named standard only;
combined/mixed standard strings are never guessed at or merged into one.

{policy_sections_text}

## Step 4 - Existing ciprofloxacin pilot under each policy

{pilot_check}

## Step 5 - Recommendation

{recommendation_text}
"""
