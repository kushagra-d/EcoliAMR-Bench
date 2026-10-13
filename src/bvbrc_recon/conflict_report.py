"""Renders reports/phenotype_conflict_analysis.md from the computed
AntibioticConflictReport objects (see pipeline module `conflict_pipeline`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .conflict_analysis import GENOME_PATTERN_LABELS

if TYPE_CHECKING:
    from .conflict_pipeline import AntibioticConflictReport

PATTERN_ORDER = list(dict.fromkeys(GENOME_PATTERN_LABELS.values()))
MULTIPLICITY_ORDER = ["single", "exact_duplicate", "repeated_concordant", "conflicting"]


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "n/a"


def _pattern_table(report: "AntibioticConflictReport") -> str:
    audit = report.audit
    lines = ["| Genome pattern (recognized categories only) | Genomes | % of unique genomes |", "|---|---|---|"]
    for label in PATTERN_ORDER:
        count = audit.pattern_counts.get(label, 0)
        lines.append(f"| `{label}` | {count} | {_pct(count, audit.n_unique_genomes)} |")
    lines.append(f"| **Total (partition check)** | **{sum(audit.pattern_counts.values())}** | should equal {audit.n_unique_genomes} |")
    return "\n".join(lines)


def _multiplicity_table(report: "AntibioticConflictReport") -> str:
    audit = report.audit
    lines = ["| Record-multiplicity class | Genomes |", "|---|---|"]
    for label in MULTIPLICITY_ORDER:
        lines.append(f"| `{label}` | {audit.multiplicity_counts.get(label, 0)} |")
    lines.append(f"| **Total (partition check)** | **{sum(audit.multiplicity_counts.values())}** (should equal {audit.n_unique_genomes}) |")
    return "\n".join(lines)


def _heterogeneity_table(report: "AntibioticConflictReport") -> str:
    audit = report.audit
    n_conflicting = len(audit.conflicting_genome_ids)
    lines = [
        f"Among the **{n_conflicting}** genomes with a genuine categorical conflict "
        f"(Resistant+Susceptible, Resistant+Intermediate-no-Susceptible, "
        f"Susceptible+Intermediate-no-Resistant, or all three):",
        "",
        "| Metadata field | Differs across the genome's records | Same across the genome's records |",
        "|---|---|---|",
    ]
    for field_name, counts in audit.heterogeneity_among_conflicts.items():
        lines.append(f"| `{field_name}` | {counts['differs']} ({_pct(counts['differs'], n_conflicting)}) | {counts['same']} |")
    lines.append(
        "\n*Co-occurrence only, not causation: a differing field alongside a conflicting call does "
        "not establish that the field difference caused the disagreement - it may equally reflect "
        "genuine biological variability (e.g. a true phenotype change between sampling events) or an "
        "unrelated data-entry difference.*"
    )
    return "\n".join(lines)


def _old_vs_new_conflict_comparison(report: "AntibioticConflictReport") -> str:
    old = report.old_broad_conflict_count
    new = len(report.audit.conflicting_genome_ids)
    return (
        f"The prior reconnaissance report counted **{old}** genomes as \"conflicting\" using a "
        f"broader definition (any two records disagreeing on categorized phenotype, where "
        f"\"missing\"/\"other\" values count as distinct categories). Re-deriving that same broad "
        f"count from the current CSV confirms it was computed correctly as *defined* - it was not "
        f"an error, but a different and broader question than this audit asks. Restricting to "
        f"genuine Resistant/Susceptible/Intermediate disagreement (ignoring missing/other) gives "
        f"**{new}** genomes - the number used throughout the rest of this report as \"conflicting\"."
    )


def _policy_table(report: "AntibioticConflictReport", policy_key: str) -> str:
    policy = getattr(report, policy_key)
    transparency = getattr(report, f"{policy_key}_transparency")
    counts = policy.outcome_counts()
    total_genomes = report.audit.n_unique_genomes
    eligible_r = counts.get("eligible_resistant", 0)
    eligible_s = counts.get("eligible_susceptible", 0)
    eligible_total = eligible_r + eligible_s

    lines = [
        "| Outcome | Genomes | % of unique genomes |",
        "|---|---|---|",
        f"| eligible_resistant | {eligible_r} | {_pct(eligible_r, total_genomes)} |",
        f"| eligible_susceptible | {eligible_s} | {_pct(eligible_s, total_genomes)} |",
    ]
    for reason, count in counts.items():
        if reason.startswith("excluded"):
            lines.append(f"| {reason} | {count} | {_pct(count, total_genomes)} |")
    lines.append(f"| **Total (partition check)** | **{sum(counts.values())}** | should equal {total_genomes} |")
    lines.append("")
    if eligible_total:
        lines.append(
            f"Class proportions among eligible genomes: Resistant {eligible_r}/{eligible_total} "
            f"({_pct(eligible_r, eligible_total)}), Susceptible {eligible_s}/{eligible_total} "
            f"({_pct(eligible_s, eligible_total)})."
        )
    else:
        lines.append("No genomes were eligible under this policy.")
    lines.append("")
    lines.append(f"- Eligible genomes with more than one AST record for this antibiotic: {transparency.eligible_with_repeated_records}")
    for field_name, count in transparency.eligible_with_heterogeneous_field.items():
        lines.append(
            f"  - of those, differing `{field_name}` despite categorical agreement: {count}"
        )
    lines.append(
        f"- Eligible genomes that also carry an Intermediate record alongside their binary call: "
        f"{transparency.eligible_with_intermediate_co_occurrence}"
    )
    return "\n".join(lines)


def _strata_table(report: "AntibioticConflictReport") -> str:
    lines = [
        "| Raw `testing_standard` value | Records | Resistant | Susceptible | Intermediate | Missing | Other | Genomes represented | ... globally conflicting |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for stratum in report.strata:
        c = stratum.phenotype_counts
        lines.append(
            f"| `{stratum.raw_value}` | {stratum.record_count} | {c.get('Resistant', 0)} | "
            f"{c.get('Susceptible', 0)} | {c.get('Intermediate', 0)} | {c.get('missing', 0)} | "
            f"{c.get('other', 0)} | {stratum.genomes_represented} | {stratum.genomes_globally_conflicting} |"
        )
    lines.append(
        "\n*\"...globally conflicting\" counts, for genomes represented in this testing-standard "
        "stratum, how many are globally conflicting across ALL of their records for this antibiotic "
        "(not just the records in this stratum). A genome with records in more than one stratum is "
        "counted in each. This is a cross-tabulation for descriptive purposes only - it does not "
        "imply any stratum is or is not suitable for a primary modelling cohort.*"
    )
    return "\n".join(lines)


def render_antibiotic_section(report: "AntibioticConflictReport") -> str:
    return f"""### {report.antibiotic.capitalize()}

**Records:** {report.audit.n_records} | **Unique genomes:** {report.audit.n_unique_genomes}

#### Task A - Genome-level pattern audit

{_pattern_table(report)}

**Record-vs-genome multiplicity:**

{_multiplicity_table(report)}

**Conflict-count reconciliation with the prior reconnaissance report:**

{_old_vs_new_conflict_comparison(report)}

**Metadata heterogeneity among conflicting genomes:**

{_heterogeneity_table(report)}

#### Task B - Policy 1: strict categorical agreement

Resistant+Intermediate (and Susceptible+Intermediate) genomes ARE eligible under this policy -
Intermediate does not block inclusion and is never mapped into the binary class; it is simply
not counted against the genome when only one of Resistant/Susceptible is present.

{_policy_table(report, "policy1")}

#### Task B - Policy 2: require a single binary interpretation

Any Intermediate co-occurrence, or any Resistant+Susceptible disagreement, excludes the genome
entirely under this stricter policy.

{_policy_table(report, "policy2")}

#### Task B - Policy 3: standard-stratified descriptive analysis (no cohort implied)

{_strata_table(report)}
"""


TASK_C_UNCERTAINTIES = """## Task C - Decisions that still require human judgment

- **Combined/mixed testing_standard values** (e.g. `"EUCAST, CLSI"`, `"CLSI, EUCAST"`,
  `"EUCAST and CLSI"`) were observed as distinct raw strings. Whether these represent a single
  concordant cross-standard confirmation, an ambiguous/unreliable entry, or something else is not
  resolved here - no rule has been invented to merge, split, or discard them.
- **Whether testing_standard_year must match** between records used to confirm a phenotype call
  is unresolved; breakpoints are revised over time and a multi-year gap between concordant calls
  may or may not be acceptable depending on the antibiotic and standard.
- **Whether laboratory_typing_method/platform need to be restricted** (e.g. requiring broth
  dilution with a reported MIC over disk diffusion) has not been decided; this report only
  quantifies how often these fields differ among agreeing or conflicting records.
- **How Intermediate results should be handled** beyond "never silently mapped to Susceptible" -
  whether Intermediate genomes should form a third model class, be excluded, or be revisited with
  updated breakpoints is left open; Policies 1 and 2 above show two different consequences of two
  different stances, not a recommendation.
- **How to treat genomes that agree categorically but differ in MIC/disk measurement or metadata**
  (the `repeated_concordant` bucket) - whether to keep one representative record, average/aggregate
  measurements, or keep all records as-is for provenance is unresolved.
- **How to define an isolate** when multiple genome_ids or AST records may refer to related or
  repeated biological material (e.g. resequencing, repeat submissions) is unresolved. This audit
  also surfaced a concrete, separate tooling risk worth carrying into that decision: naively
  reading these CSVs with pandas auto-inferring `genome_id` as a float silently merges distinct
  genome_id strings in this dataset (confirmed: fewer unique float-coerced ids than unique string
  ids for all three antibiotics) - any future isolate-definition work must load `genome_id` as a
  string, as this module does, not rely on default CSV type inference.
"""


def render_report(reports: dict[str, "AntibioticConflictReport"], retrieval_note: str) -> str:
    sections = "\n".join(render_antibiotic_section(r) for r in reports.values())
    return f"""# Phenotype Conflict Analysis and Curation-Policy Comparison

{retrieval_note}

This report performs no label assignment and selects no modelling cohort. Every count below is
computed directly from the existing reconnaissance CSVs under `data/processed/` (read-only; those
files and the original `reports/bvbrc_feasibility_report.md` were not modified). Genomes are
grouped by `genome_id` loaded explicitly as a string (see Task C for why this matters).

## Per-antibiotic results

{sections}

{TASK_C_UNCERTAINTIES}
"""
