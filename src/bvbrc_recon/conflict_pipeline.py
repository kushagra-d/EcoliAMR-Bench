"""Orchestrates the phenotype-conflict audit and policy comparison.

Reads the existing per-antibiotic reconnaissance CSVs read-only, never
touches genome sequence data, and never writes back to the original
CSVs or `reports/bvbrc_feasibility_report.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import analysis as old_analysis
from .conflict_analysis import (
    GenomeAudit,
    PolicyResult,
    PolicyTransparency,
    StandardStratum,
    apply_policy_1,
    apply_policy_2,
    audit_genomes,
    group_by_genome,
    policy_transparency_stats,
    standard_stratified_analysis,
)
from .conflict_report import render_report
from .csv_loader import load_antibiotic_csv

DEFAULT_ANTIBIOTICS = ["ciprofloxacin", "ampicillin", "gentamicin"]


@dataclass
class AntibioticConflictReport:
    antibiotic: str
    audit: GenomeAudit
    policy1: PolicyResult
    policy1_transparency: PolicyTransparency
    policy2: PolicyResult
    policy2_transparency: PolicyTransparency
    strata: list[StandardStratum]
    old_broad_conflict_count: int


def analyze_antibiotic(antibiotic: str, records: list[dict]) -> AntibioticConflictReport:
    audit = audit_genomes(records)
    groups = group_by_genome(records)

    policy1 = apply_policy_1(groups)
    policy2 = apply_policy_2(groups)

    old_multiplicity = old_analysis.summarize_genome_multiplicity(records)

    return AntibioticConflictReport(
        antibiotic=antibiotic,
        audit=audit,
        policy1=policy1,
        policy1_transparency=policy_transparency_stats(policy1, groups),
        policy2=policy2,
        policy2_transparency=policy_transparency_stats(policy2, groups),
        strata=standard_stratified_analysis(records, audit),
        old_broad_conflict_count=old_multiplicity.conflicting_phenotype_genomes,
    )


def build_summary_rows(report: AntibioticConflictReport) -> list[dict]:
    rows = []
    for genome_id in report.audit.genome_pattern:
        rows.append(
            {
                "antibiotic": report.antibiotic,
                "genome_id": genome_id,
                "n_records": report.audit.genome_record_count[genome_id],
                "pattern": report.audit.genome_pattern[genome_id],
                "multiplicity": report.audit.genome_multiplicity[genome_id],
                "policy1_outcome": report.policy1.genome_outcome.get(genome_id),
                "policy1_class": report.policy1.genome_class.get(genome_id),
                "policy2_outcome": report.policy2.genome_outcome.get(genome_id),
                "policy2_class": report.policy2.genome_class.get(genome_id),
            }
        )
    return rows


def main(
    data_dir: Path = Path("data/processed"),
    report_path: Path = Path("reports/phenotype_conflict_analysis.md"),
    summary_csv_path: Path | None = Path("data/processed/phenotype_conflict_policy_summary.csv"),
    antibiotics: list[str] = DEFAULT_ANTIBIOTICS,
) -> dict[str, AntibioticConflictReport]:
    reports: dict[str, AntibioticConflictReport] = {}
    all_summary_rows: list[dict] = []

    for antibiotic in antibiotics:
        csv_path = data_dir / f"{antibiotic}_amr_metadata.csv"
        records = load_antibiotic_csv(csv_path)
        report = analyze_antibiotic(antibiotic, records)
        reports[antibiotic] = report
        all_summary_rows.extend(build_summary_rows(report))
        print(
            f"{antibiotic}: {report.audit.n_unique_genomes} unique genomes, "
            f"{len(report.audit.conflicting_genome_ids)} genuinely conflicting "
            f"(prior broad count: {report.old_broad_conflict_count})"
        )

    if summary_csv_path is not None:
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_summary_rows).to_csv(summary_csv_path, index=False)
        print(f"Derived policy-comparison summary written -> {summary_csv_path}")

    retrieval_note = (
        f"- Analysis date (UTC): {datetime.now(timezone.utc).isoformat()}\n"
        f"- Source CSVs: `data/processed/{{antibiotic}}_amr_metadata.csv` "
        f"(read-only input; not modified by this analysis)"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(reports, retrieval_note), encoding="utf-8")
    print(f"Conflict analysis report written -> {report_path}")

    return reports


if __name__ == "__main__":
    main()
