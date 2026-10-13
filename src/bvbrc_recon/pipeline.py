"""End-to-end metadata-only reconnaissance pipeline.

For each target antibiotic: paginate laboratory-evidence AST records for
E. coli from `genome_amr`, batch-join genome metadata (MLST, cgMLST,
assembly/QC fields) from `genome`, write one CSV under data/processed/,
and render a combined feasibility report under reports/.

No genome sequence data is ever requested or written by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .analysis import GenomeMultiplicity, phenotype_value_counts, summarize_genome_multiplicity
from .client import BVBRCClient, rql_and, rql_eq
from .genome_join import GenomeJoinResult, fetch_genome_metadata
from .pagination import PaginationResult, paginate
from .report import render_report

TAXON_ID_ECOLI = 562
TARGET_ANTIBIOTICS = ["ciprofloxacin", "ampicillin", "gentamicin"]
EVIDENCE_FILTER = "Laboratory Method"

AMR_SELECT_FIELDS = [
    "id",
    "genome_id",
    "genome_name",
    "antibiotic",
    "resistant_phenotype",
    "measurement",
    "measurement_sign",
    "measurement_value",
    "measurement_unit",
    "laboratory_typing_method",
    "laboratory_typing_platform",
    "testing_standard",
    "testing_standard_year",
    "evidence",
    "source",
    "pmid",
]


@dataclass
class AntibioticResult:
    antibiotic: str
    pagination: PaginationResult
    genome_join: GenomeJoinResult
    multiplicity: GenomeMultiplicity
    phenotype_counts: dict
    merged_rows: list[dict]


def run_antibiotic(
    client: BVBRCClient,
    antibiotic: str,
    page_size: int = 1000,
    genome_batch_size: int = 200,
) -> AntibioticResult:
    rql_filter = rql_and(
        rql_eq("taxon_id", TAXON_ID_ECOLI),
        rql_eq("antibiotic", antibiotic),
        rql_eq("evidence", EVIDENCE_FILTER),
    )
    pagination = paginate(
        client, "genome_amr", rql_filter, AMR_SELECT_FIELDS, page_size=page_size
    )

    genome_ids = [r["genome_id"] for r in pagination.records if r.get("genome_id")]
    genome_join = fetch_genome_metadata(client, genome_ids, batch_size=genome_batch_size)
    multiplicity = summarize_genome_multiplicity(pagination.records)
    phenotype_counts = phenotype_value_counts(pagination.records)

    merged_rows = []
    for record in pagination.records:
        row = dict(record)
        genome_meta = genome_join.metadata_by_genome_id.get(record.get("genome_id"), {})
        for key, value in genome_meta.items():
            if key == "genome_id":
                continue
            row[f"genome_{key}"] = value
        merged_rows.append(row)

    return AntibioticResult(
        antibiotic=antibiotic,
        pagination=pagination,
        genome_join=genome_join,
        multiplicity=multiplicity,
        phenotype_counts=phenotype_counts,
        merged_rows=merged_rows,
    )


def save_csv(result: AntibioticResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{result.antibiotic}_amr_metadata.csv"
    pd.DataFrame(result.merged_rows).to_csv(out_path, index=False)
    return out_path


def main(
    out_data_dir: Path = Path("data/processed"),
    out_report_dir: Path = Path("reports"),
    client: BVBRCClient | None = None,
    antibiotics: list[str] = TARGET_ANTIBIOTICS,
) -> dict[str, AntibioticResult]:
    client = client or BVBRCClient()
    retrieval_date = datetime.now(timezone.utc).isoformat()

    results: dict[str, AntibioticResult] = {}
    for antibiotic in antibiotics:
        results[antibiotic] = run_antibiotic(client, antibiotic)
        csv_path = save_csv(results[antibiotic], out_data_dir)
        print(f"{antibiotic}: wrote {len(results[antibiotic].merged_rows)} rows -> {csv_path}")

    out_report_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_report_dir / "bvbrc_feasibility_report.md"
    report_path.write_text(render_report(results, retrieval_date), encoding="utf-8")
    print(f"Feasibility report written -> {report_path}")

    return results


if __name__ == "__main__":
    main()
