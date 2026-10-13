"""Renders the plain-language feasibility report (Markdown)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import AntibioticResult

QUERY_FILTERS_NOTE = (
    "taxon_id=562 (Escherichia coli) AND antibiotic=<name, lowercase> "
    "AND evidence=\"Laboratory Method\""
)


def _phenotype_counts_section(result: "AntibioticResult") -> str:
    lines = ["| Raw `resistant_phenotype` value | Count |", "|---|---|"]
    for value, count in sorted(result.phenotype_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{value}` | {count} |")
    return "\n".join(lines)


def _testing_standard_section(result: "AntibioticResult") -> str:
    from collections import Counter

    combo_counts: Counter = Counter()
    for record in result.pagination.records:
        standard = record.get("testing_standard") or "<missing/blank>"
        year = record.get("testing_standard_year") or "<missing/blank>"
        combo_counts[(standard, year)] += 1
    lines = ["| testing_standard | testing_standard_year | Count |", "|---|---|---|"]
    for (standard, year), count in sorted(combo_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {standard} | {year} | {count} |")
    return "\n".join(lines)


def render_antibiotic_section(result: "AntibioticResult") -> str:
    pag = result.pagination
    gj = result.genome_join
    mult = result.multiplicity

    total_genome_ids = len({r.get("genome_id") for r in pag.records if r.get("genome_id")})
    mlst_available = sum(
        1 for meta in gj.metadata_by_genome_id.values() if meta.get("mlst")
    )
    distinct_mlst = len(
        {meta["mlst"] for meta in gj.metadata_by_genome_id.values() if meta.get("mlst")}
    )
    assembly_accession_available = sum(
        1 for meta in gj.metadata_by_genome_id.values() if meta.get("assembly_accession")
    )
    from collections import Counter

    genome_status_counts = Counter(
        meta.get("genome_status") or "<missing/blank>"
        for meta in gj.metadata_by_genome_id.values()
    )
    genome_status_line = ", ".join(
        f"{status}={count}" for status, count in sorted(genome_status_counts.items())
    )

    completeness_note = (
        "Retrieval appears COMPLETE: unique record count matches the last reported total."
        if pag.complete
        else "Retrieval is **NOT confirmed complete** - see Limitations below."
    )
    drift_note = (
        f"Reported total changed during pagination (first={pag.reported_total_first}, "
        f"last={pag.reported_total_last}) - the source collection was updated mid-retrieval."
        if pag.total_drifted
        else "Reported total was stable across all pages."
    )

    return f"""### {result.antibiotic.capitalize()}

- Records retrieved (unique by `id`): **{len(pag.records)}**
- Reported total (`Content-Range`, first page): {pag.reported_total_first}
- Reported total (`Content-Range`, last page): {pag.reported_total_last}
- {drift_note}
- Duplicate record ids seen across pages (same `id`, counted once, overlap logged): {len(pag.duplicate_record_ids)}
- Pagination errors: {pag.errors or "none"}
- {completeness_note}
- Unique genome IDs referenced: **{total_genome_ids}**
- Genome metadata found for: {len(gj.metadata_by_genome_id)} / {total_genome_ids} genome IDs
- Genome IDs with NO matching genome record (metadata missing): {len(gj.missing_genome_ids)}
- Genome-metadata batch errors: {gj.errors or "none"}
- MLST available for: {mlst_available} / {len(gj.metadata_by_genome_id)} joined genomes ({distinct_mlst} distinct MLST values)
- `genome_status` distribution among joined genomes: {genome_status_line}
- NCBI `assembly_accession` populated for: {assembly_accession_available} / {len(gj.metadata_by_genome_id)} joined genomes (the rest have a BV-BRC assembly but no recorded NCBI cross-reference in this field)
- Genomes with exactly one AST record for this antibiotic: {mult.single_record_genomes}
- Genomes with multiple AST records that AGREE on phenotype (replicates, not collapsed): {mult.replicate_same_phenotype_genomes}
- Genomes with multiple AST records that DISAGREE on phenotype (conflicting calls, all rows preserved in CSV): {mult.conflicting_phenotype_genomes}

**Raw `resistant_phenotype` value distribution** (every observed value, nothing grouped):

{_phenotype_counts_section(result)}

**`testing_standard` x `testing_standard_year` distribution:**

{_testing_standard_section(result)}
"""


def render_report(results: dict, retrieval_date: str) -> str:
    sections = "\n".join(render_antibiotic_section(r) for r in results.values())
    return f"""# BV-BRC AMR Metadata Feasibility Report

- Retrieval date (UTC): {retrieval_date}
- Query filters applied: {QUERY_FILTERS_NOTE}
- Source: BV-BRC Data API (`https://www.bv-brc.org/api/genome_amr/`, `.../genome/`)
- Scope: metadata only - no genome sequence data was downloaded.

## Per-antibiotic results

{sections}

## Rules applied (recorded explicitly, per task constraints)

- Intermediate was NOT treated as Susceptible anywhere in this pipeline.
- Missing/blank/non-standard `resistant_phenotype` values were NOT treated as a valid class; they
  are counted separately under the raw-value distribution above.
- No cross-standard (e.g. EUCAST vs CLSI) combination or normalization was performed; the
  `testing_standard` x `testing_standard_year` table above is reported as-is for a later, explicit
  decision.
- Exact duplicate records (same `id` returned on more than one page, a pagination artifact) were
  deduplicated; this is the ONLY collapsing rule applied, and the count of such duplicates is
  reported per antibiotic above.
- Multiple independent AST records for the same genome_id (replicates or conflicts) were NOT
  collapsed into one row; both the CSV and the counts above preserve every record.
- No genome-quality thresholds (contig count, CheckM completeness/contamination, etc.) were
  applied to filter records; those fields are included in the CSV for a later, explicit decision.

## Limitations and unresolved questions

- "Completeness" above is evidence-based: it only reports COMPLETE when every page succeeded and
  the final unique-record count matches the most recently reported total. Any error, or any report
  marked not-complete, means the corresponding CSV may be a partial snapshot - treat it as such
  rather than as the final cohort.
- BV-BRC is a live, continuously-updated database; "reported total" drift during retrieval reflects
  genuine concurrent updates, not a client bug, but it does mean these counts are a snapshot, not a
  fixed ground truth.
- The MLST scheme is not validated against the recorded species for each genome in this script;
  an earlier manual spot-check found at least one genome labelled *Escherichia coli* with an MLST
  value from a different species' scheme, which this automated pass does not detect or correct.
- No genome-level QC filtering (assembly completeness, contamination, contig count) has been
  applied; the CSVs intentionally include genomes of unknown or poor quality for now.
- Final cohort size, lineage-split strategy, and label-standardization choices are explicitly out
  of scope for this task and are left for a dedicated follow-up decision informed by this report.
"""
