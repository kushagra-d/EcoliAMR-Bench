"""Batched lookup of genome metadata (MLST, cgMLST, assembly, QC fields)
for a set of genome_ids already identified via AST records.

Batches genome_ids with the RQL `in()` operator instead of issuing one
request per genome, per the task's efficiency requirement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .client import BVBRCClient, BVBRCQueryError, rql_in, rql_limit

GENOME_SELECT_FIELDS = [
    "genome_id",
    "genome_status",
    "contigs",
    "genome_length",
    "contig_n50",
    "checkm_completeness",
    "checkm_contamination",
    "genome_quality",
    "mlst",
    "cgmlst_id",
    "cgmlst_hc0",
    "cgmlst_hc2",
    "cgmlst_hc5",
    "cgmlst_hc10",
    "cgmlst_hc20",
    "cgmlst_hc50",
    "cgmlst_hc100",
    "assembly_accession",
    "biosample_accession",
]


@dataclass
class GenomeJoinResult:
    metadata_by_genome_id: dict[str, dict] = field(default_factory=dict)
    missing_genome_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def fetch_genome_metadata(
    client: BVBRCClient,
    genome_ids: list[str],
    select_fields: list[str] = GENOME_SELECT_FIELDS,
    batch_size: int = 200,
) -> GenomeJoinResult:
    result = GenomeJoinResult()
    unique_ids = sorted(set(genome_ids))

    for batch_start in range(0, len(unique_ids), batch_size):
        batch = unique_ids[batch_start : batch_start + batch_size]
        rql = f"{rql_in('genome_id', batch)}&select({','.join(select_fields)})&{rql_limit(len(batch))}"
        try:
            records, _ = client.query("genome", rql)
        except BVBRCQueryError as exc:
            result.errors.append(f"genome batch starting {batch[0]} failed: {exc}")
            continue

        found_ids = set()
        for record in records:
            genome_id = record.get("genome_id")
            if genome_id is None:
                continue
            result.metadata_by_genome_id[genome_id] = record
            found_ids.add(genome_id)

        for genome_id in batch:
            if genome_id not in found_ids:
                result.missing_genome_ids.append(genome_id)

    return result
