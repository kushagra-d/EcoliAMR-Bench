"""Phenotype categorization and duplicate/conflict detection.

Deliberately does not collapse anything on its own: every function here
either labels a value or reports a count, leaving collapsing decisions
(e.g. what to do with Intermediate, or with conflicting replicate calls)
to a later, explicit modelling step.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

KNOWN_PHENOTYPES = {"Resistant", "Susceptible", "Intermediate"}
MISSING_SENTINEL = "<missing/blank>"


def categorize_phenotype(value: str | None) -> str:
    """Bucket a raw ``resistant_phenotype`` value.

    Returns one of "Resistant", "Susceptible", "Intermediate", "missing"
    (None or blank/whitespace-only), or "other" (any other string, e.g.
    non-standard phenotype labels present in the source data).
    """
    if value is None:
        return "missing"
    stripped = value.strip()
    if stripped == "":
        return "missing"
    if stripped in KNOWN_PHENOTYPES:
        return stripped
    return "other"


def phenotype_value_counts(records: list[dict], field_name: str = "resistant_phenotype") -> Counter:
    """Raw-value distribution, including the exact missing/blank/unknown
    strings observed - nothing is grouped into "other" here."""
    counts: Counter = Counter()
    for record in records:
        value = record.get(field_name)
        if value is None or str(value).strip() == "":
            counts[MISSING_SENTINEL] += 1
        else:
            counts[value] += 1
    return counts


@dataclass
class GenomeMultiplicity:
    single_record_genomes: int = 0
    replicate_same_phenotype_genomes: int = 0
    conflicting_phenotype_genomes: int = 0
    conflicts: dict[str, list[str]] = field(default_factory=dict)


def summarize_genome_multiplicity(
    records: list[dict],
    genome_id_field: str = "genome_id",
    phenotype_field: str = "resistant_phenotype",
) -> GenomeMultiplicity:
    """Group AST records by genome_id and classify each genome as having
    one record, multiple records that agree, or multiple records that
    disagree on categorized phenotype (a genuine conflicting call -
    never silently resolved here)."""
    by_genome: dict[str, list[str]] = defaultdict(list)
    for record in records:
        genome_id = record.get(genome_id_field)
        if genome_id is None:
            continue
        by_genome[genome_id].append(categorize_phenotype(record.get(phenotype_field)))

    summary = GenomeMultiplicity()
    for genome_id, phenotypes in by_genome.items():
        distinct = set(phenotypes)
        if len(phenotypes) == 1:
            summary.single_record_genomes += 1
        elif len(distinct) == 1:
            summary.replicate_same_phenotype_genomes += 1
        else:
            summary.conflicting_phenotype_genomes += 1
            summary.conflicts[genome_id] = phenotypes

    return summary
