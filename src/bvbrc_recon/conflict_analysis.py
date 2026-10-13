"""Genome-level audit of conflicting AST phenotype calls, and metadata-only
comparison of hypothetical label-curation policies.

This module is deliberately independent of, and stricter than, the
``conflicting_phenotype_genomes`` count in :mod:`bvbrc_recon.analysis`
(``summarize_genome_multiplicity``). That existing function builds its
"distinct phenotype" set from *every* categorized value including
``"missing"`` and ``"other"``, so a genome with one Resistant record and
one blank record is counted as conflicting even though there is no actual
Resistant/Susceptible/Intermediate disagreement. This module instead
restricts the per-genome categorical set to the three recognized labels
only, so "conflict" here means a genuine categorical disagreement -
nothing here claims the older count was wrong in what it measured, only
that it measures something broader than "categorical conflict".

Nothing in this module assigns a final genome label or selects a cohort:
every function either counts, classifies into a labelled bucket, or
estimates the effect of a named hypothetical policy.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .analysis import MISSING_SENTINEL, categorize_phenotype

RECOGNIZED_PHENOTYPES = frozenset({"Resistant", "Susceptible", "Intermediate"})

# Partition of possible per-genome *recognized-categorical* sets. Every
# genome falls into exactly one bucket - this is a partition, verified by
# tests to sum to the total genome count.
GENOME_PATTERN_LABELS: dict[frozenset, str] = {
    frozenset(): "no_recognized_interpretation",
    frozenset({"Resistant"}): "only_resistant",
    frozenset({"Susceptible"}): "only_susceptible",
    frozenset({"Intermediate"}): "only_intermediate",
    frozenset({"Resistant", "Susceptible"}): "resistant_and_susceptible",
    frozenset({"Resistant", "Intermediate"}): "resistant_and_intermediate_no_susceptible",
    frozenset({"Susceptible", "Intermediate"}): "susceptible_and_intermediate_no_resistant",
    frozenset({"Resistant", "Susceptible", "Intermediate"}): "all_three",
}

# A genome pattern counts as a genuine categorical conflict only if its
# recognized-only set has more than one member.
CONFLICTING_PATTERNS = {
    label for pattern, label in GENOME_PATTERN_LABELS.items() if len(pattern) > 1
}

HETEROGENEITY_FIELDS = [
    "testing_standard",
    "testing_standard_year",
    "laboratory_typing_method",
    "laboratory_typing_platform",
    "measurement_unit",
]

# Fields compared to decide whether two records for the same genome are
# exact duplicates. Deliberately excludes the record `id` (BV-BRC's own
# unique key - two duplicate ingestions would have different ids) and
# includes `source`/`pmid` so two independently-published results are
# never treated as the same record even if every other field matches.
DUPLICATE_COMPARISON_FIELDS = [
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


def normalize_value(value: object) -> str:
    """Render any field value (str, None, float NaN, float year) as a
    comparable string, with a single consistent sentinel for missing/blank.
    Never used to overwrite a raw value - only for comparison/grouping."""
    if value is None:
        return MISSING_SENTINEL
    if isinstance(value, float):
        if value != value:  # NaN
            return MISSING_SENTINEL
        if value.is_integer():
            return str(int(value))
        return str(value)
    text = str(value).strip()
    return MISSING_SENTINEL if text == "" else text


def group_by_genome(records: list[dict], genome_id_field: str = "genome_id") -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        genome_id = record.get(genome_id_field)
        if genome_id is None:
            continue
        # normalize_value is a no-op (aside from whitespace stripping) for an
        # already-correct string id; it only matters if a caller passed a
        # float-coerced id (e.g. from careless pandas loading upstream).
        groups[normalize_value(genome_id)].append(record)
    return dict(groups)


def categorical_set(records: list[dict], phenotype_field: str = "resistant_phenotype") -> frozenset:
    """The set of *recognized* categorical phenotypes (Resistant/Susceptible/
    Intermediate) present among these records, ignoring missing/other."""
    categories = {categorize_phenotype(r.get(phenotype_field)) for r in records}
    return frozenset(categories & RECOGNIZED_PHENOTYPES)


def classify_genome_pattern(records: list[dict], phenotype_field: str = "resistant_phenotype") -> str:
    return GENOME_PATTERN_LABELS[categorical_set(records, phenotype_field)]


def record_signature(record: dict, fields: list[str] = DUPLICATE_COMPARISON_FIELDS) -> tuple:
    return tuple(normalize_value(record.get(f)) for f in fields)


def classify_genome_multiplicity(records: list[dict]) -> str:
    """One of "single", "exact_duplicate", "repeated_concordant", "conflicting".

    "conflicting" takes priority: any genuine categorical disagreement
    (more than one recognized phenotype) is reported as conflicting
    regardless of how the records otherwise compare. Only when the
    recognized-categorical set has at most one member do we ask whether
    the (possibly all-missing/other) records are byte-for-byte identical
    (`exact_duplicate`) or merely non-conflicting repeats that differ in
    some field (`repeated_concordant`).
    """
    if len(records) == 1:
        return "single"
    if len(categorical_set(records)) > 1:
        return "conflicting"
    signatures = {record_signature(r) for r in records}
    return "exact_duplicate" if len(signatures) == 1 else "repeated_concordant"


def field_differs_within_genome(records: list[dict], field_name: str) -> bool:
    values = {normalize_value(r.get(field_name)) for r in records}
    return len(values) > 1


@dataclass
class GenomeAudit:
    n_records: int = 0
    n_unique_genomes: int = 0
    pattern_counts: Counter = field(default_factory=Counter)
    multiplicity_counts: Counter = field(default_factory=Counter)
    record_phenotype_counts: Counter = field(default_factory=Counter)
    conflicting_genome_ids: list[str] = field(default_factory=list)
    heterogeneity_among_conflicts: dict[str, dict[str, int]] = field(default_factory=dict)
    genome_pattern: dict[str, str] = field(default_factory=dict)
    genome_multiplicity: dict[str, str] = field(default_factory=dict)
    genome_record_count: dict[str, int] = field(default_factory=dict)


def audit_genomes(records: list[dict]) -> GenomeAudit:
    audit = GenomeAudit()
    audit.n_records = len(records)
    audit.record_phenotype_counts = Counter(categorize_phenotype(r.get("resistant_phenotype")) for r in records)

    groups = group_by_genome(records)
    audit.n_unique_genomes = len(groups)

    for genome_id, genome_records in groups.items():
        pattern = classify_genome_pattern(genome_records)
        multiplicity = classify_genome_multiplicity(genome_records)
        audit.pattern_counts[pattern] += 1
        audit.multiplicity_counts[multiplicity] += 1
        audit.genome_pattern[genome_id] = pattern
        audit.genome_multiplicity[genome_id] = multiplicity
        audit.genome_record_count[genome_id] = len(genome_records)
        if pattern in CONFLICTING_PATTERNS:
            audit.conflicting_genome_ids.append(genome_id)

    for field_name in HETEROGENEITY_FIELDS:
        differs = sum(
            1
            for genome_id in audit.conflicting_genome_ids
            if field_differs_within_genome(groups[genome_id], field_name)
        )
        audit.heterogeneity_among_conflicts[field_name] = {
            "differs": differs,
            "same": len(audit.conflicting_genome_ids) - differs,
        }

    return audit


# --- Task B: hypothetical curation policies -------------------------------


@dataclass
class PolicyResult:
    name: str
    genome_outcome: dict[str, str] = field(default_factory=dict)  # genome_id -> outcome label
    genome_class: dict[str, str] = field(default_factory=dict)  # genome_id -> "Resistant"/"Susceptible" (eligible only)

    def outcome_counts(self) -> Counter:
        return Counter(self.genome_outcome.values())

    def eligible_genome_ids(self) -> list[str]:
        return list(self.genome_class)


def apply_policy_1(groups: dict[str, list[dict]]) -> PolicyResult:
    """Strict categorical agreement.

    A genome is included if its recognized R/S subset agrees on exactly
    one of Resistant or Susceptible. Intermediate does NOT block inclusion
    when it co-occurs with a single-sided R/S call (e.g. Resistant +
    Intermediate is included as Resistant) - Intermediate is kept separate,
    never mapped into the binary class, and never treated as a reason to
    exclude on its own unless no R/S interpretation exists at all.
    """
    result = PolicyResult(name="policy_1_strict_categorical_agreement")
    for genome_id, genome_records in groups.items():
        rs_subset = categorical_set(genome_records) & {"Resistant", "Susceptible"}
        if rs_subset == {"Resistant"}:
            result.genome_outcome[genome_id] = "eligible_resistant"
            result.genome_class[genome_id] = "Resistant"
        elif rs_subset == {"Susceptible"}:
            result.genome_outcome[genome_id] = "eligible_susceptible"
            result.genome_class[genome_id] = "Susceptible"
        elif rs_subset == {"Resistant", "Susceptible"}:
            result.genome_outcome[genome_id] = "excluded_categorical_disagreement"
        else:
            result.genome_outcome[genome_id] = "excluded_no_recognized_binary_interpretation"
    return result


def apply_policy_2(groups: dict[str, list[dict]]) -> PolicyResult:
    """Require a single binary interpretation: the genome's entire
    recognized-categorical set must be exactly {Resistant} or exactly
    {Susceptible} - no Intermediate, no opposite class, at all."""
    result = PolicyResult(name="policy_2_single_binary_interpretation")
    for genome_id, genome_records in groups.items():
        cats = categorical_set(genome_records)
        if cats == {"Resistant"}:
            result.genome_outcome[genome_id] = "eligible_resistant"
            result.genome_class[genome_id] = "Resistant"
        elif cats == {"Susceptible"}:
            result.genome_outcome[genome_id] = "eligible_susceptible"
            result.genome_class[genome_id] = "Susceptible"
        elif "Resistant" in cats and "Susceptible" in cats:
            result.genome_outcome[genome_id] = "excluded_both_classes_present"
        elif "Intermediate" in cats:
            result.genome_outcome[genome_id] = "excluded_intermediate_present_with_binary"
        else:
            result.genome_outcome[genome_id] = "excluded_no_binary_interpretation"
    return result


@dataclass
class PolicyTransparency:
    eligible_with_repeated_records: int = 0
    eligible_with_heterogeneous_field: dict[str, int] = field(default_factory=dict)
    eligible_with_intermediate_co_occurrence: int = 0


def policy_transparency_stats(
    policy_result: PolicyResult, groups: dict[str, list[dict]]
) -> PolicyTransparency:
    stats = PolicyTransparency()
    eligible_ids = policy_result.eligible_genome_ids()
    repeated_ids = [gid for gid in eligible_ids if len(groups[gid]) > 1]
    stats.eligible_with_repeated_records = len(repeated_ids)

    for field_name in HETEROGENEITY_FIELDS:
        stats.eligible_with_heterogeneous_field[field_name] = sum(
            1 for gid in repeated_ids if field_differs_within_genome(groups[gid], field_name)
        )

    stats.eligible_with_intermediate_co_occurrence = sum(
        1 for gid in eligible_ids if "Intermediate" in categorical_set(groups[gid])
    )
    return stats


# --- Task B, Policy 3: standard-stratified descriptive analysis ----------


@dataclass
class StandardStratum:
    raw_value: str
    record_count: int
    phenotype_counts: Counter
    genomes_represented: int
    genomes_globally_conflicting: int


def standard_stratified_analysis(
    records: list[dict], audit: GenomeAudit, standard_field: str = "testing_standard"
) -> list[StandardStratum]:
    """Groups RECORDS (not genomes) by their raw, unmodified testing_standard
    string (missing/blank is its own explicit stratum). Reports, per stratum,
    the categorized phenotype distribution and how many of the genomes
    represented in that stratum are *globally* conflicting (per `audit`,
    computed from ALL of a genome's records for this antibiotic - not
    recomputed per-stratum). This is a cross-tabulation, not a causal claim:
    a genome can appear in more than one stratum if its own records use
    different standards.
    """
    conflicting_ids = set(audit.conflicting_genome_ids)
    by_standard: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_standard[normalize_value(record.get(standard_field))].append(record)

    strata = []
    for raw_value, group_records in by_standard.items():
        genome_ids = {r.get("genome_id") for r in group_records if r.get("genome_id") is not None}
        strata.append(
            StandardStratum(
                raw_value=raw_value,
                record_count=len(group_records),
                phenotype_counts=Counter(
                    categorize_phenotype(r.get("resistant_phenotype")) for r in group_records
                ),
                genomes_represented=len(genome_ids),
                genomes_globally_conflicting=len(genome_ids & conflicting_ids),
            )
        )
    return sorted(strata, key=lambda s: -s.record_count)
