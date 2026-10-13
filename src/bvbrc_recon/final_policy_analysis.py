"""Final, cross-antibiotic phenotype-curation policy comparison.

Builds on (reuses, never reimplements) `conflict_analysis.py`:
`group_by_genome`, `categorical_set`, `apply_policy_1`, `classify_genome_multiplicity`,
`field_differs_within_genome`, `normalize_value`, `MISSING_SENTINEL`.

Policy A here is exactly `conflict_analysis.apply_policy_1` ("strict
categorical agreement": a genome is eligible if its recognized R/S subset
agrees on exactly one side; Intermediate co-occurring with a single-sided
R/S call does not exclude the genome, and is never mapped to either
class). Policies B and C add testing-standard and laboratory-method
consistency requirements on top of Policy A, each under an explicit,
stated rule - never an invented or silently-applied normalization.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .conflict_analysis import (
    MISSING_SENTINEL,
    apply_policy_1,
    categorical_set,
    classify_genome_multiplicity,
    field_differs_within_genome,
    group_by_genome,
    normalize_value,
)

RECOGNIZED_SINGLE_STANDARDS = {"EUCAST", "CLSI"}
OTHER_OR_COMBINED_OR_MISSING = "<other/combined/missing>"

STANDARD_CONSISTENT = "consistent_single_standard"
STANDARD_INCONSISTENT = "inconsistent_across_records"
STANDARD_MISSING_OR_COMBINED = "missing_or_combined_only"

METHOD_CONSISTENT = "consistent_method"
METHOD_INCONSISTENT = "inconsistent_method_or_unit"
METHOD_MISSING = "missing_method_only"


def normalize_standard_for_eligibility(raw_value) -> str:
    """Case-folds ONLY an exact, single, non-combined standard name (e.g.
    'eucast' -> 'EUCAST'). A blank/missing value, or any value naming more
    than one standard (e.g. 'EUCAST, CLSI'), becomes the explicit sentinel
    `OTHER_OR_COMBINED_OR_MISSING` - never guessed at or merged into one
    named standard. Casing is treated as a data-entry variant of the same
    intended value (a documented judgment call, not an empirical fact); a
    combined string is never treated as naming "the" standard used."""
    normalized = normalize_value(raw_value)
    if normalized == MISSING_SENTINEL:
        return OTHER_OR_COMBINED_OR_MISSING
    upper = normalized.strip().upper()
    return upper if upper in RECOGNIZED_SINGLE_STANDARDS else OTHER_OR_COMBINED_OR_MISSING


def classify_genome_standard_status(records: list[dict]) -> str:
    categories = {normalize_standard_for_eligibility(r.get("testing_standard")) for r in records}
    if len(categories) > 1:
        return STANDARD_INCONSISTENT
    only_category = next(iter(categories))
    return STANDARD_MISSING_OR_COMBINED if only_category == OTHER_OR_COMBINED_OR_MISSING else STANDARD_CONSISTENT


def classify_genome_method_status(records: list[dict]) -> str:
    """A genome's method is 'consistent' only if every record shares the
    same non-missing `laboratory_typing_method` AND, among records that
    have a populated `measurement_unit`, that unit also agrees - different
    measurement types (e.g. an MIC in mg/L vs. a disk-diffusion zone in mm)
    are never treated as interchangeable just because the method label
    happens to match."""
    methods = {normalize_value(r.get("laboratory_typing_method")) for r in records}
    if methods == {MISSING_SENTINEL}:
        return METHOD_MISSING
    if len(methods) > 1:
        return METHOD_INCONSISTENT
    units = {normalize_value(r.get("measurement_unit")) for r in records if r.get("measurement_unit")}
    if len(units) > 1:
        return METHOD_INCONSISTENT
    return METHOD_CONSISTENT


@dataclass
class MetadataCharacterization:
    antibiotic: str
    n_records: int
    n_genomes: int
    testing_standard_counts: Counter = field(default_factory=Counter)
    testing_standard_year_counts: Counter = field(default_factory=Counter)
    laboratory_typing_method_counts: Counter = field(default_factory=Counter)
    laboratory_typing_platform_counts: Counter = field(default_factory=Counter)
    measurement_unit_counts: Counter = field(default_factory=Counter)
    # genomes classified as "repeated_concordant" by classify_genome_multiplicity
    # (categorically agreeing, >1 record) whose records differ in a given field
    repeated_concordant_genomes: int = 0
    repeated_concordant_heterogeneity: dict[str, int] = field(default_factory=dict)


def characterize_metadata(records: list[dict], antibiotic: str) -> MetadataCharacterization:
    char = MetadataCharacterization(
        antibiotic=antibiotic, n_records=len(records), n_genomes=len(group_by_genome(records))
    )
    for field_name, counter_attr in [
        ("testing_standard", "testing_standard_counts"),
        ("testing_standard_year", "testing_standard_year_counts"),
        ("laboratory_typing_method", "laboratory_typing_method_counts"),
        ("laboratory_typing_platform", "laboratory_typing_platform_counts"),
        ("measurement_unit", "measurement_unit_counts"),
    ]:
        counter = Counter(normalize_value(r.get(field_name)) for r in records)
        setattr(char, counter_attr, counter)

    groups = group_by_genome(records)
    repeated_concordant_ids = [
        gid for gid, recs in groups.items() if classify_genome_multiplicity(recs) == "repeated_concordant"
    ]
    char.repeated_concordant_genomes = len(repeated_concordant_ids)
    for metadata_field in ["testing_standard", "testing_standard_year", "laboratory_typing_method", "laboratory_typing_platform", "measurement_unit"]:
        char.repeated_concordant_heterogeneity[metadata_field] = sum(
            1 for gid in repeated_concordant_ids if field_differs_within_genome(groups[gid], metadata_field)
        )
    return char


@dataclass
class PolicyOutcome:
    policy_name: str
    antibiotic: str
    n_total_genomes: int
    genome_outcome: dict[str, str] = field(default_factory=dict)
    genome_class: dict[str, str] = field(default_factory=dict)
    eligible_with_repeated_records: int = 0
    eligible_with_heterogeneous_field: dict[str, int] = field(default_factory=dict)
    eligible_with_mlst: int = 0
    eligible_distinct_mlst_groups: int = 0

    def outcome_counts(self) -> Counter:
        return Counter(self.genome_outcome.values())

    def eligible_genome_ids(self) -> list[str]:
        return list(self.genome_class)

    def class_counts(self) -> Counter:
        return Counter(self.genome_class.values())


def _transparency_stats(
    outcome: PolicyOutcome, groups: dict[str, list[dict]], mlst_by_genome: dict[str, str | None]
) -> None:
    eligible_ids = outcome.eligible_genome_ids()
    repeated_ids = [gid for gid in eligible_ids if len(groups[gid]) > 1]
    outcome.eligible_with_repeated_records = len(repeated_ids)
    for field_name in ["testing_standard", "testing_standard_year", "laboratory_typing_method", "laboratory_typing_platform", "measurement_unit"]:
        outcome.eligible_with_heterogeneous_field[field_name] = sum(
            1 for gid in repeated_ids if field_differs_within_genome(groups[gid], field_name)
        )
    mlst_values = {gid: mlst_by_genome.get(gid) for gid in eligible_ids}
    present = {gid: v for gid, v in mlst_values.items() if v is not None and str(v).strip() != ""}
    outcome.eligible_with_mlst = len(present)
    outcome.eligible_distinct_mlst_groups = len(set(present.values()))


def apply_policy_a(records: list[dict], antibiotic: str, mlst_by_genome: dict[str, str | None]) -> PolicyOutcome:
    groups = group_by_genome(records)
    policy1 = apply_policy_1(groups)
    outcome = PolicyOutcome(
        policy_name="A_categorical_agreement",
        antibiotic=antibiotic,
        n_total_genomes=len(groups),
        genome_outcome=dict(policy1.genome_outcome),
        genome_class=dict(policy1.genome_class),
    )
    _transparency_stats(outcome, groups, mlst_by_genome)
    return outcome


def apply_policy_b(records: list[dict], antibiotic: str, mlst_by_genome: dict[str, str | None]) -> PolicyOutcome:
    groups = group_by_genome(records)
    policy1 = apply_policy_1(groups)
    outcome = PolicyOutcome(
        policy_name="B_categorical_agreement_plus_standard_consistency",
        antibiotic=antibiotic,
        n_total_genomes=len(groups),
    )
    for genome_id, genome_records in groups.items():
        base_outcome = policy1.genome_outcome[genome_id]
        if base_outcome not in ("eligible_resistant", "eligible_susceptible"):
            outcome.genome_outcome[genome_id] = f"excluded_by_policy_a:{base_outcome}"
            continue
        standard_status = classify_genome_standard_status(genome_records)
        if standard_status == STANDARD_CONSISTENT:
            outcome.genome_outcome[genome_id] = base_outcome
            outcome.genome_class[genome_id] = policy1.genome_class[genome_id]
        else:
            outcome.genome_outcome[genome_id] = f"excluded_standard_{standard_status}"
    _transparency_stats(outcome, groups, mlst_by_genome)
    return outcome


def apply_policy_c(records: list[dict], antibiotic: str, mlst_by_genome: dict[str, str | None]) -> PolicyOutcome:
    groups = group_by_genome(records)
    policy1 = apply_policy_1(groups)
    outcome = PolicyOutcome(
        policy_name="C_categorical_agreement_plus_method_consistency",
        antibiotic=antibiotic,
        n_total_genomes=len(groups),
    )
    for genome_id, genome_records in groups.items():
        base_outcome = policy1.genome_outcome[genome_id]
        if base_outcome not in ("eligible_resistant", "eligible_susceptible"):
            outcome.genome_outcome[genome_id] = f"excluded_by_policy_a:{base_outcome}"
            continue
        method_status = classify_genome_method_status(genome_records)
        if method_status == METHOD_CONSISTENT:
            outcome.genome_outcome[genome_id] = base_outcome
            outcome.genome_class[genome_id] = policy1.genome_class[genome_id]
        else:
            outcome.genome_outcome[genome_id] = f"excluded_method_{method_status}"
    _transparency_stats(outcome, groups, mlst_by_genome)
    return outcome


POLICY_FUNCS = {"A": apply_policy_a, "B": apply_policy_b, "C": apply_policy_c}


def run_all_policies(
    records: list[dict], antibiotic: str, mlst_by_genome: dict[str, str | None]
) -> dict[str, PolicyOutcome]:
    return {name: func(records, antibiotic, mlst_by_genome) for name, func in POLICY_FUNCS.items()}
