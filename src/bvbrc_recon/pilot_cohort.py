"""Selects a small, reproducible pilot cohort for one antibiotic.

Label-curation rule (documented, not invented here): reuses
``conflict_analysis.apply_policy_2`` ("require a single binary
interpretation") verbatim as the genome-eligibility rule - a genome is
eligible only if ALL of its recognized AST records for this antibiotic
are exactly Resistant, or exactly Susceptible. Any Intermediate
co-occurrence or any Resistant/Susceptible disagreement excludes the
genome; this is the policy already defined, tested, and reported on in
``reports/phenotype_conflict_analysis.md`` - reused here rather than
reimplemented, per Policy 2's own exclusion-reason labels.

This module never resolves a conflict (by majority vote, recency, or
otherwise) and never invents a label: every genome not meeting Policy 2
is excluded, with its Policy-2 reason preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .conflict_analysis import apply_policy_1, apply_policy_2, group_by_genome


@dataclass
class PilotCohort:
    antibiotic: str
    seed: int
    target_size: int
    selected_genome_ids: list[str] = field(default_factory=list)
    selected_labels: dict[str, str] = field(default_factory=dict)  # genome_id -> "Resistant"/"Susceptible"
    eligible_resistant_pool_size: int = 0
    eligible_susceptible_pool_size: int = 0
    n_selected_resistant: int = 0
    n_selected_susceptible: int = 0
    genome_decision_table: pd.DataFrame = None  # one row per genome in the FULL antibiotic cohort
    limitations: list[str] = field(default_factory=list)


def select_pilot_cohort(
    records: list[dict],
    antibiotic: str,
    target_size: int = 100,
    seed: int = 20261009,
) -> PilotCohort:
    """Builds the full per-genome eligibility/decision table for this
    antibiotic (every genome, included or excluded, with its Policy 2
    reason), then draws a reproducible stratified sample from the
    eligible pool, proportional to the eligible pool's own Resistant:
    Susceptible ratio (not forced to 50/50 - an enforced balance would be
    just as arbitrary an intervention as the disagreement it is trying to
    avoid, and proportional sampling makes no representativeness claim
    beyond "matches the eligible pool's own observed ratio").
    """
    if not all(r.get("antibiotic") == antibiotic for r in records):
        raise ValueError(
            "select_pilot_cohort received records for more than one antibiotic - "
            "refusing to proceed to avoid mislabeling genomes"
        )

    groups = group_by_genome(records)
    policy2 = apply_policy_2(groups)
    policy1 = apply_policy_1(groups)  # computed for the decision table's transparency, not used to select

    decision_rows = []
    for genome_id, genome_records in groups.items():
        decision_rows.append(
            {
                "genome_id": genome_id,
                "n_records": len(genome_records),
                "policy2_outcome": policy2.genome_outcome[genome_id],
                "policy2_class": policy2.genome_class.get(genome_id),
                "policy1_outcome": policy1.genome_outcome[genome_id],
            }
        )
    decision_table = pd.DataFrame(decision_rows)

    eligible_resistant = decision_table[decision_table["policy2_outcome"] == "eligible_resistant"]
    eligible_susceptible = decision_table[decision_table["policy2_outcome"] == "eligible_susceptible"]

    cohort = PilotCohort(antibiotic=antibiotic, seed=seed, target_size=target_size)
    cohort.eligible_resistant_pool_size = len(eligible_resistant)
    cohort.eligible_susceptible_pool_size = len(eligible_susceptible)
    cohort.genome_decision_table = decision_table

    total_eligible = cohort.eligible_resistant_pool_size + cohort.eligible_susceptible_pool_size
    if total_eligible == 0:
        cohort.limitations.append(
            "BLOCKER: zero genomes satisfy the Policy 2 label-curation rule for this antibiotic - "
            "cannot build any pilot cohort without inventing labels."
        )
        return cohort

    effective_target = min(target_size, total_eligible)
    if effective_target < target_size:
        cohort.limitations.append(
            f"Only {total_eligible} genomes satisfy Policy 2 for {antibiotic} - "
            f"using all of them instead of the requested {target_size}."
        )

    resistant_fraction = cohort.eligible_resistant_pool_size / total_eligible
    n_resistant = round(effective_target * resistant_fraction)
    n_susceptible = effective_target - n_resistant

    # guarantee at least one of each class is present if both pools are nonempty,
    # since "include both Resistant and Susceptible genomes" is an explicit requirement
    if n_resistant == 0 and cohort.eligible_resistant_pool_size > 0:
        n_resistant, n_susceptible = 1, max(effective_target - 1, 0)
    if n_susceptible == 0 and cohort.eligible_susceptible_pool_size > 0:
        n_susceptible, n_resistant = 1, max(effective_target - 1, 0)

    n_resistant = min(n_resistant, cohort.eligible_resistant_pool_size)
    n_susceptible = min(n_susceptible, cohort.eligible_susceptible_pool_size)

    sampled_resistant = eligible_resistant.sample(n=n_resistant, random_state=seed)
    sampled_susceptible = eligible_susceptible.sample(n=n_susceptible, random_state=seed)

    for _, row in pd.concat([sampled_resistant, sampled_susceptible]).iterrows():
        cohort.selected_genome_ids.append(row["genome_id"])
        cohort.selected_labels[row["genome_id"]] = row["policy2_class"]

    cohort.n_selected_resistant = n_resistant
    cohort.n_selected_susceptible = n_susceptible

    if resistant_fraction < 0.5:
        cohort.limitations.append(
            f"Sampling is proportional to the eligible pool's natural class ratio "
            f"({cohort.eligible_resistant_pool_size} Resistant : {cohort.eligible_susceptible_pool_size} "
            f"Susceptible), not rebalanced - the pilot is therefore imbalanced "
            f"({n_resistant} Resistant : {n_susceptible} Susceptible) by design, reflecting the "
            f"underlying data rather than an invented balance."
        )
    cohort.limitations.append(
        "This pilot cohort is a small, non-lineage-aware random sample from genomes passing a strict "
        "single-interpretation label rule; it is NOT claimed to be representative of the full "
        "candidate dataset, and genomes were not selected based on ease of sequence download."
    )
    return cohort
