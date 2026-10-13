"""Selects additional, bounded ciprofloxacin genomes under Policy A
(the officially recommended cross-antibiotic eligibility rule -
`final_policy_analysis.apply_policy_a`, see
`reports/final_phenotype_policy_analysis.md`) for an incremental,
real-data cohort expansion beyond the existing 100-genome pilot.

Explicitly distinct from `pilot_cohort.py`: the ORIGINAL 100-genome
pilot was selected under the earlier, stricter "Policy 2" rule
(`conflict_analysis.apply_policy_2` - excludes a genome on ANY
Intermediate co-occurrence), before Policy A (`apply_policy_1` under a
new name - categorical R/S agreement alone, tolerating a co-occurring
Intermediate call) was adopted as the final cross-antibiotic
recommendation. This expansion deliberately uses Policy A, per this
task's explicit instruction - the combined ~200-genome pool therefore
spans two different, both-documented eligibility policies, not one
consistent rule. This is stated here, not hidden; the original 100 are
never re-selected, re-labeled, or re-checked under Policy A.

Selection is deterministic (seeded, sorted pools before sampling) and
NEVER includes a genome already in the pilot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .final_policy_analysis import apply_policy_a


@dataclass
class CohortExpansionSelection:
    antibiotic: str
    seed: int
    target_additional: int
    selected_genome_ids: list[str] = field(default_factory=list)
    selected_labels: dict[str, str] = field(default_factory=dict)
    eligible_pool_size_after_pilot_exclusion: int = 0
    eligible_resistant_pool_size: int = 0
    eligible_susceptible_pool_size: int = 0
    n_selected_resistant: int = 0
    n_selected_susceptible: int = 0
    limitations: list[str] = field(default_factory=list)


def select_expansion_cohort(
    records: list[dict],
    mlst_by_genome: dict[str, str | None],
    antibiotic: str,
    pilot_genome_ids: set[str],
    target_additional: int = 100,
    seed: int = 20261009,
) -> CohortExpansionSelection:
    """`pilot_genome_ids` must be the EXACT string genome IDs already in
    the 100-genome pilot - every one of them is excluded from the
    candidate pool before sampling, regardless of its Policy A outcome."""
    outcome = apply_policy_a(records, antibiotic, mlst_by_genome)
    eligible_ids = set(outcome.eligible_genome_ids())
    candidate_ids = eligible_ids - set(pilot_genome_ids)

    selection = CohortExpansionSelection(antibiotic=antibiotic, seed=seed, target_additional=target_additional)

    resistant_pool = sorted(gid for gid in candidate_ids if outcome.genome_class[gid] == "Resistant")
    susceptible_pool = sorted(gid for gid in candidate_ids if outcome.genome_class[gid] == "Susceptible")
    selection.eligible_resistant_pool_size = len(resistant_pool)
    selection.eligible_susceptible_pool_size = len(susceptible_pool)
    selection.eligible_pool_size_after_pilot_exclusion = len(candidate_ids)

    if not candidate_ids:
        selection.limitations.append(
            "BLOCKER: zero Policy-A-eligible genomes remain after excluding the pilot's genomes."
        )
        return selection

    effective_target = min(target_additional, len(candidate_ids))
    if effective_target < target_additional:
        selection.limitations.append(
            f"Only {len(candidate_ids)} Policy-A-eligible genomes remain outside the pilot - "
            f"using all of them instead of the requested {target_additional}."
        )

    total = len(candidate_ids)
    resistant_fraction = len(resistant_pool) / total
    n_resistant = round(effective_target * resistant_fraction)
    n_susceptible = effective_target - n_resistant
    if n_resistant == 0 and resistant_pool:
        n_resistant, n_susceptible = 1, max(effective_target - 1, 0)
    if n_susceptible == 0 and susceptible_pool:
        n_susceptible, n_resistant = 1, max(effective_target - 1, 0)
    n_resistant = min(n_resistant, len(resistant_pool))
    n_susceptible = min(n_susceptible, len(susceptible_pool))

    # Deterministic: pools are sorted (above) before a seeded sample is
    # drawn, so the result depends only on pool MEMBERSHIP and the seed,
    # never on input dict/record ordering.
    sampled_resistant = (
        pd.Series(resistant_pool).sample(n=n_resistant, random_state=seed).tolist() if n_resistant else []
    )
    sampled_susceptible = (
        pd.Series(susceptible_pool).sample(n=n_susceptible, random_state=seed).tolist() if n_susceptible else []
    )

    selection.selected_genome_ids = sorted(sampled_resistant + sampled_susceptible)
    for gid in selection.selected_genome_ids:
        selection.selected_labels[gid] = outcome.genome_class[gid]
    selection.n_selected_resistant = n_resistant
    selection.n_selected_susceptible = n_susceptible

    selection.limitations.append(
        "This expansion uses Policy A (the officially recommended cross-antibiotic rule), NOT the "
        "stricter Policy 2 rule used to select the original 100-genome pilot - the combined pool "
        "spans two different eligibility policies; the original pilot genomes are not retroactively "
        "re-checked or re-labeled under Policy A."
    )
    return selection
