"""Reconciles the original 100-genome pilot (selected under Policy 2 -
`conflict_analysis.apply_policy_2`) and the 100-genome expansion
(selected under Policy A - `final_policy_analysis.apply_policy_a`) into
a SINGLE Policy-A-consistent validation cohort.

This module ONLY re-evaluates the 200 genomes already retrieved by the
two prior tasks against Policy A - it never selects a genome outside
that union of 200, and never retrieves anything. If more than 200 of
them are Policy-A-eligible, this can't happen here by construction
(there are only 200 candidates total); if fewer than 200 are eligible,
the reconciled cohort is simply smaller - reported honestly, not padded.

Reuses `final_policy_analysis.apply_policy_a` verbatim - the eligibility
rule itself is never reimplemented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .final_policy_analysis import apply_policy_a


@dataclass
class ReconciliationEntry:
    genome_id: str
    original_cohort: str  # "pilot" or "expansion"
    original_policy: str  # "policy_2" or "policy_a"
    original_label: str
    policy_a_eligible: bool
    policy_a_label: str | None
    policy_a_outcome: str
    label_matches_original: bool | None  # None if not eligible (no Policy A label to compare)
    included: bool = False
    exclusion_reason: str | None = None


@dataclass
class ReconciliationResult:
    antibiotic: str
    seed: int
    target_size: int
    entries: list[ReconciliationEntry] = field(default_factory=list)
    n_pilot_eligible: int = 0
    n_pilot_ineligible: int = 0
    n_expansion_eligible: int = 0
    n_expansion_ineligible: int = 0
    n_label_mismatches: int = 0
    n_duplicate_genome_ids: int = 0
    included_genome_ids: list[str] = field(default_factory=list)
    included_labels: dict[str, str] = field(default_factory=dict)
    n_included_resistant: int = 0
    n_included_susceptible: int = 0
    limitations: list[str] = field(default_factory=list)


def reconcile_cohort(
    records: list[dict],
    mlst_by_genome: dict[str, str | None],
    antibiotic: str,
    pilot_genome_ids: list[str],
    pilot_labels: dict[str, str],
    expansion_genome_ids: list[str],
    expansion_labels: dict[str, str],
    target_size: int = 200,
    seed: int = 20261009,
) -> ReconciliationResult:
    outcome = apply_policy_a(records, antibiotic, mlst_by_genome)
    result = ReconciliationResult(antibiotic=antibiotic, seed=seed, target_size=target_size)

    seen_ids: set[str] = set()
    for gid_list, cohort_name, policy_name, label_map in [
        (pilot_genome_ids, "pilot", "policy_2", pilot_labels),
        (expansion_genome_ids, "expansion", "policy_a", expansion_labels),
    ]:
        for gid in gid_list:
            if gid in seen_ids:
                result.n_duplicate_genome_ids += 1
                continue
            seen_ids.add(gid)

            eligible = gid in outcome.genome_class
            policy_a_label = outcome.genome_class.get(gid)
            original_label = label_map[gid]
            label_matches = (policy_a_label == original_label) if eligible else None
            if eligible and not label_matches:
                result.n_label_mismatches += 1

            entry = ReconciliationEntry(
                genome_id=gid,
                original_cohort=cohort_name,
                original_policy=policy_name,
                original_label=original_label,
                policy_a_eligible=eligible,
                policy_a_label=policy_a_label,
                policy_a_outcome=outcome.genome_outcome.get(gid, "genome_id_not_found_in_metadata"),
                label_matches_original=label_matches,
            )
            if not eligible:
                entry.exclusion_reason = f"fails Policy A: {entry.policy_a_outcome}"
            result.entries.append(entry)

            if cohort_name == "pilot":
                if eligible:
                    result.n_pilot_eligible += 1
                else:
                    result.n_pilot_ineligible += 1
            else:
                if eligible:
                    result.n_expansion_eligible += 1
                else:
                    result.n_expansion_ineligible += 1

    eligible_entries = [e for e in result.entries if e.policy_a_eligible]
    total_eligible = len(eligible_entries)

    if total_eligible < target_size:
        result.limitations.append(
            f"Only {total_eligible} of the 200 already-retrieved genomes satisfy Policy A - "
            f"retaining the largest valid cohort ({total_eligible}) instead of {target_size}. "
            f"No replacement genomes were retrieved, per this task's explicit scope."
        )

    # deterministic: sort eligible entries by genome_id before including -
    # at most `target_size` here is a no-op cap since the candidate pool
    # (200 already-retrieved genomes) can never exceed target_size=200 by
    # construction, but the explicit sort still makes "which ones" fully
    # reproducible and independent of input ordering.
    eligible_entries_sorted = sorted(eligible_entries, key=lambda e: e.genome_id)
    included = eligible_entries_sorted[:target_size]

    for entry in included:
        entry.included = True
        result.included_genome_ids.append(entry.genome_id)
        result.included_labels[entry.genome_id] = entry.policy_a_label
        if entry.policy_a_label == "Resistant":
            result.n_included_resistant += 1
        else:
            result.n_included_susceptible += 1

    for entry in eligible_entries_sorted[target_size:]:
        entry.exclusion_reason = "Policy A eligible but beyond target_size cap"

    result.included_genome_ids.sort()  # final, fully deterministic order
    return result
