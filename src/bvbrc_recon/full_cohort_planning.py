"""Builds per-antibiotic Policy A cohort manifests, cross-antibiotic
overlap analysis, assembly-availability assessment, and a deterministic
batched retrieval plan - all PLANNING ONLY, no network access, no
genome downloads, no model training.

Reuses `final_policy_analysis.apply_policy_a` verbatim - the Policy A
eligibility rule is never reimplemented here. A genome's label is
always kept PER ANTIBIOTIC (a dict keyed by (genome_id, antibiotic),
never collapsed into one cross-antibiotic label) - the same genome can
be Resistant for one antibiotic and Susceptible for another, and this
module never assumes otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .final_policy_analysis import PolicyOutcome, apply_policy_a


def build_policy_a_manifest(records: list[dict], antibiotic: str, mlst_by_genome: dict[str, str | None]) -> pd.DataFrame:
    """One row per genome recognized for this antibiotic's metadata -
    eligible genomes get their Policy A label; excluded genomes get an
    explicit, machine-readable exclusion reason (Policy A's own outcome
    string, e.g. 'excluded_both_classes_present')."""
    outcome = apply_policy_a(records, antibiotic, mlst_by_genome)
    rows = []
    for genome_id, policy_outcome in outcome.genome_outcome.items():
        eligible = genome_id in outcome.genome_class
        rows.append({
            "genome_id": genome_id,
            "antibiotic": antibiotic,
            "eligible": eligible,
            "label": outcome.genome_class.get(genome_id),
            "exclusion_reason": None if eligible else policy_outcome,
        })
    df = pd.DataFrame(rows)
    duplicate_count = df["genome_id"].duplicated().sum()
    if duplicate_count:
        raise ValueError(
            f"INTERNAL INVARIANT VIOLATED: {duplicate_count} duplicate genome_id rows in the "
            f"{antibiotic} Policy A manifest - group_by_genome should make this impossible; "
            f"refusing to silently proceed with a corrupted manifest."
        )
    return df


@dataclass
class OverlapResult:
    antibiotics: list[str]
    eligible_counts: dict[str, int] = field(default_factory=dict)
    pairwise_overlap: dict[str, int] = field(default_factory=dict)  # "cipro&amp" -> count
    three_way_overlap: int = 0
    total_antibiotic_specific_records: int = 0
    n_unique_genomes: int = 0
    per_genome_antibiotics: dict[str, list[str]] = field(default_factory=dict)  # genome_id -> [antibiotics it's eligible for]
    per_genome_labels: dict[str, dict[str, str]] = field(default_factory=dict)  # genome_id -> {antibiotic: label}


def compute_cross_antibiotic_overlap(eligible_ids_by_antibiotic: dict[str, set[str]], labels_by_antibiotic: dict[str, dict[str, str]]) -> OverlapResult:
    antibiotics = sorted(eligible_ids_by_antibiotic)
    result = OverlapResult(antibiotics=antibiotics)
    for ab in antibiotics:
        result.eligible_counts[ab] = len(eligible_ids_by_antibiotic[ab])
    result.total_antibiotic_specific_records = sum(result.eligible_counts.values())

    for i, a in enumerate(antibiotics):
        for b in antibiotics[i + 1:]:
            key = f"{a}&{b}"
            result.pairwise_overlap[key] = len(eligible_ids_by_antibiotic[a] & eligible_ids_by_antibiotic[b])

    if len(antibiotics) >= 3:
        common = eligible_ids_by_antibiotic[antibiotics[0]]
        for ab in antibiotics[1:]:
            common = common & eligible_ids_by_antibiotic[ab]
        result.three_way_overlap = len(common)

    all_genome_ids: set[str] = set()
    for ab in antibiotics:
        all_genome_ids |= eligible_ids_by_antibiotic[ab]
    result.n_unique_genomes = len(all_genome_ids)

    for gid in all_genome_ids:
        member_of = [ab for ab in antibiotics if gid in eligible_ids_by_antibiotic[ab]]
        result.per_genome_antibiotics[gid] = member_of
        result.per_genome_labels[gid] = {ab: labels_by_antibiotic[ab][gid] for ab in member_of}

    return result


@dataclass
class AssemblyAvailability:
    antibiotic: str
    n_eligible: int = 0
    already_validated: list[str] = field(default_factory=list)
    previously_failed: list[str] = field(default_factory=list)
    missing_never_attempted: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)  # present in a manifest with an ambiguous/unexpected status


COMPLETE_STATUSES = {"complete", "already_present_skipped"}


def assess_assembly_availability(
    eligible_genome_ids: set[str],
    raw_fasta_dirs: list[Path],
    retrieval_manifest_paths: list[Path],
) -> AssemblyAvailability:
    """Compares the eligible pool against on-disk FASTA files (any of
    `raw_fasta_dirs`) and any existing retrieval manifests (any of
    `retrieval_manifest_paths` - genome_id/status columns expected,
    same convention as `sequence_retrieval.py`'s status constants)."""
    on_disk_ids: set[str] = set()
    for d in raw_fasta_dirs:
        if d.exists():
            on_disk_ids |= {p.stem for p in d.glob("*.fna")}

    status_by_genome: dict[str, str] = {}
    for p in retrieval_manifest_paths:
        if not p.exists():
            continue
        df = pd.read_csv(p, dtype={"genome_id": str})
        for _, row in df.iterrows():
            # a genome could appear in more than one manifest (e.g. pilot + expansion
            # are disjoint by construction, but this stays correct even if not) -
            # a later COMPLETE status always wins over an earlier failure record.
            gid, status = row["genome_id"], row["status"]
            if gid not in status_by_genome or status in COMPLETE_STATUSES:
                status_by_genome[gid] = status

    result = AssemblyAvailability(antibiotic="", n_eligible=len(eligible_genome_ids))
    for gid in sorted(eligible_genome_ids):
        has_file = gid in on_disk_ids
        status = status_by_genome.get(gid)
        status_is_complete = status is not None and status in COMPLETE_STATUSES
        status_is_failure = status is not None and status not in COMPLETE_STATUSES

        if has_file and status_is_failure:
            # file on disk but its OWN manifest says incomplete - a genuine
            # inconsistency, checked FIRST so it is never silently absorbed
            # into either "validated" or "failed".
            result.unresolved.append(gid)
        elif has_file:
            # has_file and (status_is_complete or status is None)
            result.already_validated.append(gid)
        elif status_is_failure:
            result.previously_failed.append(gid)
        else:
            result.missing_never_attempted.append(gid)
    return result


def plan_retrieval_batches(
    needed_genome_antibiotics: dict[str, list[str]],
    batch_size: int = 100,
    seed: int = 20261009,
    max_retries: int = 3,
) -> list[dict]:
    """Deterministic batching: genome IDs sorted (not input-order-
    dependent), chunked into fixed-size batches. Each genome appears in
    EXACTLY ONE planned download job regardless of how many antibiotics
    need it - `antibiotics_needed` records all of them on that one job,
    so nothing is fetched twice for a shared genome."""
    sorted_ids = sorted(needed_genome_antibiotics)
    rows = []
    for batch_idx in range(0, len(sorted_ids), batch_size):
        batch_ids = sorted_ids[batch_idx : batch_idx + batch_size]
        batch_number = batch_idx // batch_size
        for gid in batch_ids:
            rows.append({
                "batch_id": batch_number,
                "genome_id": gid,
                "antibiotics_needed": ";".join(sorted(needed_genome_antibiotics[gid])),
                "status": "planned",
                "retry_count": 0,
                "max_retries": max_retries,
                "last_error": "",
            })
    return rows
