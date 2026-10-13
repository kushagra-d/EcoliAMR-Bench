import pandas as pd
import pytest

from bvbrc_recon.cohort_expansion import select_expansion_cohort


def _make_records(n_resistant=20, n_susceptible=60, antibiotic="ciprofloxacin", offset=0):
    records = []
    for i in range(n_resistant):
        gid = f"genome_r{i + offset}"
        records.append({"genome_id": gid, "antibiotic": antibiotic, "resistant_phenotype": "Resistant"})
    for i in range(n_susceptible):
        gid = f"genome_s{i + offset}"
        records.append({"genome_id": gid, "antibiotic": antibiotic, "resistant_phenotype": "Susceptible"})
    return records


# --- deterministic cohort selection ---


def test_selection_is_deterministic_given_same_seed():
    records = _make_records()
    mlst = {}
    sel1 = select_expansion_cohort(records, mlst, "ciprofloxacin", pilot_genome_ids=set(), target_additional=10, seed=42)
    sel2 = select_expansion_cohort(records, mlst, "ciprofloxacin", pilot_genome_ids=set(), target_additional=10, seed=42)
    assert sel1.selected_genome_ids == sel2.selected_genome_ids
    assert sel1.selected_labels == sel2.selected_labels


def test_different_seeds_can_give_different_selections():
    records = _make_records(n_resistant=20, n_susceptible=60)
    mlst = {}
    sel1 = select_expansion_cohort(records, mlst, "ciprofloxacin", pilot_genome_ids=set(), target_additional=10, seed=1)
    sel2 = select_expansion_cohort(records, mlst, "ciprofloxacin", pilot_genome_ids=set(), target_additional=10, seed=2)
    assert sel1.selected_genome_ids != sel2.selected_genome_ids  # astronomically unlikely to collide


# --- pilot exclusion ---


def test_pilot_genomes_are_never_selected():
    records = _make_records(n_resistant=10, n_susceptible=10)
    mlst = {}
    pilot_ids = {f"genome_r{i}" for i in range(8)} | {f"genome_s{i}" for i in range(8)}
    sel = select_expansion_cohort(records, mlst, "ciprofloxacin", pilot_genome_ids=pilot_ids, target_additional=5, seed=1)
    assert set(sel.selected_genome_ids).isdisjoint(pilot_ids)


def test_pilot_exclusion_shrinks_available_pool_correctly():
    records = _make_records(n_resistant=5, n_susceptible=5)
    mlst = {}
    all_ids = {f"genome_r{i}" for i in range(5)} | {f"genome_s{i}" for i in range(5)}
    sel = select_expansion_cohort(records, mlst, "ciprofloxacin", pilot_genome_ids=all_ids, target_additional=5, seed=1)
    assert sel.eligible_pool_size_after_pilot_exclusion == 0
    assert any("BLOCKER" in w for w in sel.limitations)


# --- genome-ID string preservation ---


def test_genome_ids_remain_strings_not_coerced_to_numeric():
    records = _make_records(n_resistant=5, n_susceptible=5)
    # use numeric-looking IDs, like the real "562.xxxxx" format
    for i, r in enumerate(records):
        r["genome_id"] = f"562.{100000 + i}"
    mlst = {}
    sel = select_expansion_cohort(records, mlst, "ciprofloxacin", pilot_genome_ids=set(), target_additional=5, seed=1)
    for gid in sel.selected_genome_ids:
        assert isinstance(gid, str)
        assert gid.startswith("562.")


# --- class balance / proportional sampling ---


def test_selection_respects_pool_size_when_smaller_than_target():
    records = _make_records(n_resistant=2, n_susceptible=3)
    mlst = {}
    sel = select_expansion_cohort(records, mlst, "ciprofloxacin", pilot_genome_ids=set(), target_additional=100, seed=1)
    assert len(sel.selected_genome_ids) == 5
    assert any("Only 5" in w for w in sel.limitations)


def test_both_classes_present_when_both_pools_nonempty():
    records = _make_records(n_resistant=1, n_susceptible=50)
    mlst = {}
    sel = select_expansion_cohort(records, mlst, "ciprofloxacin", pilot_genome_ids=set(), target_additional=10, seed=1)
    assert sel.n_selected_resistant >= 1
    assert sel.n_selected_susceptible >= 1


# --- policy-difference documentation (mechanical check, not just prose) ---


def test_limitations_mention_policy_a_vs_policy_2_distinction():
    records = _make_records(n_resistant=5, n_susceptible=5)
    mlst = {}
    sel = select_expansion_cohort(records, mlst, "ciprofloxacin", pilot_genome_ids=set(), target_additional=5, seed=1)
    assert any("Policy A" in w and "Policy 2" in w for w in sel.limitations)
