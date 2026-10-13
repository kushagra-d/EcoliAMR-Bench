from bvbrc_recon.cohort_reconciliation import reconcile_cohort


def _make_records(label_by_genome, antibiotic="ciprofloxacin"):
    """One record per genome with the given SINGLE categorical label -
    i.e. unanimously that label, satisfying both Policy 2 and Policy A."""
    return [
        {"genome_id": gid, "antibiotic": antibiotic, "resistant_phenotype": label}
        for gid, label in label_by_genome.items()
    ]


def _make_mixed_record_genome(genome_id, labels, antibiotic="ciprofloxacin"):
    """Multiple records for one genome, with possibly different
    categorical labels - used to construct Policy-A-eligible-but-not-
    Policy-2 and Policy-A-ineligible fixtures."""
    return [{"genome_id": genome_id, "antibiotic": antibiotic, "resistant_phenotype": label} for label in labels]


# --- Policy A reconciliation + exclusion-reason completeness ---


def test_pure_resistant_and_susceptible_genomes_are_eligible_and_included():
    records = _make_records({"g1": "Resistant", "g2": "Susceptible"})
    result = reconcile_cohort(
        records, {}, "ciprofloxacin",
        pilot_genome_ids=["g1"], pilot_labels={"g1": "Resistant"},
        expansion_genome_ids=["g2"], expansion_labels={"g2": "Susceptible"},
        target_size=200, seed=1,
    )
    assert set(result.included_genome_ids) == {"g1", "g2"}
    assert result.n_pilot_eligible == 1
    assert result.n_expansion_eligible == 1


def test_genome_with_intermediate_plus_resistant_is_policy_a_eligible_not_policy_2():
    """Exactly the case Policy A tolerates and Policy 2 does not -
    Intermediate co-occurring with a single-sided R/S call."""
    records = (
        _make_mixed_record_genome("g1", ["Resistant", "Intermediate"])
        + _make_records({"g2": "Susceptible"})
    )
    result = reconcile_cohort(
        records, {}, "ciprofloxacin",
        pilot_genome_ids=["g1"], pilot_labels={"g1": "Resistant"},  # was included under policy 2 elsewhere, hypothetically
        expansion_genome_ids=["g2"], expansion_labels={"g2": "Susceptible"},
        target_size=200, seed=1,
    )
    entry_g1 = next(e for e in result.entries if e.genome_id == "g1")
    assert entry_g1.policy_a_eligible is True
    assert entry_g1.policy_a_label == "Resistant"


def test_conflicting_genome_is_excluded_with_explicit_reason():
    records = _make_mixed_record_genome("g1", ["Resistant", "Susceptible"])
    result = reconcile_cohort(
        records, {}, "ciprofloxacin",
        pilot_genome_ids=["g1"], pilot_labels={"g1": "Resistant"},
        expansion_genome_ids=[], expansion_labels={},
        target_size=200, seed=1,
    )
    entry = result.entries[0]
    assert entry.policy_a_eligible is False
    assert entry.included is False
    assert entry.exclusion_reason is not None and "fails Policy A" in entry.exclusion_reason


def test_genome_missing_from_metadata_entirely_is_excluded_with_reason():
    records = _make_records({"g_known": "Resistant"})
    result = reconcile_cohort(
        records, {}, "ciprofloxacin",
        pilot_genome_ids=["g_known", "g_ghost"], pilot_labels={"g_known": "Resistant", "g_ghost": "Susceptible"},
        expansion_genome_ids=[], expansion_labels={},
        target_size=200, seed=1,
    )
    ghost_entry = next(e for e in result.entries if e.genome_id == "g_ghost")
    assert ghost_entry.policy_a_eligible is False
    assert ghost_entry.exclusion_reason is not None
    assert "genome_id_not_found_in_metadata" in ghost_entry.policy_a_outcome


# --- deterministic cohort selection ---


def test_reconciliation_is_deterministic_given_same_inputs():
    records = _make_records({f"g{i}": "Resistant" if i % 3 == 0 else "Susceptible" for i in range(50)})
    pilot_ids = [f"g{i}" for i in range(25)]
    exp_ids = [f"g{i}" for i in range(25, 50)]
    pilot_labels = {gid: ("Resistant" if int(gid[1:]) % 3 == 0 else "Susceptible") for gid in pilot_ids}
    exp_labels = {gid: ("Resistant" if int(gid[1:]) % 3 == 0 else "Susceptible") for gid in exp_ids}

    r1 = reconcile_cohort(records, {}, "ciprofloxacin", pilot_ids, pilot_labels, exp_ids, exp_labels, target_size=200, seed=1)
    r2 = reconcile_cohort(records, {}, "ciprofloxacin", pilot_ids, pilot_labels, exp_ids, exp_labels, target_size=200, seed=1)
    assert r1.included_genome_ids == r2.included_genome_ids


def test_target_size_cap_is_applied_deterministically():
    records = _make_records({f"g{i}": "Resistant" if i % 2 == 0 else "Susceptible" for i in range(10)})
    result = reconcile_cohort(
        records, {}, "ciprofloxacin",
        pilot_genome_ids=[f"g{i}" for i in range(5)],
        pilot_labels={f"g{i}": ("Resistant" if i % 2 == 0 else "Susceptible") for i in range(5)},
        expansion_genome_ids=[f"g{i}" for i in range(5, 10)],
        expansion_labels={f"g{i}": ("Resistant" if i % 2 == 0 else "Susceptible") for i in range(5, 10)},
        target_size=6, seed=1,
    )
    assert len(result.included_genome_ids) == 6
    assert result.included_genome_ids == sorted(result.included_genome_ids)


# --- fewer-than-target handling ---


def test_fewer_than_target_retains_largest_valid_cohort_and_reports_it():
    records = _make_mixed_record_genome("g1", ["Resistant", "Susceptible"]) + _make_records({"g2": "Resistant"})
    result = reconcile_cohort(
        records, {}, "ciprofloxacin",
        pilot_genome_ids=["g1", "g2"], pilot_labels={"g1": "Resistant", "g2": "Resistant"},
        expansion_genome_ids=[], expansion_labels={},
        target_size=200, seed=1,
    )
    assert len(result.included_genome_ids) == 1
    assert result.included_genome_ids == ["g2"]
    assert any("Only 1" in w for w in result.limitations)


# --- duplicate prevention ---


def test_duplicate_genome_id_across_pilot_and_expansion_is_counted_and_not_double_included():
    records = _make_records({"g1": "Resistant"})
    result = reconcile_cohort(
        records, {}, "ciprofloxacin",
        pilot_genome_ids=["g1"], pilot_labels={"g1": "Resistant"},
        expansion_genome_ids=["g1"], expansion_labels={"g1": "Resistant"},  # accidental duplicate
        target_size=200, seed=1,
    )
    assert result.n_duplicate_genome_ids == 1
    assert result.included_genome_ids.count("g1") == 1


# --- label and genome-ID integrity ---


def test_genome_ids_remain_strings():
    records = _make_records({"562.100011": "Resistant"})
    result = reconcile_cohort(
        records, {}, "ciprofloxacin",
        pilot_genome_ids=["562.100011"], pilot_labels={"562.100011": "Resistant"},
        expansion_genome_ids=[], expansion_labels={},
        target_size=200, seed=1,
    )
    for gid in result.included_genome_ids:
        assert isinstance(gid, str)


def test_label_mismatch_between_original_and_policy_a_is_detected():
    """Artificial case: original_label disagrees with what Policy A
    itself would assign - must be flagged, never silently overwritten
    without a record of the mismatch."""
    records = _make_records({"g1": "Susceptible"})  # Policy A would say Susceptible
    result = reconcile_cohort(
        records, {}, "ciprofloxacin",
        pilot_genome_ids=["g1"], pilot_labels={"g1": "Resistant"},  # original label disagrees (contrived)
        expansion_genome_ids=[], expansion_labels={},
        target_size=200, seed=1,
    )
    entry = result.entries[0]
    assert entry.label_matches_original is False
    assert result.n_label_mismatches == 1


# --- Resistant/Susceptible counts reported ---


def test_included_resistant_susceptible_counts_are_correct():
    records = _make_records({"g1": "Resistant", "g2": "Resistant", "g3": "Susceptible"})
    result = reconcile_cohort(
        records, {}, "ciprofloxacin",
        pilot_genome_ids=["g1", "g2"], pilot_labels={"g1": "Resistant", "g2": "Resistant"},
        expansion_genome_ids=["g3"], expansion_labels={"g3": "Susceptible"},
        target_size=200, seed=1,
    )
    assert result.n_included_resistant == 2
    assert result.n_included_susceptible == 1
