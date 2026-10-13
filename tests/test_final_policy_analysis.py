from bvbrc_recon.final_policy_analysis import (
    METHOD_CONSISTENT,
    METHOD_INCONSISTENT,
    METHOD_MISSING,
    OTHER_OR_COMBINED_OR_MISSING,
    STANDARD_CONSISTENT,
    STANDARD_INCONSISTENT,
    STANDARD_MISSING_OR_COMBINED,
    apply_policy_a,
    apply_policy_b,
    apply_policy_c,
    characterize_metadata,
    classify_genome_method_status,
    classify_genome_standard_status,
    normalize_standard_for_eligibility,
    run_all_policies,
)


def rec(genome_id, phenotype, **kwargs):
    base = {
        "genome_id": genome_id,
        "resistant_phenotype": phenotype,
        "antibiotic": "ciprofloxacin",
        "testing_standard": "EUCAST",
        "testing_standard_year": 2021,
        "laboratory_typing_method": "Disk diffusion",
        "laboratory_typing_platform": "",
        "measurement_unit": "",
        "measurement": "",
        "measurement_sign": "",
        "measurement_value": "",
        "evidence": "Laboratory Method",
        "source": "",
        "pmid": "[]",
    }
    base.update(kwargs)
    return base


# --- normalize_standard_for_eligibility -----------------------------------


def test_normalize_standard_folds_casing_of_single_standard():
    assert normalize_standard_for_eligibility("EUCAST") == "EUCAST"
    assert normalize_standard_for_eligibility("eucast") == "EUCAST"
    assert normalize_standard_for_eligibility("CLSI") == "CLSI"
    assert normalize_standard_for_eligibility("clsi") == "CLSI"


def test_normalize_standard_treats_missing_as_sentinel():
    assert normalize_standard_for_eligibility(None) == OTHER_OR_COMBINED_OR_MISSING
    assert normalize_standard_for_eligibility("") == OTHER_OR_COMBINED_OR_MISSING


def test_normalize_standard_never_merges_combined_values():
    assert normalize_standard_for_eligibility("EUCAST, CLSI") == OTHER_OR_COMBINED_OR_MISSING
    assert normalize_standard_for_eligibility("EUCAST and CLSI") == OTHER_OR_COMBINED_OR_MISSING


# --- classify_genome_standard_status ---------------------------------------


def test_genome_standard_status_consistent_same_casing():
    records = [rec("g1", "Resistant", testing_standard="EUCAST"), rec("g1", "Resistant", testing_standard="EUCAST")]
    assert classify_genome_standard_status(records) == STANDARD_CONSISTENT


def test_genome_standard_status_consistent_different_casing_still_folds():
    records = [rec("g1", "Resistant", testing_standard="EUCAST"), rec("g1", "Resistant", testing_standard="eucast")]
    assert classify_genome_standard_status(records) == STANDARD_CONSISTENT


def test_genome_standard_status_inconsistent_across_named_standards():
    records = [rec("g1", "Resistant", testing_standard="EUCAST"), rec("g1", "Resistant", testing_standard="CLSI")]
    assert classify_genome_standard_status(records) == STANDARD_INCONSISTENT


def test_genome_standard_status_missing_only():
    records = [rec("g1", "Resistant", testing_standard=""), rec("g1", "Resistant", testing_standard=None)]
    assert classify_genome_standard_status(records) == STANDARD_MISSING_OR_COMBINED


def test_genome_standard_status_combined_only():
    records = [rec("g1", "Resistant", testing_standard="EUCAST, CLSI")]
    assert classify_genome_standard_status(records) == STANDARD_MISSING_OR_COMBINED


def test_genome_standard_status_named_vs_missing_is_inconsistent():
    records = [rec("g1", "Resistant", testing_standard="EUCAST"), rec("g1", "Resistant", testing_standard="")]
    assert classify_genome_standard_status(records) == STANDARD_INCONSISTENT


# --- classify_genome_method_status -----------------------------------------


def test_genome_method_status_consistent():
    records = [
        rec("g1", "Resistant", laboratory_typing_method="Disk diffusion"),
        rec("g1", "Resistant", laboratory_typing_method="Disk diffusion"),
    ]
    assert classify_genome_method_status(records) == METHOD_CONSISTENT


def test_genome_method_status_inconsistent_method():
    records = [
        rec("g1", "Resistant", laboratory_typing_method="Disk diffusion"),
        rec("g1", "Resistant", laboratory_typing_method="Broth dilution"),
    ]
    assert classify_genome_method_status(records) == METHOD_INCONSISTENT


def test_genome_method_status_missing_only():
    records = [rec("g1", "Resistant", laboratory_typing_method=""), rec("g1", "Resistant", laboratory_typing_method=None)]
    assert classify_genome_method_status(records) == METHOD_MISSING


def test_genome_method_status_same_method_differing_units_is_inconsistent():
    records = [
        rec("g1", "Resistant", laboratory_typing_method="Broth dilution", measurement_unit="mg/L"),
        rec("g1", "Resistant", laboratory_typing_method="Broth dilution", measurement_unit="mm"),
    ]
    assert classify_genome_method_status(records) == METHOD_INCONSISTENT


def test_genome_method_status_same_method_one_missing_unit_is_consistent():
    records = [
        rec("g1", "Resistant", laboratory_typing_method="Broth dilution", measurement_unit="mg/L"),
        rec("g1", "Resistant", laboratory_typing_method="Broth dilution", measurement_unit=""),
    ]
    assert classify_genome_method_status(records) == METHOD_CONSISTENT


# --- characterize_metadata --------------------------------------------------


def test_characterize_metadata_preserves_raw_casing_distinctly():
    records = [
        rec("g1", "Resistant", testing_standard="EUCAST"),
        rec("g2", "Susceptible", testing_standard="eucast"),
    ]
    char = characterize_metadata(records, "ciprofloxacin")
    assert char.testing_standard_counts["EUCAST"] == 1
    assert char.testing_standard_counts["eucast"] == 1


def test_characterize_metadata_counts_repeated_concordant_heterogeneity():
    records = [
        rec("g1", "Resistant", testing_standard="EUCAST"),
        rec("g1", "Resistant", testing_standard="CLSI"),  # same phenotype, differing standard
    ]
    char = characterize_metadata(records, "ciprofloxacin")
    assert char.repeated_concordant_genomes == 1
    assert char.repeated_concordant_heterogeneity["testing_standard"] == 1
    assert char.repeated_concordant_heterogeneity["laboratory_typing_method"] == 0


def test_characterize_metadata_totals_reconcile():
    records = [rec("g1", "Resistant"), rec("g2", "Susceptible"), rec("g2", "Susceptible")]
    char = characterize_metadata(records, "ciprofloxacin")
    assert char.n_records == 3
    assert char.n_genomes == 2


# --- apply_policy_a ----------------------------------------------------


def test_policy_a_includes_resistant_plus_intermediate():
    records = [rec("g1", "Resistant"), rec("g1", "Intermediate")]
    outcome = apply_policy_a(records, "ciprofloxacin", {})
    assert outcome.genome_class["g1"] == "Resistant"


def test_policy_a_excludes_resistant_and_susceptible_conflict():
    records = [rec("g1", "Resistant"), rec("g1", "Susceptible")]
    outcome = apply_policy_a(records, "ciprofloxacin", {})
    assert "g1" not in outcome.genome_class
    assert outcome.genome_outcome["g1"] == "excluded_categorical_disagreement"


# --- apply_policy_b ----------------------------------------------------


def test_policy_b_excludes_missing_standard():
    records = [rec("g1", "Resistant", testing_standard="")]
    outcome = apply_policy_b(records, "ciprofloxacin", {})
    assert "g1" not in outcome.genome_class
    assert "missing_or_combined" in outcome.genome_outcome["g1"]


def test_policy_b_excludes_inconsistent_standard():
    records = [rec("g1", "Resistant", testing_standard="EUCAST"), rec("g1", "Resistant", testing_standard="CLSI")]
    outcome = apply_policy_b(records, "ciprofloxacin", {})
    assert "g1" not in outcome.genome_class
    assert "inconsistent" in outcome.genome_outcome["g1"]


def test_policy_b_excludes_combined_standard():
    records = [rec("g1", "Resistant", testing_standard="EUCAST, CLSI")]
    outcome = apply_policy_b(records, "ciprofloxacin", {})
    assert "g1" not in outcome.genome_class


def test_policy_b_includes_consistent_standard():
    records = [rec("g1", "Resistant", testing_standard="EUCAST"), rec("g1", "Resistant", testing_standard="eucast")]
    outcome = apply_policy_b(records, "ciprofloxacin", {})
    assert outcome.genome_class["g1"] == "Resistant"


def test_policy_b_excludes_genomes_already_excluded_by_policy_a():
    records = [rec("g1", "Intermediate", testing_standard="EUCAST")]
    outcome = apply_policy_b(records, "ciprofloxacin", {})
    assert "g1" not in outcome.genome_class
    assert outcome.genome_outcome["g1"].startswith("excluded_by_policy_a:")


def test_policy_b_partition_sums_reconcile_with_total_genomes():
    records = [
        rec("g1", "Resistant", testing_standard="EUCAST"),
        rec("g2", "Susceptible", testing_standard=""),
        rec("g3", "Intermediate"),
        rec("g4", "Resistant"),
        rec("g4", "Susceptible"),
    ]
    outcome = apply_policy_b(records, "ciprofloxacin", {})
    assert sum(outcome.outcome_counts().values()) == outcome.n_total_genomes == 4


# --- apply_policy_c ----------------------------------------------------


def test_policy_c_excludes_missing_method():
    records = [rec("g1", "Resistant", laboratory_typing_method="")]
    outcome = apply_policy_c(records, "ciprofloxacin", {})
    assert "g1" not in outcome.genome_class


def test_policy_c_excludes_inconsistent_method():
    records = [
        rec("g1", "Resistant", laboratory_typing_method="Disk diffusion"),
        rec("g1", "Resistant", laboratory_typing_method="Broth dilution"),
    ]
    outcome = apply_policy_c(records, "ciprofloxacin", {})
    assert "g1" not in outcome.genome_class


def test_policy_c_includes_consistent_method():
    records = [rec("g1", "Resistant", laboratory_typing_method="Disk diffusion")]
    outcome = apply_policy_c(records, "ciprofloxacin", {})
    assert outcome.genome_class["g1"] == "Resistant"


def test_policy_c_partition_sums_reconcile():
    records = [
        rec("g1", "Resistant", laboratory_typing_method="Disk diffusion"),
        rec("g2", "Susceptible", laboratory_typing_method=""),
        rec("g3", "Resistant"),
        rec("g3", "Susceptible"),
    ]
    outcome = apply_policy_c(records, "ciprofloxacin", {})
    assert sum(outcome.outcome_counts().values()) == outcome.n_total_genomes == 3


# --- MLST transparency stats -------------------------------------------


def test_mlst_transparency_counts_only_eligible_genomes_with_mlst():
    records = [rec("g1", "Resistant"), rec("g2", "Susceptible"), rec("g3", "Intermediate")]
    mlst_by_genome = {"g1": "MLST.ecoli_achtman_4.73", "g2": None, "g3": "MLST.ecoli_achtman_4.95"}
    outcome = apply_policy_a(records, "ciprofloxacin", mlst_by_genome)
    assert outcome.eligible_with_mlst == 1  # only g1 is eligible AND has mlst (g3 excluded by policy A)
    assert outcome.eligible_distinct_mlst_groups == 1


# --- run_all_policies ----------------------------------------------------


def test_run_all_policies_returns_all_three():
    records = [rec("g1", "Resistant"), rec("g2", "Susceptible")]
    outcomes = run_all_policies(records, "ciprofloxacin", {})
    assert set(outcomes) == {"A", "B", "C"}


def test_policy_c_is_at_least_as_strict_as_policy_a():
    records = [
        rec("g1", "Resistant", laboratory_typing_method="Disk diffusion"),
        rec("g2", "Susceptible", laboratory_typing_method=""),
        rec("g3", "Intermediate"),
    ]
    outcomes = run_all_policies(records, "ciprofloxacin", {})
    eligible_a = set(outcomes["A"].eligible_genome_ids())
    eligible_c = set(outcomes["C"].eligible_genome_ids())
    assert eligible_c <= eligible_a
