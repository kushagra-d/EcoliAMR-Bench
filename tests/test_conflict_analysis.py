import math

import pytest

from bvbrc_recon.conflict_analysis import (
    apply_policy_1,
    apply_policy_2,
    audit_genomes,
    categorical_set,
    classify_genome_multiplicity,
    classify_genome_pattern,
    field_differs_within_genome,
    group_by_genome,
    normalize_value,
    policy_transparency_stats,
    standard_stratified_analysis,
)


def rec(genome_id, phenotype, **kwargs):
    base = {
        "genome_id": genome_id,
        "resistant_phenotype": phenotype,
        "antibiotic": "ciprofloxacin",
        "measurement": "",
        "measurement_sign": "",
        "measurement_value": "",
        "measurement_unit": "",
        "laboratory_typing_method": "Disk diffusion",
        "laboratory_typing_platform": "",
        "testing_standard": "EUCAST",
        "testing_standard_year": 2021.0,
        "evidence": "Laboratory Method",
        "source": "",
        "pmid": "[]",
    }
    base.update(kwargs)
    return base


# --- normalize_value ----------------------------------------------------


def test_normalize_value_handles_none_nan_blank_and_floats():
    assert normalize_value(None) == "<missing/blank>"
    assert normalize_value(float("nan")) == "<missing/blank>"
    assert normalize_value("") == "<missing/blank>"
    assert normalize_value("   ") == "<missing/blank>"
    assert normalize_value(2021.0) == "2021"
    assert normalize_value("EUCAST") == "EUCAST"


# --- group_by_genome -----------------------------------------------------


def test_group_by_genome_ignores_missing_genome_id():
    records = [rec("g1", "Resistant"), rec(None, "Resistant")]
    groups = group_by_genome(records)
    assert set(groups) == {"g1"}


# --- categorical_set / classify_genome_pattern ---------------------------


@pytest.mark.parametrize(
    "phenotypes,expected_pattern",
    [
        (["Resistant"], "only_resistant"),
        (["Susceptible"], "only_susceptible"),
        (["Intermediate"], "only_intermediate"),
        (["Resistant", "Susceptible"], "resistant_and_susceptible"),
        (["Resistant", "Intermediate"], "resistant_and_intermediate_no_susceptible"),
        (["Susceptible", "Intermediate"], "susceptible_and_intermediate_no_resistant"),
        (["Resistant", "Susceptible", "Intermediate"], "all_three"),
        ([""], "no_recognized_interpretation"),
        ([None, "Non-susceptible"], "no_recognized_interpretation"),
    ],
)
def test_classify_genome_pattern_all_buckets(phenotypes, expected_pattern):
    records = [rec("g1", p) for p in phenotypes]
    assert classify_genome_pattern(records) == expected_pattern


def test_categorical_set_ignores_missing_and_other():
    records = [rec("g1", "Resistant"), rec("g1", ""), rec("g1", "Non-susceptible")]
    assert categorical_set(records) == frozenset({"Resistant"})


# --- classify_genome_multiplicity ----------------------------------------


def test_single_record_is_single():
    assert classify_genome_multiplicity([rec("g1", "Resistant")]) == "single"


def test_one_resistant_plus_one_blank_is_not_conflicting():
    """Regression check for the exact flaw being audited: a Resistant record
    plus a blank/missing record must NOT be reported as a categorical
    conflict, unlike the older, broader `summarize_genome_multiplicity`."""
    records = [rec("g1", "Resistant"), rec("g1", "")]
    assert classify_genome_multiplicity(records) == "repeated_concordant"


def test_exact_duplicate_records_are_detected():
    records = [rec("g1", "Resistant"), rec("g1", "Resistant")]
    assert classify_genome_multiplicity(records) == "exact_duplicate"


def test_differing_metadata_with_same_phenotype_is_repeated_concordant():
    records = [
        rec("g1", "Resistant", testing_standard="EUCAST"),
        rec("g1", "Resistant", testing_standard="CLSI"),
    ]
    assert classify_genome_multiplicity(records) == "repeated_concordant"


def test_disagreeing_phenotype_is_conflicting_even_with_identical_metadata():
    records = [rec("g1", "Resistant"), rec("g1", "Susceptible")]
    assert classify_genome_multiplicity(records) == "conflicting"


def test_field_differs_within_genome():
    records = [
        rec("g1", "Resistant", testing_standard="EUCAST"),
        rec("g1", "Resistant", testing_standard="CLSI"),
    ]
    assert field_differs_within_genome(records, "testing_standard") is True
    assert field_differs_within_genome(records, "laboratory_typing_method") is False


# --- audit_genomes: partition integrity -----------------------------------


def _mixed_fixture():
    return [
        rec("g_single", "Resistant"),
        rec("g_replicate", "Susceptible"),
        rec("g_replicate", "Susceptible"),
        rec("g_conflict", "Resistant"),
        rec("g_conflict", "Susceptible"),
        rec("g_missing_only", ""),
        rec("g_r_plus_blank", "Resistant"),
        rec("g_r_plus_blank", ""),
    ]


def test_audit_genomes_pattern_counts_sum_to_unique_genomes():
    audit = audit_genomes(_mixed_fixture())
    assert audit.n_unique_genomes == 5
    assert sum(audit.pattern_counts.values()) == audit.n_unique_genomes


def test_audit_genomes_multiplicity_counts_sum_to_unique_genomes():
    audit = audit_genomes(_mixed_fixture())
    assert sum(audit.multiplicity_counts.values()) == audit.n_unique_genomes


def test_audit_genomes_only_true_categorical_disagreement_counts_as_conflicting():
    audit = audit_genomes(_mixed_fixture())
    assert audit.conflicting_genome_ids == ["g_conflict"]
    assert "g_r_plus_blank" not in audit.conflicting_genome_ids


def test_audit_genomes_heterogeneity_fields_present_for_conflicting_only():
    records = _mixed_fixture() + [
        rec("g_conflict2", "Resistant", testing_standard="EUCAST"),
        rec("g_conflict2", "Susceptible", testing_standard="CLSI"),
    ]
    audit = audit_genomes(records)
    het = audit.heterogeneity_among_conflicts["testing_standard"]
    # g_conflict: same standard (EUCAST/EUCAST) both records -> "same"
    # g_conflict2: differing standard -> "differs"
    assert het["differs"] == 1
    assert het["same"] == 1


# --- Policy 1 --------------------------------------------------------------


def test_policy1_resistant_plus_intermediate_is_eligible_as_resistant():
    groups = group_by_genome([rec("g1", "Resistant"), rec("g1", "Intermediate")])
    result = apply_policy_1(groups)
    assert result.genome_outcome["g1"] == "eligible_resistant"
    assert result.genome_class["g1"] == "Resistant"


def test_policy1_susceptible_plus_intermediate_is_eligible_as_susceptible():
    groups = group_by_genome([rec("g1", "Susceptible"), rec("g1", "Intermediate")])
    result = apply_policy_1(groups)
    assert result.genome_class["g1"] == "Susceptible"


def test_policy1_resistant_and_susceptible_is_excluded():
    groups = group_by_genome([rec("g1", "Resistant"), rec("g1", "Susceptible")])
    result = apply_policy_1(groups)
    assert result.genome_outcome["g1"] == "excluded_categorical_disagreement"
    assert "g1" not in result.genome_class


def test_policy1_only_intermediate_is_excluded():
    groups = group_by_genome([rec("g1", "Intermediate")])
    result = apply_policy_1(groups)
    assert result.genome_outcome["g1"] == "excluded_no_recognized_binary_interpretation"


def test_policy1_all_three_is_excluded_as_disagreement():
    groups = group_by_genome(
        [rec("g1", "Resistant"), rec("g1", "Susceptible"), rec("g1", "Intermediate")]
    )
    result = apply_policy_1(groups)
    assert result.genome_outcome["g1"] == "excluded_categorical_disagreement"


def test_policy1_partition_covers_every_genome_exactly_once():
    groups = group_by_genome(_mixed_fixture())
    result = apply_policy_1(groups)
    assert set(result.genome_outcome) == set(groups)
    assert sum(result.outcome_counts().values()) == len(groups)


# --- Policy 2 ----------------------------------------------------------


def test_policy2_resistant_plus_intermediate_is_excluded():
    groups = group_by_genome([rec("g1", "Resistant"), rec("g1", "Intermediate")])
    result = apply_policy_2(groups)
    assert result.genome_outcome["g1"] == "excluded_intermediate_present_with_binary"
    assert "g1" not in result.genome_class


def test_policy2_pure_resistant_is_eligible():
    groups = group_by_genome([rec("g1", "Resistant"), rec("g1", "Resistant")])
    result = apply_policy_2(groups)
    assert result.genome_class["g1"] == "Resistant"


def test_policy2_resistant_and_susceptible_is_excluded_both_classes():
    groups = group_by_genome([rec("g1", "Resistant"), rec("g1", "Susceptible")])
    result = apply_policy_2(groups)
    assert result.genome_outcome["g1"] == "excluded_both_classes_present"


def test_policy2_no_recognized_interpretation_is_excluded_no_binary():
    groups = group_by_genome([rec("g1", "")])
    result = apply_policy_2(groups)
    assert result.genome_outcome["g1"] == "excluded_no_binary_interpretation"


def test_policy2_is_strictly_more_exclusive_than_policy1():
    groups = group_by_genome([rec("g1", "Resistant"), rec("g1", "Intermediate")])
    p1 = apply_policy_1(groups)
    p2 = apply_policy_2(groups)
    assert "g1" in p1.genome_class
    assert "g1" not in p2.genome_class


# --- policy_transparency_stats --------------------------------------------


def test_policy_transparency_counts_repeated_records_and_heterogeneity():
    groups = group_by_genome(
        [
            rec("g1", "Resistant", testing_standard="EUCAST"),
            rec("g1", "Resistant", testing_standard="CLSI"),
            rec("g2", "Susceptible"),
        ]
    )
    result = apply_policy_1(groups)
    stats = policy_transparency_stats(result, groups)
    assert stats.eligible_with_repeated_records == 1
    assert stats.eligible_with_heterogeneous_field["testing_standard"] == 1
    assert stats.eligible_with_heterogeneous_field["laboratory_typing_method"] == 0


def test_policy2_transparency_never_shows_intermediate_co_occurrence():
    groups = group_by_genome(
        [rec("g1", "Resistant"), rec("g2", "Resistant"), rec("g2", "Intermediate")]
    )
    result = apply_policy_2(groups)
    stats = policy_transparency_stats(result, groups)
    assert stats.eligible_with_intermediate_co_occurrence == 0


def test_policy1_transparency_shows_intermediate_co_occurrence():
    groups = group_by_genome([rec("g1", "Resistant"), rec("g1", "Intermediate")])
    result = apply_policy_1(groups)
    stats = policy_transparency_stats(result, groups)
    assert stats.eligible_with_intermediate_co_occurrence == 1


# --- standard_stratified_analysis -----------------------------------------


def test_standard_stratified_analysis_preserves_raw_casing_and_combined_values():
    records = [
        rec("g1", "Resistant", testing_standard="EUCAST"),
        rec("g2", "Susceptible", testing_standard="eucast"),
        rec("g3", "Resistant", testing_standard="EUCAST, CLSI"),
    ]
    audit = audit_genomes(records)
    strata = standard_stratified_analysis(records, audit)
    raw_values = {s.raw_value for s in strata}
    assert "EUCAST" in raw_values
    assert "eucast" in raw_values  # NOT merged with "EUCAST"
    assert "EUCAST, CLSI" in raw_values


def test_standard_stratified_analysis_cross_tabulates_global_conflicts():
    records = [
        rec("g1", "Resistant", testing_standard="EUCAST"),
        rec("g1", "Susceptible", testing_standard="CLSI"),  # g1 is globally conflicting
        rec("g2", "Susceptible", testing_standard="EUCAST"),
    ]
    audit = audit_genomes(records)
    strata = {s.raw_value: s for s in standard_stratified_analysis(records, audit)}
    assert strata["EUCAST"].genomes_represented == 2
    assert strata["EUCAST"].genomes_globally_conflicting == 1  # only g1
    assert strata["CLSI"].genomes_globally_conflicting == 1
