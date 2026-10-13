import pytest

from bvbrc_recon.analysis import (
    MISSING_SENTINEL,
    categorize_phenotype,
    phenotype_value_counts,
    summarize_genome_multiplicity,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Resistant", "Resistant"),
        ("Susceptible", "Susceptible"),
        ("Intermediate", "Intermediate"),
        (None, "missing"),
        ("", "missing"),
        ("   ", "missing"),
        ("Non-susceptible", "other"),
        ("Reduced Susceptibility", "other"),
    ],
)
def test_categorize_phenotype(value, expected):
    assert categorize_phenotype(value) == expected


def test_categorize_phenotype_never_maps_intermediate_to_susceptible():
    assert categorize_phenotype("Intermediate") != "Susceptible"


def test_phenotype_value_counts_preserves_raw_values_and_groups_missing():
    records = [
        {"resistant_phenotype": "Resistant"},
        {"resistant_phenotype": "Resistant"},
        {"resistant_phenotype": ""},
        {"resistant_phenotype": None},
        {"resistant_phenotype": "Non-susceptible"},
    ]
    counts = phenotype_value_counts(records)
    assert counts["Resistant"] == 2
    assert counts["Non-susceptible"] == 1
    assert counts[MISSING_SENTINEL] == 2


def test_summarize_genome_multiplicity_single_replicate_and_conflict():
    records = [
        {"genome_id": "g1", "resistant_phenotype": "Resistant"},  # single
        {"genome_id": "g2", "resistant_phenotype": "Susceptible"},  # replicate agree
        {"genome_id": "g2", "resistant_phenotype": "Susceptible"},
        {"genome_id": "g3", "resistant_phenotype": "Resistant"},  # conflict
        {"genome_id": "g3", "resistant_phenotype": "Susceptible"},
    ]
    summary = summarize_genome_multiplicity(records)
    assert summary.single_record_genomes == 1
    assert summary.replicate_same_phenotype_genomes == 1
    assert summary.conflicting_phenotype_genomes == 1
    assert summary.conflicts["g3"] == ["Resistant", "Susceptible"]


def test_summarize_genome_multiplicity_ignores_records_without_genome_id():
    records = [{"resistant_phenotype": "Resistant"}]
    summary = summarize_genome_multiplicity(records)
    assert summary.single_record_genomes == 0
    assert summary.conflicting_phenotype_genomes == 0
