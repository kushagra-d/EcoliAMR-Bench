import pandas as pd

from bvbrc_recon.csv_loader import load_antibiotic_csv


def test_genome_id_stays_a_distinct_string_even_when_numerically_equal(tmp_path):
    """562.100 and 562.1 are numerically identical as floats but are
    different genome_id strings - naive pandas inference would merge them."""
    df = pd.DataFrame(
        {
            "genome_id": ["562.100", "562.1"],
            "id": ["abc-1", "abc-2"],
            "resistant_phenotype": ["Resistant", "Susceptible"],
            "testing_standard_year": [2021, None],
        }
    )
    path = tmp_path / "fixture.csv"
    df.to_csv(path, index=False)

    records = load_antibiotic_csv(path)
    genome_ids = {r["genome_id"] for r in records}
    assert genome_ids == {"562.100", "562.1"}
    assert all(isinstance(r["genome_id"], str) for r in records)


def test_missing_values_become_none_not_nan(tmp_path):
    df = pd.DataFrame(
        {
            "genome_id": ["562.1"],
            "id": ["abc-1"],
            "resistant_phenotype": ["Resistant"],
            "testing_standard_year": [None],
        }
    )
    path = tmp_path / "fixture.csv"
    df.to_csv(path, index=False)

    records = load_antibiotic_csv(path)
    assert records[0]["testing_standard_year"] is None
