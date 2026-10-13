"""Integration test for src/run_full_cohort_planning.py - specifically
regression-tests the bug caught during this task's own real run: a
genome's assembly is antibiotic-independent (one FASTA file serves
every antibiotic it's eligible for), so a genome already retrieved
under one antibiotic must be recognized as available when assessing a
DIFFERENT antibiotic's eligible pool, not reported as missing just
because that antibiotic's own directory list was historically empty.
"""

import json

import pandas as pd
import pytest

import run_full_cohort_planning as script


def _write_metadata(path, label_by_genome, antibiotic):
    pd.DataFrame(
        [{"genome_id": gid, "antibiotic": antibiotic, "resistant_phenotype": label, "genome_mlst": ""} for gid, label in label_by_genome.items()]
    ).to_csv(path, index=False)


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    data_processed = tmp_path / "data_processed"
    raw_dir = tmp_path / "raw_shared"
    data_processed.mkdir()
    raw_dir.mkdir()

    # g1 is eligible for BOTH ciprofloxacin and ampicillin, and its
    # assembly was already retrieved under a ciprofloxacin-named directory.
    _write_metadata(data_processed / "ciprofloxacin_amr_metadata.csv", {"g1": "Resistant", "g2": "Susceptible"}, "ciprofloxacin")
    _write_metadata(data_processed / "ampicillin_amr_metadata.csv", {"g1": "Resistant", "g3": "Susceptible"}, "ampicillin")
    _write_metadata(data_processed / "gentamicin_amr_metadata.csv", {"g4": "Resistant"}, "gentamicin")

    (raw_dir / "g1.fna").write_text(">accn|g1\nACGT\n")
    manifest_path = data_processed / "manifest.csv"
    manifest_path.write_text("genome_id,status\ng1,complete\n")

    monkeypatch.setattr(script, "DATA_PROCESSED", data_processed)
    monkeypatch.setattr(script, "ALL_RAW_DIRS", [raw_dir])
    monkeypatch.setattr(script, "ALL_RETRIEVAL_MANIFESTS", [manifest_path])
    monkeypatch.setattr(script, "RAW_DIRS_BY_ANTIBIOTIC", {ab: [raw_dir] for ab in script.ANTIBIOTICS})
    monkeypatch.setattr(script, "MANIFESTS_BY_ANTIBIOTIC", {ab: [manifest_path] for ab in script.ANTIBIOTICS})

    return data_processed


def test_genome_validated_under_one_antibiotic_counts_as_available_for_another(fake_env):
    script.main()
    availability = json.loads((fake_env / "full_cohort_assembly_availability.json").read_text())
    # g1 is eligible for ampicillin too, and its file already exists -
    # must NOT appear in ampicillin's missing list.
    assert availability["ampicillin"]["n_already_validated"] == 1
    assert availability["ciprofloxacin"]["n_already_validated"] == 1


def test_shared_genome_is_not_double_planned_for_retrieval(fake_env):
    script.main()
    plan = pd.read_csv(fake_env / "full_cohort_retrieval_plan.csv", dtype={"genome_id": str})
    assert plan["genome_id"].duplicated().sum() == 0
    # g1 already has a file - must not appear in the retrieval plan at all
    assert "g1" not in set(plan["genome_id"])


def test_manifest_to_source_metadata_consistency(fake_env):
    script.main()
    cipro_manifest = pd.read_csv(fake_env / "full_cohort_ciprofloxacin_policy_a_manifest.csv", dtype={"genome_id": str})
    assert set(cipro_manifest["genome_id"]) == {"g1", "g2"}
    assert cipro_manifest.set_index("genome_id").loc["g1", "label"] == "Resistant"
