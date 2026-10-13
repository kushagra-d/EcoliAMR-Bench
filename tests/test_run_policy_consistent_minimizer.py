import json

import pandas as pd
import pytest
from scipy import sparse

import run_policy_consistent_minimizer as script

GENOME_IDS = ["g1", "g2", "g3"]
SEQ = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    data_processed = tmp_path / "data_processed"
    raw_pilot_dir = tmp_path / "raw_pilot"
    raw_expansion_dir = tmp_path / "raw_expansion"
    data_processed.mkdir()
    raw_pilot_dir.mkdir()
    raw_expansion_dir.mkdir()

    (raw_pilot_dir / "g1.fna").write_text(f">accn|g1 desc\n{SEQ}\n")
    (raw_pilot_dir / "g2.fna").write_text(f">accn|g2 desc\n{SEQ}\n")
    (raw_expansion_dir / "g3.fna").write_text(f">accn|g3 desc\n{SEQ}\n")

    pd.DataFrame({"genome_id": GENOME_IDS, "label": ["Resistant", "Susceptible", "Resistant"]}).to_csv(
        data_processed / "policy_consistent_ciprofloxacin_selected_cohort.csv", index=False
    )

    monkeypatch.setattr(script, "DATA_PROCESSED", data_processed)
    monkeypatch.setattr(script, "DATA_RAW_PILOT_DIR", raw_pilot_dir)
    monkeypatch.setattr(script, "DATA_RAW_EXPANSION_DIR", raw_expansion_dir)
    monkeypatch.setattr(script, "LOG_PATH", data_processed / "test.log")
    monkeypatch.setattr(script, "K", 5)
    monkeypatch.setattr(script, "W", 3)

    return data_processed


def test_row_alignment_matches_reconciled_manifest(fake_env):
    script.main()
    config = json.loads((fake_env / "policy_consistent_ciprofloxacin_minimizer_config_k5_w3.json").read_text())
    assert config["row_alignment_matches_manifest"] is True
    assert config["n_genomes_processed"] == 3


def test_matrix_shape_matches_reconciled_cohort(fake_env):
    script.main()
    matrix = sparse.load_npz(fake_env / "policy_consistent_ciprofloxacin_minimizer_matrix_k5_w3.npz")
    assert matrix.shape[0] == 3


def test_does_not_touch_other_minimizer_artifacts(fake_env):
    script.main()
    assert not (fake_env / "pilot_ciprofloxacin_minimizer_matrix_k5_w3.npz").exists()
    assert not (fake_env / "cohort_expansion_ciprofloxacin_minimizer_matrix_k5_w3.npz").exists()
