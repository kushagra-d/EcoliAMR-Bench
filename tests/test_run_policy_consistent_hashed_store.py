import json

import pandas as pd
import pytest

import run_policy_consistent_hashed_store as script

GENOME_IDS = ["g1", "g2", "g3"]
SEQ = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"


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
    monkeypatch.setattr(script, "DB_PATH", data_processed / "policy_consistent_ciprofloxacin_rows_k15_d262144.db")
    monkeypatch.setattr(script, "FAILURES_PATH", data_processed / "policy_consistent_ciprofloxacin_hashed_failures_k15_d262144.jsonl")
    monkeypatch.setattr(script, "CONFIG_PATH", data_processed / "policy_consistent_ciprofloxacin_hashed_build_config_k15_d262144.json")
    monkeypatch.setattr(script, "GENOME_ORDER_PATH", data_processed / "policy_consistent_ciprofloxacin_genome_order.csv")
    monkeypatch.setattr(script, "N_BUCKETS", 20)
    monkeypatch.setattr(script, "K", 5)

    return data_processed


def test_builds_from_both_pilot_and_expansion_dirs(fake_env):
    script.main()
    config = json.loads((fake_env / "policy_consistent_ciprofloxacin_hashed_build_config_k15_d262144.json").read_text())
    assert config["n_rows_total"] == 3
    assert config["complete"] is True
    assert config["row_alignment_mismatches"] == 0


def test_genome_order_matches_reconciled_manifest(fake_env):
    script.main()
    order = pd.read_csv(fake_env / "policy_consistent_ciprofloxacin_genome_order.csv", dtype={"genome_id": str})
    assert order["genome_id"].tolist() == GENOME_IDS


def test_does_not_touch_pilot_or_expansion_store_paths(fake_env):
    script.main()
    assert not (fake_env / "out_of_core_v2_ciprofloxacin_rows_k15_d262144.db").exists()
    assert not (fake_env / "cohort_expansion_ciprofloxacin_rows_k15_d262144.db").exists()
