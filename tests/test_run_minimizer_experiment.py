"""Integration test for src/run_minimizer_experiment.py - exercises the
real script logic (artifact naming/isolation, row-order preservation,
split alignment) against a tiny synthetic pilot.
"""

import json

import pandas as pd
import pytest
from scipy import sparse

import run_minimizer_experiment as script

GENOME_IDS = ["g1", "g2", "g3"]
LABELS = {"g1": "Resistant", "g2": "Susceptible", "g3": "Susceptible"}
SEQUENCES = {
    "g1": "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT",
    "g2": "TTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGG",
    "g3": "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGA",
}


@pytest.fixture
def fake_pilot(tmp_path, monkeypatch):
    data_processed = tmp_path / "data_processed"
    raw_pilot_dir = tmp_path / "raw_pilot"
    data_processed.mkdir()
    raw_pilot_dir.mkdir()

    for gid, seq in SEQUENCES.items():
        (raw_pilot_dir / f"{gid}.fna").write_text(f">accn|{gid} desc\n{seq}\n")

    pd.DataFrame({"genome_id": GENOME_IDS}).to_csv(
        data_processed / "pilot_ciprofloxacin_genome_order.csv", index=False
    )
    pd.DataFrame({"genome_id": GENOME_IDS, "label": [LABELS[g] for g in GENOME_IDS]}).to_csv(
        data_processed / "pilot_ciprofloxacin_selected_cohort.csv", index=False
    )
    pd.DataFrame({"genome_id": ["g1", "g2", "g3"], "partition": ["train", "train", "test"]}).to_csv(
        data_processed / "pilot_ciprofloxacin_random_split.csv", index=False
    )
    pd.DataFrame({"genome_id": ["g1", "g3", "g2"], "partition": ["train", "train", "test"]}).to_csv(
        data_processed / "pilot_ciprofloxacin_lineage_split.csv", index=False
    )

    monkeypatch.setattr(script, "DATA_PROCESSED", data_processed)
    monkeypatch.setattr(script, "DATA_RAW_PILOT_DIR", raw_pilot_dir)
    monkeypatch.setattr(script, "RAW_K", 5)  # small k so the tiny synthetic contigs produce real windows
    return data_processed


def test_writes_w_specific_artifacts_only(fake_pilot):
    script.main(w=3)

    assert (fake_pilot / "pilot_ciprofloxacin_minimizer_matrix_k5_w3.npz").exists()
    assert (fake_pilot / "pilot_ciprofloxacin_minimizer_vocabulary_k5_w3.csv").exists()
    assert (fake_pilot / "pilot_ciprofloxacin_minimizer_config_k5_w3.json").exists()
    assert (fake_pilot / "pilot_ciprofloxacin_minimizer_lr_results_k5_w3.json").exists()

    # must never touch any raw/hashed k-mer artifact naming
    assert not (fake_pilot / "pilot_ciprofloxacin_kmer_matrix_k31.npz").exists()


def test_preserves_genome_row_order(fake_pilot):
    script.main(w=3)
    matrix = sparse.load_npz(fake_pilot / "pilot_ciprofloxacin_minimizer_matrix_k5_w3.npz")
    assert matrix.shape[0] == len(GENOME_IDS)


def test_config_reports_complete_and_per_genome_stats(fake_pilot):
    script.main(w=3)
    config = json.loads((fake_pilot / "pilot_ciprofloxacin_minimizer_config_k5_w3.json").read_text())
    assert config["complete"] is True
    assert config["n_genomes_processed"] == 3
    assert config["k"] == 5
    assert config["w"] == 3
    assert "mean_selected_features_per_genome" in config


def test_lr_results_cover_both_splits(fake_pilot):
    script.main(w=3)
    results = json.loads((fake_pilot / "pilot_ciprofloxacin_minimizer_lr_results_k5_w3.json").read_text())
    assert set(results) == {"random", "lineage"}


def test_does_not_mutate_input_split_files(fake_pilot):
    before = (fake_pilot / "pilot_ciprofloxacin_random_split.csv").read_text()
    script.main(w=3)
    after = (fake_pilot / "pilot_ciprofloxacin_random_split.csv").read_text()
    assert before == after


def test_memory_abort_produces_incomplete_config(fake_pilot, monkeypatch):
    monkeypatch.setattr(script, "MEMORY_LIMIT_MB", -1.0)
    script.main(w=3)
    config = json.loads((fake_pilot / "pilot_ciprofloxacin_minimizer_config_k5_w3.json").read_text())
    assert config["complete"] is False
    assert config["aborted_for_memory"] is True
    assert not (fake_pilot / "pilot_ciprofloxacin_minimizer_lr_results_k5_w3.json").exists()
