"""Integration test for src/run_intermediate_k_experiment.py itself -
exercises the real script logic (artifact naming/isolation, genome-row
order preservation, split alignment) against a tiny synthetic pilot,
not just the already-tested underlying functions it composes.
"""

import json

import pandas as pd
import pytest
from scipy import sparse

import run_intermediate_k_experiment as script

GENOME_IDS = ["g1", "g2", "g3"]
LABELS = {"g1": "Resistant", "g2": "Susceptible", "g3": "Susceptible"}
SEQUENCES = {
    "g1": "ACGTACGTACGTACGTACGT",
    "g2": "TTTTGGGGCCCCAAAATTTT",
    "g3": "ACGTACGTACGTACGTACGA",
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
    # random split: g1,g2 train / g3 test
    pd.DataFrame(
        {"genome_id": ["g1", "g2", "g3"], "partition": ["train", "train", "test"]}
    ).to_csv(data_processed / "pilot_ciprofloxacin_random_split.csv", index=False)
    # lineage split: g1,g3 train / g2 test
    pd.DataFrame(
        {"genome_id": ["g1", "g3", "g2"], "partition": ["train", "train", "test"]}
    ).to_csv(data_processed / "pilot_ciprofloxacin_lineage_split.csv", index=False)

    monkeypatch.setattr(script, "DATA_PROCESSED", data_processed)
    monkeypatch.setattr(script, "DATA_RAW_PILOT_DIR", raw_pilot_dir)
    return data_processed


def test_writes_k_specific_artifacts_only(fake_pilot):
    script.main(k=5)

    assert (fake_pilot / "pilot_ciprofloxacin_kmer_matrix_k5.npz").exists()
    assert (fake_pilot / "pilot_ciprofloxacin_kmer_vocabulary_k5.csv").exists()
    assert (fake_pilot / "pilot_ciprofloxacin_kmer_config_k5_comparison.json").exists()
    assert (fake_pilot / "pilot_ciprofloxacin_kmer_k5_lr_results.json").exists()

    # must never create or touch anything named for k=10/k=31
    for other_k in (10, 31):
        assert not (fake_pilot / f"pilot_ciprofloxacin_kmer_matrix_k{other_k}.npz").exists()


def test_preserves_genome_row_order(fake_pilot):
    script.main(k=5)
    matrix = sparse.load_npz(fake_pilot / "pilot_ciprofloxacin_kmer_matrix_k5.npz")
    assert matrix.shape[0] == len(GENOME_IDS)


def test_config_reports_complete_and_dimensions(fake_pilot):
    script.main(k=5)
    config = json.loads((fake_pilot / "pilot_ciprofloxacin_kmer_config_k5_comparison.json").read_text())
    assert config["complete"] is True
    assert config["n_genomes_processed"] == 3
    assert config["k"] == 5
    assert config["n_features"] > 0
    assert config["nnz"] > 0


def test_lr_results_cover_both_splits(fake_pilot):
    script.main(k=5)
    results = json.loads((fake_pilot / "pilot_ciprofloxacin_kmer_k5_lr_results.json").read_text())
    assert set(results) == {"random", "lineage"}
    assert results["random"]["n_train"] == 2
    assert results["random"]["n_test"] == 1
    assert results["lineage"]["n_train"] == 2
    assert results["lineage"]["n_test"] == 1


def test_does_not_mutate_input_split_files(fake_pilot):
    before = (fake_pilot / "pilot_ciprofloxacin_random_split.csv").read_text()
    script.main(k=5)
    after = (fake_pilot / "pilot_ciprofloxacin_random_split.csv").read_text()
    assert before == after


def test_memory_abort_produces_incomplete_config(fake_pilot, monkeypatch):
    # force every memory check to report "way over limit" so the very
    # first genome already triggers the safety abort
    monkeypatch.setattr(script, "MEMORY_LIMIT_MB", -1.0)
    script.main(k=5)
    config = json.loads((fake_pilot / "pilot_ciprofloxacin_kmer_config_k5_comparison.json").read_text())
    assert config["complete"] is False
    assert config["aborted_for_memory"] is True
    # no LR results should be produced for an incomplete matrix
    assert not (fake_pilot / "pilot_ciprofloxacin_kmer_k5_lr_results.json").exists()
