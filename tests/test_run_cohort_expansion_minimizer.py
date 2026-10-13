"""Integration test for src/run_cohort_expansion_minimizer.py - verifies
combined pilot+expansion genome ordering, usable-expansion filtering,
artifact isolation, and that the existing pilot minimizer artifact is
never overwritten. Uses a tiny k/w and tiny synthetic contigs - no real
network calls and no dependency on the real 200-genome retrieval.
"""

import json

import pandas as pd
import pytest
from scipy import sparse

import run_cohort_expansion_minimizer as script

PILOT_IDS = ["p1", "p2"]
EXPANSION_IDS = ["e1", "e2", "e3"]
SEQ = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"


def _write_fasta(path, genome_id):
    path.write_text(f">accn|{genome_id} desc\n{SEQ}\n")


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    data_processed = tmp_path / "data_processed"
    raw_pilot_dir = tmp_path / "raw_pilot"
    raw_expansion_dir = tmp_path / "raw_expansion"
    data_processed.mkdir()
    raw_pilot_dir.mkdir()
    raw_expansion_dir.mkdir()

    for gid in PILOT_IDS:
        _write_fasta(raw_pilot_dir / f"{gid}.fna", gid)
    for gid in EXPANSION_IDS:
        _write_fasta(raw_expansion_dir / f"{gid}.fna", gid)

    pd.DataFrame({"genome_id": PILOT_IDS}).to_csv(data_processed / "pilot_ciprofloxacin_genome_order.csv", index=False)
    pd.DataFrame({"genome_id": PILOT_IDS, "label": ["Resistant", "Susceptible"]}).to_csv(
        data_processed / "pilot_ciprofloxacin_selected_cohort.csv", index=False
    )
    pd.DataFrame({"genome_id": EXPANSION_IDS, "label": ["Resistant", "Susceptible", "Resistant"]}).to_csv(
        data_processed / "cohort_expansion_ciprofloxacin_selected.csv", index=False
    )
    # e3 is marked failed - must be EXCLUDED from the combined genome order
    pd.DataFrame({
        "genome_id": EXPANSION_IDS,
        "status": ["complete", "already_present_skipped", "incomplete"],
    }).to_csv(data_processed / "cohort_expansion_ciprofloxacin_retrieval_manifest.csv", index=False)

    monkeypatch.setattr(script, "DATA_PROCESSED", data_processed)
    monkeypatch.setattr(script, "DATA_RAW_PILOT_DIR", raw_pilot_dir)
    monkeypatch.setattr(script, "DATA_RAW_EXPANSION_DIR", raw_expansion_dir)
    monkeypatch.setattr(script, "LOG_PATH", data_processed / "test.log")
    monkeypatch.setattr(script, "K", 5)
    monkeypatch.setattr(script, "W", 3)

    return data_processed


def test_combined_genome_order_excludes_failed_expansion_genomes(fake_env):
    script.main()
    config = json.loads((fake_env / "cohort_expansion_ciprofloxacin_minimizer_config_k5_w3.json").read_text())
    assert config["n_pilot_genomes"] == 2
    assert config["n_expansion_genomes_usable"] == 2  # e3 (incomplete) excluded
    assert config["n_genomes_processed"] == 4


def test_matrix_row_count_matches_combined_usable_cohort(fake_env):
    script.main()
    matrix = sparse.load_npz(fake_env / "cohort_expansion_ciprofloxacin_minimizer_matrix_k5_w3.npz")
    assert matrix.shape[0] == 4  # 2 pilot + 2 usable expansion (e3 excluded)


def test_does_not_write_to_pilot_artifact_paths(fake_env):
    script.main()
    assert not (fake_env / "pilot_ciprofloxacin_minimizer_matrix_k5_w3.npz").exists()
    assert (fake_env / "cohort_expansion_ciprofloxacin_minimizer_matrix_k5_w3.npz").exists()
