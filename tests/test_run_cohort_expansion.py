"""Integration tests for src/run_cohort_expansion.py - exercises the
real script logic (selection idempotency, retrieval resumability,
failure-limit handling, QC gating, artifact isolation) against a tiny
synthetic antibiotic-metadata CSV. No real network calls: `fetch_complete_assembly`
is monkeypatched (same pattern this project already uses in
`tests/test_sequence_retrieval.py`).
"""

from __future__ import annotations

import pandas as pd
import pytest

import run_cohort_expansion as script
from bvbrc_recon.sequence_retrieval import (
    ASSEMBLY_STATUS_COMPLETE,
    ASSEMBLY_STATUS_INCOMPLETE,
    CompleteAssemblyResult,
    ContigRecord,
)


def _write_metadata_csv(path, pilot_ids, extra_resistant, extra_susceptible):
    rows = []
    for gid in pilot_ids:
        rows.append({"genome_id": gid, "antibiotic": "ciprofloxacin", "resistant_phenotype": "Resistant", "genome_contigs": 1, "genome_mlst": ""})
    for gid in extra_resistant:
        rows.append({"genome_id": gid, "antibiotic": "ciprofloxacin", "resistant_phenotype": "Resistant", "genome_contigs": 1, "genome_mlst": ""})
    for gid in extra_susceptible:
        rows.append({"genome_id": gid, "antibiotic": "ciprofloxacin", "resistant_phenotype": "Susceptible", "genome_contigs": 1, "genome_mlst": ""})
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    data_processed = tmp_path / "data_processed"
    raw_expansion_dir = tmp_path / "raw_expansion"
    data_processed.mkdir()

    pilot_ids = [f"pilot_g{i}" for i in range(5)]
    extra_resistant = [f"extra_r{i}" for i in range(10)]
    extra_susceptible = [f"extra_s{i}" for i in range(10)]

    pd.DataFrame({"genome_id": pilot_ids, "label": ["Resistant"] * 5}).to_csv(
        data_processed / "pilot_ciprofloxacin_selected_cohort.csv", index=False
    )
    _write_metadata_csv(
        data_processed / "ciprofloxacin_amr_metadata.csv", pilot_ids, extra_resistant, extra_susceptible
    )

    monkeypatch.setattr(script, "DATA_PROCESSED", data_processed)
    monkeypatch.setattr(script, "DATA_RAW_EXPANSION_DIR", raw_expansion_dir)
    monkeypatch.setattr(script, "BVBRCClient", lambda: object())  # never actually used by the fake fetch below

    return {
        "data_processed": data_processed,
        "raw_expansion_dir": raw_expansion_dir,
        "pilot_ids": pilot_ids,
        "extra_resistant": extra_resistant,
        "extra_susceptible": extra_susceptible,
    }


def _fake_fetch_always_succeeds(client, genome_id, expected_contigs=None):
    return CompleteAssemblyResult(
        genome_id=genome_id,
        status=ASSEMBLY_STATUS_COMPLETE,
        contigs=[ContigRecord(sequence_id="c1", header=f">accn|{genome_id}", sequence="ACGTACGT")],
        n_contigs_in_index=1,
        n_contigs_retrieved=1,
        total_length_retrieved=8,
    )


def _fake_fetch_always_fails(client, genome_id, expected_contigs=None):
    return CompleteAssemblyResult(genome_id=genome_id, status=ASSEMBLY_STATUS_INCOMPLETE, errors=["simulated failure"])


# --- deterministic selection + pilot exclusion (via the real CLI path) ---


def test_selection_excludes_pilot_and_is_deterministic(fake_env):
    df1 = script.step1_select_or_load(target_additional=8, seed=1)
    assert set(df1["genome_id"]).isdisjoint(set(fake_env["pilot_ids"]))

    # a SECOND call must reuse the persisted selection exactly (idempotent)
    df2 = script.step1_select_or_load(target_additional=999, seed=999)  # different args, ignored
    assert df1["genome_id"].tolist() == df2["genome_id"].tolist()


# --- resume / idempotent retrieval ---


def test_retrieval_skips_already_retrieved_genomes_on_resume(fake_env, monkeypatch):
    monkeypatch.setattr(script, "fetch_complete_assembly", _fake_fetch_always_succeeds)
    selected_df = script.step1_select_or_load(target_additional=4, seed=1)

    manifest1 = script.step2_retrieve(selected_df, max_failures=20)
    assert (manifest1["status"] == ASSEMBLY_STATUS_COMPLETE).all()

    call_count = {"n": 0}

    def counting_fetch(client, genome_id, expected_contigs=None):
        call_count["n"] += 1
        return _fake_fetch_always_succeeds(client, genome_id, expected_contigs)

    monkeypatch.setattr(script, "fetch_complete_assembly", counting_fetch)
    manifest2 = script.step2_retrieve(selected_df, max_failures=20)
    assert call_count["n"] == 0  # every genome already had a FASTA file - none re-fetched
    assert (manifest2["status"] == "already_present_skipped").all()


# --- failure tracking and max_failures budget ---


def test_failures_are_recorded_and_never_written_as_fasta(fake_env, monkeypatch):
    monkeypatch.setattr(script, "fetch_complete_assembly", _fake_fetch_always_fails)
    selected_df = script.step1_select_or_load(target_additional=4, seed=1)
    manifest = script.step2_retrieve(selected_df, max_failures=20)

    assert (manifest["status"] == ASSEMBLY_STATUS_INCOMPLETE).all()
    assert (manifest["fasta_path"] == "").all()
    for gid in selected_df["genome_id"]:
        assert not (fake_env["raw_expansion_dir"] / f"{gid}.fna").exists()


def test_max_failures_budget_stops_further_attempts_this_run(fake_env, monkeypatch):
    attempted = []

    def fetch_that_counts(client, genome_id, expected_contigs=None):
        attempted.append(genome_id)
        return _fake_fetch_always_fails(client, genome_id, expected_contigs)

    monkeypatch.setattr(script, "fetch_complete_assembly", fetch_that_counts)
    selected_df = script.step1_select_or_load(target_additional=8, seed=1)
    manifest = script.step2_retrieve(selected_df, max_failures=3)

    assert len(attempted) == 3  # stopped after exactly 3 new failures
    assert (manifest["status"] == "skipped_max_failures_reached").sum() == len(selected_df) - 3


# --- prevention of incomplete cohorts being marked complete ---


def test_incomplete_assembly_is_not_qcd_or_marked_usable(fake_env, monkeypatch):
    monkeypatch.setattr(script, "fetch_complete_assembly", _fake_fetch_always_fails)
    selected_df = script.step1_select_or_load(target_additional=4, seed=1)
    manifest = script.step2_retrieve(selected_df, max_failures=20)
    qc_df = script.step3_qc(manifest)
    assert len(qc_df) == 0  # nothing usable, nothing silently QC'd as if it were


# --- genome-ID string preservation through the full pipeline ---


def test_genome_ids_remain_strings_through_selection_and_manifest(fake_env, monkeypatch):
    monkeypatch.setattr(script, "fetch_complete_assembly", _fake_fetch_always_succeeds)
    selected_df = script.step1_select_or_load(target_additional=4, seed=1)
    assert selected_df["genome_id"].map(type).eq(str).all()
    manifest = script.step2_retrieve(selected_df, max_failures=20)
    assert manifest["genome_id"].map(type).eq(str).all()


# --- artifact isolation: pilot untouched, new paths only ---


def test_pilot_selected_cohort_file_is_never_modified(fake_env, monkeypatch):
    pilot_path = fake_env["data_processed"] / "pilot_ciprofloxacin_selected_cohort.csv"
    before = pilot_path.read_text()
    monkeypatch.setattr(script, "fetch_complete_assembly", _fake_fetch_always_succeeds)
    selected_df = script.step1_select_or_load(target_additional=4, seed=1)
    script.step2_retrieve(selected_df, max_failures=20)
    assert pilot_path.read_text() == before


def test_expansion_writes_only_to_new_suffixed_paths(fake_env, monkeypatch):
    monkeypatch.setattr(script, "fetch_complete_assembly", _fake_fetch_always_succeeds)
    selected_df = script.step1_select_or_load(target_additional=4, seed=1)
    manifest = script.step2_retrieve(selected_df, max_failures=20)
    script.step3_qc(manifest)

    data_processed = fake_env["data_processed"]
    assert (data_processed / "cohort_expansion_ciprofloxacin_selected.csv").exists()
    assert (data_processed / "cohort_expansion_ciprofloxacin_retrieval_manifest.csv").exists()
    assert (data_processed / "cohort_expansion_ciprofloxacin_sequence_qc.csv").exists()
    # the raw expansion dir is SEPARATE from any pilot directory
    assert fake_env["raw_expansion_dir"] != data_processed.parent / "raw" / "ciprofloxacin_pilot"
