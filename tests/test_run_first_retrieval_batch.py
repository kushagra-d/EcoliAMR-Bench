"""Integration tests for src/run_first_retrieval_batch.py - no real
network calls (fetch_complete_assembly is monkeypatched). Covers bounded
retry, atomic writes, resume behavior, status transitions, per-genome
failure isolation, and QC gating.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import run_first_retrieval_batch as script
from bvbrc_recon.sequence_retrieval import ASSEMBLY_STATUS_COMPLETE, ASSEMBLY_STATUS_INCOMPLETE, CompleteAssemblyResult, ContigRecord


def _plan_df(genome_ids, batch_id=0):
    return pd.DataFrame([
        {"batch_id": batch_id, "genome_id": gid, "antibiotics_needed": "ciprofloxacin",
         "status": "planned", "retry_count": 0, "max_retries": 3, "last_error": ""}
        for gid in genome_ids
    ])


def _write_metadata(path, genome_ids, antibiotic="ciprofloxacin"):
    columns = ["genome_id", "antibiotic", "resistant_phenotype", "genome_contigs", "genome_mlst"]
    rows = [
        {"genome_id": gid, "antibiotic": antibiotic, "resistant_phenotype": "Resistant", "genome_contigs": 1, "genome_mlst": ""}
        for gid in genome_ids
    ]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    data_processed = tmp_path / "data_processed"
    raw_shared = tmp_path / "raw_shared"
    data_processed.mkdir()

    genome_ids = ["g1", "g2", "g3"]
    _plan_df(genome_ids).to_csv(data_processed / "plan.csv", index=False)
    for ab in ["ciprofloxacin", "ampicillin", "gentamicin"]:
        _write_metadata(data_processed / f"{ab}_amr_metadata.csv", genome_ids if ab == "ciprofloxacin" else [])

    monkeypatch.setattr(script, "DATA_PROCESSED", data_processed)
    monkeypatch.setattr(script, "DATA_RAW_SHARED", raw_shared)
    monkeypatch.setattr(script, "PLAN_PATH", data_processed / "plan.csv")
    monkeypatch.setattr(script, "BATCH_MANIFEST_PATH", data_processed / "batch_manifest.csv")
    monkeypatch.setattr(script, "SUMMARY_PATH", data_processed / "summary.json")
    monkeypatch.setattr(script, "RETRY_BACKOFF_SECONDS", 0.0)  # no real sleeping in tests

    return {"data_processed": data_processed, "raw_shared": raw_shared, "genome_ids": genome_ids}


def _good_result(genome_id):
    return CompleteAssemblyResult(
        genome_id=genome_id, status=ASSEMBLY_STATUS_COMPLETE,
        contigs=[ContigRecord(sequence_id="c1", header=f">accn|{genome_id}", sequence="ACGTACGT")],
        n_contigs_in_index=1, n_contigs_retrieved=1, total_length_retrieved=8,
    )


def _bad_result(genome_id):
    return CompleteAssemblyResult(genome_id=genome_id, status=ASSEMBLY_STATUS_INCOMPLETE, errors=["simulated failure"])


# --- accurate status transitions / successful path ---


def test_successful_genomes_marked_complete_in_plan_and_manifest(fake_env, monkeypatch):
    monkeypatch.setattr(script, "fetch_complete_assembly", lambda client, gid, expected_contigs=None: _good_result(gid))
    script.main()

    plan = pd.read_csv(fake_env["data_processed"] / "plan.csv", dtype={"genome_id": str})
    assert (plan["status"] == "complete").all()

    manifest = pd.read_csv(fake_env["data_processed"] / "batch_manifest.csv", dtype={"genome_id": str})
    assert (manifest["status"] == ASSEMBLY_STATUS_COMPLETE).all()
    assert manifest["qc_passed"].all()


def test_successful_assembly_file_is_written_and_parseable(fake_env, monkeypatch):
    monkeypatch.setattr(script, "fetch_complete_assembly", lambda client, gid, expected_contigs=None: _good_result(gid))
    script.main()
    for gid in fake_env["genome_ids"]:
        path = fake_env["raw_shared"] / f"{gid}.fna"
        assert path.exists()
        assert path.read_text().count(">") == 1


# --- per-genome failure isolation ---


def test_one_genome_failing_does_not_abort_the_batch(fake_env, monkeypatch):
    def fetch(client, gid, expected_contigs=None):
        if gid == "g2":
            return _bad_result(gid)
        return _good_result(gid)

    monkeypatch.setattr(script, "fetch_complete_assembly", fetch)
    script.main()

    manifest = pd.read_csv(fake_env["data_processed"] / "batch_manifest.csv", dtype={"genome_id": str})
    assert len(manifest) == 3  # all three genomes were processed, not aborted after g2
    g1_g3 = manifest[manifest["genome_id"].isin(["g1", "g3"])]
    assert (g1_g3["status"] == ASSEMBLY_STATUS_COMPLETE).all()
    g2 = manifest[manifest["genome_id"] == "g2"].iloc[0]
    assert g2["status"] == ASSEMBLY_STATUS_INCOMPLETE


def test_unexpected_exception_for_one_genome_is_isolated(fake_env, monkeypatch):
    def fetch(client, gid, expected_contigs=None):
        if gid == "g2":
            raise RuntimeError("simulated unexpected crash")
        return _good_result(gid)

    monkeypatch.setattr(script, "fetch_complete_assembly", fetch)
    script.main()  # must not raise / must not abort

    manifest = pd.read_csv(fake_env["data_processed"] / "batch_manifest.csv", dtype={"genome_id": str})
    assert len(manifest) == 3
    g1_g3 = manifest[manifest["genome_id"].isin(["g1", "g3"])]
    assert (g1_g3["status"] == ASSEMBLY_STATUS_COMPLETE).all()


# --- bounded retry behavior ---


def test_bounded_retries_then_gives_up(fake_env, monkeypatch):
    call_count = {"n": 0}

    def always_fails(client, gid, expected_contigs=None):
        call_count["n"] += 1
        return _bad_result(gid)

    monkeypatch.setattr(script, "fetch_complete_assembly", always_fails)
    script.main()

    # 3 genomes x MAX_RETRIES_PER_GENOME(3) attempts each = 9 calls
    assert call_count["n"] == 3 * script.MAX_RETRIES_PER_GENOME

    plan = pd.read_csv(fake_env["data_processed"] / "plan.csv", dtype={"genome_id": str})
    assert (plan["status"] == "failed").all()
    assert (plan["retry_count"] == script.MAX_RETRIES_PER_GENOME).all()


def test_retry_succeeds_before_exhausting_the_bound(fake_env, monkeypatch):
    attempts_by_genome = {}

    def fails_once_then_succeeds(client, gid, expected_contigs=None):
        attempts_by_genome[gid] = attempts_by_genome.get(gid, 0) + 1
        if attempts_by_genome[gid] == 1:
            return _bad_result(gid)
        return _good_result(gid)

    monkeypatch.setattr(script, "fetch_complete_assembly", fails_once_then_succeeds)
    script.main()

    plan = pd.read_csv(fake_env["data_processed"] / "plan.csv", dtype={"genome_id": str})
    assert (plan["status"] == "complete").all()
    assert (plan["retry_count"] == 2).all()  # succeeded on the 2nd attempt


# --- resume behavior ---


def test_resume_skips_already_retrieved_genomes(fake_env, monkeypatch):
    fake_env["raw_shared"].mkdir(parents=True, exist_ok=True)
    (fake_env["raw_shared"] / "g1.fna").write_text(">accn|g1\nACGT\n")

    call_count = {"n": 0}

    def fetch(client, gid, expected_contigs=None):
        call_count["n"] += 1
        return _good_result(gid)

    monkeypatch.setattr(script, "fetch_complete_assembly", fetch)
    script.main()

    assert call_count["n"] == 2  # g1 was skipped (already on disk), only g2/g3 fetched
    manifest = pd.read_csv(fake_env["data_processed"] / "batch_manifest.csv", dtype={"genome_id": str})
    g1_row = manifest[manifest["genome_id"] == "g1"].iloc[0]
    assert g1_row["status"] == "already_present_skipped"


def test_resume_skips_rows_not_in_planned_status(fake_env, monkeypatch):
    plan_path = fake_env["data_processed"] / "plan.csv"
    plan = pd.read_csv(plan_path, dtype={"genome_id": str})
    plan.loc[plan["genome_id"] == "g1", "status"] = "complete"
    plan.to_csv(plan_path, index=False)

    call_count = {"n": 0}

    def fetch(client, gid, expected_contigs=None):
        call_count["n"] += 1
        return _good_result(gid)

    monkeypatch.setattr(script, "fetch_complete_assembly", fetch)
    script.main()
    assert call_count["n"] == 2  # g1 already resolved, not re-fetched


# --- QC validation / rejection, and cleanup of rejected files ---


def test_qc_rejected_assembly_is_not_left_on_disk(fake_env, monkeypatch):
    from bvbrc_recon.sequence_qc import AssemblyQC

    monkeypatch.setattr(script, "fetch_complete_assembly", lambda client, gid, expected_contigs=None: _good_result(gid))

    def fake_qc(genome_id, fasta_path):
        qc = AssemblyQC(genome_id=genome_id, fasta_path=str(fasta_path))
        qc.n_contigs = 0  # simulate a QC failure (structurally invalid)
        qc.has_valid_nucleotide_data = False
        return qc

    monkeypatch.setattr(script, "qc_assembly", fake_qc)
    script.main()

    manifest = pd.read_csv(fake_env["data_processed"] / "batch_manifest.csv", dtype={"genome_id": str})
    assert (manifest["status"] == "qc_rejected").all()
    for gid in fake_env["genome_ids"]:
        assert not (fake_env["raw_shared"] / f"{gid}.fna").exists()

    plan = pd.read_csv(fake_env["data_processed"] / "plan.csv", dtype={"genome_id": str})
    assert (plan["status"] == "failed").all()


# --- batch-size guard ---


def test_refuses_to_process_an_oversized_batch(fake_env):
    big_ids = [f"g{i}" for i in range(150)]
    _plan_df(big_ids).to_csv(fake_env["data_processed"] / "plan.csv", index=False)
    with pytest.raises(RuntimeError):
        script.main()
