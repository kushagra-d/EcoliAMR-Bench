"""Integration tests for src/run_course_project_additional_retrieval.py -
no real network calls (fetch_complete_assembly is monkeypatched). Mirrors
test_run_first_retrieval_batch.py's coverage, adapted for this script's
flat (non-batched) genome list, disk-space check, and consecutive-failure
circuit breaker.
"""

from __future__ import annotations

from collections import namedtuple

import pandas as pd
import pytest

import run_course_project_additional_retrieval as script
from bvbrc_recon.sequence_retrieval import ASSEMBLY_STATUS_COMPLETE, ASSEMBLY_STATUS_INCOMPLETE, CompleteAssemblyResult, ContigRecord

_DiskUsage = namedtuple("_DiskUsage", ["total", "used", "free"])


def _plan_df(genome_ids):
    return pd.DataFrame([
        {"genome_id": gid, "status": "planned", "retry_count": 0, "max_retries": 3, "last_error": ""}
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
    monkeypatch.setattr(script, "MANIFEST_PATH", data_processed / "manifest.csv")
    monkeypatch.setattr(script, "SUMMARY_PATH", data_processed / "summary.json")
    monkeypatch.setattr(script, "RETRY_BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(script, "EXPECTED_PLAN_SIZE", len(genome_ids))
    monkeypatch.setattr(script.shutil, "disk_usage", lambda path: _DiskUsage(100 * 1024**3, 0, 10 * 1024**3))

    return {"data_processed": data_processed, "raw_shared": raw_shared, "genome_ids": genome_ids}


def _good_result(genome_id):
    return CompleteAssemblyResult(
        genome_id=genome_id, status=ASSEMBLY_STATUS_COMPLETE,
        contigs=[ContigRecord(sequence_id="c1", header=f">accn|{genome_id}", sequence="ACGTACGT")],
        n_contigs_in_index=1, n_contigs_retrieved=1, total_length_retrieved=8,
    )


def _bad_result(genome_id):
    return CompleteAssemblyResult(genome_id=genome_id, status=ASSEMBLY_STATUS_INCOMPLETE, errors=["simulated failure"])


def test_successful_genomes_marked_complete(fake_env, monkeypatch):
    monkeypatch.setattr(script, "fetch_complete_assembly", lambda client, gid, expected_contigs=None: _good_result(gid))
    script.main()

    plan = pd.read_csv(fake_env["data_processed"] / "plan.csv", dtype={"genome_id": str})
    assert (plan["status"] == "complete").all()
    manifest = pd.read_csv(fake_env["data_processed"] / "manifest.csv", dtype={"genome_id": str})
    assert (manifest["status"] == ASSEMBLY_STATUS_COMPLETE).all()
    assert manifest["qc_passed"].all()
    for gid in fake_env["genome_ids"]:
        path = fake_env["raw_shared"] / f"{gid}.fna"
        assert path.exists()
        assert path.read_text().count(">") == 1


def test_one_genome_failing_does_not_abort_the_run(fake_env, monkeypatch):
    def fetch(client, gid, expected_contigs=None):
        return _bad_result(gid) if gid == "g2" else _good_result(gid)

    monkeypatch.setattr(script, "fetch_complete_assembly", fetch)
    script.main()

    manifest = pd.read_csv(fake_env["data_processed"] / "manifest.csv", dtype={"genome_id": str})
    assert len(manifest) == 3
    g1_g3 = manifest[manifest["genome_id"].isin(["g1", "g3"])]
    assert (g1_g3["status"] == ASSEMBLY_STATUS_COMPLETE).all()
    g2 = manifest[manifest["genome_id"] == "g2"].iloc[0]
    assert g2["status"] == ASSEMBLY_STATUS_INCOMPLETE

    plan = pd.read_csv(fake_env["data_processed"] / "plan.csv", dtype={"genome_id": str})
    g2_plan = plan[plan["genome_id"] == "g2"].iloc[0]
    assert g2_plan["status"] == "failed"  # never silently swapped for a substitute genome


def test_bounded_retries_then_gives_up(fake_env, monkeypatch):
    call_count = {"n": 0}

    def always_fails(client, gid, expected_contigs=None):
        call_count["n"] += 1
        return _bad_result(gid)

    monkeypatch.setattr(script, "fetch_complete_assembly", always_fails)
    script.main()  # only 3 genomes - below CONSECUTIVE_FAILURE_LIMIT, so the circuit breaker doesn't trip here

    plan = pd.read_csv(fake_env["data_processed"] / "plan.csv", dtype={"genome_id": str})
    assert (plan["status"] == "failed").all()
    assert (plan["retry_count"] == script.MAX_RETRIES_PER_GENOME).all()


def test_retry_succeeds_before_exhausting_the_bound(fake_env, monkeypatch):
    attempts_by_genome = {}

    def fails_once_then_succeeds(client, gid, expected_contigs=None):
        attempts_by_genome[gid] = attempts_by_genome.get(gid, 0) + 1
        return _bad_result(gid) if attempts_by_genome[gid] == 1 else _good_result(gid)

    monkeypatch.setattr(script, "fetch_complete_assembly", fails_once_then_succeeds)
    script.main()

    plan = pd.read_csv(fake_env["data_processed"] / "plan.csv", dtype={"genome_id": str})
    assert (plan["status"] == "complete").all()
    assert (plan["retry_count"] == 2).all()


def test_resume_skips_already_retrieved_genomes(fake_env, monkeypatch):
    fake_env["raw_shared"].mkdir(parents=True, exist_ok=True)
    (fake_env["raw_shared"] / "g1.fna").write_text(">accn|g1\nACGT\n")

    call_count = {"n": 0}

    def fetch(client, gid, expected_contigs=None):
        call_count["n"] += 1
        return _good_result(gid)

    monkeypatch.setattr(script, "fetch_complete_assembly", fetch)
    script.main()

    assert call_count["n"] == 2
    manifest = pd.read_csv(fake_env["data_processed"] / "manifest.csv", dtype={"genome_id": str})
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
    assert call_count["n"] == 2


def test_qc_rejected_assembly_is_not_left_on_disk(fake_env, monkeypatch):
    from bvbrc_recon.sequence_qc import AssemblyQC

    monkeypatch.setattr(script, "fetch_complete_assembly", lambda client, gid, expected_contigs=None: _good_result(gid))

    def fake_qc(genome_id, fasta_path):
        qc = AssemblyQC(genome_id=genome_id, fasta_path=str(fasta_path))
        qc.n_contigs = 0
        qc.has_valid_nucleotide_data = False
        return qc

    monkeypatch.setattr(script, "qc_assembly", fake_qc)
    script.main()

    manifest = pd.read_csv(fake_env["data_processed"] / "manifest.csv", dtype={"genome_id": str})
    assert (manifest["status"] == "qc_rejected").all()
    for gid in fake_env["genome_ids"]:
        assert not (fake_env["raw_shared"] / f"{gid}.fna").exists()
    plan = pd.read_csv(fake_env["data_processed"] / "plan.csv", dtype={"genome_id": str})
    assert (plan["status"] == "failed").all()


def test_refuses_plan_with_wrong_row_count(fake_env):
    # fixture set EXPECTED_PLAN_SIZE to 3 (len of the original genome_ids); a
    # plan with a different row count must be refused, not silently resized.
    big_ids = [f"g{i}" for i in range(5)]
    _plan_df(big_ids).to_csv(fake_env["data_processed"] / "plan.csv", index=False)
    with pytest.raises(RuntimeError):
        script.main()


def test_refuses_plan_with_duplicate_genome_ids(fake_env):
    dup_ids = ["g1", "g1", "g2"]
    _plan_df(dup_ids).to_csv(fake_env["data_processed"] / "plan.csv", index=False)
    with pytest.raises(RuntimeError):
        script.main()


def test_consecutive_failure_circuit_breaker_trips(fake_env, monkeypatch):
    genome_ids = [f"g{i}" for i in range(script.CONSECUTIVE_FAILURE_LIMIT + 2)]
    monkeypatch.setattr(script, "EXPECTED_PLAN_SIZE", len(genome_ids))
    _plan_df(genome_ids).to_csv(fake_env["data_processed"] / "plan.csv", index=False)

    call_count = {"n": 0}

    def always_fails(client, gid, expected_contigs=None):
        call_count["n"] += 1
        return _bad_result(gid)

    monkeypatch.setattr(script, "fetch_complete_assembly", always_fails)
    with pytest.raises(script.SystemicFailureError):
        script.main()

    manifest = pd.read_csv(fake_env["data_processed"] / "manifest.csv", dtype={"genome_id": str})
    # stopped at the failure limit, not after processing every genome in the plan
    assert len(manifest) == script.CONSECUTIVE_FAILURE_LIMIT


def test_insufficient_disk_space_stops_before_any_retrieval(fake_env, monkeypatch):
    monkeypatch.setattr(script.shutil, "disk_usage", lambda path: _DiskUsage(100 * 1024**3, 100 * 1024**3, 0))
    monkeypatch.setattr(
        script, "fetch_complete_assembly",
        lambda client, gid, expected_contigs=None: pytest.fail("should never be called"),
    )
    with pytest.raises(script.SystemicFailureError):
        script.main()
