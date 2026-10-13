"""Executes ONLY the first pending batch (`batch_id == 0`) of the
full-cohort retrieval plan (`full_cohort_retrieval_plan.csv`, built by
`run_full_cohort_planning.py`) - never any later batch. Reuses the
existing, now-safety-reviewed `sequence_retrieval.fetch_complete_assembly`
(sequence-identity validated this task) and `sequence_qc.qc_assembly`
unchanged.

CONCURRENCY: strictly sequential (bounded at 1), matching every prior
successful retrieval run in this project (the 100-genome pilot and the
100-genome expansion, both 0% failure). This is a deliberate choice,
not an oversight: introducing real thread/async concurrency against a
shared public endpoint would be a new, unvalidated risk (rate-limit
pressure, harder-to-diagnose partial failures), not a "defect fix" this
task's safety review calls for - "bounded concurrency appropriate for
the existing endpoint" is satisfied by keeping the bound at the value
already proven safe.

Writes retrieved assemblies to `data/raw/full_cohort_assemblies/` - a
single shared directory, deliberately NOT antibiotic-named, since one
assembly file serves every antibiotic that genome happens to be
eligible for (the same principle behind this task's cross-antibiotic
availability fix in the planning task).

Usage (from the project root, with the project venv active):

    python src/run_first_retrieval_batch.py

Writes:
    data/raw/full_cohort_assemblies/<genome_id>.fna  (one per success)
    data/processed/full_cohort_batch_0_retrieval_manifest.csv
    data/processed/full_cohort_batch_0_execution_summary.json
    data/processed/full_cohort_retrieval_plan.csv  (batch 0 rows updated in place)
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bvbrc_recon.client import BVBRCClient
from bvbrc_recon.csv_loader import load_antibiotic_csv
from bvbrc_recon.sequence_qc import qc_assembly
from bvbrc_recon.sequence_retrieval import (
    ASSEMBLY_STATUS_COMPLETE,
    CompleteAssemblyResult,
    fetch_complete_assembly,
    write_assembly_fasta_atomic,
)

ANTIBIOTICS = ["ciprofloxacin", "ampicillin", "gentamicin"]
TARGET_BATCH_ID = 0
MAX_RETRIES_PER_GENOME = 3
RETRY_BACKOFF_SECONDS = 2.0

DATA_PROCESSED = Path("data/processed")
DATA_RAW_SHARED = Path("data/raw") / "full_cohort_assemblies"
PLAN_PATH = DATA_PROCESSED / "full_cohort_retrieval_plan.csv"
BATCH_MANIFEST_PATH = DATA_PROCESSED / f"full_cohort_batch_{TARGET_BATCH_ID}_retrieval_manifest.csv"
SUMMARY_PATH = DATA_PROCESSED / f"full_cohort_batch_{TARGET_BATCH_ID}_execution_summary.json"


def load_genome_contigs_map() -> dict[str, int]:
    """Merged across all three antibiotics' metadata - genome_contigs is
    a property of the assembly, not of any one antibiotic's phenotype
    test, so the first non-null value seen for a genome_id is used."""
    genome_contigs: dict[str, int] = {}
    for antibiotic in ANTIBIOTICS:
        records = load_antibiotic_csv(DATA_PROCESSED / f"{antibiotic}_amr_metadata.csv")
        for r in records:
            gid, gc = r.get("genome_id"), r.get("genome_contigs")
            if gid and gc is not None and gid not in genome_contigs:
                genome_contigs[gid] = int(gc)
    return genome_contigs


def main() -> None:
    plan = pd.read_csv(PLAN_PATH, dtype={"genome_id": str, "status": str, "last_error": str})
    plan["last_error"] = plan["last_error"].fillna("")
    batch = plan[plan["batch_id"] == TARGET_BATCH_ID].copy()
    if len(batch) == 0:
        print(f"No rows found for batch_id={TARGET_BATCH_ID} - nothing to do.")
        return
    if len(batch) > 100:
        raise RuntimeError(f"REFUSING TO PROCEED: batch {TARGET_BATCH_ID} has {len(batch)} genomes, exceeding the 100-genome bound.")
    if not (batch["status"] == "planned").all():
        print(f"WARNING: batch {TARGET_BATCH_ID} contains rows not in 'planned' status - "
              f"this run will still only process batch {TARGET_BATCH_ID}, and will skip any "
              f"row already resolved (status not 'planned').")

    print(f"=== Executing batch {TARGET_BATCH_ID}: {len(batch)} genomes ===")
    DATA_RAW_SHARED.mkdir(parents=True, exist_ok=True)
    genome_contigs = load_genome_contigs_map()
    client = BVBRCClient()

    manifest_rows = []
    plan = plan.set_index(["batch_id", "genome_id"])
    run_t0 = time.perf_counter()

    for i, row in enumerate(batch.itertuples(), start=1):
        genome_id = row.genome_id
        fasta_path = DATA_RAW_SHARED / f"{genome_id}.fna"
        key = (TARGET_BATCH_ID, genome_id)

        if row.status != "planned":
            print(f"[{i}/{len(batch)}] {genome_id}: skipping - status already '{row.status}'")
            continue

        if fasta_path.exists():
            # resumable: a genome whose file already exists from an
            # EARLIER interrupted run of THIS SAME batch is reused, not
            # re-fetched - write_assembly_fasta_atomic guarantees that
            # any file at this path is either complete or absent, never
            # a valid-looking partial.
            print(f"[{i}/{len(batch)}] {genome_id}: already present on disk, reusing (resumed)")
            manifest_rows.append({
                "genome_id": genome_id, "status": "already_present_skipped", "n_contigs_retrieved": None,
                "n_contigs_in_index": None, "n_contigs_expected_from_metadata": genome_contigs.get(genome_id),
                "total_length_retrieved": None, "duplicate_sequence_ids": "", "errors": "",
                "retrieval_seconds": None, "attempts": 0, "fasta_path": str(fasta_path),
                "retrieved_at_utc": None, "qc_passed": None,
            })
            plan.loc[key, "status"] = "complete"
            continue

        expected = genome_contigs.get(genome_id)
        last_error = ""
        result = None
        t0 = time.perf_counter()
        for attempt in range(1, MAX_RETRIES_PER_GENOME + 1):
            try:
                result = fetch_complete_assembly(client, genome_id, expected_contigs=expected)
            except Exception as exc:  # noqa: BLE001 - per-genome isolation: this genome's
                # failure must never abort the rest of the batch, whatever the exact cause.
                last_error = f"{type(exc).__name__}: {exc}"
                result = CompleteAssemblyResult(genome_id=genome_id, status="unexpected_exception", errors=[last_error])
                if attempt < MAX_RETRIES_PER_GENOME:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue
            if result.status == ASSEMBLY_STATUS_COMPLETE:
                break
            last_error = " | ".join(result.errors) if result.errors else result.status
            if attempt < MAX_RETRIES_PER_GENOME:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        elapsed = time.perf_counter() - t0

        qc_passed = None
        if result.status == ASSEMBLY_STATUS_COMPLETE:
            write_assembly_fasta_atomic(fasta_path, result.contigs)
            qc = qc_assembly(genome_id, fasta_path)
            qc_passed = qc.has_valid_nucleotide_data and qc.n_contigs > 0 and qc.duplicate_sequence_ids == []
            if not qc_passed:
                last_error = f"QC rejected: warnings={qc.warnings}, error={qc.error}"
            plan.loc[key, "status"] = "complete" if qc_passed else "failed"
        else:
            plan.loc[key, "status"] = "failed"

        plan.loc[key, "retry_count"] = attempt
        plan.loc[key, "last_error"] = last_error

        print(f"[{i}/{len(batch)}] {genome_id}: status={result.status} attempts={attempt} "
              f"contigs={result.n_contigs_retrieved}/{result.n_contigs_in_index} qc_passed={qc_passed} ({elapsed:.1f}s)")

        manifest_rows.append({
            "genome_id": genome_id,
            "status": result.status if qc_passed is not False else "qc_rejected",
            "n_contigs_retrieved": result.n_contigs_retrieved, "n_contigs_in_index": result.n_contigs_in_index,
            "n_contigs_expected_from_metadata": result.n_contigs_expected_from_metadata,
            "total_length_retrieved": result.total_length_retrieved,
            "duplicate_sequence_ids": ";".join(result.duplicate_sequence_ids),
            "errors": " | ".join(result.errors), "retrieval_seconds": elapsed, "attempts": attempt,
            "fasta_path": str(fasta_path) if (result.status == ASSEMBLY_STATUS_COMPLETE and qc_passed) else "",
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "qc_passed": qc_passed,
        })

        # a QC-rejected file must not remain on disk looking "available"
        if result.status == ASSEMBLY_STATUS_COMPLETE and not qc_passed and fasta_path.exists():
            fasta_path.unlink()

    total_elapsed = time.perf_counter() - run_t0
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(BATCH_MANIFEST_PATH, index=False)

    plan = plan.reset_index()
    plan.to_csv(PLAN_PATH, index=False)

    n_success = manifest_df["status"].isin([ASSEMBLY_STATUS_COMPLETE, "already_present_skipped"]).sum()
    n_qc_rejected = (manifest_df["status"] == "qc_rejected").sum()
    n_failed = len(manifest_df) - n_success - n_qc_rejected
    total_attempts = manifest_df["attempts"].fillna(0).sum()
    total_retries = (manifest_df["attempts"].fillna(1) - 1).clip(lower=0).sum()

    summary = {
        "batch_id": TARGET_BATCH_ID, "n_genomes_in_batch": len(batch),
        "n_attempted": int((manifest_df["attempts"].fillna(0) > 0).sum()),
        "n_successful": int(n_success), "n_qc_rejected": int(n_qc_rejected), "n_failed": int(n_failed),
        "total_attempts": int(total_attempts), "total_retries": int(total_retries),
        "total_elapsed_seconds": total_elapsed,
        "started_at_utc": datetime.fromtimestamp(time.time() - total_elapsed, tz=timezone.utc).isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Batch {TARGET_BATCH_ID} complete: {n_success} successful, {n_qc_rejected} QC-rejected, "
          f"{n_failed} failed, {total_elapsed:.1f}s total ===")
    print(f"wrote {BATCH_MANIFEST_PATH}, {SUMMARY_PATH}, updated {PLAN_PATH}")


if __name__ == "__main__":
    main()
