"""Retrieves ONLY the 700 genomes selected for the course-project
~1,000-genome cohort (`data/processed/course_project_additional_700_retrieval_plan.csv`,
a fixed, pre-verified list - see `reports/course_project_split_design.md`
and `reports/course_project_cohort_design.md`). Does NOT touch or
execute the original full-cohort retrieval plan
(`full_cohort_retrieval_plan.csv`) - that plan's 8,410 remaining rows
are untouched by this script.

Reuses the existing, already safety-reviewed retrieval/QC/atomic-write
implementation unchanged: `sequence_retrieval.fetch_complete_assembly`
(sequence-identity validated), `sequence_qc.qc_assembly`, and
`sequence_retrieval.write_assembly_fasta_atomic`. Concurrency stays
strictly sequential (bounded at 1), matching every prior successful
retrieval run in this project - this task does not increase
concurrency to finish faster.

Two safety additions specific to this long (~90+ minute), unattended
run, beyond what the single 100-genome batch script needed:
  - a disk-space check before starting and every 50 genomes
  - a consecutive-failure circuit breaker: if
    CONSECUTIVE_FAILURE_LIMIT genomes in a row fail outright (not
    counting the per-genome bounded retries, which already happen
    before a genome counts as "failed"), retrieval stops immediately
    rather than burning through the remaining list blindly - this is
    exactly the "systemic problem" stop condition this task calls for.

The fixed 700-genome list is never regenerated or substituted: it is
read once from the persisted plan CSV (which was itself built and
verified against the proposal's claimed counts before this script was
written) and reused across any resume, so a failed genome is recorded
as failed, never silently swapped for a different one.

Usage (from the project root, with the project venv active):

    python src/run_course_project_additional_retrieval.py

Writes incrementally (so partial progress survives an interruption):
    data/raw/full_cohort_assemblies/<genome_id>.fna  (one per success)
    data/processed/course_project_additional_700_manifest.csv
    data/processed/course_project_additional_700_retrieval_plan.csv  (status column updated in place)
    data/processed/course_project_additional_700_execution_summary.json
"""

from __future__ import annotations

import json
import shutil
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
EXPECTED_PLAN_SIZE = 700
MAX_RETRIES_PER_GENOME = 3
RETRY_BACKOFF_SECONDS = 2.0
CONSECUTIVE_FAILURE_LIMIT = 10
MIN_FREE_DISK_BYTES = 1 * 1024**3  # 1 GiB
DISK_CHECK_INTERVAL = 50

DATA_PROCESSED = Path("data/processed")
DATA_RAW_SHARED = Path("data/raw") / "full_cohort_assemblies"
PLAN_PATH = DATA_PROCESSED / "course_project_additional_700_retrieval_plan.csv"
MANIFEST_PATH = DATA_PROCESSED / "course_project_additional_700_manifest.csv"
SUMMARY_PATH = DATA_PROCESSED / "course_project_additional_700_execution_summary.json"


class SystemicFailureError(RuntimeError):
    pass


def load_genome_contigs_map() -> dict[str, int]:
    genome_contigs: dict[str, int] = {}
    for antibiotic in ANTIBIOTICS:
        records = load_antibiotic_csv(DATA_PROCESSED / f"{antibiotic}_amr_metadata.csv")
        for r in records:
            gid, gc = r.get("genome_id"), r.get("genome_contigs")
            if gid and gc is not None and gid not in genome_contigs:
                genome_contigs[gid] = int(gc)
    return genome_contigs


def check_disk_space() -> int:
    free_bytes = shutil.disk_usage(DATA_RAW_SHARED.parent if DATA_RAW_SHARED.exists() else Path(".")).free
    if free_bytes < MIN_FREE_DISK_BYTES:
        raise SystemicFailureError(
            f"Only {free_bytes / 1024**3:.2f} GiB free disk space remains "
            f"(below the {MIN_FREE_DISK_BYTES / 1024**3:.0f} GiB safety floor) - stopping safely."
        )
    return free_bytes


def main() -> None:
    if not PLAN_PATH.exists():
        raise RuntimeError(f"REFUSING TO PROCEED: {PLAN_PATH} does not exist - the 700-genome plan must be built and verified first.")

    plan = pd.read_csv(PLAN_PATH, dtype={"genome_id": str, "status": str, "last_error": str})
    plan["last_error"] = plan["last_error"].fillna("")
    if len(plan) != EXPECTED_PLAN_SIZE or plan["genome_id"].duplicated().any():
        raise RuntimeError(
            f"REFUSING TO PROCEED: plan has {len(plan)} rows "
            f"(duplicates: {plan['genome_id'].duplicated().sum()}) - expected exactly {EXPECTED_PLAN_SIZE} unique genome IDs."
        )

    print(f"=== Retrieving the course-project additional cohort: {len(plan)} genomes ===")
    print(f"Initial free disk space: {check_disk_space() / 1024**3:.2f} GiB")
    DATA_RAW_SHARED.mkdir(parents=True, exist_ok=True)
    genome_contigs = load_genome_contigs_map()
    client = BVBRCClient()

    manifest_rows = []
    plan = plan.set_index("genome_id")
    run_t0 = time.perf_counter()
    consecutive_failures = 0

    genome_ids = list(plan.index)
    for i, genome_id in enumerate(genome_ids, start=1):
        if i % DISK_CHECK_INTERVAL == 0:
            free = check_disk_space()
            print(f"  [disk check @ {i}/{len(genome_ids)}] {free / 1024**3:.2f} GiB free")

        row = plan.loc[genome_id]
        fasta_path = DATA_RAW_SHARED / f"{genome_id}.fna"

        if row["status"] != "planned":
            print(f"[{i}/{len(genome_ids)}] {genome_id}: skipping - status already '{row['status']}'")
            continue

        if fasta_path.exists():
            print(f"[{i}/{len(genome_ids)}] {genome_id}: already present on disk, reusing (resumed)")
            manifest_rows.append({
                "genome_id": genome_id, "status": "already_present_skipped", "n_contigs_retrieved": None,
                "n_contigs_in_index": None, "n_contigs_expected_from_metadata": genome_contigs.get(genome_id),
                "total_length_retrieved": None, "duplicate_sequence_ids": "", "errors": "",
                "retrieval_seconds": None, "attempts": 0, "fasta_path": str(fasta_path),
                "retrieved_at_utc": None, "qc_passed": None,
            })
            plan.loc[genome_id, "status"] = "complete"
            consecutive_failures = 0
            continue

        expected = genome_contigs.get(genome_id)
        last_error = ""
        result = None
        t0 = time.perf_counter()
        for attempt in range(1, MAX_RETRIES_PER_GENOME + 1):
            try:
                result = fetch_complete_assembly(client, genome_id, expected_contigs=expected)
            except Exception as exc:  # noqa: BLE001 - per-genome isolation: this genome's
                # failure must never abort the rest of the run on its own; the
                # consecutive-failure circuit breaker (below) is what catches a
                # genuinely systemic problem.
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
            plan.loc[genome_id, "status"] = "complete" if qc_passed else "failed"
        else:
            plan.loc[genome_id, "status"] = "failed"

        plan.loc[genome_id, "retry_count"] = attempt
        plan.loc[genome_id, "last_error"] = last_error

        genome_succeeded = result.status == ASSEMBLY_STATUS_COMPLETE and qc_passed
        consecutive_failures = 0 if genome_succeeded else consecutive_failures + 1

        print(f"[{i}/{len(genome_ids)}] {genome_id}: status={result.status} attempts={attempt} "
              f"contigs={result.n_contigs_retrieved}/{result.n_contigs_in_index} qc_passed={qc_passed} "
              f"({elapsed:.1f}s) consecutive_failures={consecutive_failures}")

        manifest_rows.append({
            "genome_id": genome_id,
            "status": result.status if qc_passed is not False else "qc_rejected",
            "n_contigs_retrieved": result.n_contigs_retrieved, "n_contigs_in_index": result.n_contigs_in_index,
            "n_contigs_expected_from_metadata": result.n_contigs_expected_from_metadata,
            "total_length_retrieved": result.total_length_retrieved,
            "duplicate_sequence_ids": ";".join(result.duplicate_sequence_ids),
            "errors": " | ".join(result.errors), "retrieval_seconds": elapsed, "attempts": attempt,
            "fasta_path": str(fasta_path) if genome_succeeded else "",
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "qc_passed": qc_passed,
        })

        if result.status == ASSEMBLY_STATUS_COMPLETE and not qc_passed and fasta_path.exists():
            fasta_path.unlink()

        # incremental persistence - a kill mid-run loses at most the in-flight genome,
        # never previously-completed work, and the plan's status column is the
        # resume source of truth on the next invocation.
        pd.DataFrame(manifest_rows).to_csv(MANIFEST_PATH, index=False)
        plan.reset_index().to_csv(PLAN_PATH, index=False)

        if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
            raise SystemicFailureError(
                f"STOPPING SAFELY: {consecutive_failures} consecutive genome failures "
                f"(last: {genome_id}, error: {last_error}) - this looks like a systemic problem "
                f"(network, endpoint, or storage), not isolated per-genome issues. "
                f"{i}/{len(genome_ids)} genomes attempted so far; manifest and plan are saved "
                f"up to this point and the run can be resumed after the underlying issue is fixed."
            )

    total_elapsed = time.perf_counter() - run_t0
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(MANIFEST_PATH, index=False)
    plan.reset_index().to_csv(PLAN_PATH, index=False)

    final_plan = pd.read_csv(PLAN_PATH, dtype={"genome_id": str})
    n_success = (final_plan["status"] == "complete").sum()
    n_failed = (final_plan["status"] == "failed").sum()
    n_planned_remaining = (final_plan["status"] == "planned").sum()
    n_qc_rejected = int((manifest_df["status"] == "qc_rejected").sum()) if len(manifest_df) else 0

    summary = {
        "n_genomes_in_plan": len(final_plan), "n_successful": int(n_success),
        "n_failed": int(n_failed), "n_qc_rejected": n_qc_rejected,
        "n_planned_remaining_unresolved": int(n_planned_remaining),
        "total_attempts_this_run": int(manifest_df["attempts"].fillna(0).sum()) if len(manifest_df) else 0,
        "total_retries_this_run": int((manifest_df["attempts"].fillna(1) - 1).clip(lower=0).sum()) if len(manifest_df) else 0,
        "total_elapsed_seconds_this_run": total_elapsed,
        "final_free_disk_gib": check_disk_space() / 1024**3,
        "started_at_utc": datetime.fromtimestamp(time.time() - total_elapsed, tz=timezone.utc).isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Run complete: {n_success}/{len(final_plan)} complete, {n_failed} failed, "
          f"{n_qc_rejected} QC-rejected, {n_planned_remaining} still unresolved, {total_elapsed:.1f}s this run ===")
    print(f"wrote {MANIFEST_PATH}, {SUMMARY_PATH}, updated {PLAN_PATH}")


if __name__ == "__main__":
    main()
