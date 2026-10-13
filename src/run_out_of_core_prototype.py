"""Out-of-core hashed-matrix prototype on the EXISTING 100-genome
ciprofloxacin pilot assemblies, splits, and cohort. Reads only existing
artifacts under data/processed/ and data/raw/ciprofloxacin_pilot/ - never
downloads genomes, never modifies the proposal/spreadsheet, never
overwrites any existing raw/hashed/minimizer artifact.

Key architectural difference from every prior pipeline in this project:
`contigs_loader` below loads ONE genome's FASTA file per call. No dict of
all 100 genomes' contigs is ever built (contrast with
`run_minimizer_experiment.py`'s `contigs_by_genome = {gid: ... for gid in
genome_order}`, which loads everything upfront). Combined with
`streaming_hashed_matrix.build_streaming_hashed_matrix` hashing directly
from each genome's sequence stream (no prerequisite raw k-mer matrix) and
appending each row to disk immediately, peak memory is bounded by one
genome's own data, not by genome count or total k-mer diversity.

k=15 is reused (not re-chosen) to match the existing in-memory
`pilot_ciprofloxacin_kmer_hashed_matrix_d262144.npz` pilot, so this run's
peak-RSS figure is directly comparable to a known, already-measured
full-in-memory baseline for the same (k, n_buckets) pair.

Usage (from the project root, with the project venv active):

    python src/run_out_of_core_prototype.py

Writes (new files only):
    data/processed/out_of_core_ciprofloxacin_rows_k15_d262144.jsonl
    data/processed/out_of_core_ciprofloxacin_failures_k15_d262144.jsonl
    data/processed/out_of_core_ciprofloxacin_config_k15_d262144.json
    data/processed/out_of_core_ciprofloxacin_incremental_model_results_k15_d262144.json
    data/processed/out_of_core_ciprofloxacin_k15_d262144_experiment.log
    reports/out_of_core_prototype.md
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

from bvbrc_recon.feature_hashing import hash_kmer_to_bucket
from bvbrc_recon.incremental_model import evaluate_streaming_hashed_matrix
from bvbrc_recon.kmer_features import canonical_kmer
from bvbrc_recon.streaming_hashed_matrix import (
    DiskBackedRowStore,
    FailureLog,
    ResourceLimits,
    build_streaming_hashed_matrix,
    check_disk_space,
    hash_genome_to_bucket_counts,
)
from bvbrc_recon.streaming_kmer_build import DEFAULT_MEMORY_LIMIT_MB, peak_memory_mb

ANTIBIOTIC = "ciprofloxacin"
SEED = 20261009
K = 15
N_BUCKETS = 2**18  # 262144 - matches existing in-memory d=262144 pilot for comparability
MEMORY_LIMIT_MB = DEFAULT_MEMORY_LIMIT_MB
BATCH_SIZE = 16
MAX_FAILED_ROWS = 5
MIN_FREE_DISK_BYTES = 1_000_000_000

DATA_PROCESSED = Path("data/processed")
DATA_RAW_PILOT_DIR = Path("data/raw") / f"{ANTIBIOTIC}_pilot"
REPORTS_DIR = Path("reports")

PREFIX = f"out_of_core_{ANTIBIOTIC}_k{K}_d{N_BUCKETS}"
ROWS_PATH = DATA_PROCESSED / f"out_of_core_{ANTIBIOTIC}_rows_k{K}_d{N_BUCKETS}.jsonl"
FAILURES_PATH = DATA_PROCESSED / f"out_of_core_{ANTIBIOTIC}_failures_k{K}_d{N_BUCKETS}.jsonl"
CONFIG_PATH = DATA_PROCESSED / f"out_of_core_{ANTIBIOTIC}_config_k{K}_d{N_BUCKETS}.json"
MODEL_RESULTS_PATH = DATA_PROCESSED / f"out_of_core_{ANTIBIOTIC}_incremental_model_results_k{K}_d{N_BUCKETS}.json"
LOG_PATH = DATA_PROCESSED / f"{PREFIX}_experiment.log"
REPORT_PATH = REPORTS_DIR / "out_of_core_prototype.md"


def logline(fh, msg: str) -> None:
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


def independent_fixture_check(k: int, n_buckets: int) -> dict:
    """A small synthetic fixture whose expected canonical-kmer counts are
    derived by an INDEPENDENT loop (not by calling
    `hash_genome_to_bucket_counts`), then mapped through the same
    deterministic hash used in production and compared against the
    production function's actual output. Exercises: canonical k-mer
    collapsing, ambiguous-base skipping, and a too-short contig
    contributing nothing."""
    contigs = ["ACGTACGT", "NNNACGTNNN", "AC"]
    valid_bases = set("ACGT")

    expected_kmer_counts: dict[str, int] = {}
    for contig in contigs:
        seq = contig.upper()
        for i in range(len(seq) - k + 1):
            window = seq[i : i + k]
            if not valid_bases.issuperset(window):
                continue
            canon = canonical_kmer(window)
            expected_kmer_counts[canon] = expected_kmer_counts.get(canon, 0) + 1

    expected_bucket_counts: dict[int, int] = {}
    for kmer, count in expected_kmer_counts.items():
        bucket = hash_kmer_to_bucket(kmer, n_buckets)
        expected_bucket_counts[bucket] = expected_bucket_counts.get(bucket, 0) + count

    actual_bucket_counts = hash_genome_to_bucket_counts(contigs, k, n_buckets)

    passed = actual_bucket_counts == expected_bucket_counts
    return {
        "fixture_contigs": contigs,
        "k": k,
        "n_buckets": n_buckets,
        "expected_canonical_kmer_counts": expected_kmer_counts,
        "expected_bucket_counts": expected_bucket_counts,
        "actual_bucket_counts": actual_bucket_counts,
        "passed": passed,
    }


def contigs_loader(genome_id: str) -> list[str]:
    """Loads exactly ONE genome's contigs per call - never a dict of all
    genomes' contigs. This is what makes extraction out-of-core."""
    fasta_path = DATA_RAW_PILOT_DIR / f"{genome_id}.fna"
    return [str(record.seq) for record in SeqIO.parse(str(fasta_path), "fasta")]


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_fh = open(LOG_PATH, "w")
    logline(log_fh, f"=== out-of-core hashed matrix prototype: k={K}, n_buckets={N_BUCKETS} ===")

    fixture_result = independent_fixture_check(k=4, n_buckets=16)
    logline(log_fh, f"synthetic fixture check: passed={fixture_result['passed']}")
    if not fixture_result["passed"]:
        logline(log_fh, "ABORT: synthetic fixture check failed - refusing to run on real data.")
        CONFIG_PATH.write_text(json.dumps({"fixture_check": fixture_result}, indent=2))
        log_fh.close()
        return

    genome_order = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_genome_order.csv", dtype={"genome_id": str}
    )["genome_id"].tolist()
    selected = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_selected_cohort.csv", dtype={"genome_id": str}
    )
    label_by_genome = dict(zip(selected["genome_id"], selected["label"]))
    random_split = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_random_split.csv", dtype={"genome_id": str}
    )
    lineage_split = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_lineage_split.csv", dtype={"genome_id": str}
    )
    logline(log_fh, f"loaded genome_order ({len(genome_order)} genomes) and both splits - read-only")

    row_store = DiskBackedRowStore(ROWS_PATH, n_buckets=N_BUCKETS)
    failure_log = FailureLog(FAILURES_PATH)
    limits = ResourceLimits(
        memory_limit_mb=MEMORY_LIMIT_MB,
        min_free_disk_bytes=MIN_FREE_DISK_BYTES,
        max_failed_rows=MAX_FAILED_ROWS,
        batch_size=BATCH_SIZE,
    )
    disk_space_ok_before = check_disk_space(ROWS_PATH, MIN_FREE_DISK_BYTES)
    logline(log_fh, f"disk space check before run: ok={disk_space_ok_before}")

    def progress(row_idx, genome_id, n_written, mem_mb):
        if (n_written % 20 == 0) or n_written == 1:
            logline(log_fh, f"  genome row_index={row_idx} ({genome_id}): rows_written={n_written} peak_rss_mb={mem_mb:.0f}")

    t0 = time.perf_counter()
    status1 = build_streaming_hashed_matrix(
        contigs_loader, genome_order, k=K, n_buckets=N_BUCKETS,
        row_store=row_store, failure_log=failure_log, limits=limits, progress_callback=progress,
    )
    elapsed1 = time.perf_counter() - t0
    logline(
        log_fh,
        f"run 1: complete={status1.complete} rows_written={status1.n_rows_written_this_run} "
        f"failures={status1.n_failures_this_run} aborted_reason={status1.aborted_reason} "
        f"elapsed={elapsed1:.1f}s peak_rss_mb={status1.peak_rss_mb:.0f}",
    )

    # --- resumption-safety demonstration: re-run against the SAME store/log ---
    t0 = time.perf_counter()
    status2 = build_streaming_hashed_matrix(
        contigs_loader, genome_order, k=K, n_buckets=N_BUCKETS,
        row_store=row_store, failure_log=failure_log, limits=limits,
    )
    elapsed2 = time.perf_counter() - t0
    logline(
        log_fh,
        f"run 2 (resumption check): rows_written_this_run={status2.n_rows_written_this_run} "
        f"skipped_already_present={status2.n_genomes_skipped_already_present} elapsed={elapsed2:.1f}s",
    )
    resumption_safe = (
        status2.n_rows_written_this_run == 0
        and status2.n_genomes_skipped_already_present == status1.n_rows_written_this_run
        and row_store.n_rows_written() == status1.n_rows_total
    )
    logline(log_fh, f"resumption safe (no duplicate rows, no re-processing): {resumption_safe}")

    # --- reopen the store fresh and consume it strictly in batches, never as one matrix ---
    reopened_store = DiskBackedRowStore(ROWS_PATH, n_buckets=N_BUCKETS)
    total_rows_seen = 0
    total_nnz = 0
    genome_ids_seen: set[str] = set()
    max_batch_rows_seen = 0
    for genome_ids, row_indices, batch_matrix in reopened_store.iter_batches(batch_size=BATCH_SIZE):
        total_rows_seen += batch_matrix.shape[0]
        total_nnz += int(batch_matrix.nnz)
        genome_ids_seen.update(genome_ids)
        max_batch_rows_seen = max(max_batch_rows_seen, batch_matrix.shape[0])
    density = total_nnz / (total_rows_seen * N_BUCKETS) if total_rows_seen else 0.0
    logline(
        log_fh,
        f"reopened-and-batched read: rows={total_rows_seen} nnz={total_nnz:,} density={density:.6%} "
        f"max_batch_rows_materialized={max_batch_rows_seen} (never {total_rows_seen})",
    )

    disk_usage_bytes = ROWS_PATH.stat().st_size + (FAILURES_PATH.stat().st_size if FAILURES_PATH.exists() else 0)

    # --- downstream compatibility: SGDClassifier/partial_fit on existing splits ---
    label_by_row_index = {i: label_by_genome[gid] for i, gid in enumerate(genome_order) if gid in label_by_genome}
    model_results = {}
    for split_name, split_df in [("random", random_split), ("lineage", lineage_split)]:
        genome_to_partition = dict(zip(split_df["genome_id"], split_df["partition"]))
        train_idx = {i for i, gid in enumerate(genome_order) if genome_to_partition.get(gid) == "train"}
        test_idx = {i for i, gid in enumerate(genome_order) if genome_to_partition.get(gid) == "test"}
        result = evaluate_streaming_hashed_matrix(
            reopened_store, train_idx, test_idx, label_by_row_index, seed=SEED, batch_size=BATCH_SIZE, n_epochs=5
        )
        model_results[split_name] = {
            "n_train": result.n_train,
            "n_test": result.n_test,
            "train_class_counts": result.train_class_counts,
            "test_class_counts": result.test_class_counts,
            "accuracy": result.accuracy,
            "roc_auc": result.roc_auc,
            "average_precision": result.average_precision,
            "recall_per_class": result.recall_per_class,
            "confusion_matrix": result.confusion_matrix,
            "warnings": result.warnings,
            "max_batch_rows_held_in_memory": result.max_batch_rows_held_in_memory,
        }
        logline(
            log_fh,
            f"[{split_name}] SGDClassifier(partial_fit) acc={result.accuracy} auc={result.roc_auc} "
            f"recall={result.recall_per_class} max_batch_rows={result.max_batch_rows_held_in_memory}",
        )

    final_peak_rss_mb = max(status1.peak_rss_mb, status2.peak_rss_mb, peak_memory_mb())

    config = {
        "k": K,
        "n_buckets": N_BUCKETS,
        "antibiotic": ANTIBIOTIC,
        "resource_limits": {
            "memory_limit_mb": MEMORY_LIMIT_MB,
            "min_free_disk_bytes": MIN_FREE_DISK_BYTES,
            "max_failed_rows": MAX_FAILED_ROWS,
            "batch_size": BATCH_SIZE,
        },
        "fixture_check": fixture_result,
        "run1": {
            "n_genomes_requested": status1.n_genomes_requested,
            "n_rows_written_this_run": status1.n_rows_written_this_run,
            "n_rows_total": status1.n_rows_total,
            "n_failures_this_run": status1.n_failures_this_run,
            "n_failures_total": status1.n_failures_total,
            "aborted_reason": status1.aborted_reason,
            "peak_rss_mb": status1.peak_rss_mb,
            "elapsed_seconds": elapsed1,
            "complete": status1.complete,
        },
        "run2_resumption_check": {
            "n_rows_written_this_run": status2.n_rows_written_this_run,
            "n_genomes_skipped_already_present": status2.n_genomes_skipped_already_present,
            "elapsed_seconds": elapsed2,
            "resumption_safe_no_duplication": resumption_safe,
        },
        "reopened_batched_read": {
            "n_rows": total_rows_seen,
            "n_features": N_BUCKETS,
            "nnz": total_nnz,
            "density": density,
            "max_batch_rows_materialized_at_once": max_batch_rows_seen,
        },
        "disk_usage_bytes": disk_usage_bytes,
        "final_peak_rss_mb": final_peak_rss_mb,
        "comparison_to_existing_in_memory_pipeline": {
            "note": (
                "pilot_ciprofloxacin_kmer_hashed_matrix_d262144.npz (same k=15, same n_buckets=262144) "
                "was built by first materializing a full raw k=15 matrix in RAM "
                "(peak_rss_mb=7158.17, per pilot_ciprofloxacin_kmer_config_k15_comparison.json) and then "
                "hashing it (peak_rss_mb=7837.27, per pilot_ciprofloxacin_kmer_hashed_config_d262144.json). "
                "This out-of-core run never builds a raw matrix and never holds more than one genome's "
                "sequence plus a <=262144-entry bucket-count dict at a time."
            ),
            "existing_raw_k15_peak_rss_mb": 7158.1696,
            "existing_in_memory_hashed_d262144_peak_rss_mb": 7837.270016,
            "this_run_peak_rss_mb": final_peak_rss_mb,
        },
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    MODEL_RESULTS_PATH.write_text(json.dumps(model_results, indent=2))
    logline(log_fh, f"wrote config to {CONFIG_PATH}")
    logline(log_fh, f"wrote model results to {MODEL_RESULTS_PATH}")
    logline(log_fh, "Done.")
    log_fh.close()


if __name__ == "__main__":
    main()
