"""Validates the new SQLite-backed row store (`sqlite_row_store.py`)
against the real 100-genome ciprofloxacin pilot: exact row-level
agreement with the existing in-memory hashed matrix, storage/time/RSS/
lookup-performance comparison against the JSONL store from the prior
out-of-core prototype task, bounded-memory batch reads after reopening,
and incremental-model evaluation on both existing splits using FIXED,
UNTUNED settings (this task explicitly does not tune the classifier).

Reads only existing, already-saved artifacts (genome order, splits,
cohort, raw FASTA, and the existing hashed-matrix .npz/JSONL store) -
never modifies them. Writes only to new `out_of_core_v2_*` paths.

Usage (from the project root, with the project venv active):

    python src/run_sqlite_row_store_validation.py

Writes:
    data/processed/out_of_core_v2_ciprofloxacin_rows_k15_d262144.db
    data/processed/out_of_core_v2_ciprofloxacin_failures_k15_d262144.jsonl
    data/processed/out_of_core_v2_ciprofloxacin_config_k15_d262144.json
    data/processed/out_of_core_v2_ciprofloxacin_incremental_model_results_k15_d262144.json
    data/processed/out_of_core_v2_ciprofloxacin_k15_d262144_experiment.log
    reports/disk_backed_store_validation.md
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy import sparse

from bvbrc_recon.incremental_model import evaluate_streaming_hashed_matrix
from bvbrc_recon.sqlite_row_store import (
    FailureLog,
    ResourceLimits,
    SqliteRowStore,
    build_sqlite_backed_matrix,
)
from bvbrc_recon.streaming_hashed_matrix import DiskBackedRowStore
from bvbrc_recon.streaming_kmer_build import DEFAULT_MEMORY_LIMIT_MB, peak_memory_mb

ANTIBIOTIC = "ciprofloxacin"
SEED = 20261009
K = 15
N_BUCKETS = 2**18
MEMORY_LIMIT_MB = DEFAULT_MEMORY_LIMIT_MB
BATCH_SIZE = 16
MAX_FAILED_ROWS = 5
MIN_FREE_DISK_BYTES = 1_000_000_000

DATA_PROCESSED = Path("data/processed")
DATA_RAW_PILOT_DIR = Path("data/raw") / f"{ANTIBIOTIC}_pilot"
REPORTS_DIR = Path("reports")

PREFIX = f"out_of_core_v2_{ANTIBIOTIC}_k{K}_d{N_BUCKETS}"
DB_PATH = DATA_PROCESSED / f"out_of_core_v2_{ANTIBIOTIC}_rows_k{K}_d{N_BUCKETS}.db"
FAILURES_PATH = DATA_PROCESSED / f"out_of_core_v2_{ANTIBIOTIC}_failures_k{K}_d{N_BUCKETS}.jsonl"
CONFIG_PATH = DATA_PROCESSED / f"out_of_core_v2_{ANTIBIOTIC}_config_k{K}_d{N_BUCKETS}.json"
MODEL_RESULTS_PATH = DATA_PROCESSED / f"out_of_core_v2_{ANTIBIOTIC}_incremental_model_results_k{K}_d{N_BUCKETS}.json"
LOG_PATH = DATA_PROCESSED / f"{PREFIX}_experiment.log"

# the prior task's JSONL artifact - read-only, used for the size/lookup comparison
EXISTING_JSONL_ROWS_PATH = DATA_PROCESSED / f"out_of_core_{ANTIBIOTIC}_rows_k{K}_d{N_BUCKETS}.jsonl"
EXISTING_HASHED_MATRIX_PATH = DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_hashed_matrix_d{N_BUCKETS}.npz"
REPORT_PATH = REPORTS_DIR / "disk_backed_store_validation.md"


def logline(fh, msg: str) -> None:
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


def contigs_loader(genome_id: str) -> list[str]:
    fasta_path = DATA_RAW_PILOT_DIR / f"{genome_id}.fna"
    return [str(record.seq) for record in SeqIO.parse(str(fasta_path), "fasta")]


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_fh = open(LOG_PATH, "w")
    logline(log_fh, f"=== sqlite row store validation: k={K}, n_buckets={N_BUCKETS} ===")

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

    # --- build the new store ---
    row_store = SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS)
    failure_log = FailureLog(FAILURES_PATH)
    limits = ResourceLimits(
        memory_limit_mb=MEMORY_LIMIT_MB,
        min_free_disk_bytes=MIN_FREE_DISK_BYTES,
        max_failed_rows=MAX_FAILED_ROWS,
        batch_size=BATCH_SIZE,
    )

    def progress(row_idx, genome_id, n_written, mem_mb):
        if (n_written % 20 == 0) or n_written == 1:
            logline(log_fh, f"  genome row_index={row_idx} ({genome_id}): rows_written={n_written} peak_rss_mb={mem_mb:.0f}")

    t0 = time.perf_counter()
    status = build_sqlite_backed_matrix(
        contigs_loader, genome_order, k=K, n_buckets=N_BUCKETS,
        row_store=row_store, failure_log=failure_log, limits=limits, progress_callback=progress,
    )
    elapsed = time.perf_counter() - t0
    logline(
        log_fh,
        f"build: complete={status.complete} rows_written={status.n_rows_written_this_run} "
        f"failures={status.n_failures_this_run} elapsed={elapsed:.1f}s peak_rss_mb={status.peak_rss_mb:.0f}",
    )

    # --- exact row-level agreement with the EXISTING in-memory hashed matrix ---
    # Established in the prior task: pilot_ciprofloxacin_kmer_hashed_matrix_d262144.npz's
    # row i corresponds to genome_order[i] exactly (that build completed with
    # n_genomes_processed == n_requested == len(genome_order), no reordering).
    existing_matrix = sparse.load_npz(EXISTING_HASHED_MATRIX_PATH).tocsr()
    row_mismatches = []
    for row_index in range(len(genome_order)):
        genome_id, cols, vals = row_store.get_row(row_index)
        assert genome_id == genome_order[row_index]
        new_row = dict(zip(cols.tolist(), vals.tolist()))

        existing_row = existing_matrix.getrow(row_index).tocoo()
        existing_dict = dict(zip(existing_row.col.tolist(), existing_row.data.tolist()))

        if new_row != existing_dict:
            row_mismatches.append(row_index)

    exact_row_agreement = len(row_mismatches) == 0
    logline(
        log_fh,
        f"exact row-level agreement with existing hashed matrix: {exact_row_agreement} "
        f"({len(genome_order) - len(row_mismatches)}/{len(genome_order)} rows match exactly, "
        f"mismatches at row_index={row_mismatches})",
    )

    # --- storage / lookup-performance comparison against the prior JSONL store ---
    new_db_bytes = DB_PATH.stat().st_size
    old_jsonl_exists = EXISTING_JSONL_ROWS_PATH.exists()
    comparison = {"old_jsonl_available_for_comparison": old_jsonl_exists}
    if old_jsonl_exists:
        old_jsonl_bytes = EXISTING_JSONL_ROWS_PATH.stat().st_size

        t0 = time.perf_counter()
        old_store = DiskBackedRowStore(EXISTING_JSONL_ROWS_PATH, n_buckets=N_BUCKETS)
        old_ids = old_store.existing_genome_ids()
        old_full_scan_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        new_ids = row_store.existing_genome_ids()
        new_full_scan_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        for gid in genome_order:
            row_store.row_exists(gid)
        new_indexed_lookup_seconds = time.perf_counter() - t0

        comparison.update(
            {
                "old_jsonl_bytes": old_jsonl_bytes,
                "new_db_bytes": new_db_bytes,
                "size_ratio_new_over_old": new_db_bytes / old_jsonl_bytes,
                "old_full_file_scan_seconds_for_100_genome_ids": old_full_scan_seconds,
                "new_full_table_scan_seconds_for_100_genome_ids": new_full_scan_seconds,
                "new_100_indexed_point_lookups_seconds": new_indexed_lookup_seconds,
                "old_genome_ids_match_new": old_ids == new_ids,
            }
        )
        logline(
            log_fh,
            f"storage: old_jsonl={old_jsonl_bytes/1e6:.1f}MB new_db={new_db_bytes/1e6:.1f}MB "
            f"ratio={new_db_bytes/old_jsonl_bytes:.3f}",
        )
        logline(
            log_fh,
            f"lookup: old full-file scan={old_full_scan_seconds*1000:.2f}ms, "
            f"new full-table scan={new_full_scan_seconds*1000:.2f}ms, "
            f"new 100 indexed point lookups={new_indexed_lookup_seconds*1000:.2f}ms",
        )
    else:
        comparison["new_db_bytes"] = new_db_bytes
        logline(log_fh, "WARNING: prior JSONL artifact not found - size/lookup comparison skipped, not fabricated.")

    # --- reopen fresh, consume strictly in bounded batches ---
    reopened_store = SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS)
    total_rows_seen = 0
    total_nnz = 0
    max_batch_rows_seen = 0
    for genome_ids, row_indices, batch_matrix in reopened_store.iter_batches(batch_size=BATCH_SIZE):
        total_rows_seen += batch_matrix.shape[0]
        total_nnz += int(batch_matrix.nnz)
        max_batch_rows_seen = max(max_batch_rows_seen, batch_matrix.shape[0])
    density = total_nnz / (total_rows_seen * N_BUCKETS) if total_rows_seen else 0.0
    logline(
        log_fh,
        f"reopened-and-batched read: rows={total_rows_seen} nnz={total_nnz:,} density={density:.6%} "
        f"max_batch_rows_materialized={max_batch_rows_seen}",
    )

    # --- model evaluation: fixed, documented, UNTUNED settings (not this task's job to tune) ---
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
            f"recall={result.recall_per_class}",
        )

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
        "build": {
            "n_genomes_requested": status.n_genomes_requested,
            "n_rows_written_this_run": status.n_rows_written_this_run,
            "n_rows_total": status.n_rows_total,
            "n_failures_total": status.n_failures_total,
            "aborted_reason": status.aborted_reason,
            "peak_rss_mb": status.peak_rss_mb,
            "elapsed_seconds": elapsed,
            "complete": status.complete,
        },
        "exact_row_level_agreement_with_existing_hashed_matrix": {
            "passed": exact_row_agreement,
            "n_rows_compared": len(genome_order),
            "n_mismatches": len(row_mismatches),
            "mismatched_row_indices": row_mismatches,
        },
        "storage_and_lookup_comparison_vs_jsonl": comparison,
        "reopened_batched_read": {
            "n_rows": total_rows_seen,
            "n_features": N_BUCKETS,
            "nnz": total_nnz,
            "density": density,
            "max_batch_rows_materialized_at_once": max_batch_rows_seen,
        },
        "final_peak_rss_mb": max(status.peak_rss_mb, peak_memory_mb()),
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    MODEL_RESULTS_PATH.write_text(json.dumps(model_results, indent=2))
    logline(log_fh, f"wrote config to {CONFIG_PATH}")
    logline(log_fh, f"wrote model results to {MODEL_RESULTS_PATH}")
    logline(log_fh, "Done.")
    log_fh.close()


if __name__ == "__main__":
    main()
