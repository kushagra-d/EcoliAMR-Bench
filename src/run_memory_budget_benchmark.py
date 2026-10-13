"""Benchmarks the memory-budgeted `CachedRowReader` across several
cache budgets on the real 100-genome ciprofloxacin pilot (k=15,
n_buckets=262144). Also re-verifies exact equivalence (Step 4) and
records data for the full-scale feasibility estimate (Step 5, written
in the report, not computed here beyond the raw measured numbers this
script captures).

Reuses the ALREADY-BUILT SQLite store - does not rebuild it, does not
re-extract anything, does not download genomes. Reads existing
splits/labels read-only.

Usage (from the project root, with the project venv active, AFTER the
SQLite store from the prior tasks has been built):

    python src/run_memory_budget_benchmark.py

Writes:
    data/processed/memory_budget_benchmark_ciprofloxacin_k15_d262144.json
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize as sk_normalize

from bvbrc_recon.cached_row_reader import CachedRowReader
from bvbrc_recon.incremental_model import evaluate_streaming_hashed_matrix
from bvbrc_recon.sqlite_row_store import SqliteRowStore
from bvbrc_recon.streaming_kmer_build import peak_memory_mb

ANTIBIOTIC = "ciprofloxacin"
SEED = 20261009
K = 15
N_BUCKETS = 2**18
BATCH_SIZE = 16
N_EPOCHS = 50

DATA_PROCESSED = Path("data/processed")
DB_PATH = DATA_PROCESSED / f"out_of_core_v2_{ANTIBIOTIC}_rows_k{K}_d{N_BUCKETS}.db"
OUTPUT_PATH = DATA_PROCESSED / f"memory_budget_benchmark_{ANTIBIOTIC}_k{K}_d{N_BUCKETS}.json"

BUDGETS = [
    {"name": "16MB", "max_cache_bytes": 16_000_000},
    {"name": "64MB", "max_cache_bytes": 64_000_000},
    {"name": "128MB", "max_cache_bytes": 128_000_000},
    {"name": "256MB", "max_cache_bytes": 256_000_000},
]


def load_pilot():
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
    label_by_row_index = {i: label_by_genome[gid] for i, gid in enumerate(genome_order) if gid in label_by_genome}
    return genome_order, label_by_row_index, random_split, lineage_split


def split_indices(genome_order, split_df):
    genome_to_partition = dict(zip(split_df["genome_id"], split_df["partition"]))
    train_idx = {i for i, gid in enumerate(genome_order) if genome_to_partition.get(gid) == "train"}
    test_idx = {i for i, gid in enumerate(genome_order) if genome_to_partition.get(gid) == "test"}
    return train_idx, test_idx


def median(values):
    return statistics.median(values)


def measure_read_path(store, train_idx, max_cache_bytes, n_passes=10, n_reps=3):
    rep_times = []
    reader = None
    for _ in range(n_reps):
        reader = CachedRowReader(store, max_cache_bytes=max_cache_bytes)
        t0 = time.perf_counter()
        for _ in range(n_passes):
            list(reader.read_rows_for_indices(train_idx, batch_size=BATCH_SIZE))
        rep_times.append(time.perf_counter() - t0)
    return {
        "rep_seconds": rep_times,
        "median_seconds": median(rep_times),
        "cache_stats": vars(reader.stats),
        "within_cache_budget": reader.stats.n_bytes_cached <= max_cache_bytes,
    }


def measure_end_to_end(train_idx, test_idx, label_by_row_index, max_cache_bytes, n_reps=2):
    rep_times = []
    last_result = None
    reader = None
    mem_before = peak_memory_mb()
    for _ in range(n_reps):
        store = SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS)
        reader = CachedRowReader(store, max_cache_bytes=max_cache_bytes)
        t0 = time.perf_counter()
        last_result = evaluate_streaming_hashed_matrix(
            reader, train_idx, test_idx, label_by_row_index,
            seed=SEED, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, normalize_rows=True,
        )
        rep_times.append(time.perf_counter() - t0)
    mem_after = peak_memory_mb()
    return {
        "rep_seconds": rep_times,
        "median_seconds": median(rep_times),
        "peak_rss_mb_before": mem_before,
        "peak_rss_mb_after": mem_after,
        "within_cache_budget": reader.stats.n_bytes_cached <= max_cache_bytes,
        "cache_stats": vars(reader.stats),
        "accuracy": last_result.accuracy,
        "roc_auc": last_result.roc_auc,
        "confusion_matrix": last_result.confusion_matrix,
        "max_batch_rows_held_in_memory": last_result.max_batch_rows_held_in_memory,
    }


def measure_disabled_baseline(store, train_idx, test_idx, label_by_row_index, n_passes=10, n_reps=3, n_reps_e2e=2):
    """Direct SQLite reads, caching disabled entirely - the raw
    `SqliteRowStore`, no wrapper at all."""
    rep_times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        for _ in range(n_passes):
            list(store.read_rows_for_indices(train_idx, batch_size=BATCH_SIZE))
        rep_times.append(time.perf_counter() - t0)

    rep_times_e2e = []
    last_result = None
    mem_before = peak_memory_mb()
    for _ in range(n_reps_e2e):
        fresh_store = SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS)
        t0 = time.perf_counter()
        last_result = evaluate_streaming_hashed_matrix(
            fresh_store, train_idx, test_idx, label_by_row_index,
            seed=SEED, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, normalize_rows=True,
        )
        rep_times_e2e.append(time.perf_counter() - t0)
    mem_after = peak_memory_mb()

    return {
        "read_path": {"rep_seconds": rep_times, "median_seconds": median(rep_times)},
        "end_to_end": {
            "rep_seconds": rep_times_e2e,
            "median_seconds": median(rep_times_e2e),
            "peak_rss_mb_before": mem_before,
            "peak_rss_mb_after": mem_after,
            "accuracy": last_result.accuracy,
            "roc_auc": last_result.roc_auc,
            "confusion_matrix": last_result.confusion_matrix,
            "max_batch_rows_held_in_memory": last_result.max_batch_rows_held_in_memory,
        },
    }


def verify_equivalence_all_genomes(store, genome_order, max_cache_bytes):
    all_idx = set(range(len(genome_order)))
    reader = CachedRowReader(store, max_cache_bytes=max_cache_bytes)
    direct = list(store.read_rows_for_indices(all_idx, batch_size=BATCH_SIZE))
    via_reader_pass1 = list(reader.read_rows_for_indices(all_idx, batch_size=BATCH_SIZE))
    via_reader_pass2 = list(reader.read_rows_for_indices(all_idx, batch_size=BATCH_SIZE))

    mismatches = []
    normalized_mismatches = []
    for (gd, rd, md), (g1, r1, m1), (g2, r2, m2) in zip(direct, via_reader_pass1, via_reader_pass2):
        if gd != g1 or gd != g2:
            mismatches.append({"kind": "genome_id", "row_indices": rd})
        if rd != r1 or rd != r2:
            mismatches.append({"kind": "row_index", "row_indices": rd})
        if (md != m1).nnz != 0 or (md != m2).nnz != 0:
            mismatches.append({"kind": "sparse_values", "row_indices": rd})
        nd = sk_normalize(md, norm="l2")
        n1 = sk_normalize(m1, norm="l2")
        if not np.allclose(nd.toarray(), n1.toarray()):
            normalized_mismatches.append({"row_indices": rd})

    return {
        "n_genomes_checked": len(genome_order),
        "n_mismatches": len(mismatches),
        "mismatches": mismatches,
        "n_normalized_row_mismatches": len(normalized_mismatches),
        "second_pass_cache_hits": reader.stats.n_hits,
        "passed": len(mismatches) == 0 and len(normalized_mismatches) == 0,
    }


def main() -> None:
    genome_order, label_by_row_index, random_split, lineage_split = load_pilot()
    store = SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS)
    train_idx, test_idx = split_indices(genome_order, random_split)
    train_idx_lineage, test_idx_lineage = split_indices(genome_order, lineage_split)
    db_size_bytes = DB_PATH.stat().st_size
    one_dense_batch_bytes = None
    for _, _, m in store.read_rows_for_indices(set(range(16)), batch_size=16):
        one_dense_batch_bytes = int(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes)
    train_partition_total_bytes = sum(
        CachedRowReader._csr_bytes(m)
        for _, _, m in store.read_rows_for_indices(train_idx, batch_size=BATCH_SIZE)
    )

    print("=== Step 4: exact equivalence (256MB budget) ===")
    equivalence = verify_equivalence_all_genomes(store, genome_order, max_cache_bytes=256_000_000)
    print(f"passed={equivalence['passed']} n_mismatches={equivalence['n_mismatches']}")

    print("=== disabled (direct SQLite, no cache wrapper) - random split ===")
    disabled = measure_disabled_baseline(store, train_idx, test_idx, label_by_row_index)
    print(f"  read_path median={disabled['read_path']['median_seconds']:.2f}s "
          f"e2e median={disabled['end_to_end']['median_seconds']:.2f}s")

    budget_results = {}
    for cfg in BUDGETS:
        print(f"=== budget {cfg['name']} - random split ===")
        read_path = measure_read_path(store, train_idx, cfg["max_cache_bytes"])
        e2e = measure_end_to_end(train_idx, test_idx, label_by_row_index, cfg["max_cache_bytes"])
        print(f"  read_path median={read_path['median_seconds']:.2f}s cache_stats={read_path['cache_stats']}")
        print(f"  e2e median={e2e['median_seconds']:.2f}s peak_rss_after={e2e['peak_rss_mb_after']:.0f}MB "
              f"within_budget={e2e['within_cache_budget']} cache_stats={e2e['cache_stats']}")
        budget_results[cfg["name"]] = {
            "max_cache_bytes": cfg["max_cache_bytes"],
            "read_path": read_path,
            "end_to_end_random_split": e2e,
        }

    print("=== lineage-split spot check: disabled vs 256MB ===")
    e2e_lineage_disabled = evaluate_streaming_hashed_matrix(
        SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS), train_idx_lineage, test_idx_lineage, label_by_row_index,
        seed=SEED, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, normalize_rows=True,
    )
    t0 = time.perf_counter()
    reader_256 = CachedRowReader(SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS), max_cache_bytes=256_000_000)
    e2e_lineage_256 = evaluate_streaming_hashed_matrix(
        reader_256, train_idx_lineage, test_idx_lineage, label_by_row_index,
        seed=SEED, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, normalize_rows=True,
    )
    lineage_256_seconds = time.perf_counter() - t0
    lineage_spot_check = {
        "disabled_metrics_match_256MB_metrics": (
            e2e_lineage_disabled.accuracy == e2e_lineage_256.accuracy
            and e2e_lineage_disabled.roc_auc == e2e_lineage_256.roc_auc
        ),
        "256MB_seconds": lineage_256_seconds,
        "256MB_cache_stats": vars(reader_256.stats),
    }
    print(f"  metrics match: {lineage_spot_check['disabled_metrics_match_256MB_metrics']}, "
          f"256MB seconds={lineage_256_seconds:.2f}")

    output = {
        "k": K, "n_buckets": N_BUCKETS, "antibiotic": ANTIBIOTIC,
        "db_size_bytes": db_size_bytes,
        "one_dense_16row_batch_bytes": one_dense_batch_bytes,
        "train_partition_total_bytes_random_split": train_partition_total_bytes,
        "equivalence_check": equivalence,
        "disabled_baseline_random_split": disabled,
        "budgets_random_split": budget_results,
        "lineage_split_spot_check": lineage_spot_check,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
