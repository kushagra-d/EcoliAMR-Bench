"""Fair comparison of the hybrid disk+memory reader against the three
existing data-access strategies on the real 100-genome ciprofloxacin
pilot (k=15, n_buckets=262144, random split, 50 epochs, same model
configuration throughout).

Critically, the hybrid strategy is measured in BOTH regimes explicitly,
never comparing a warmed cache to a cold baseline without labeling both:
  - "hybrid_cold": the disk tier does not exist yet - this run pays the
    one-time build cost (SQL read + reconstruct + npz compress + write)
    for every batch, same as `PersistentBatchCache`'s cold build.
  - "hybrid_warm_fresh_instance": the disk tier was ALREADY built by a
    PRIOR run; this run constructs a brand-new `HybridTrainingReader`
    instance (simulating a fresh process attaching to the same
    `cache_dir` - e.g. a resumed or repeated training run) and measures
    how fast it gets to full speed. This is the hybrid design's actual
    value proposition: `NonEvictingCachedRowReader` has NO persistent
    state across processes at all, so a new process always pays the
    full SQL+reconstruct cost on its first pass; the hybrid's new
    process only pays cheap `load_npz` decompression on its first pass,
    because the disk tier survives from the prior run.

Reuses the ALREADY-BUILT SQLite store - does not rebuild it, does not
re-extract anything, does not download genomes. Model, labels, splits,
and feature mapping are NOT changed.

Usage (from the project root, with the project venv active, AFTER the
SQLite store from the prior tasks has been built):

    python src/run_hybrid_pipeline_benchmark.py

Writes:
    data/processed/hybrid_pipeline_benchmark_ciprofloxacin_k15_d262144.json
"""

from __future__ import annotations

import json
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize as sk_normalize

from bvbrc_recon.cached_row_reader import NonEvictingCachedRowReader
from bvbrc_recon.epoch_batch_cache import MultiPartitionPersistentReader, PersistentBatchCache
from bvbrc_recon.hybrid_training_reader import HybridTrainingReader, MultiPartitionHybridReader
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
OUTPUT_PATH = DATA_PROCESSED / f"hybrid_pipeline_benchmark_{ANTIBIOTIC}_k{K}_d{N_BUCKETS}.json"
HYBRID_CACHE_DIR = DATA_PROCESSED / f"hybrid_cache_{ANTIBIOTIC}_k{K}_d{N_BUCKETS}"
PERSISTENT_CACHE_DIR = DATA_PROCESSED / f"epoch_batch_cache_{ANTIBIOTIC}_k{K}_d{N_BUCKETS}"  # from the prior task


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


def run_once(reader, train_idx, test_idx, label_by_row_index):
    mem_before = peak_memory_mb()
    t0 = time.perf_counter()
    result = evaluate_streaming_hashed_matrix(
        reader, train_idx, test_idx, label_by_row_index,
        seed=SEED, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, normalize_rows=True,
    )
    elapsed = time.perf_counter() - t0
    mem_after = peak_memory_mb()
    return elapsed, result, mem_before, mem_after


def measure_disabled(train_idx, test_idx, label_by_row_index, n_reps=2):
    rep_times = []
    last_result = None
    mem_before = peak_memory_mb()
    for _ in range(n_reps):
        store = SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS)
        elapsed, last_result, _, mem_after = run_once(store, train_idx, test_idx, label_by_row_index)
        rep_times.append(elapsed)
    return {
        "rep_seconds": rep_times, "median_seconds": median(rep_times),
        "peak_rss_mb_before": mem_before, "peak_rss_mb_after": mem_after,
        "accuracy": last_result.accuracy, "roc_auc": last_result.roc_auc,
        "confusion_matrix": last_result.confusion_matrix,
    }


def measure_non_evicting(train_idx, test_idx, label_by_row_index, max_cache_bytes=256_000_000, n_reps=3):
    rep_times = []
    last_result = None
    reader = None
    mem_before = peak_memory_mb()
    for _ in range(n_reps):
        reader = NonEvictingCachedRowReader(SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS), max_cache_bytes=max_cache_bytes)
        elapsed, last_result, _, mem_after = run_once(reader, train_idx, test_idx, label_by_row_index)
        rep_times.append(elapsed)
    return {
        "rep_seconds": rep_times, "median_seconds": median(rep_times),
        "peak_rss_mb_before": mem_before, "peak_rss_mb_after": mem_after,
        "accuracy": last_result.accuracy, "roc_auc": last_result.roc_auc,
        "confusion_matrix": last_result.confusion_matrix,
        "cache_stats": vars(reader.stats),
    }


def measure_persistent(train_idx, test_idx, label_by_row_index, n_reps_warm=3):
    if PERSISTENT_CACHE_DIR.exists():
        shutil.rmtree(PERSISTENT_CACHE_DIR)
    store = SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS)
    train_cache = PersistentBatchCache(PERSISTENT_CACHE_DIR / "train", store, train_idx, batch_size=BATCH_SIZE)
    test_cache = PersistentBatchCache(PERSISTENT_CACHE_DIR / "test", store, test_idx, batch_size=BATCH_SIZE)

    t0 = time.perf_counter()
    list(train_cache.read_rows_for_indices(train_idx, batch_size=BATCH_SIZE))
    list(test_cache.read_rows_for_indices(test_idx, batch_size=BATCH_SIZE))
    cold_build_seconds = time.perf_counter() - t0
    disk_bytes = train_cache.stats.artifact_bytes + test_cache.stats.artifact_bytes

    rep_times = []
    last_result = None
    mem_before = peak_memory_mb()
    for _ in range(n_reps_warm):
        reader = MultiPartitionPersistentReader([train_cache, test_cache])
        elapsed, last_result, _, mem_after = run_once(reader, train_idx, test_idx, label_by_row_index)
        rep_times.append(elapsed)
    return {
        "cold_build_seconds": cold_build_seconds, "disk_bytes": disk_bytes,
        "warm_rep_seconds": rep_times, "warm_median_seconds": median(rep_times),
        "peak_rss_mb_before": mem_before, "peak_rss_mb_after": mem_after,
        "accuracy": last_result.accuracy, "roc_auc": last_result.roc_auc,
        "confusion_matrix": last_result.confusion_matrix,
    }


def measure_hybrid(train_idx, test_idx, label_by_row_index, max_memory_cache_bytes=256_000_000, n_reps_warm=3):
    if HYBRID_CACHE_DIR.exists():
        shutil.rmtree(HYBRID_CACHE_DIR)

    # --- COLD: disk tier does not exist yet, pays the full build cost ---
    store = SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS)
    train_hybrid_cold = HybridTrainingReader(
        HYBRID_CACHE_DIR / "train", store, train_idx, batch_size=BATCH_SIZE, max_memory_cache_bytes=max_memory_cache_bytes
    )
    test_hybrid_cold = HybridTrainingReader(
        HYBRID_CACHE_DIR / "test", store, test_idx, batch_size=BATCH_SIZE, max_memory_cache_bytes=max_memory_cache_bytes
    )
    reader_cold = MultiPartitionHybridReader([train_hybrid_cold, test_hybrid_cold])
    mem_before_cold = peak_memory_mb()
    cold_elapsed, cold_result, _, mem_after_cold = run_once(reader_cold, train_idx, test_idx, label_by_row_index)
    disk_bytes = train_hybrid_cold.stats.disk_artifact_bytes + test_hybrid_cold.stats.disk_artifact_bytes
    cold_cache_stats = {"train": vars(train_hybrid_cold.stats), "test": vars(test_hybrid_cold.stats)}

    # --- WARM: disk tier already built (from the cold run above);
    # construct FRESH reader instances (simulating a new process) ---
    rep_times = []
    last_result = None
    warm_cache_stats = None
    mem_before_warm = peak_memory_mb()
    for _ in range(n_reps_warm):
        store_fresh = SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS)
        train_hybrid_warm = HybridTrainingReader(
            HYBRID_CACHE_DIR / "train", store_fresh, train_idx, batch_size=BATCH_SIZE, max_memory_cache_bytes=max_memory_cache_bytes
        )
        test_hybrid_warm = HybridTrainingReader(
            HYBRID_CACHE_DIR / "test", store_fresh, test_idx, batch_size=BATCH_SIZE, max_memory_cache_bytes=max_memory_cache_bytes
        )
        reader_warm = MultiPartitionHybridReader([train_hybrid_warm, test_hybrid_warm])
        elapsed, last_result, _, mem_after_warm = run_once(reader_warm, train_idx, test_idx, label_by_row_index)
        rep_times.append(elapsed)
        warm_cache_stats = {"train": vars(train_hybrid_warm.stats), "test": vars(test_hybrid_warm.stats)}

    return {
        "cold_seconds": cold_elapsed,
        "cold_peak_rss_mb_before": mem_before_cold,
        "cold_peak_rss_mb_after": mem_after_cold,
        "cold_cache_stats": cold_cache_stats,
        "disk_bytes": disk_bytes,
        "warm_rep_seconds": rep_times,
        "warm_median_seconds": median(rep_times),
        "warm_peak_rss_mb_before": mem_before_warm,
        "warm_peak_rss_mb_after": mem_after_warm,
        "warm_cache_stats": warm_cache_stats,
        "accuracy": last_result.accuracy, "roc_auc": last_result.roc_auc,
        "confusion_matrix": last_result.confusion_matrix,
        "cold_accuracy": cold_result.accuracy, "cold_roc_auc": cold_result.roc_auc,
    }


def verify_equivalence(store, train_idx, max_memory_cache_bytes=256_000_000) -> dict:
    direct = list(store.read_rows_for_indices(train_idx, batch_size=BATCH_SIZE))

    non_evict = NonEvictingCachedRowReader(SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS), max_cache_bytes=max_memory_cache_bytes)
    non_evict_batches = list(non_evict.read_rows_for_indices(train_idx, batch_size=BATCH_SIZE))

    persistent_dir = HYBRID_CACHE_DIR / "equivalence_persistent"
    if persistent_dir.exists():
        shutil.rmtree(persistent_dir)
    persistent = PersistentBatchCache(persistent_dir, SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS), train_idx, batch_size=BATCH_SIZE)
    persistent_batches = list(persistent.read_rows_for_indices(train_idx, batch_size=BATCH_SIZE))

    hybrid_dir = HYBRID_CACHE_DIR / "equivalence_hybrid"
    if hybrid_dir.exists():
        shutil.rmtree(hybrid_dir)
    hybrid = HybridTrainingReader(hybrid_dir, SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS), train_idx, batch_size=BATCH_SIZE, max_memory_cache_bytes=max_memory_cache_bytes)
    hybrid_batches = list(hybrid.read_rows_for_indices(train_idx, batch_size=BATCH_SIZE))
    hybrid_batches_pass2 = list(hybrid.read_rows_for_indices(train_idx, batch_size=BATCH_SIZE))  # exercise memory-hit path too

    # also verify normalized rows match exactly
    normalized_mismatches = []
    mismatches = []
    for (gd, rd, md), (gn, rn, mn), (gp, rp, mp), (gh, rh, mh), (gh2, rh2, mh2) in zip(
        direct, non_evict_batches, persistent_batches, hybrid_batches, hybrid_batches_pass2
    ):
        for label, g, r, m in [("non_evicting", gn, rn, mn), ("persistent", gp, rp, mp), ("hybrid_pass1", gh, rh, mh), ("hybrid_pass2", gh2, rh2, mh2)]:
            if gd != g or rd != r or (md != m).nnz != 0:
                mismatches.append({"strategy": label, "row_indices": rd})
        nd = sk_normalize(md, norm="l2")
        nh = sk_normalize(mh, norm="l2")
        if not np.allclose(nd.toarray(), nh.toarray()):
            normalized_mismatches.append({"row_indices": rd})

    return {
        "n_batches_compared": len(direct),
        "n_mismatches": len(mismatches),
        "mismatches": mismatches,
        "n_normalized_row_mismatches": len(normalized_mismatches),
        "passed": len(mismatches) == 0 and len(normalized_mismatches) == 0,
    }


def main() -> None:
    genome_order, label_by_row_index, random_split, lineage_split = load_pilot()
    store = SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS)
    train_idx, test_idx = split_indices(genome_order, random_split)
    train_idx_lineage, test_idx_lineage = split_indices(genome_order, lineage_split)

    print("=== equivalence check ===")
    equivalence = verify_equivalence(store, train_idx)
    print(f"passed={equivalence['passed']} n_mismatches={equivalence['n_mismatches']}")

    print("=== disabled ===")
    disabled = measure_disabled(train_idx, test_idx, label_by_row_index)
    print(f"  median={disabled['median_seconds']:.2f}s")

    print("=== non-evicting @256MB ===")
    non_evicting = measure_non_evicting(train_idx, test_idx, label_by_row_index)
    print(f"  median={non_evicting['median_seconds']:.2f}s")

    print("=== persistent artifact ===")
    persistent = measure_persistent(train_idx, test_idx, label_by_row_index)
    print(f"  cold={persistent['cold_build_seconds']:.2f}s warm_median={persistent['warm_median_seconds']:.2f}s disk_bytes={persistent['disk_bytes']}")

    print("=== hybrid (cold + warm-fresh-instance) ===")
    hybrid = measure_hybrid(train_idx, test_idx, label_by_row_index)
    print(f"  cold={hybrid['cold_seconds']:.2f}s warm_median={hybrid['warm_median_seconds']:.2f}s disk_bytes={hybrid['disk_bytes']}")
    print(f"  cold cache stats: {hybrid['cold_cache_stats']}")
    print(f"  warm cache stats: {hybrid['warm_cache_stats']}")

    print("=== lineage-split spot check: disabled vs hybrid (warm, fresh instance) ===")
    e2e_lineage_disabled = evaluate_streaming_hashed_matrix(
        SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS), train_idx_lineage, test_idx_lineage, label_by_row_index,
        seed=SEED, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, normalize_rows=True,
    )
    lineage_dir = HYBRID_CACHE_DIR / "lineage"
    if lineage_dir.exists():
        shutil.rmtree(lineage_dir)
    train_hybrid_lineage = HybridTrainingReader(lineage_dir / "train", SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS), train_idx_lineage, batch_size=BATCH_SIZE, max_memory_cache_bytes=256_000_000)
    test_hybrid_lineage = HybridTrainingReader(lineage_dir / "test", SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS), test_idx_lineage, batch_size=BATCH_SIZE, max_memory_cache_bytes=256_000_000)
    # build once (cold), then a fresh instance for the "warm" measurement
    list(train_hybrid_lineage.read_rows_for_indices(train_idx_lineage, batch_size=BATCH_SIZE))
    list(test_hybrid_lineage.read_rows_for_indices(test_idx_lineage, batch_size=BATCH_SIZE))
    train_hybrid_lineage_fresh = HybridTrainingReader(lineage_dir / "train", SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS), train_idx_lineage, batch_size=BATCH_SIZE, max_memory_cache_bytes=256_000_000)
    test_hybrid_lineage_fresh = HybridTrainingReader(lineage_dir / "test", SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS), test_idx_lineage, batch_size=BATCH_SIZE, max_memory_cache_bytes=256_000_000)
    reader_lineage = MultiPartitionHybridReader([train_hybrid_lineage_fresh, test_hybrid_lineage_fresh])
    t0 = time.perf_counter()
    e2e_lineage_hybrid = evaluate_streaming_hashed_matrix(
        reader_lineage, train_idx_lineage, test_idx_lineage, label_by_row_index,
        seed=SEED, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, normalize_rows=True,
    )
    lineage_hybrid_seconds = time.perf_counter() - t0
    lineage_spot_check = {
        "metrics_match": (
            e2e_lineage_disabled.accuracy == e2e_lineage_hybrid.accuracy
            and e2e_lineage_disabled.roc_auc == e2e_lineage_hybrid.roc_auc
        ),
        "hybrid_warm_seconds": lineage_hybrid_seconds,
    }
    print(f"  metrics_match={lineage_spot_check['metrics_match']} seconds={lineage_hybrid_seconds:.2f}")

    output = {
        "k": K, "n_buckets": N_BUCKETS, "antibiotic": ANTIBIOTIC,
        "n_epochs": N_EPOCHS, "batch_size": BATCH_SIZE,
        "equivalence_check": equivalence,
        "strategies": {
            "disabled": disabled,
            "non_evicting_256MB": non_evicting,
            "persistent_artifact": persistent,
            "hybrid": hybrid,
        },
        "lineage_split_spot_check": lineage_spot_check,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
