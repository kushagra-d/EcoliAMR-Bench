"""Fair, same-conditions comparison of four data-access strategies for
repeated-scan incremental training on the real 100-genome ciprofloxacin
pilot (k=15, n_buckets=262144):

  1. disabled    - direct `SqliteRowStore` reads, no cache wrapper.
  2. lru         - `CachedRowReader` (deterministic LRU eviction).
  3. non_evicting - `NonEvictingCachedRowReader` (caches what fits once,
                    never evicts).
  4. persistent  - `PersistentBatchCache` / `MultiPartitionPersistentReader`
                    (disk-backed artifact, materialized once, reused
                    every epoch with no SQL query at all).

Each of strategies 2 and 3 is run at TWO budgets: 64 MB (known,
from the prior task, to be smaller than the random split's 220.2 MB
training-partition working set) and 256 MB (known to be sufficient) -
so both the "insufficient" and "sufficient" regimes are reported for
each, never comparing a warmed cache against a cold baseline without
labeling both explicitly.

Reuses the ALREADY-BUILT SQLite store - does not rebuild it, does not
re-extract anything, does not download genomes. Model, labels, splits,
and feature mapping are NOT changed by this task - `evaluate_streaming_hashed_matrix`
and `SqliteRowStore._rows_to_csr` are called exactly as before.

Usage (from the project root, with the project venv active, AFTER the
SQLite store from the prior tasks has been built):

    python src/run_repeated_scan_cache_benchmark.py

Writes:
    data/processed/repeated_scan_cache_benchmark_ciprofloxacin_k15_d262144.json
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

from bvbrc_recon.cached_row_reader import CachedRowReader, NonEvictingCachedRowReader
from bvbrc_recon.epoch_batch_cache import MultiPartitionPersistentReader, PersistentBatchCache
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
OUTPUT_PATH = DATA_PROCESSED / f"repeated_scan_cache_benchmark_{ANTIBIOTIC}_k{K}_d{N_BUCKETS}.json"
PERSISTENT_CACHE_DIR = DATA_PROCESSED / f"epoch_batch_cache_{ANTIBIOTIC}_k{K}_d{N_BUCKETS}"


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


def run_end_to_end(reader_factory, train_idx, test_idx, label_by_row_index, n_reps):
    rep_times = []
    last_result = None
    extra_stats = None
    mem_before = peak_memory_mb()
    for _ in range(n_reps):
        reader = reader_factory()
        t0 = time.perf_counter()
        last_result = evaluate_streaming_hashed_matrix(
            reader, train_idx, test_idx, label_by_row_index,
            seed=SEED, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, normalize_rows=True,
        )
        rep_times.append(time.perf_counter() - t0)
        extra_stats = reader
    mem_after = peak_memory_mb()
    result = {
        "rep_seconds": rep_times,
        "median_seconds": median(rep_times),
        "peak_rss_mb_before": mem_before,
        "peak_rss_mb_after": mem_after,
        "accuracy": last_result.accuracy,
        "roc_auc": last_result.roc_auc,
        "confusion_matrix": last_result.confusion_matrix,
        "max_batch_rows_held_in_memory": last_result.max_batch_rows_held_in_memory,
    }
    return result, extra_stats


def measure_disabled(train_idx, test_idx, label_by_row_index, n_reps=2):
    result, _ = run_end_to_end(lambda: SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS), train_idx, test_idx, label_by_row_index, n_reps)
    return result


def measure_lru(train_idx, test_idx, label_by_row_index, max_cache_bytes, n_reps):
    def factory():
        return CachedRowReader(SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS), max_cache_bytes=max_cache_bytes)
    result, reader = run_end_to_end(factory, train_idx, test_idx, label_by_row_index, n_reps)
    result["cache_stats"] = vars(reader.stats)
    return result


def measure_non_evicting(train_idx, test_idx, label_by_row_index, max_cache_bytes, n_reps):
    def factory():
        return NonEvictingCachedRowReader(SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS), max_cache_bytes=max_cache_bytes)
    result, reader = run_end_to_end(factory, train_idx, test_idx, label_by_row_index, n_reps)
    result["cache_stats"] = vars(reader.stats)
    return result


def measure_persistent(train_idx, test_idx, label_by_row_index, n_reps_warm=3):
    """Distinguishes the ONE-TIME build (cold) cost from the per-epoch
    reuse (warm) cost explicitly - never reports only the warm number."""
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

    # warm: artifact already materialized, measure repeated full end-to-end runs
    rep_times = []
    last_result = None
    mem_before = peak_memory_mb()
    for _ in range(n_reps_warm):
        reader = MultiPartitionPersistentReader([train_cache, test_cache])
        t0 = time.perf_counter()
        last_result = evaluate_streaming_hashed_matrix(
            reader, train_idx, test_idx, label_by_row_index,
            seed=SEED, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, normalize_rows=True,
        )
        rep_times.append(time.perf_counter() - t0)
    mem_after = peak_memory_mb()

    return {
        "cold_build_seconds": cold_build_seconds,
        "disk_bytes": disk_bytes,
        "warm_rep_seconds": rep_times,
        "warm_median_seconds": median(rep_times),
        "peak_rss_mb_before": mem_before,
        "peak_rss_mb_after": mem_after,
        "accuracy": last_result.accuracy,
        "roc_auc": last_result.roc_auc,
        "confusion_matrix": last_result.confusion_matrix,
        "max_batch_rows_held_in_memory": last_result.max_batch_rows_held_in_memory,
        "n_rebuilds_train": train_cache.stats.n_rebuilds,
        "n_rebuilds_test": test_cache.stats.n_rebuilds,
    }


def verify_equivalence(store, genome_order, train_idx, test_idx, label_by_row_index) -> dict:
    direct_train = list(store.read_rows_for_indices(train_idx, batch_size=BATCH_SIZE))

    lru = CachedRowReader(store, max_cache_bytes=256_000_000)
    non_evict = NonEvictingCachedRowReader(SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS), max_cache_bytes=256_000_000)
    persistent_dir = PERSISTENT_CACHE_DIR / "equivalence_check"
    if persistent_dir.exists():
        shutil.rmtree(persistent_dir)
    persistent = PersistentBatchCache(persistent_dir, SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS), train_idx, batch_size=BATCH_SIZE)

    lru_batches = list(lru.read_rows_for_indices(train_idx, batch_size=BATCH_SIZE))
    non_evict_batches = list(non_evict.read_rows_for_indices(train_idx, batch_size=BATCH_SIZE))
    persistent_batches = list(persistent.read_rows_for_indices(train_idx, batch_size=BATCH_SIZE))

    mismatches = []
    for (gd, rd, md), (gl, rl, ml), (gn, rn, mn), (gp, rp, mp) in zip(
        direct_train, lru_batches, non_evict_batches, persistent_batches
    ):
        for label, g, r, m in [("lru", gl, rl, ml), ("non_evicting", gn, rn, mn), ("persistent", gp, rp, mp)]:
            if gd != g or rd != r or (md != m).nnz != 0:
                mismatches.append({"strategy": label, "row_indices": rd})

    return {"n_batches_compared": len(direct_train), "n_mismatches": len(mismatches), "mismatches": mismatches, "passed": len(mismatches) == 0}


def main() -> None:
    genome_order, label_by_row_index, random_split, lineage_split = load_pilot()
    store = SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS)
    train_idx, test_idx = split_indices(genome_order, random_split)
    train_idx_lineage, test_idx_lineage = split_indices(genome_order, lineage_split)

    print("=== equivalence check (random split train partition) ===")
    equivalence = verify_equivalence(store, genome_order, train_idx, test_idx, label_by_row_index)
    print(f"passed={equivalence['passed']} n_mismatches={equivalence['n_mismatches']}")

    print("=== disabled (direct SQLite) ===")
    disabled = measure_disabled(train_idx, test_idx, label_by_row_index, n_reps=2)
    print(f"  median={disabled['median_seconds']:.2f}s")

    print("=== LRU @ 64MB (insufficient) ===")
    lru_64 = measure_lru(train_idx, test_idx, label_by_row_index, 64_000_000, n_reps=2)
    print(f"  median={lru_64['median_seconds']:.2f}s hits={lru_64['cache_stats']['n_hits']}")

    print("=== LRU @ 256MB (sufficient) ===")
    lru_256 = measure_lru(train_idx, test_idx, label_by_row_index, 256_000_000, n_reps=3)
    print(f"  median={lru_256['median_seconds']:.2f}s hits={lru_256['cache_stats']['n_hits']}")

    print("=== non-evicting @ 64MB (insufficient) ===")
    ne_64 = measure_non_evicting(train_idx, test_idx, label_by_row_index, 64_000_000, n_reps=3)
    print(f"  median={ne_64['median_seconds']:.2f}s hits={ne_64['cache_stats']['n_hits']}")

    print("=== non-evicting @ 256MB (sufficient) ===")
    ne_256 = measure_non_evicting(train_idx, test_idx, label_by_row_index, 256_000_000, n_reps=3)
    print(f"  median={ne_256['median_seconds']:.2f}s hits={ne_256['cache_stats']['n_hits']}")

    print("=== persistent disk-backed artifact ===")
    persistent = measure_persistent(train_idx, test_idx, label_by_row_index, n_reps_warm=3)
    print(f"  cold_build={persistent['cold_build_seconds']:.2f}s warm_median={persistent['warm_median_seconds']:.2f}s "
          f"disk_bytes={persistent['disk_bytes']}")

    print("=== lineage-split spot check: disabled vs persistent ===")
    e2e_lineage_disabled = evaluate_streaming_hashed_matrix(
        SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS), train_idx_lineage, test_idx_lineage, label_by_row_index,
        seed=SEED, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, normalize_rows=True,
    )
    lineage_dir = PERSISTENT_CACHE_DIR / "lineage"
    if lineage_dir.exists():
        shutil.rmtree(lineage_dir)
    train_cache_lineage = PersistentBatchCache(lineage_dir / "train", SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS), train_idx_lineage, batch_size=BATCH_SIZE)
    test_cache_lineage = PersistentBatchCache(lineage_dir / "test", SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS), test_idx_lineage, batch_size=BATCH_SIZE)
    reader_lineage = MultiPartitionPersistentReader([train_cache_lineage, test_cache_lineage])
    t0 = time.perf_counter()
    e2e_lineage_persistent = evaluate_streaming_hashed_matrix(
        reader_lineage, train_idx_lineage, test_idx_lineage, label_by_row_index,
        seed=SEED, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, normalize_rows=True,
    )
    lineage_persistent_seconds = time.perf_counter() - t0
    lineage_spot_check = {
        "metrics_match": (
            e2e_lineage_disabled.accuracy == e2e_lineage_persistent.accuracy
            and e2e_lineage_disabled.roc_auc == e2e_lineage_persistent.roc_auc
        ),
        "persistent_seconds_including_cold_build": lineage_persistent_seconds,
    }
    print(f"  metrics_match={lineage_spot_check['metrics_match']} seconds={lineage_persistent_seconds:.2f}")

    output = {
        "k": K, "n_buckets": N_BUCKETS, "antibiotic": ANTIBIOTIC,
        "n_epochs": N_EPOCHS, "batch_size": BATCH_SIZE,
        "equivalence_check": equivalence,
        "strategies": {
            "disabled": disabled,
            "lru_64MB": lru_64,
            "lru_256MB": lru_256,
            "non_evicting_64MB": ne_64,
            "non_evicting_256MB": ne_256,
            "persistent_artifact": persistent,
        },
        "lineage_split_spot_check": lineage_spot_check,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
