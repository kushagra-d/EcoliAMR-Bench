"""Profiles the out-of-core read path's bottleneck (Step 2) and
benchmarks the fix (Step 5): vectorized CSR reconstruction
(`SqliteRowStore._rows_to_csr`, fixed in place) plus the new bounded
`CachedRowReader` batch cache. Also re-verifies exact equivalence
(Step 4) across every real pilot genome, not just the unit-test
fixtures.

Reuses the ALREADY-BUILT SQLite store from the prior tasks - does not
rebuild it, does not re-extract anything, does not download genomes.
Reads existing splits/labels read-only.

Usage (from the project root, with the project venv active, AFTER the
SQLite store from the prior tasks has been built):

    python src/run_read_optimization_benchmark.py

Writes:
    data/processed/read_optimization_profile_ciprofloxacin_k15_d262144.json
    data/processed/read_optimization_benchmark_ciprofloxacin_k15_d262144.json
    reports/out_of_core_read_optimization.md (written separately, by hand, after reading this script's output)
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
PROFILE_PATH = DATA_PROCESSED / f"read_optimization_profile_{ANTIBIOTIC}_k{K}_d{N_BUCKETS}.json"
BENCHMARK_PATH = DATA_PROCESSED / f"read_optimization_benchmark_{ANTIBIOTIC}_k{K}_d{N_BUCKETS}.json"


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
    return genome_order, label_by_genome, label_by_row_index, random_split, lineage_split


def split_indices(genome_order, split_df):
    genome_to_partition = dict(zip(split_df["genome_id"], split_df["partition"]))
    train_idx = {i for i, gid in enumerate(genome_order) if genome_to_partition.get(gid) == "train"}
    test_idx = {i for i, gid in enumerate(genome_order) if genome_to_partition.get(gid) == "test"}
    return train_idx, test_idx


def profile_step2(store: SqliteRowStore, train_idx: set[int]) -> dict:
    """Breaks down where time goes: SQL fetch, naive reconstruction,
    vectorized reconstruction, normalization, and (separately) one
    representative `partial_fit` call - all on the SAME real rows."""
    cur = store._conn.execute(
        "SELECT row_index, genome_id, n_entries, cols, vals FROM rows WHERE row_index IN ({}) ORDER BY row_index".format(
            ",".join(str(i) for i in sorted(train_idx))
        )
    )
    t0 = time.perf_counter()
    rows = cur.fetchall()
    sql_fetch_seconds = time.perf_counter() - t0

    chunks = [rows[i : i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]

    t0 = time.perf_counter()
    for chunk in chunks:
        store._rows_to_csr_naive(chunk)
    naive_reconstruction_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    for chunk in chunks:
        store._rows_to_csr(chunk)
    fast_reconstruction_seconds = time.perf_counter() - t0

    _, _, one_batch_matrix = store._rows_to_csr(chunks[0])
    t0 = time.perf_counter()
    for _ in range(100):
        sk_normalize(one_batch_matrix, norm="l2")
    normalize_seconds_per_100_calls = time.perf_counter() - t0

    from sklearn.linear_model import SGDClassifier

    model = SGDClassifier(random_state=SEED, loss="log_loss")
    normalized = sk_normalize(one_batch_matrix, norm="l2")
    y_dummy = np.array([0, 1] * (normalized.shape[0] // 2) + [0] * (normalized.shape[0] % 2))
    t0 = time.perf_counter()
    for _ in range(100):
        model.partial_fit(normalized, y_dummy, classes=[0, 1])
    partial_fit_seconds_per_100_calls = time.perf_counter() - t0

    return {
        "n_rows_profiled": len(rows),
        "n_batches": len(chunks),
        "sql_fetch_seconds_one_pass": sql_fetch_seconds,
        "naive_reconstruction_seconds_one_pass": naive_reconstruction_seconds,
        "fast_reconstruction_seconds_one_pass": fast_reconstruction_seconds,
        "reconstruction_speedup_factor": (
            naive_reconstruction_seconds / fast_reconstruction_seconds if fast_reconstruction_seconds > 0 else None
        ),
        "normalize_seconds_per_100_calls": normalize_seconds_per_100_calls,
        "partial_fit_seconds_per_100_calls": partial_fit_seconds_per_100_calls,
        "extrapolated_50_epoch_naive_read_seconds": naive_reconstruction_seconds * N_EPOCHS + sql_fetch_seconds * N_EPOCHS,
        "extrapolated_50_epoch_fast_uncached_read_seconds": fast_reconstruction_seconds * N_EPOCHS
        + sql_fetch_seconds * N_EPOCHS,
    }


def median(values):
    return statistics.median(values)


def benchmark_read_path(store: SqliteRowStore, wanted_idx: set[int], n_passes: int, n_reps: int) -> dict:
    """Times `n_passes` repeated reads of the SAME wanted rows - the
    exact access pattern incremental training performs across epochs -
    under (a) the naive reconstruction forced per-chunk, (b) the
    now-default fast reconstruction uncached, (c) the fast
    reconstruction wrapped in CachedRowReader."""

    def naive_pass():
        wanted_sorted = sorted(wanted_idx)
        for i in range(0, len(wanted_sorted), BATCH_SIZE):
            chunk = wanted_sorted[i : i + BATCH_SIZE]
            placeholders = ",".join("?" * len(chunk))
            cur = store._conn.execute(
                f"SELECT row_index, genome_id, n_entries, cols, vals FROM rows WHERE row_index IN ({placeholders})",
                chunk,
            )
            store._rows_to_csr_naive(cur.fetchall())

    def fast_uncached_pass():
        list(store.read_rows_for_indices(wanted_idx, batch_size=BATCH_SIZE))

    results = {}
    for label, fn in [("naive_forced", naive_pass), ("fast_uncached", fast_uncached_pass)]:
        rep_times = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            for _ in range(n_passes):
                fn()
            rep_times.append(time.perf_counter() - t0)
        results[label] = {"rep_seconds": rep_times, "median_seconds": median(rep_times)}

    rep_times = []
    for _ in range(n_reps):
        reader = CachedRowReader(store, max_cache_bytes=64_000_000)
        t0 = time.perf_counter()
        for _ in range(n_passes):
            list(reader.read_rows_for_indices(wanted_idx, batch_size=BATCH_SIZE))
        rep_times.append(time.perf_counter() - t0)
    results["cached"] = {
        "rep_seconds": rep_times,
        "median_seconds": median(rep_times),
        "final_cache_stats": {
            "n_hits": reader.stats.n_hits,
            "n_misses": reader.stats.n_misses,
            "n_bytes_cached": reader.stats.n_bytes_cached,
        },
    }
    return results


def verify_exact_equivalence_all_genomes(store: SqliteRowStore, genome_order: list[str]) -> dict:
    """Step 4: compares the cached reader's output against the direct
    store reader for EVERY pilot genome - genome IDs, row ordering,
    sparse column indices, feature values, and normalized rows."""
    all_idx = set(range(len(genome_order)))
    reader = CachedRowReader(store, max_cache_bytes=64_000_000)

    direct_batches = list(store.read_rows_for_indices(all_idx, batch_size=BATCH_SIZE))
    cached_batches = list(reader.read_rows_for_indices(all_idx, batch_size=BATCH_SIZE))
    # second pass through the cache, to actually exercise cache HITS
    cached_batches_pass2 = list(reader.read_rows_for_indices(all_idx, batch_size=BATCH_SIZE))

    mismatches = []
    normalized_mismatches = []
    for (gid_d, ri_d, m_d), (gid_c, ri_c, m_c), (gid_c2, ri_c2, m_c2) in zip(
        direct_batches, cached_batches, cached_batches_pass2
    ):
        if gid_d != gid_c or gid_d != gid_c2:
            mismatches.append({"kind": "genome_id", "row_indices": ri_d})
        if ri_d != ri_c or ri_d != ri_c2:
            mismatches.append({"kind": "row_index", "row_indices": ri_d})
        if (m_d != m_c).nnz != 0 or (m_d != m_c2).nnz != 0:
            mismatches.append({"kind": "sparse_values", "row_indices": ri_d})
        n_d = sk_normalize(m_d, norm="l2")
        n_c = sk_normalize(m_c, norm="l2")
        if not np.allclose(n_d.toarray(), n_c.toarray()):
            normalized_mismatches.append({"row_indices": ri_d})

    return {
        "n_genomes_checked": len(genome_order),
        "n_batches_compared": len(direct_batches),
        "n_mismatches": len(mismatches),
        "mismatches": mismatches,
        "n_normalized_row_mismatches": len(normalized_mismatches),
        "second_pass_cache_hits": reader.stats.n_hits,
        "passed": len(mismatches) == 0 and len(normalized_mismatches) == 0,
    }


def benchmark_end_to_end_training(train_idx, test_idx, label_by_row_index, n_reps: int) -> dict:
    results = {}
    for label, build_reader in [
        ("uncached_fast_store", lambda: SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS)),
        ("cached_reader", lambda: CachedRowReader(SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS), max_cache_bytes=64_000_000)),
    ]:
        rep_times = []
        last_result = None
        mem_before = peak_memory_mb()
        for _ in range(n_reps):
            reader = build_reader()
            t0 = time.perf_counter()
            last_result = evaluate_streaming_hashed_matrix(
                reader, train_idx, test_idx, label_by_row_index,
                seed=SEED, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, normalize_rows=True,
            )
            rep_times.append(time.perf_counter() - t0)
        mem_after = peak_memory_mb()
        results[label] = {
            "rep_seconds": rep_times,
            "median_seconds": median(rep_times),
            "peak_rss_mb_before": mem_before,
            "peak_rss_mb_after": mem_after,
            "accuracy": last_result.accuracy,
            "roc_auc": last_result.roc_auc,
            "confusion_matrix": last_result.confusion_matrix,
            "max_batch_rows_held_in_memory": last_result.max_batch_rows_held_in_memory,
        }
    results["metrics_identical"] = (
        results["uncached_fast_store"]["accuracy"] == results["cached_reader"]["accuracy"]
        and results["uncached_fast_store"]["roc_auc"] == results["cached_reader"]["roc_auc"]
        and results["uncached_fast_store"]["confusion_matrix"] == results["cached_reader"]["confusion_matrix"]
    )
    return results


def main() -> None:
    genome_order, label_by_genome, label_by_row_index, random_split, lineage_split = load_pilot()
    store = SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS)
    train_idx_random, test_idx_random = split_indices(genome_order, random_split)
    train_idx_lineage, test_idx_lineage = split_indices(genome_order, lineage_split)

    print("=== Step 2: profiling ===")
    profile = profile_step2(store, train_idx_random)
    print(json.dumps(profile, indent=2))

    print("=== Step 4: exact equivalence across all genomes ===")
    equivalence = verify_exact_equivalence_all_genomes(store, genome_order)
    print(f"passed={equivalence['passed']} n_mismatches={equivalence['n_mismatches']}")

    print("=== Step 5: read-path benchmark (10 repeated passes x 3 reps, random split train) ===")
    db_size_bytes = DB_PATH.stat().st_size
    read_benchmark = benchmark_read_path(store, train_idx_random, n_passes=10, n_reps=3)
    for label, r in read_benchmark.items():
        print(f"  {label}: median={r['median_seconds']:.4f}s over reps={r['rep_seconds']}")

    print("=== Step 5: end-to-end training benchmark (50 epochs x 2 reps, both splits) ===")
    e2e_random = benchmark_end_to_end_training(train_idx_random, test_idx_random, label_by_row_index, n_reps=2)
    e2e_lineage = benchmark_end_to_end_training(train_idx_lineage, test_idx_lineage, label_by_row_index, n_reps=2)
    for split_name, e2e in [("random", e2e_random), ("lineage", e2e_lineage)]:
        print(f"[{split_name}] metrics_identical={e2e['metrics_identical']}")
        for label in ("uncached_fast_store", "cached_reader"):
            print(f"  [{split_name}][{label}] median_seconds={e2e[label]['median_seconds']:.2f} "
                  f"peak_rss_after={e2e[label]['peak_rss_mb_after']:.0f}MB")

    PROFILE_PATH.write_text(json.dumps(profile, indent=2))
    BENCHMARK_PATH.write_text(json.dumps(
        {
            "k": K, "n_buckets": N_BUCKETS, "antibiotic": ANTIBIOTIC,
            "db_size_bytes": db_size_bytes,
            "equivalence_check": equivalence,
            "read_path_benchmark_random_split_train": read_benchmark,
            "end_to_end_training_benchmark": {"random": e2e_random, "lineage": e2e_lineage},
        },
        indent=2,
    ))
    print(f"wrote {PROFILE_PATH}")
    print(f"wrote {BENCHMARK_PATH}")


if __name__ == "__main__":
    main()
