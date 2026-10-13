"""Feature-hashing experiment: folds the EXISTING raw k=15 pilot matrix
into fixed dimensions of 2^16, 2^18, and 2^20 columns, evaluates Logistic
Regression on the EXISTING saved splits, and records resource/collision
diagnostics. Does not re-extract k-mers from any genome and does not
modify the raw k=15 matrix or any saved split.

Usage (from the project root, with the project venv active, AFTER the
existing k=15 raw matrix has been built - it already has been, by the
prior intermediate-k task):

    python src/run_feature_hashing_experiment.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from bvbrc_recon.feature_hashing import build_hashed_representation
from bvbrc_recon.pilot_model import evaluate_with_precomputed_split
from bvbrc_recon.streaming_kmer_build import peak_memory_mb

ANTIBIOTIC = "ciprofloxacin"
SEED = 20261009
RAW_K = 15
HASH_DIMENSIONS = [2**16, 2**18, 2**20]

DATA_PROCESSED = Path("data/processed")
LOG_PATH = DATA_PROCESSED / "pilot_ciprofloxacin_feature_hashing_experiment.log"


def logline(fh, msg: str) -> None:
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


def main() -> None:
    log_fh = open(LOG_PATH, "w")
    logline(log_fh, f"=== feature hashing experiment: raw k={RAW_K}, dimensions={HASH_DIMENSIONS} ===")

    genome_order = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_genome_order.csv", dtype={"genome_id": str}
    )["genome_id"].tolist()
    selected = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_selected_cohort.csv", dtype={"genome_id": str}
    )
    label_by_genome = dict(zip(selected["genome_id"], selected["label"]))
    labels = [label_by_genome[g] for g in genome_order]

    raw_matrix = sparse.load_npz(DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_matrix_k{RAW_K}.npz")
    vocab_df = pd.read_csv(DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_vocabulary_k{RAW_K}.csv")
    vocabulary = dict(zip(vocab_df["kmer"], vocab_df["column_index"]))
    logline(
        log_fh,
        f"loaded raw k={RAW_K} matrix: {raw_matrix.shape}, nnz={raw_matrix.nnz:,}, "
        f"peak_rss_mb={peak_memory_mb():.0f}",
    )
    if raw_matrix.shape[0] != len(genome_order):
        raise RuntimeError("ALIGNMENT BLOCKER: raw matrix row count does not match genome_order length")

    random_split = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_random_split.csv", dtype={"genome_id": str}
    )
    lineage_split = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_lineage_split.csv", dtype={"genome_id": str}
    )
    genome_to_row = {gid: i for i, gid in enumerate(genome_order)}

    def idx_for(split_df, partition):
        return np.array(
            [genome_to_row[g] for g in split_df.loc[split_df["partition"] == partition, "genome_id"]]
        )

    splits = {
        "random": (idx_for(random_split, "train"), idx_for(random_split, "test")),
        "lineage": (idx_for(lineage_split, "train"), idx_for(lineage_split, "test")),
    }

    for n_buckets in HASH_DIMENSIONS:
        logline(log_fh, f"\n--- n_buckets=2^{int(np.log2(n_buckets))}={n_buckets:,} ---")
        t0 = time.perf_counter()
        result = build_hashed_representation(raw_matrix, vocabulary, n_buckets)
        extraction_seconds = time.perf_counter() - t0
        mem_mb = peak_memory_mb()

        hashed = result.matrix
        density = hashed.nnz / (hashed.shape[0] * hashed.shape[1])
        in_memory_csr_bytes = hashed.data.nbytes + hashed.indices.nbytes + hashed.indptr.nbytes

        npz_path = DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_hashed_matrix_d{n_buckets}.npz"
        sparse.save_npz(npz_path, hashed)
        storage_bytes = npz_path.stat().st_size

        logline(
            log_fh,
            f"  built in {extraction_seconds:.2f}s, peak_rss_mb={mem_mb:.0f}, shape={hashed.shape}, "
            f"nnz={hashed.nnz:,}, density={density:.4%}, npz={storage_bytes/1e6:.1f}MB, "
            f"in-memory CSR={in_memory_csr_bytes/1e6:.1f}MB",
        )
        logline(
            log_fh,
            f"  collisions: {result.collisions.n_colliding_buckets:,} colliding buckets / "
            f"{n_buckets:,} total ({result.collisions.n_empty_buckets:,} empty, "
            f"{result.collisions.n_lonely_buckets:,} lonely), "
            f"{result.collisions.n_raw_features_in_a_collision:,}/{result.collisions.n_raw_features:,} "
            f"raw k-mers fall in a collision, max distinct k-mers sharing one bucket="
            f"{result.collisions.max_distinct_kmers_in_one_bucket:,}, "
            f"mean distinct k-mers per occupied bucket="
            f"{result.collisions.mean_distinct_kmers_per_occupied_bucket:.2f}",
        )

        model_results = {}
        for split_name, (train_idx, test_idx) in splits.items():
            lr = evaluate_with_precomputed_split(hashed, labels, train_idx, test_idx, seed=SEED)
            model_results[split_name] = {
                "n_train": lr.n_train,
                "n_test": lr.n_test,
                "train_class_counts": lr.train_class_counts,
                "test_class_counts": lr.test_class_counts,
                "accuracy": lr.accuracy,
                "roc_auc": lr.roc_auc,
                "average_precision": lr.average_precision,
                "recall_per_class": lr.recall_per_class,
                "confusion_matrix": lr.confusion_matrix,
                "warnings": lr.warnings,
            }
            logline(
                log_fh,
                f"  [{split_name}] LogisticRegression acc={lr.accuracy} auc={lr.roc_auc} "
                f"recall={lr.recall_per_class}",
            )

        config = {
            "n_buckets": n_buckets,
            "raw_k": RAW_K,
            "antibiotic": ANTIBIOTIC,
            "n_genomes": hashed.shape[0],
            "n_features": hashed.shape[1],
            "nnz": int(hashed.nnz),
            "density": density,
            "build_seconds": extraction_seconds,
            "peak_rss_mb": mem_mb,
            "npz_storage_bytes": storage_bytes,
            "in_memory_csr_bytes": int(in_memory_csr_bytes),
            "collisions": {
                "n_buckets": result.collisions.n_buckets,
                "n_raw_features": result.collisions.n_raw_features,
                "n_empty_buckets": result.collisions.n_empty_buckets,
                "n_lonely_buckets": result.collisions.n_lonely_buckets,
                "n_colliding_buckets": result.collisions.n_colliding_buckets,
                "n_raw_features_in_a_collision": result.collisions.n_raw_features_in_a_collision,
                "max_distinct_kmers_in_one_bucket": result.collisions.max_distinct_kmers_in_one_bucket,
                "mean_distinct_kmers_per_occupied_bucket": result.collisions.mean_distinct_kmers_per_occupied_bucket,
            },
            "model_results": model_results,
        }
        config_path = DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_hashed_config_d{n_buckets}.json"
        config_path.write_text(json.dumps(config, indent=2))

        del hashed, result
        logline(log_fh, f"  config written -> {config_path}")

    logline(log_fh, f"\nAnalysis date (UTC): {datetime.now(timezone.utc).isoformat()}")
    logline(log_fh, "Done.")
    log_fh.close()


if __name__ == "__main__":
    main()
