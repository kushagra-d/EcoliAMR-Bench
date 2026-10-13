"""Small, reproducible k-mer configuration comparison (k=10 vs k=31) on
the EXISTING 100-genome ciprofloxacin pilot assemblies only.

Reuses kmer_features.py's `count_kmers_in_contig`/`canonical_kmer` (same
canonical/ambiguity/contig-boundary handling for any k) and
model_comparison.py's model evaluators unchanged. Does NOT overwrite the
existing k=10 pilot matrix - all new outputs use a `_k{K}` suffix. Does
NOT download any genome.

MEMORY NOTE (discovered empirically, not assumed): a naive
`{genome_id: count_kmers_in_assembly(...) for ...}` dict comprehension -
holding all 100 genomes' full per-genome k-mer Counters in memory at once
- OOM-killed this process at k=31 (confirmed: a single genome's k=31
Counter alone uses ~500-700MB RSS; 100 of them at once exceeds this
machine's 17GB RAM). This script instead processes ONE genome at a time,
folding each genome's counts directly into a shared COO accumulator and
vocabulary dict, then discarding that genome's Counter before moving on -
peak memory is bounded by one genome's Counter plus the cumulative
(unavoidable) vocabulary, not by genome count times Counter size. Progress
is logged to a file (not just stdout) so a crash still leaves real,
inspectable partial measurements - a process can be SIGKILLed without any
buffered stdout ever reaching disk.

This is explicitly exploratory: a single ~100-genome comparison with no
train/test tuning, reusing the EXISTING saved random and lineage splits
(never recomputed, never peeked at for feature/model selection).

Usage (from the project root, with the project venv active):

    python src/run_kmer_k_comparison.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy import sparse

from bvbrc_recon.model_comparison import evaluate_random_forest, evaluate_xgboost
from bvbrc_recon.pilot_model import evaluate_with_precomputed_split
from bvbrc_recon.streaming_kmer_build import DEFAULT_MEMORY_LIMIT_MB, build_matrix_streaming, peak_memory_mb

ANTIBIOTIC = "ciprofloxacin"
SEED = 20261009
K_VALUES = [10, 31]
MEMORY_LIMIT_MB = DEFAULT_MEMORY_LIMIT_MB  # safety abort threshold, well under this machine's 17GB total RAM

DATA_PROCESSED = Path("data/processed")
DATA_RAW_PILOT_DIR = Path("data/raw") / f"{ANTIBIOTIC}_pilot"
LOG_PATH = Path("data/processed/pilot_ciprofloxacin_kmer_k_comparison.log")


def load_contig_sequences(fasta_path: Path) -> list[str]:
    return [str(record.seq) for record in SeqIO.parse(str(fasta_path), "fasta")]


def logline(fh, msg: str) -> None:
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    log_fh = open(LOG_PATH, "w")

    genome_order = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_genome_order.csv", dtype={"genome_id": str}
    )["genome_id"].tolist()
    selected = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_selected_cohort.csv", dtype={"genome_id": str}
    )
    label_by_genome = dict(zip(selected["genome_id"], selected["label"]))

    logline(log_fh, "Loading contig sequences for all 100 pilot genomes (shared across both k values)...")
    t0 = time.perf_counter()
    contigs_by_genome = {
        gid: load_contig_sequences(DATA_RAW_PILOT_DIR / f"{gid}.fna") for gid in genome_order
    }
    logline(log_fh, f"  loaded in {time.perf_counter() - t0:.1f}s, peak_rss_mb={peak_memory_mb():.0f}")

    random_split = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_random_split.csv", dtype={"genome_id": str}
    )
    lineage_split = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_lineage_split.csv", dtype={"genome_id": str}
    )

    for k in K_VALUES:
        logline(log_fh, f"\n=== k={k} ===")

        def progress(row_idx, genome_id, vocab_size, nnz_so_far, mem_mb):
            if (row_idx + 1) % 10 == 0 or row_idx == 0:
                logline(
                    log_fh,
                    f"    [k={k}] genome {row_idx + 1}/{len(genome_order)} ({genome_id}): "
                    f"vocab_size={vocab_size:,} nnz_so_far={nnz_so_far:,} peak_rss_mb={mem_mb:.0f}",
                )
            if mem_mb > MEMORY_LIMIT_MB:
                logline(
                    log_fh,
                    f"    [k={k}] SAFETY ABORT: peak RSS {mem_mb:.0f} MB exceeded limit "
                    f"{MEMORY_LIMIT_MB:.0f} MB after {row_idx + 1}/{len(genome_order)} genomes - "
                    f"stopping here rather than risking an OS OOM-kill.",
                )

        t0 = time.perf_counter()
        build_result = build_matrix_streaming(
            contigs_by_genome, genome_order, k, memory_limit_mb=MEMORY_LIMIT_MB, progress_callback=progress
        )
        extraction_seconds = time.perf_counter() - t0
        mem_after_mb = peak_memory_mb()

        processed_ids = build_result.processed_genome_ids
        matrix = build_result.matrix
        vocabulary = build_result.vocabulary
        n_genomes_used = len(processed_ids)

        logline(
            log_fh,
            f"  [k={k}] {'COMPLETE' if build_result.complete else 'INCOMPLETE'}: "
            f"{n_genomes_used}/{build_result.n_requested_genomes} genomes processed in "
            f"{extraction_seconds:.1f}s, peak_rss_mb={mem_after_mb:.0f}",
        )

        npz_path = DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_matrix_k{k}.npz"
        sparse.save_npz(npz_path, matrix)
        storage_bytes = npz_path.stat().st_size

        vocab_path = DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_vocabulary_k{k}.csv"
        pd.DataFrame({"kmer": list(vocabulary), "column_index": list(vocabulary.values())}).to_csv(
            vocab_path, index=False
        )

        density = matrix.nnz / (matrix.shape[0] * matrix.shape[1]) if matrix.shape[1] else 0.0
        in_memory_csr_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes

        config = {
            "k": k,
            "antibiotic": ANTIBIOTIC,
            "n_genomes_requested": build_result.n_requested_genomes,
            "n_genomes_processed": n_genomes_used,
            "complete": build_result.complete,
            "aborted_for_memory": build_result.aborted_for_memory,
            "n_features": matrix.shape[1],
            "nnz": int(matrix.nnz),
            "density": density,
            "extraction_seconds": extraction_seconds,
            "npz_storage_bytes": storage_bytes,
            "in_memory_csr_bytes": int(in_memory_csr_bytes),
            "peak_rss_mb": mem_after_mb,
        }
        (DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_config_k{k}_comparison.json").write_text(
            json.dumps(config, indent=2)
        )
        logline(
            log_fh,
            f"  matrix: {matrix.shape[0]}x{matrix.shape[1]}, nnz={matrix.nnz:,}, density={density:.4%}, "
            f"npz={storage_bytes/1e6:.1f}MB, in-memory CSR={in_memory_csr_bytes/1e6:.1f}MB",
        )

        if not build_result.complete:
            logline(
                log_fh,
                f"  [k={k}] SKIPPING model evaluation - the matrix only covers {n_genomes_used} of "
                f"{build_result.n_requested_genomes} genomes and cannot be aligned with the "
                f"saved full-cohort splits/labels.",
            )
            continue

        genome_to_row = {gid: i for i, gid in enumerate(processed_ids)}

        def idx_for(split_df, partition):
            return np.array(
                [genome_to_row[g] for g in split_df.loc[split_df["partition"] == partition, "genome_id"]]
            )

        labels = [label_by_genome[g] for g in processed_ids]
        for split_name, split_df in [("random", random_split), ("lineage", lineage_split)]:
            train_idx = idx_for(split_df, "train")
            test_idx = idx_for(split_df, "test")
            lr = evaluate_with_precomputed_split(matrix, labels, train_idx, test_idx, seed=SEED)
            rf = evaluate_random_forest(matrix, labels, train_idx, test_idx, seed=SEED)
            xgb = evaluate_xgboost(matrix, labels, train_idx, test_idx, seed=SEED)
            logline(
                log_fh,
                f"  [{split_name}] LR acc={lr.accuracy} auc={lr.roc_auc} | "
                f"RF acc={rf.accuracy} auc={rf.roc_auc} | XGB acc={xgb.accuracy} auc={xgb.roc_auc}",
            )

    logline(log_fh, "\nDone.")
    log_fh.close()


if __name__ == "__main__":
    main()
