"""Controlled intermediate-k experiment: extracts ONE k value's canonical
k-mer feature matrix from the EXISTING 100-genome ciprofloxacin pilot
assemblies, using the EXISTING memory-safe streaming builder, then
evaluates Logistic Regression on the EXISTING saved random and lineage
splits. Run once per k value (see below for why).

Reuses, unmodified:
  - `bvbrc_recon.streaming_kmer_build.build_matrix_streaming` (same
    canonical/ambiguous-base/contig-boundary rules as every other k
    already computed in this project).
  - `bvbrc_recon.pilot_model.evaluate_with_precomputed_split` (same LR
    configuration already used for k=10/k=31).
  - The existing saved `pilot_ciprofloxacin_{genome_order,selected_cohort,
    random_split,lineage_split}.csv` - none of these are modified, and
    none is regenerated.

Why one k per PROCESS (not one process looping over all k values): the
k=10/k=31 comparison already showed peak RSS accumulates across k values
processed in the same Python process (del'ing a matrix does not always
return memory to the OS promptly). Running each k as a fresh process
guarantees memory is fully released between k values - a stronger
resource-safety guarantee than relying on in-process cleanup alone.

Usage (from the project root, with the project venv active):

    python src/run_intermediate_k_experiment.py <k>

Writes (new files only, never overwriting k=10/k=31 or any existing
pilot artifact):
    data/processed/pilot_ciprofloxacin_kmer_matrix_k<k>.npz
    data/processed/pilot_ciprofloxacin_kmer_vocabulary_k<k>.csv
    data/processed/pilot_ciprofloxacin_kmer_config_k<k>_comparison.json
    data/processed/pilot_ciprofloxacin_kmer_k<k>_experiment.log
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy import sparse

from bvbrc_recon.pilot_model import evaluate_with_precomputed_split
from bvbrc_recon.streaming_kmer_build import DEFAULT_MEMORY_LIMIT_MB, build_matrix_streaming, peak_memory_mb

ANTIBIOTIC = "ciprofloxacin"
SEED = 20261009
MEMORY_LIMIT_MB = DEFAULT_MEMORY_LIMIT_MB

DATA_PROCESSED = Path("data/processed")
DATA_RAW_PILOT_DIR = Path("data/raw") / f"{ANTIBIOTIC}_pilot"


def load_contig_sequences(fasta_path: Path) -> list[str]:
    return [str(record.seq) for record in SeqIO.parse(str(fasta_path), "fasta")]


def logline(fh, msg: str) -> None:
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


def main(k: int) -> None:
    log_path = DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_k{k}_experiment.log"
    log_fh = open(log_path, "w")
    logline(log_fh, f"=== intermediate-k experiment: k={k} ===")

    genome_order = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_genome_order.csv", dtype={"genome_id": str}
    )["genome_id"].tolist()
    selected = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_selected_cohort.csv", dtype={"genome_id": str}
    )
    label_by_genome = dict(zip(selected["genome_id"], selected["label"]))

    t0 = time.perf_counter()
    contigs_by_genome = {
        gid: load_contig_sequences(DATA_RAW_PILOT_DIR / f"{gid}.fna") for gid in genome_order
    }
    logline(log_fh, f"loaded {len(genome_order)} genomes' contigs in {time.perf_counter() - t0:.1f}s")

    def progress(row_idx, genome_id, vocab_size, nnz_so_far, mem_mb):
        if (row_idx + 1) % 20 == 0 or row_idx == 0:
            logline(
                log_fh,
                f"  genome {row_idx + 1}/{len(genome_order)} ({genome_id}): "
                f"vocab_size={vocab_size:,} nnz_so_far={nnz_so_far:,} peak_rss_mb={mem_mb:.0f}",
            )
        if mem_mb > MEMORY_LIMIT_MB:
            logline(
                log_fh,
                f"  SAFETY ABORT: peak RSS {mem_mb:.0f} MB exceeded limit {MEMORY_LIMIT_MB:.0f} MB "
                f"after {row_idx + 1}/{len(genome_order)} genomes.",
            )

    t0 = time.perf_counter()
    try:
        build_result = build_matrix_streaming(
            contigs_by_genome, genome_order, k, memory_limit_mb=MEMORY_LIMIT_MB, progress_callback=progress
        )
        failure = None
    except MemoryError as exc:
        build_result = None
        failure = f"MemoryError: {exc}"
        logline(log_fh, f"  HARD FAILURE: {failure}")
    extraction_seconds = time.perf_counter() - t0
    mem_after_mb = peak_memory_mb()

    if build_result is None:
        config = {
            "k": k,
            "antibiotic": ANTIBIOTIC,
            "n_genomes_requested": len(genome_order),
            "n_genomes_processed": 0,
            "complete": False,
            "aborted_for_memory": True,
            "failure": failure,
            "extraction_seconds": extraction_seconds,
            "peak_rss_mb": mem_after_mb,
        }
        (DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_config_k{k}_comparison.json").write_text(
            json.dumps(config, indent=2)
        )
        logline(log_fh, "Done (failed).")
        log_fh.close()
        return

    matrix = build_result.matrix
    vocabulary = build_result.vocabulary
    processed_ids = build_result.processed_genome_ids
    n_genomes_used = len(processed_ids)

    logline(
        log_fh,
        f"{'COMPLETE' if build_result.complete else 'INCOMPLETE'}: {n_genomes_used}/"
        f"{build_result.n_requested_genomes} genomes in {extraction_seconds:.1f}s, "
        f"peak_rss_mb={mem_after_mb:.0f}",
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
        "failure": None,
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
        f"matrix: {matrix.shape[0]}x{matrix.shape[1]}, nnz={matrix.nnz:,}, density={density:.4%}, "
        f"npz={storage_bytes/1e6:.1f}MB, in-memory CSR={in_memory_csr_bytes/1e6:.1f}MB",
    )

    if not build_result.complete:
        logline(log_fh, "SKIPPING model evaluation - incomplete matrix cannot align with full-cohort splits.")
        logline(log_fh, "Done (incomplete).")
        log_fh.close()
        return

    random_split = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_random_split.csv", dtype={"genome_id": str}
    )
    lineage_split = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_lineage_split.csv", dtype={"genome_id": str}
    )
    genome_to_row = {gid: i for i, gid in enumerate(processed_ids)}

    def idx_for(split_df, partition):
        return np.array(
            [genome_to_row[g] for g in split_df.loc[split_df["partition"] == partition, "genome_id"]]
        )

    labels = [label_by_genome[g] for g in processed_ids]
    model_results = {}
    for split_name, split_df in [("random", random_split), ("lineage", lineage_split)]:
        train_idx = idx_for(split_df, "train")
        test_idx = idx_for(split_df, "test")
        result = evaluate_with_precomputed_split(matrix, labels, train_idx, test_idx, seed=SEED)
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
        }
        logline(
            log_fh,
            f"[{split_name}] LogisticRegression acc={result.accuracy} auc={result.roc_auc} "
            f"recall={result.recall_per_class}",
        )

    (DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_k{k}_lr_results.json").write_text(
        json.dumps(model_results, indent=2)
    )
    logline(log_fh, "Done.")
    log_fh.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/run_intermediate_k_experiment.py <k>", file=sys.stderr)
        sys.exit(1)
    main(int(sys.argv[1]))
