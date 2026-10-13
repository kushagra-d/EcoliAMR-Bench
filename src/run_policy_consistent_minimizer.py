"""Builds the k=31/w=15 minimizer representation for the Policy-A-
reconciled 200-genome cohort, using the EXISTING, already-retrieved
FASTA assemblies and the EXISTING, unmodified
`minimizer_extraction.build_minimizer_matrix_streaming`.

Usage (from the project root, with the project venv active, AFTER
run_cohort_reconciliation.py has written the reconciled selected-cohort
CSV):

    python src/run_policy_consistent_minimizer.py

Writes:
    data/processed/policy_consistent_ciprofloxacin_minimizer_matrix_k31_w15.npz
    data/processed/policy_consistent_ciprofloxacin_minimizer_vocabulary_k31_w15.csv
    data/processed/policy_consistent_ciprofloxacin_minimizer_config_k31_w15.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy import sparse

from bvbrc_recon.minimizer_extraction import build_minimizer_matrix_streaming
from bvbrc_recon.streaming_kmer_build import DEFAULT_MEMORY_LIMIT_MB, peak_memory_mb

ANTIBIOTIC = "ciprofloxacin"
K, W = 31, 15
MEMORY_LIMIT_MB = DEFAULT_MEMORY_LIMIT_MB

DATA_PROCESSED = Path("data/processed")
DATA_RAW_PILOT_DIR = Path("data/raw") / f"{ANTIBIOTIC}_pilot"
DATA_RAW_EXPANSION_DIR = Path("data/raw") / f"{ANTIBIOTIC}_expansion"
LOG_PATH = DATA_PROCESSED / f"policy_consistent_{ANTIBIOTIC}_minimizer_k{K}_w{W}_experiment.log"


def load_contig_sequences(fasta_path: Path) -> list[str]:
    return [str(record.seq) for record in SeqIO.parse(str(fasta_path), "fasta")]


def logline(fh, msg: str) -> None:
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


def main() -> None:
    log_fh = open(LOG_PATH, "w")
    logline(log_fh, f"=== policy-consistent minimizer experiment: k={K}, w={W} ===")

    selected = pd.read_csv(
        DATA_PROCESSED / f"policy_consistent_{ANTIBIOTIC}_selected_cohort.csv", dtype={"genome_id": str}
    )
    genome_order = selected["genome_id"].tolist()
    label_by_genome = dict(zip(selected["genome_id"], selected["label"]))
    logline(log_fh, f"reconciled cohort: {len(genome_order)} genomes")

    t0 = time.perf_counter()
    contigs_by_genome = {}
    for gid in genome_order:
        fasta_path = DATA_RAW_PILOT_DIR / f"{gid}.fna"
        if not fasta_path.exists():
            fasta_path = DATA_RAW_EXPANSION_DIR / f"{gid}.fna"
        contigs_by_genome[gid] = load_contig_sequences(fasta_path)
    logline(log_fh, f"loaded {len(genome_order)} genomes' contigs in {time.perf_counter() - t0:.1f}s, "
            f"peak_rss_mb={peak_memory_mb():.0f}")

    def progress(row_idx, genome_id, vocab_size, nnz_so_far, mem_mb):
        if (row_idx + 1) % 25 == 0 or row_idx == 0:
            logline(log_fh, f"  genome {row_idx + 1}/{len(genome_order)} ({genome_id}): "
                    f"vocab_size={vocab_size:,} nnz_so_far={nnz_so_far:,} peak_rss_mb={mem_mb:.0f}")
        if mem_mb > MEMORY_LIMIT_MB:
            logline(log_fh, f"  SAFETY ABORT: peak RSS {mem_mb:.0f} MB exceeded limit {MEMORY_LIMIT_MB:.0f} MB "
                    f"after {row_idx + 1}/{len(genome_order)} genomes.")

    t0 = time.perf_counter()
    build_result = build_minimizer_matrix_streaming(
        contigs_by_genome, genome_order, k=K, w=W, memory_limit_mb=MEMORY_LIMIT_MB, progress_callback=progress
    )
    extraction_seconds = time.perf_counter() - t0
    mem_after_mb = peak_memory_mb()

    matrix = build_result.matrix
    vocabulary = build_result.vocabulary
    processed_ids = build_result.processed_genome_ids
    n_genomes_used = len(processed_ids)

    logline(log_fh, f"{'COMPLETE' if build_result.complete else 'INCOMPLETE'}: {n_genomes_used}/"
            f"{build_result.n_requested_genomes} genomes in {extraction_seconds:.1f}s, peak_rss_mb={mem_after_mb:.0f}")

    # row-alignment confirmation against the reconciled manifest
    row_alignment_ok = processed_ids == genome_order[:n_genomes_used]
    logline(log_fh, f"row-alignment matches reconciled manifest order: {row_alignment_ok}")

    npz_path = DATA_PROCESSED / f"policy_consistent_{ANTIBIOTIC}_minimizer_matrix_k{K}_w{W}.npz"
    sparse.save_npz(npz_path, matrix)
    storage_bytes = npz_path.stat().st_size

    vocab_path = DATA_PROCESSED / f"policy_consistent_{ANTIBIOTIC}_minimizer_vocabulary_k{K}_w{W}.csv"
    pd.DataFrame({"kmer": list(vocabulary), "column_index": list(vocabulary.values())}).to_csv(vocab_path, index=False)

    density = matrix.nnz / (matrix.shape[0] * matrix.shape[1]) if matrix.shape[1] else 0.0
    in_memory_csr_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    per_genome_nnz = np.diff(matrix.indptr).tolist()

    config = {
        "k": K, "w": W, "antibiotic": ANTIBIOTIC, "cohort": "policy_a_reconciled_200",
        "n_genomes_requested": build_result.n_requested_genomes, "n_genomes_processed": n_genomes_used,
        "complete": build_result.complete, "aborted_for_memory": build_result.aborted_for_memory,
        "row_alignment_matches_manifest": row_alignment_ok,
        "n_features": matrix.shape[1], "nnz": int(matrix.nnz), "density": density,
        "extraction_seconds": extraction_seconds, "npz_storage_bytes": storage_bytes,
        "in_memory_csr_bytes": int(in_memory_csr_bytes), "peak_rss_mb": mem_after_mb,
        "mean_selected_features_per_genome": float(np.mean(per_genome_nnz)) if per_genome_nnz else 0.0,
        "min_selected_features_per_genome": int(min(per_genome_nnz)) if per_genome_nnz else 0,
        "max_selected_features_per_genome": int(max(per_genome_nnz)) if per_genome_nnz else 0,
    }
    config_path = DATA_PROCESSED / f"policy_consistent_{ANTIBIOTIC}_minimizer_config_k{K}_w{W}.json"
    config_path.write_text(json.dumps(config, indent=2))
    logline(log_fh, f"matrix: {matrix.shape[0]}x{matrix.shape[1]}, nnz={matrix.nnz:,}, density={density:.4%}, "
            f"npz={storage_bytes/1e6:.1f}MB, mean_features/genome={config['mean_selected_features_per_genome']:.0f}")
    logline(log_fh, f"wrote {config_path}")
    logline(log_fh, "Done.")
    log_fh.close()


if __name__ == "__main__":
    main()
