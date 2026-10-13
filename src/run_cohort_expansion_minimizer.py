"""Step 4 of the cohort-expansion task: builds the k=31/w=15 minimizer
representation on the COMBINED 200-genome cohort (the existing
100-genome pilot + the 100 newly-retrieved expansion genomes), using
the EXISTING, already-tested `minimizer_extraction.build_minimizer_matrix_streaming`
unchanged - a genuinely exploratory comparison point against the
existing 100-genome pilot's own k=31/w=15 minimizer artifact (never
overwritten).

Does not modify any existing raw/hashed/minimizer/hybrid-cache artifact.
Writes only to new `cohort_expansion_`-prefixed paths.

Usage (from the project root, with the project venv active, AFTER
run_cohort_expansion.py has retrieved the expansion genomes):

    python src/run_cohort_expansion_minimizer.py

Writes:
    data/processed/cohort_expansion_ciprofloxacin_minimizer_matrix_k31_w15.npz
    data/processed/cohort_expansion_ciprofloxacin_minimizer_vocabulary_k31_w15.csv
    data/processed/cohort_expansion_ciprofloxacin_minimizer_config_k31_w15.json
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
LOG_PATH = DATA_PROCESSED / f"cohort_expansion_{ANTIBIOTIC}_minimizer_k{K}_w{W}_experiment.log"


def load_contig_sequences(fasta_path: Path) -> list[str]:
    return [str(record.seq) for record in SeqIO.parse(str(fasta_path), "fasta")]


def logline(fh, msg: str) -> None:
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


def main() -> None:
    log_fh = open(LOG_PATH, "w")
    logline(log_fh, f"=== cohort-expansion minimizer experiment: k={K}, w={W} ===")

    pilot_genome_order = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_genome_order.csv", dtype={"genome_id": str}
    )["genome_id"].tolist()
    pilot_selected = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_selected_cohort.csv", dtype={"genome_id": str}
    )
    expansion_selected = pd.read_csv(
        DATA_PROCESSED / f"cohort_expansion_{ANTIBIOTIC}_selected.csv", dtype={"genome_id": str}
    )
    expansion_manifest = pd.read_csv(
        DATA_PROCESSED / f"cohort_expansion_{ANTIBIOTIC}_retrieval_manifest.csv", dtype={"genome_id": str}
    )
    usable_expansion_ids = set(
        expansion_manifest.loc[
            expansion_manifest["status"].isin(["complete", "already_present_skipped"]), "genome_id"
        ]
    )
    expansion_genome_order = [gid for gid in expansion_selected["genome_id"] if gid in usable_expansion_ids]
    logline(log_fh, f"pilot genomes: {len(pilot_genome_order)}, usable expansion genomes: "
            f"{len(expansion_genome_order)}/{len(expansion_selected)}")

    combined_genome_order = pilot_genome_order + expansion_genome_order
    label_by_genome = dict(zip(pilot_selected["genome_id"], pilot_selected["label"]))
    label_by_genome.update(dict(zip(expansion_selected["genome_id"], expansion_selected["label"])))

    logline(log_fh, f"combined cohort size: {len(combined_genome_order)}")

    t0 = time.perf_counter()
    contigs_by_genome = {}
    for gid in combined_genome_order:
        fasta_path = DATA_RAW_PILOT_DIR / f"{gid}.fna"
        if not fasta_path.exists():
            fasta_path = DATA_RAW_EXPANSION_DIR / f"{gid}.fna"
        contigs_by_genome[gid] = load_contig_sequences(fasta_path)
    logline(log_fh, f"loaded {len(combined_genome_order)} genomes' contigs in {time.perf_counter() - t0:.1f}s, "
            f"peak_rss_mb={peak_memory_mb():.0f}")

    def progress(row_idx, genome_id, vocab_size, nnz_so_far, mem_mb):
        if (row_idx + 1) % 20 == 0 or row_idx == 0:
            logline(log_fh, f"  genome {row_idx + 1}/{len(combined_genome_order)} ({genome_id}): "
                    f"vocab_size={vocab_size:,} nnz_so_far={nnz_so_far:,} peak_rss_mb={mem_mb:.0f}")
        if mem_mb > MEMORY_LIMIT_MB:
            logline(log_fh, f"  SAFETY ABORT: peak RSS {mem_mb:.0f} MB exceeded limit {MEMORY_LIMIT_MB:.0f} MB "
                    f"after {row_idx + 1}/{len(combined_genome_order)} genomes.")

    t0 = time.perf_counter()
    build_result = build_minimizer_matrix_streaming(
        contigs_by_genome, combined_genome_order, k=K, w=W, memory_limit_mb=MEMORY_LIMIT_MB, progress_callback=progress
    )
    extraction_seconds = time.perf_counter() - t0
    mem_after_mb = peak_memory_mb()

    matrix = build_result.matrix
    vocabulary = build_result.vocabulary
    processed_ids = build_result.processed_genome_ids
    n_genomes_used = len(processed_ids)

    logline(log_fh, f"{'COMPLETE' if build_result.complete else 'INCOMPLETE'}: {n_genomes_used}/"
            f"{build_result.n_requested_genomes} genomes in {extraction_seconds:.1f}s, peak_rss_mb={mem_after_mb:.0f}")

    npz_path = DATA_PROCESSED / f"cohort_expansion_{ANTIBIOTIC}_minimizer_matrix_k{K}_w{W}.npz"
    sparse.save_npz(npz_path, matrix)
    storage_bytes = npz_path.stat().st_size

    vocab_path = DATA_PROCESSED / f"cohort_expansion_{ANTIBIOTIC}_minimizer_vocabulary_k{K}_w{W}.csv"
    pd.DataFrame({"kmer": list(vocabulary), "column_index": list(vocabulary.values())}).to_csv(vocab_path, index=False)

    density = matrix.nnz / (matrix.shape[0] * matrix.shape[1]) if matrix.shape[1] else 0.0
    in_memory_csr_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    per_genome_nnz = np.diff(matrix.indptr).tolist()

    # comparison against the EXISTING 100-genome pilot's own k=31/w=15 artifact
    pilot_config_path = DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_minimizer_config_k{K}_w{W}.json"
    pilot_config = json.loads(pilot_config_path.read_text()) if pilot_config_path.exists() else None

    config = {
        "k": K, "w": W, "antibiotic": ANTIBIOTIC,
        "cohort": "combined_pilot_plus_expansion",
        "n_pilot_genomes": len(pilot_genome_order),
        "n_expansion_genomes_usable": len(expansion_genome_order),
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
        "mean_selected_features_per_genome": float(np.mean(per_genome_nnz)) if per_genome_nnz else 0.0,
        "min_selected_features_per_genome": int(min(per_genome_nnz)) if per_genome_nnz else 0,
        "max_selected_features_per_genome": int(max(per_genome_nnz)) if per_genome_nnz else 0,
        "comparison_to_existing_100_genome_pilot": {
            "pilot_n_features": pilot_config["n_features"] if pilot_config else None,
            "pilot_nnz": pilot_config["nnz"] if pilot_config else None,
            "pilot_density": pilot_config["density"] if pilot_config else None,
            "pilot_mean_selected_features_per_genome": pilot_config["mean_selected_features_per_genome"] if pilot_config else None,
            "pilot_extraction_seconds": pilot_config["extraction_seconds"] if pilot_config else None,
            "pilot_peak_rss_mb": pilot_config["peak_rss_mb"] if pilot_config else None,
            "vocabulary_growth_ratio": (matrix.shape[1] / pilot_config["n_features"]) if pilot_config else None,
            "mean_features_per_genome_ratio": (
                (float(np.mean(per_genome_nnz)) / pilot_config["mean_selected_features_per_genome"])
                if pilot_config and per_genome_nnz else None
            ),
        },
    }
    config_path = DATA_PROCESSED / f"cohort_expansion_{ANTIBIOTIC}_minimizer_config_k{K}_w{W}.json"
    config_path.write_text(json.dumps(config, indent=2))
    logline(log_fh, f"matrix: {matrix.shape[0]}x{matrix.shape[1]}, nnz={matrix.nnz:,}, density={density:.4%}, "
            f"npz={storage_bytes/1e6:.1f}MB, mean_features/genome={config['mean_selected_features_per_genome']:.0f}")
    logline(log_fh, f"wrote {config_path}")
    logline(log_fh, "Done.")
    log_fh.close()


if __name__ == "__main__":
    main()
