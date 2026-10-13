"""Builds the k=15/n_buckets=262144 hashed-feature SQLite store for the
Policy-A-reconciled 200-genome cohort, using the EXISTING, already-
retrieved FASTA assemblies (never re-retrieves anything) and the
EXISTING, unmodified `sqlite_row_store.build_sqlite_backed_matrix`.

Genome order is the reconciled manifest's own canonical order
(alphabetically sorted genome IDs - see `cohort_reconciliation.py`),
NOT the prior task's pilot-then-expansion concatenation order - this is
a genuinely new, independently-verifiable build, not a copy.

Usage (from the project root, with the project venv active, AFTER
run_cohort_reconciliation.py has written the reconciled selected-cohort
CSV):

    python src/run_policy_consistent_hashed_store.py

Writes:
    data/processed/policy_consistent_ciprofloxacin_rows_k15_d262144.db
    data/processed/policy_consistent_ciprofloxacin_hashed_build_config_k15_d262144.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from bvbrc_recon.sqlite_row_store import FailureLog, ResourceLimits, SqliteRowStore, build_sqlite_backed_matrix
from bvbrc_recon.streaming_kmer_build import DEFAULT_MEMORY_LIMIT_MB, peak_memory_mb

ANTIBIOTIC = "ciprofloxacin"
K = 15
N_BUCKETS = 2**18
MEMORY_LIMIT_MB = DEFAULT_MEMORY_LIMIT_MB

DATA_PROCESSED = Path("data/processed")
DATA_RAW_PILOT_DIR = Path("data/raw") / f"{ANTIBIOTIC}_pilot"
DATA_RAW_EXPANSION_DIR = Path("data/raw") / f"{ANTIBIOTIC}_expansion"
DB_PATH = DATA_PROCESSED / f"policy_consistent_{ANTIBIOTIC}_rows_k{K}_d{N_BUCKETS}.db"
FAILURES_PATH = DATA_PROCESSED / f"policy_consistent_{ANTIBIOTIC}_hashed_failures_k{K}_d{N_BUCKETS}.jsonl"
CONFIG_PATH = DATA_PROCESSED / f"policy_consistent_{ANTIBIOTIC}_hashed_build_config_k{K}_d{N_BUCKETS}.json"
GENOME_ORDER_PATH = DATA_PROCESSED / f"policy_consistent_{ANTIBIOTIC}_genome_order.csv"


def contigs_loader(genome_id: str) -> list[str]:
    fasta_path = DATA_RAW_PILOT_DIR / f"{genome_id}.fna"
    if not fasta_path.exists():
        fasta_path = DATA_RAW_EXPANSION_DIR / f"{genome_id}.fna"
    return [str(record.seq) for record in SeqIO.parse(str(fasta_path), "fasta")]


def main() -> None:
    selected = pd.read_csv(
        DATA_PROCESSED / f"policy_consistent_{ANTIBIOTIC}_selected_cohort.csv", dtype={"genome_id": str}
    )
    genome_order = selected["genome_id"].tolist()  # already the reconciled manifest's canonical (sorted) order
    print(f"reconciled cohort: {len(genome_order)} genomes")
    pd.DataFrame({"genome_id": genome_order}).to_csv(GENOME_ORDER_PATH, index=False)

    row_store = SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS)
    failure_log = FailureLog(FAILURES_PATH)
    limits = ResourceLimits(memory_limit_mb=MEMORY_LIMIT_MB, min_free_disk_bytes=1_000_000_000, max_failed_rows=5, batch_size=16)

    def progress(row_idx, genome_id, n_written, mem_mb):
        if (n_written % 25 == 0) or n_written == 1:
            print(f"  genome row_index={row_idx} ({genome_id}): rows_written={n_written} peak_rss_mb={mem_mb:.0f}", flush=True)

    status = build_sqlite_backed_matrix(
        contigs_loader, genome_order, k=K, n_buckets=N_BUCKETS,
        row_store=row_store, failure_log=failure_log, limits=limits, progress_callback=progress,
    )
    print(f"build: complete={status.complete} rows_written={status.n_rows_written_this_run} "
          f"failures={status.n_failures_total} elapsed={status.elapsed_seconds:.1f}s peak_rss_mb={status.peak_rss_mb:.0f}")

    # row-alignment confirmation: row_index i must correspond to genome_order[i]
    mismatches = 0
    for i, gid in enumerate(genome_order):
        row = row_store.get_row(i)
        if row is None or row[0] != gid:
            mismatches += 1
    print(f"row-alignment check against reconciled manifest: {len(genome_order) - mismatches}/{len(genome_order)} match, "
          f"mismatches={mismatches}")

    db_bytes = DB_PATH.stat().st_size
    CONFIG_PATH.write_text(json.dumps({
        "k": K, "n_buckets": N_BUCKETS, "antibiotic": ANTIBIOTIC,
        "n_genomes_requested": status.n_genomes_requested, "n_rows_total": status.n_rows_total,
        "n_failures_total": status.n_failures_total, "complete": status.complete,
        "elapsed_seconds": status.elapsed_seconds, "peak_rss_mb": status.peak_rss_mb,
        "db_size_bytes": db_bytes, "row_alignment_mismatches": mismatches,
    }, indent=2))
    print(f"wrote {CONFIG_PATH}")


if __name__ == "__main__":
    main()
