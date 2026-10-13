"""Generates the PRIMARY feature representation for the frozen
956-genome course-project cohort (`course_project_final_cohort_manifest.csv`,
see reports/course_project_final_cohort.md).

Representation selected (see reports/course_project_feature_generation.md
for the comparison): direct feature hashing, k=15, n_buckets=262144,
built with the EXISTING, already-validated out-of-core SQLite row store
(`bvbrc_recon.sqlite_row_store.build_sqlite_backed_matrix` -
genuinely streaming: hashes directly during the per-genome sequence
walk, one genome's contigs in memory at a time, never a prerequisite
raw k-mer matrix) - the same implementation already exercised on the
200-genome cohort-expansion and policy-consistent cohorts
(`run_cohort_expansion_hashed_store.py`, `run_policy_consistent_hashed_store.py`),
reused here unmodified at the 956-genome scale.

One feature matrix is built for the whole shared cohort (genomic
features do not depend on antibiotic); antibiotic-specific labels are
read separately from `course_project_final_cohort_{antibiotic}_manifest.csv`
and are NEVER written into this matrix or its row store.

Does not download any genome, does not touch any pilot/expansion/
cohort_expansion/policy_consistent artifact, and does not train a
model.

Usage (from the project root, with the project venv active):

    python src/run_course_project_feature_generation.py

Writes:
    data/processed/course_project_final_cohort_rows_k15_d262144.db
    data/processed/course_project_final_cohort_hashed_failures_k15_d262144.jsonl
    data/processed/course_project_final_cohort_hashed_build_config_k15_d262144.json
    data/processed/course_project_final_cohort_genome_order.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from bvbrc_recon.sqlite_row_store import FailureLog, ResourceLimits, SqliteRowStore, build_sqlite_backed_matrix
from bvbrc_recon.streaming_kmer_build import DEFAULT_MEMORY_LIMIT_MB

K = 15
N_BUCKETS = 2**18

DATA_PROCESSED = Path("data/processed")
RAW_DIRS = [
    Path("data/raw") / "ciprofloxacin_pilot",
    Path("data/raw") / "ciprofloxacin_expansion",
    Path("data/raw") / "full_cohort_assemblies",
]
COHORT_MANIFEST_PATH = DATA_PROCESSED / "course_project_final_cohort_manifest.csv"
DB_PATH = DATA_PROCESSED / f"course_project_final_cohort_rows_k{K}_d{N_BUCKETS}.db"
FAILURES_PATH = DATA_PROCESSED / f"course_project_final_cohort_hashed_failures_k{K}_d{N_BUCKETS}.jsonl"
CONFIG_PATH = DATA_PROCESSED / f"course_project_final_cohort_hashed_build_config_k{K}_d{N_BUCKETS}.json"
GENOME_ORDER_PATH = DATA_PROCESSED / "course_project_final_cohort_genome_order.csv"


def contigs_loader(genome_id: str) -> list[str]:
    for d in RAW_DIRS:
        fasta_path = d / f"{genome_id}.fna"
        if fasta_path.exists():
            return [str(record.seq) for record in SeqIO.parse(str(fasta_path), "fasta")]
    raise FileNotFoundError(f"no assembly found for {genome_id} in any of {[str(d) for d in RAW_DIRS]}")


def main() -> None:
    cohort = pd.read_csv(COHORT_MANIFEST_PATH, dtype={"genome_id": str})
    genome_order = sorted(cohort["genome_id"].tolist())  # deterministic ordering, independent of manifest row order
    if len(genome_order) != len(set(genome_order)):
        raise RuntimeError("REFUSING TO PROCEED: duplicate genome_id in the frozen cohort manifest.")
    print(f"frozen cohort: {len(genome_order)} genomes")
    pd.DataFrame({"genome_id": genome_order, "row_index": range(len(genome_order))}).to_csv(GENOME_ORDER_PATH, index=False)

    row_store = SqliteRowStore(DB_PATH, n_buckets=N_BUCKETS)
    failure_log = FailureLog(FAILURES_PATH)
    limits = ResourceLimits(
        memory_limit_mb=DEFAULT_MEMORY_LIMIT_MB, min_free_disk_bytes=1_000_000_000, max_failed_rows=5, batch_size=16
    )

    def progress(row_idx, genome_id, n_written, mem_mb):
        if (n_written % 50 == 0) or n_written == 1:
            print(f"  genome row_index={row_idx} ({genome_id}): rows_written={n_written} peak_rss_mb={mem_mb:.0f}", flush=True)

    status = build_sqlite_backed_matrix(
        contigs_loader, genome_order, k=K, n_buckets=N_BUCKETS,
        row_store=row_store, failure_log=failure_log, limits=limits, progress_callback=progress,
    )
    print(f"build: complete={status.complete} rows_written={status.n_rows_written_this_run} "
          f"failures={status.n_failures_total} elapsed={status.elapsed_seconds:.1f}s peak_rss_mb={status.peak_rss_mb:.0f}")

    # row-alignment confirmation: row_index i must correspond to genome_order[i] - same
    # check pattern used by the cohort_expansion/policy_consistent hashed-store builds.
    mismatches = 0
    for i, gid in enumerate(genome_order):
        row = row_store.get_row(i)
        if row is None or row[0] != gid:
            mismatches += 1
    print(f"row-alignment check against frozen cohort manifest: {len(genome_order) - mismatches}/{len(genome_order)} match, "
          f"mismatches={mismatches}")

    total_nnz = row_store._conn.execute("SELECT SUM(n_entries) FROM rows").fetchone()[0] or 0
    density = total_nnz / (len(genome_order) * N_BUCKETS) if genome_order else 0.0

    db_bytes = DB_PATH.stat().st_size
    CONFIG_PATH.write_text(json.dumps({
        "k": K, "n_buckets": N_BUCKETS, "cohort": "course_project_final_956",
        "n_genomes_requested": status.n_genomes_requested, "n_rows_total": status.n_rows_total,
        "n_failures_total": status.n_failures_total, "complete": status.complete,
        "elapsed_seconds": status.elapsed_seconds, "peak_rss_mb": status.peak_rss_mb,
        "nnz": int(total_nnz), "density": density,
        "db_size_bytes": db_bytes, "row_alignment_mismatches": mismatches,
    }, indent=2))
    print(f"nnz={total_nnz:,} density={density:.4%} db_size={db_bytes/1e6:.1f}MB")
    print(f"wrote {CONFIG_PATH}, {GENOME_ORDER_PATH}, {DB_PATH}")
    row_store.close()


if __name__ == "__main__":
    main()
