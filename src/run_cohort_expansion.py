"""Reproducible, bounded, resumable cohort-expansion CLI: retrieves,
validates, and QCs a configurable number of ADDITIONAL real ciprofloxacin
genomes beyond the existing 100-genome pilot, using Policy A eligibility
(the officially recommended cross-antibiotic rule) and reusing the
existing retrieval/QC infrastructure (`sequence_retrieval.py`,
`sequence_qc.py`) exactly as the original pilot did.

Does NOT touch the original pilot's FASTA directory, selected-cohort
CSV, matrices, or splits - everything here writes to new,
`_expansion`-suffixed paths. Does NOT retrieve the full 8,710-genome
cohort - bounded to `--target-additional` (default 100).

RESUMABILITY AND IDEMPOTENCY:
  - Selection is persisted on first run
    (`cohort_expansion_ciprofloxacin_selected.csv`) and REUSED on every
    later run, even if the underlying metadata CSV changes in the
    meantime - the first run's selection is authoritative, so re-running
    never silently re-samples a different cohort.
  - Retrieval is resumable by FASTA-file existence: a genome whose
    `.fna` file already exists under the expansion's raw directory is
    never re-fetched. A genome that failed in a prior run is retried on
    the next run (BV-BRC availability can be transient) - failures are
    never permanently blacklisted, but a single run stops once
    `--max-failures` NEW failures have occurred, to avoid a hung/looping
    run against a systemically failing endpoint.
  - The manifest is rewritten fresh each run from the CURRENT on-disk
    state plus this run's attempts - a genome is marked "complete" ONLY
    if its FASTA file exists AND was validated complete by
    `fetch_complete_assembly` (or was already complete going in); a
    genome is never silently marked successful just because a previous
    run touched it.

Usage (from the project root, with the project venv active):

    python src/run_cohort_expansion.py                      # default: 100 additional genomes
    python src/run_cohort_expansion.py --target-additional 50
    python src/run_cohort_expansion.py --max-failures 10
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from bvbrc_recon.client import BVBRCClient
from bvbrc_recon.cohort_expansion import select_expansion_cohort
from bvbrc_recon.csv_loader import load_antibiotic_csv
from bvbrc_recon.sequence_qc import qc_assembly
from bvbrc_recon.sequence_retrieval import ASSEMBLY_STATUS_COMPLETE, fetch_complete_assembly, write_assembly_fasta_atomic

ANTIBIOTIC = "ciprofloxacin"
SEED = 20261009
DEFAULT_TARGET_ADDITIONAL = 100
DEFAULT_MAX_FAILURES = 20

DATA_PROCESSED = Path("data/processed")
DATA_RAW_PILOT_DIR = Path("data/raw") / f"{ANTIBIOTIC}_pilot"
DATA_RAW_EXPANSION_DIR = Path("data/raw") / f"{ANTIBIOTIC}_expansion"

# NOTE: the paths below are computed INSIDE each function (from the
# module-global DATA_PROCESSED/DATA_RAW_EXPANSION_DIR at call time),
# not frozen here at import time - this is what lets tests monkeypatch
# DATA_PROCESSED/DATA_RAW_EXPANSION_DIR and have every derived path
# follow, matching this project's established test pattern (e.g.
# `run_minimizer_experiment.py`).


def load_existing_pilot_genome_ids() -> set[str]:
    df = pd.read_csv(DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_selected_cohort.csv", dtype={"genome_id": str})
    return set(df["genome_id"])


def step1_select_or_load(target_additional: int, seed: int) -> pd.DataFrame:
    """Idempotent: if a selection was already persisted, reuse it
    verbatim (ignoring `target_additional`/`seed` on this call) rather
    than re-deriving a possibly-different sample."""
    selected_path = DATA_PROCESSED / f"cohort_expansion_{ANTIBIOTIC}_selected.csv"
    decision_notes_path = DATA_PROCESSED / f"cohort_expansion_{ANTIBIOTIC}_selection_notes.json"
    if selected_path.exists():
        print(f"[select] reusing already-persisted selection at {selected_path}")
        return pd.read_csv(selected_path, dtype={"genome_id": str})

    print(f"=== Step 1: selecting up to {target_additional} additional genomes (Policy A, seed={seed}) ===")
    pilot_ids = load_existing_pilot_genome_ids()
    print(f"  excluding {len(pilot_ids)} genome IDs already in the 100-genome pilot")

    records = load_antibiotic_csv(DATA_PROCESSED / f"{ANTIBIOTIC}_amr_metadata.csv")
    mlst_by_genome: dict[str, str | None] = {}
    for r in records:
        gid = r.get("genome_id")
        if gid is not None and gid not in mlst_by_genome:
            mlst_by_genome[gid] = r.get("genome_mlst")

    selection = select_expansion_cohort(
        records, mlst_by_genome, ANTIBIOTIC, pilot_ids, target_additional=target_additional, seed=seed
    )
    for note in selection.limitations:
        print(f"  NOTE: {note}")

    selected_df = pd.DataFrame(
        {
            "genome_id": selection.selected_genome_ids,
            "label": [selection.selected_labels[g] for g in selection.selected_genome_ids],
        }
    )
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(selected_path, index=False)

    import json

    decision_notes_path.write_text(json.dumps(
        {
            "antibiotic": ANTIBIOTIC, "seed": seed, "target_additional": target_additional,
            "eligible_resistant_pool_size": selection.eligible_resistant_pool_size,
            "eligible_susceptible_pool_size": selection.eligible_susceptible_pool_size,
            "eligible_pool_size_after_pilot_exclusion": selection.eligible_pool_size_after_pilot_exclusion,
            "n_selected_resistant": selection.n_selected_resistant,
            "n_selected_susceptible": selection.n_selected_susceptible,
            "limitations": selection.limitations,
        },
        indent=2,
    ))
    print(f"  selected {len(selected_df)}: {selection.n_selected_resistant} Resistant, "
          f"{selection.n_selected_susceptible} Susceptible -> wrote {selected_path}")
    return selected_df


def step2_retrieve(selected_df: pd.DataFrame, max_failures: int) -> pd.DataFrame:
    print(f"\n=== Step 2: complete-assembly retrieval (max {max_failures} new failures this run) ===")
    DATA_RAW_EXPANSION_DIR.mkdir(parents=True, exist_ok=True)
    client = BVBRCClient()

    records = load_antibiotic_csv(DATA_PROCESSED / f"{ANTIBIOTIC}_amr_metadata.csv")
    contigs_by_genome: dict[str, int] = {}
    for r in records:
        gid, gc = r.get("genome_id"), r.get("genome_contigs")
        if gid and gc is not None and gid not in contigs_by_genome:
            contigs_by_genome[gid] = int(gc)

    manifest_rows = []
    n_new_failures = 0
    genome_ids = selected_df["genome_id"].tolist()

    for i, genome_id in enumerate(genome_ids, start=1):
        fasta_path = DATA_RAW_EXPANSION_DIR / f"{genome_id}.fna"
        if fasta_path.exists():
            print(f"[{i}/{len(genome_ids)}] {genome_id}: already present, skipping fetch")
            manifest_rows.append({
                "genome_id": genome_id, "status": "already_present_skipped",
                "n_contigs_retrieved": None, "n_contigs_in_index": None,
                "n_contigs_expected_from_metadata": contigs_by_genome.get(genome_id),
                "total_length_retrieved": None, "duplicate_sequence_ids": "", "errors": "",
                "retrieval_seconds": None, "fasta_path": str(fasta_path),
            })
            continue

        if n_new_failures >= max_failures:
            print(f"[{i}/{len(genome_ids)}] {genome_id}: SKIPPED - max_failures ({max_failures}) reached this run")
            manifest_rows.append({
                "genome_id": genome_id, "status": "skipped_max_failures_reached",
                "n_contigs_retrieved": None, "n_contigs_in_index": None,
                "n_contigs_expected_from_metadata": contigs_by_genome.get(genome_id),
                "total_length_retrieved": None, "duplicate_sequence_ids": "", "errors": "",
                "retrieval_seconds": None, "fasta_path": "",
            })
            continue

        expected = contigs_by_genome.get(genome_id)
        t0 = time.time()
        result = fetch_complete_assembly(client, genome_id, expected_contigs=expected)
        elapsed = time.time() - t0

        if result.status == ASSEMBLY_STATUS_COMPLETE:
            write_assembly_fasta_atomic(fasta_path, result.contigs)
        else:
            n_new_failures += 1

        print(f"[{i}/{len(genome_ids)}] {genome_id}: status={result.status} "
              f"contigs={result.n_contigs_retrieved}/{result.n_contigs_in_index} ({elapsed:.1f}s)")
        manifest_rows.append({
            "genome_id": genome_id, "status": result.status,
            "n_contigs_retrieved": result.n_contigs_retrieved, "n_contigs_in_index": result.n_contigs_in_index,
            "n_contigs_expected_from_metadata": result.n_contigs_expected_from_metadata,
            "total_length_retrieved": result.total_length_retrieved,
            "duplicate_sequence_ids": ";".join(result.duplicate_sequence_ids),
            "errors": " | ".join(result.errors), "retrieval_seconds": elapsed,
            "fasta_path": str(fasta_path) if result.status == ASSEMBLY_STATUS_COMPLETE else "",
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = DATA_PROCESSED / f"cohort_expansion_{ANTIBIOTIC}_retrieval_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    n_complete = (manifest_df["status"].isin([ASSEMBLY_STATUS_COMPLETE, "already_present_skipped"])).sum()
    print(f"  retrieval complete: {n_complete}/{len(genome_ids)} usable, {n_new_failures} new failures this run")
    return manifest_df


def step3_qc(manifest_df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Step 3: sequence QC ===")
    rows = []
    for _, row in manifest_df.iterrows():
        if row["status"] not in (ASSEMBLY_STATUS_COMPLETE, "already_present_skipped"):
            continue
        fasta_path = Path(row["fasta_path"]) if row["fasta_path"] else DATA_RAW_EXPANSION_DIR / f"{row['genome_id']}.fna"
        if not fasta_path.exists():
            continue
        qc = qc_assembly(row["genome_id"], fasta_path)
        rows.append({
            "genome_id": qc.genome_id, "n_contigs": qc.n_contigs, "total_length": qc.total_length,
            "ambiguous_base_count": qc.ambiguous_base_count,
            "ambiguous_base_proportion": qc.ambiguous_base_proportion,
            "duplicate_sequence_ids": ";".join(qc.duplicate_sequence_ids),
            "empty_contig_count": qc.empty_contig_count, "is_nonempty": qc.is_nonempty,
            "has_valid_nucleotide_data": qc.has_valid_nucleotide_data,
            "warnings": " | ".join(qc.warnings), "error": qc.error,
        })
    qc_df = pd.DataFrame(rows)
    qc_path = DATA_PROCESSED / f"cohort_expansion_{ANTIBIOTIC}_sequence_qc.csv"
    qc_df.to_csv(qc_path, index=False)
    print(f"  QC'd {len(qc_df)} assemblies -> wrote {qc_path}")
    return qc_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-additional", type=int, default=DEFAULT_TARGET_ADDITIONAL)
    parser.add_argument("--max-failures", type=int, default=DEFAULT_MAX_FAILURES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    selected_df = step1_select_or_load(args.target_additional, args.seed)
    manifest_df = step2_retrieve(selected_df, args.max_failures)
    step3_qc(manifest_df)


if __name__ == "__main__":
    main()
