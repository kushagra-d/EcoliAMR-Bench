"""End-to-end ciprofloxacin pilot: cohort selection, complete-assembly
retrieval, sequence QC, canonical k-mer features, and one preliminary
Logistic Regression model.

PRELIMINARY PILOT ONLY - see reports/pilot_ciprofloxacin_report.md for
every caveat. This script does not select the final modelling cohort,
does not implement lineage-aware splitting, and its metrics are NOT the
three-antibiotic benchmark.

Usage (from the project root, with the project venv active):

    python src/run_pilot.py

Re-running is safe: it never overwrites an already-retrieved FASTA file,
and cohort selection is deterministic given the fixed seed.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from bvbrc_recon.client import BVBRCClient
from bvbrc_recon.csv_loader import load_antibiotic_csv
from bvbrc_recon.kmer_features import build_feature_matrix, count_kmers_in_assembly, estimate_dense_bytes
from bvbrc_recon.pilot_cohort import select_pilot_cohort
from bvbrc_recon.pilot_model import train_and_evaluate_pilot_model
from bvbrc_recon.sequence_qc import qc_assembly
from bvbrc_recon.sequence_retrieval import ASSEMBLY_STATUS_COMPLETE, fetch_complete_assembly, write_assembly_fasta_atomic

ANTIBIOTIC = "ciprofloxacin"
SEED = 20261009
TARGET_SIZE = 100
# Pilot k-mer size: the submitted proposal states k is approximately 10;
# instructor feedback separately raised a concern about k=31. This is NOT
# resolved here - k=10 is used for this pilot because it is cheap enough to
# run quickly on ~100 genomes and matches the literal proposal value, not
# because the k=31 concern has been settled. See the report's Step 5 section.
PILOT_K = 10
TEST_SIZE = 0.3

DATA_PROCESSED = Path("data/processed")
DATA_RAW_PILOT_DIR = Path("data/raw") / f"{ANTIBIOTIC}_pilot"
REPORT_PATH = Path("reports") / f"pilot_{ANTIBIOTIC}_report.md"


def load_contig_sequences(fasta_path: Path) -> list[str]:
    return [str(record.seq) for record in SeqIO.parse(str(fasta_path), "fasta")]


def step2_select_cohort() -> tuple:
    print("=== Step 2: pilot cohort selection ===")
    records = load_antibiotic_csv(DATA_PROCESSED / f"{ANTIBIOTIC}_amr_metadata.csv")
    cohort = select_pilot_cohort(records, ANTIBIOTIC, target_size=TARGET_SIZE, seed=SEED)

    contigs_by_genome: dict[str, int] = {}
    for r in records:
        gid = r.get("genome_id")
        gc = r.get("genome_contigs")
        if gid and gc is not None and gid not in contigs_by_genome:
            contigs_by_genome[gid] = int(gc)

    cohort.genome_decision_table.to_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_cohort_decision_table.csv", index=False
    )
    selected_df = pd.DataFrame(
        {
            "genome_id": cohort.selected_genome_ids,
            "label": [cohort.selected_labels[g] for g in cohort.selected_genome_ids],
        }
    )
    selected_df.to_csv(DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_selected_cohort.csv", index=False)

    print(
        f"Selected {len(cohort.selected_genome_ids)} genomes: "
        f"{cohort.n_selected_resistant} Resistant, {cohort.n_selected_susceptible} Susceptible"
    )
    for note in cohort.limitations:
        print(f"  NOTE: {note}")
    return cohort, contigs_by_genome


def step3_retrieve_assemblies(cohort, contigs_by_genome: dict[str, int]) -> pd.DataFrame:
    print("\n=== Step 3: complete-assembly retrieval ===")
    DATA_RAW_PILOT_DIR.mkdir(parents=True, exist_ok=True)
    client = BVBRCClient()
    manifest_rows = []

    for i, genome_id in enumerate(cohort.selected_genome_ids, start=1):
        fasta_path = DATA_RAW_PILOT_DIR / f"{genome_id}.fna"
        if fasta_path.exists():
            print(f"[{i}/{len(cohort.selected_genome_ids)}] {genome_id}: already present, skipping fetch")
            manifest_rows.append(
                {
                    "genome_id": genome_id,
                    "status": "already_present_skipped",
                    "n_contigs_retrieved": None,
                    "n_contigs_in_index": None,
                    "n_contigs_expected_from_metadata": contigs_by_genome.get(genome_id),
                    "total_length_retrieved": None,
                    "duplicate_sequence_ids": "",
                    "errors": "",
                    "fasta_path": str(fasta_path),
                }
            )
            continue

        expected = contigs_by_genome.get(genome_id)
        t0 = time.time()
        result = fetch_complete_assembly(client, genome_id, expected_contigs=expected)
        elapsed = time.time() - t0

        if result.status == ASSEMBLY_STATUS_COMPLETE:
            write_assembly_fasta_atomic(fasta_path, result.contigs)

        print(
            f"[{i}/{len(cohort.selected_genome_ids)}] {genome_id}: status={result.status} "
            f"contigs={result.n_contigs_retrieved}/{result.n_contigs_in_index} "
            f"({elapsed:.1f}s)"
        )
        manifest_rows.append(
            {
                "genome_id": genome_id,
                "status": result.status,
                "n_contigs_retrieved": result.n_contigs_retrieved,
                "n_contigs_in_index": result.n_contigs_in_index,
                "n_contigs_expected_from_metadata": result.n_contigs_expected_from_metadata,
                "total_length_retrieved": result.total_length_retrieved,
                "duplicate_sequence_ids": ";".join(result.duplicate_sequence_ids),
                "errors": " | ".join(result.errors),
                "fasta_path": str(fasta_path) if result.status == ASSEMBLY_STATUS_COMPLETE else "",
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_retrieval_manifest.csv", index=False)
    return manifest_df


def step4_qc_assemblies(manifest_df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Step 4: sequence QC ===")
    rows = []
    for _, row in manifest_df.iterrows():
        if row["status"] not in (ASSEMBLY_STATUS_COMPLETE, "already_present_skipped"):
            continue
        fasta_path = Path(row["fasta_path"]) if row["fasta_path"] else DATA_RAW_PILOT_DIR / f"{row['genome_id']}.fna"
        if not fasta_path.exists():
            continue
        qc = qc_assembly(row["genome_id"], fasta_path)
        rows.append(
            {
                "genome_id": qc.genome_id,
                "n_contigs": qc.n_contigs,
                "total_length": qc.total_length,
                "ambiguous_base_count": qc.ambiguous_base_count,
                "ambiguous_base_proportion": qc.ambiguous_base_proportion,
                "duplicate_sequence_ids": ";".join(qc.duplicate_sequence_ids),
                "empty_contig_count": qc.empty_contig_count,
                "is_nonempty": qc.is_nonempty,
                "has_valid_nucleotide_data": qc.has_valid_nucleotide_data,
                "warnings": " | ".join(qc.warnings),
                "error": qc.error,
            }
        )
    qc_df = pd.DataFrame(rows)
    qc_df.to_csv(DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_sequence_qc.csv", index=False)
    print(f"QC'd {len(qc_df)} assemblies")
    return qc_df


def step5_build_kmer_matrix(manifest_df: pd.DataFrame, cohort):
    print(f"\n=== Step 5: canonical k-mer features (k={PILOT_K}) ===")
    usable_genome_ids = [
        gid
        for gid, status in zip(manifest_df["genome_id"], manifest_df["status"])
        if status in (ASSEMBLY_STATUS_COMPLETE, "already_present_skipped")
        and (DATA_RAW_PILOT_DIR / f"{gid}.fna").exists()
    ]
    print(f"Usable assemblies for feature extraction: {len(usable_genome_ids)}/{len(cohort.selected_genome_ids)}")

    # Exact theoretical max: every k-mer pairs with its reverse complement
    # except self-complementary ("palindromic") k-mers, which exist only for
    # even k (there are 4**(k/2) of them) and do not get halved.
    if PILOT_K % 2 == 0:
        max_possible_canonical_kmers = (4**PILOT_K + 4 ** (PILOT_K // 2)) // 2
    else:
        max_possible_canonical_kmers = 4**PILOT_K // 2
    print(f"Theoretical max canonical {PILOT_K}-mer vocabulary size: {max_possible_canonical_kmers:,}")
    print(
        f"Estimated DENSE matrix size at that upper bound for {len(usable_genome_ids)} genomes: "
        f"{estimate_dense_bytes(len(usable_genome_ids), max_possible_canonical_kmers) / 1e9:.2f} GB "
        f"(this is why the matrix is built sparse, never densified)"
    )

    genome_kmer_counts = {}
    for gid in usable_genome_ids:
        contigs = load_contig_sequences(DATA_RAW_PILOT_DIR / f"{gid}.fna")
        genome_kmer_counts[gid] = count_kmers_in_assembly(contigs, PILOT_K)

    genome_order = usable_genome_ids
    result = build_feature_matrix(genome_kmer_counts, k=PILOT_K, genome_order=genome_order)
    print(
        f"Matrix: {result.n_genomes} genomes x {result.n_features} observed features, "
        f"nnz={result.nnz} (sparsity={1 - result.nnz / (result.n_genomes * result.n_features):.4%})"
    )

    from scipy import sparse

    sparse.save_npz(DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_matrix.npz", result.matrix)
    pd.DataFrame({"genome_id": result.genome_order}).to_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_genome_order.csv", index=False
    )
    pd.DataFrame({"kmer": list(result.vocabulary), "column_index": list(result.vocabulary.values())}).to_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_vocabulary.csv", index=False
    )
    config = {
        "k": PILOT_K,
        "seed": SEED,
        "antibiotic": ANTIBIOTIC,
        "n_genomes": result.n_genomes,
        "n_features": result.n_features,
        "nnz": result.nnz,
        "input_fasta_dir": str(DATA_RAW_PILOT_DIR),
    }
    (DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_config.json").write_text(json.dumps(config, indent=2))
    return result


def step6_train_model(kmer_result, cohort):
    print("\n=== Step 6: preliminary Logistic Regression pilot model ===")
    labels = [cohort.selected_labels[gid] for gid in kmer_result.genome_order]
    model_result = train_and_evaluate_pilot_model(kmer_result.matrix, labels, seed=SEED, test_size=TEST_SIZE)
    print(f"Train: {model_result.n_train} ({model_result.train_class_counts})")
    print(f"Test:  {model_result.n_test} ({model_result.test_class_counts})")
    print(f"Accuracy: {model_result.accuracy}")
    print(f"ROC-AUC: {model_result.roc_auc}")
    print(f"Average precision: {model_result.average_precision}")
    print(f"Recall per class: {model_result.recall_per_class}")
    print(f"Confusion matrix [[TN,FP],[FN,TP]]: {model_result.confusion_matrix}")
    for w in model_result.warnings:
        print(f"  WARNING: {w}")
    return model_result


def main():
    cohort, contigs_by_genome = step2_select_cohort()
    manifest_df = step3_retrieve_assemblies(cohort, contigs_by_genome)
    qc_df = step4_qc_assemblies(manifest_df)
    kmer_result = step5_build_kmer_matrix(manifest_df, cohort)
    model_result = step6_train_model(kmer_result, cohort)

    from bvbrc_recon.pilot_report import render_pilot_report

    report_text = render_pilot_report(
        antibiotic=ANTIBIOTIC,
        seed=SEED,
        k=PILOT_K,
        analysis_date=datetime.now(timezone.utc).isoformat(),
        cohort=cohort,
        manifest_df=manifest_df,
        qc_df=qc_df,
        kmer_result=kmer_result,
        model_result=model_result,
        test_size=TEST_SIZE,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"\nPilot report written -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
