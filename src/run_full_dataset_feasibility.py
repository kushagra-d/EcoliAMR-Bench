"""Full-dataset feasibility assessment and pipeline design (metadata-only
+ a small, already-run k=10-vs-k=31 pilot comparison). Does not download
any additional genome, does not run full-scale extraction, does not
train a full benchmark model.

Usage (from the project root, with the project venv active, AFTER
running src/run_kmer_k_comparison.py):

    python src/run_full_dataset_feasibility.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bvbrc_recon.csv_loader import load_antibiotic_csv
from bvbrc_recon.final_policy_analysis import run_all_policies
from bvbrc_recon.full_dataset_report import (
    render_cohort_section,
    render_full_dataset_report,
    render_k_comparison_table,
    render_model_comparison_table,
    render_resource_table,
)
from bvbrc_recon.resource_estimation import (
    estimate_fasta_storage_bytes,
    estimate_matrix_resources,
    estimate_vocabulary_scaling,
)

ANTIBIOTICS = ["ciprofloxacin", "ampicillin", "gentamicin"]
DATA_PROCESSED = Path("data/processed")
REPORT_PATH = Path("reports/full_dataset_feasibility_and_pipeline_plan.md")
MEASURED_PILOT_N_GENOMES = 100
MEASURED_FASTA_MB_PER_GENOME = 495.0 / 100.0  # measured from data/raw/ciprofloxacin_pilot/


def compute_policy_a_and_characterization() -> tuple[dict, dict, set[str]]:
    outcomes_by_antibiotic = {}
    char_by_antibiotic = {}
    eligible_union: set[str] = set()

    for antibiotic in ANTIBIOTICS:
        path = DATA_PROCESSED / f"{antibiotic}_amr_metadata.csv"
        records = load_antibiotic_csv(path)
        mlst_by_genome: dict[str, str | None] = {}
        for r in records:
            gid = r.get("genome_id")
            if gid is not None and gid not in mlst_by_genome:
                mlst_by_genome[gid] = r.get("genome_mlst")

        outcomes = run_all_policies(records, antibiotic, mlst_by_genome)
        outcome_a = outcomes["A"]
        outcomes_by_antibiotic[antibiotic] = outcome_a
        eligible_union |= set(outcome_a.eligible_genome_ids())

        raw_df = pd.read_csv(path, dtype={"genome_id": str})
        eligible_ids = set(outcome_a.eligible_genome_ids())
        eligible_raw = raw_df[raw_df["genome_id"].isin(eligible_ids)].drop_duplicates("genome_id")
        char_by_antibiotic[antibiotic] = {
            "mean_length": eligible_raw["genome_genome_length"].mean(),
            "mean_contigs": eligible_raw["genome_contigs"].mean(),
            "has_assembly": int(eligible_raw["genome_assembly_accession"].notna().sum()),
            "has_biosample": int(eligible_raw["genome_biosample_accession"].notna().sum()),
        }

    return outcomes_by_antibiotic, char_by_antibiotic, eligible_union


def load_k_comparison_configs() -> dict[int, dict]:
    configs = {}
    for k in (10, 31):
        path = DATA_PROCESSED / f"pilot_ciprofloxacin_kmer_config_k{k}_comparison.json"
        configs[k] = json.loads(path.read_text())
    return configs


def main() -> None:
    outcomes, char, eligible_union = compute_policy_a_and_characterization()

    cohort_sections = "\n".join(
        render_cohort_section(antibiotic, outcomes[antibiotic], char[antibiotic])
        for antibiotic in ANTIBIOTICS
    )

    k_configs = load_k_comparison_configs()

    resource_rows = []
    targets = {antibiotic: len(outcomes[antibiotic].eligible_genome_ids()) for antibiotic in ANTIBIOTICS}
    targets["union (unique genomes, all 3 antibiotics)"] = len(eligible_union)
    for k, cfg in k_configs.items():
        for label, n in targets.items():
            vocab = estimate_vocabulary_scaling(
                k=k, pilot_n_features=cfg["n_features"], pilot_n_genomes=MEASURED_PILOT_N_GENOMES, target_n_genomes=n
            )
            mat = estimate_matrix_resources(
                vocabulary=vocab, pilot_n_genomes=MEASURED_PILOT_N_GENOMES, pilot_nnz=cfg["nnz"], target_n_genomes=n
            )
            resource_rows.append(
                {
                    "label": label,
                    "n": n,
                    "k": k,
                    "regime": vocab.regime,
                    "n_features_low": vocab.estimated_n_features_low,
                    "n_features_high": vocab.estimated_n_features_high,
                    "nnz": mat.estimated_nnz,
                    "csr_bytes": mat.estimated_csr_memory_bytes,
                }
            )
    resource_table = render_resource_table(resource_rows)

    fasta_rows = []
    for label, n in targets.items():
        b = estimate_fasta_storage_bytes(n, MEASURED_FASTA_MB_PER_GENOME * 1e6)
        fasta_rows.append(f"| {label} | {n:,} | {b/1e9:.1f} GB |")
    fasta_storage_table = "| Antibiotic (or union) | n genomes | Estimated FASTA storage |\n|---|---|---|\n" + "\n".join(
        fasta_rows
    )

    k_comparison_table = render_k_comparison_table(k_configs)

    log_text = (DATA_PROCESSED / "pilot_ciprofloxacin_kmer_k_comparison.log").read_text()
    model_lines = []
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if line.startswith("[random]") or line.startswith("[lineage]"):
            split_name = "random" if line.startswith("[random]") else "lineage"
            # crude parse of "LR acc=X auc=Y | RF acc=X auc=Y | XGB acc=X auc=Y"
            parts = line.split("|")
            current_k = model_lines  # placeholder, k determined by position in file
            for part, model_name in zip(parts, ["LogisticRegression", "RandomForest", "XGBoost"]):
                try:
                    acc = part.split("acc=")[1].split(" ")[0]
                    auc = part.split("auc=")[1].strip()
                except IndexError:
                    continue
                model_lines.append(f"| ? | {split_name} | {model_name} | {acc} | {auc} |")

    # fix up k labels: first half of model_lines (6 lines) are k=10, second half k=31
    half = len(model_lines) // 2
    model_lines = [line.replace("| ? |", "| 10 |", 1) for line in model_lines[:half]] + [
        line.replace("| ? |", "| 31 |", 1) for line in model_lines[half:]
    ]
    model_comparison_table = render_model_comparison_table(model_lines)

    k_comparison_narrative = """
**Observed pattern (measured on this one 100-genome pilot - not a general conclusion)**: k=31
shows somewhat higher accuracy/ROC-AUC than k=10 for most model/split combinations in this specific
run, consistent with the CARD interpretation task's earlier finding that k=10 produces very weakly
specific sequence matches (CARD's own reference set saturates ~87% of the k=10 space). A larger k
gives each feature more specific sequence information, which is plausible motivation - but this is
one measurement on 100 genomes with no cross-validation, and is not proof that k=31 generalizes
better. The feature-dimensionality risk (32.6M features for 100 genomes) is far more severe than
k=10's, and cannot be dismissed by one favorable-looking run.
"""

    pipeline_design = """
A full-dataset pipeline for all three antibiotics should reuse these EXISTING, already-tested
components as-is, orchestrated by thin per-stage scripts (as already done for the pilot) rather
than a new framework:

1. **Cohort selection** (`bvbrc_recon.final_policy_analysis.apply_policy_a`, already implemented
   and tested): run once per antibiotic against the existing reconnaissance CSVs. Output: a
   decision table per antibiotic (genome_id, outcome, assigned_class), exactly as already produced
   for the pilot and for this task's own Step 2/3.

2. **Deduplicated assembly retrieval** (`bvbrc_recon.sequence_retrieval.fetch_complete_assembly`,
   already implemented and tested): retrieve the UNION of eligible genome_ids across all three
   antibiotics ONCE (not per antibiotic - see the measured overlap above), reusing the existing
   batched, retry-safe `genome_sequence` FASTA fetch. Retry/resume: already implemented (skips a
   genome whose FASTA file already exists on disk, per the existing pilot retrieval script's
   "never overwrite existing user data" behavior) - this makes a full run resumable for free after
   an interruption.
   **Manifest and integrity validation**: extend the existing per-genome manifest pattern
   (`pilot_{antibiotic}_retrieval_manifest.csv`) with one row per UNIQUE genome (not per antibiotic),
   recording status, contig-count-vs-metadata agreement (already implemented), and any duplicate
   contig ids (already implemented) - no new validation logic is needed, only applying the existing
   logic across the larger, deduplicated genome set.

3. **Sequence QC** (`bvbrc_recon.sequence_qc.qc_assembly`, already implemented and tested): run
   once per unique retrieved genome, exactly as for the pilot. Output: one QC row per genome,
   reusable across all three antibiotics' downstream analyses.

4. **Configurable k-mer extraction** (`bvbrc_recon.kmer_features`, already k-configurable): given
   this task's finding that BOTH k=10 and k=31 produce an impractically large matrix at full scale
   (see Step 2/3 above), this stage should NOT proceed to full-scale extraction until a specific k
   (or an alternative feature-dimensionality strategy) is chosen and separately validated at a
   larger-than-100-genome but still bounded scale (e.g. 500-1,000 genomes) - a natural next small
   step, not performed in this task. The memory-safety pattern discovered and implemented in this
   task's `run_kmer_k_comparison.py` (process one genome at a time, never hold all genomes' raw
   counts simultaneously, log progress to a file, abort safely before an OS OOM-kill) should be
   adopted as the standard extraction pattern regardless of which k is ultimately chosen.

5. **Sparse feature-matrix storage and genome-row mapping**
   (`bvbrc_recon.kmer_features.build_feature_matrix`, already implemented and tested): unchanged -
   already saves the matrix, the explicit genome-row order, and the k-mer-to-column vocabulary as
   separate, explicit files, exactly as needed for a larger cohort.

6. **Split generation**: random split via
   `bvbrc_recon.pilot_model.compute_random_split_indices` (already implemented, already proven
   equivalent-by-test to the stratified split used for evaluation) and MLST-grouped lineage-aware
   split via `bvbrc_recon.lineage_split.split_by_lineage_group` (already implemented and tested,
   including the "report the blocker, never silently fall back to random" behavior) - both already
   scale to any genome count; no new code needed.

7. **Model training/evaluation**: `bvbrc_recon.pilot_model.evaluate_with_precomputed_split` and
   `bvbrc_recon.model_comparison.evaluate_random_forest`/`evaluate_xgboost` (all already implemented
   and tested, including explicit missing-class-in-partition handling and sparse-input
   verification). **Confidence intervals** are not yet implemented anywhere in this project and
   would need to be added (e.g. bootstrap resampling of the test set) before full benchmark
   reporting - flagged here as a genuine gap, not assumed to exist.

8. **CARD-based exploratory interpretation**
   (`bvbrc_recon.card_reference`/`card_matching`/`kmer_candidates`, already implemented and
   tested): unchanged in design, though the k=10-saturation caveat already documented in
   `reports/pilot_ciprofloxacin_biological_interpretation.md` would need revisiting if a different
   k is chosen for the full run.

9. **Provenance, seeds, configuration, logging, failure recovery**: every stage above already
   writes its configuration (seed, k, antibiotic, dimensions) to a small JSON file alongside its
   output (established pattern: `pilot_{antibiotic}_kmer_config.json`, etc.) - this pattern should
   simply be continued per-stage for the full run. Failure recovery is already handled per-stage
   (retrieval resumes by skipping existing files; this task's k-mer extraction now aborts safely
   above a memory threshold rather than risking an uncontrolled OOM-kill) - the remaining gap is a
   single top-level run manifest tying stage outputs together by run id/timestamp, which does not
   exist yet and is a reasonable addition, not built in this task.
"""

    recommendation = f"""
**Recommended next step: do NOT commit to k=10 or k=31 for the full run yet.** Both are estimated,
from measured pilot characteristics, to produce an impractically large feature matrix at full
dataset scale (tens of GB at k=10 because the matrix becomes nearly dense; potentially hundreds of
GB at k=31 because the vocabulary has not saturated even at pilot scale). This is evidence, not a
final verdict - the k=31 upper-bound estimate in particular is a known overestimate (it assumes no
additional cross-genome sharing as more genomes are added, which is unlikely for genomes of the
same species).

**Concrete, bounded next step**: run the SAME memory-safe k-mer extraction pattern used in this
task, on an intermediate genome count (e.g. 500-1,000 genomes, still far below the full ~{len(eligible_union):,}-genome
union) and at an intermediate k (e.g. k=15-21) in addition to k=10/k=31, to measure - not assume -
where vocabulary saturation actually levels off for this dataset, before deciding the final k for
the full three-antibiotic benchmark. This directly addresses the instructor's k=31 concern and the
proposal's k=10 figure without assuming either is correct.

**Caveats**: this recommendation is itself a judgment call about risk tolerance, not a measured
fact - a reviewer willing to accept a very large (but not impossible, given sufficient compute
infrastructure beyond this development machine) matrix could instead proceed directly with k=10 or
k=31 at full scale. The 495MB/genome and 4.95MB/genome figures used throughout this report are
measured from ONE antibiotic's pilot cohort; genome size for the other two antibiotics' eligible
genomes was not independently re-measured (their mean `genome_genome_length` values are reported
above and are very close to ciprofloxacin's, which is expected since all three draw from the same
*E. coli* genome collection, but this was not re-verified by downloading their sequences).
"""

    report_text = render_full_dataset_report(
        analysis_date=datetime.now(timezone.utc).isoformat(),
        cohort_sections=cohort_sections,
        resource_table=resource_table,
        pilot_fasta_measured_mb_per_genome=MEASURED_FASTA_MB_PER_GENOME,
        fasta_storage_table=fasta_storage_table,
        k_comparison_table=k_comparison_table,
        model_comparison_table=model_comparison_table,
        k_comparison_narrative=k_comparison_narrative,
        pipeline_design=pipeline_design,
        recommendation=recommendation,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"Report written -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
