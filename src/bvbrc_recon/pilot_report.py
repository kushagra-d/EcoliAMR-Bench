"""Renders reports/pilot_<antibiotic>_report.md."""

from __future__ import annotations

import pandas as pd


def render_pilot_report(
    antibiotic: str,
    seed: int,
    k: int,
    analysis_date: str,
    cohort,
    manifest_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    kmer_result,
    model_result,
    test_size: float,
) -> str:
    status_counts = manifest_df["status"].value_counts().to_dict()
    n_selected = len(cohort.selected_genome_ids)
    n_complete = sum(
        1 for s in manifest_df["status"] if s in ("complete", "already_present_skipped")
    )

    limitations_md = "\n".join(f"- {note}" for note in cohort.limitations)
    status_table = "\n".join(f"| `{status}` | {count} |" for status, count in status_counts.items())

    qc_summary = ""
    if len(qc_df):
        qc_summary = f"""
- Assemblies QC'd: {len(qc_df)}
- Contig count range: {qc_df['n_contigs'].min()}-{qc_df['n_contigs'].max()}
- Total length range: {qc_df['total_length'].min():,}-{qc_df['total_length'].max():,} bp
- Mean ambiguous-base proportion: {qc_df['ambiguous_base_proportion'].mean():.6f}
- Assemblies with any duplicate contig ids: {(qc_df['duplicate_sequence_ids'] != '').sum()}
- Assemblies with any empty contigs: {(qc_df['empty_contig_count'] > 0).sum()}
- Assemblies with parse errors: {qc_df['error'].notna().sum()}

No quality threshold was applied to exclude any assembly from the k-mer/model steps based on these
QC characteristics - they are reported for a later, informed decision, per the task constraints.
"""

    model_section = "Model step was not reached (insufficient usable assemblies)."
    if model_result is not None:
        model_section = f"""
**PRELIMINARY PILOT RESULT - not the final benchmark, and this is an ordinary random train/test
split, NOT a lineage-independent split.**

- Train: {model_result.n_train} genomes {model_result.train_class_counts}
- Test: {model_result.n_test} genomes {model_result.test_class_counts}
- Accuracy: {model_result.accuracy}
- ROC-AUC: {model_result.roc_auc}
- Average precision: {model_result.average_precision}
- Recall per class: {model_result.recall_per_class}
- Confusion matrix `[[TN, FP], [FN, TP]]` (0=Susceptible, 1=Resistant): {model_result.confusion_matrix}
- Warnings: {model_result.warnings or "none"}
"""

    return f"""# Ciprofloxacin Pilot Report (PRELIMINARY)

- Analysis date (UTC): {analysis_date}
- Antibiotic: {antibiotic}
- Random seed (cohort sampling + train/test split): {seed}

**This is a small, preliminary, non-lineage-aware pilot for ONE antibiotic. It does not represent
the full candidate dataset, does not select a final modelling cohort, and its model metrics are
not the three-antibiotic benchmark's expected performance.**

## Step 2 - Cohort selection

Label-curation rule: reused `conflict_analysis.apply_policy_2` verbatim ("require a single binary
interpretation" - a genome is eligible only if every recognized AST record for this antibiotic is
exactly Resistant, or exactly Susceptible; any Intermediate co-occurrence or any Resistant/
Susceptible disagreement excludes the genome). This is the same policy already defined and tested
in `reports/phenotype_conflict_analysis.md` - reused here, not reinvented.

- Eligible pool: {cohort.eligible_resistant_pool_size} Resistant, {cohort.eligible_susceptible_pool_size} Susceptible
- Selected: {n_selected} genomes - {cohort.n_selected_resistant} Resistant, {cohort.n_selected_susceptible} Susceptible
- Sampling: a reproducible random sample (`random_state={seed}`) from each eligible class pool,
  proportional to the pool's own Resistant:Susceptible ratio - not forced to 50/50.
- Every genome in the full antibiotic cohort was decided (included or excluded) and recorded, with
  its reason, in `data/processed/pilot_{antibiotic}_cohort_decision_table.csv`.

**Limitations (stated, not hidden):**

{limitations_md}

## Step 3 - Assembly retrieval

Used the already-verified `fetch_complete_assembly` route (batched `genome_sequence` FASTA
retrieval, `reports/bvbrc_assembly_retrieval_feasibility.md`), extended minimally to fetch EVERY
contig per genome rather than one probe contig, with completeness cross-checked against both the
live JSON-mode contig index and the already-known `genome_contigs` metadata value.

| Retrieval status | Genomes |
|---|---|
{status_table}

- **{n_complete} / {n_selected}** selected genomes have a verified-complete assembly available for
  feature extraction.
- Full per-genome retrieval status, contig counts, and any errors are recorded in
  `data/processed/pilot_{antibiotic}_retrieval_manifest.csv`.
- FASTA files were written under `data/raw/{antibiotic}_pilot/` (one file per genome_id); an
  already-present file was never overwritten - re-running this pilot will skip genomes already
  retrieved.

## Step 4 - Sequence QC (Biopython)
{qc_summary or "No assemblies were available to QC."}

## Step 5 - k-mer feature extraction

- **k = {k}.** The submitted proposal states k is approximately 10; instructor feedback separately
  raised a concern about k=31. **This discrepancy is not resolved here** - k={k} was used for this
  pilot because it is cheap to run on ~100 genomes and matches the literal proposal value, not
  because the k=31 question has been settled. k is a configurable parameter in
  `bvbrc_recon.kmer_features`, so the pilot can be rerun at k=31 (or any other value) on request.
- Canonical k-mers: a k-mer and its reverse complement are counted as one feature.
- Any k-mer window containing a non-ACGT character was skipped entirely (never counted under
  either orientation).
- Contigs were processed independently - no k-mer spans a contig boundary.
- Feature matrix: **{kmer_result.n_genomes} genomes x {kmer_result.n_features} observed canonical
  k-mers**, {kmer_result.nnz:,} nonzero entries. Stored as a SciPy CSR matrix
  (`data/processed/pilot_{antibiotic}_kmer_matrix.npz`) - never densified, regardless of the
  observed density below.
- **Observed density is high at k={k}, not low**: {kmer_result.nnz / (kmer_result.n_genomes * kmer_result.n_features):.1%}
  of matrix entries are nonzero (only {1 - kmer_result.nnz / (kmer_result.n_genomes * kmer_result.n_features):.1%} are zero/"sparse").
  This is expected, not a bug: a ~5 Mbp *E. coli* genome contains far more k=10 windows than the
  ~{(4**k + 4**(k//2))//2 if k % 2 == 0 else 4**k//2:,} possible canonical 10-mers, so by the
  pigeonhole principle almost every possible 10-mer appears in almost every genome - at this k, the
  feature space is nearly saturated per genome and carries correspondingly little genome-specific
  information per individual feature. This would look very different at a larger k (e.g. the
  proposal-vs-instructor k=31 question above) - the matrix would be enormous in nominal
  dimensionality and genuinely sparse, since no single genome could saturate a k=31 space. This
  density/dimensionality trade-off is itself relevant evidence for resolving the open k question,
  though this report does not resolve it.
- Genome-id-to-row mapping saved explicitly: `data/processed/pilot_{antibiotic}_genome_order.csv`.
- k-mer-to-column mapping saved explicitly: `data/processed/pilot_{antibiotic}_kmer_vocabulary.csv`.
- Full configuration (k, seed, input directory, dimensions): `data/processed/pilot_{antibiotic}_kmer_config.json`.

## Step 6 - Preliminary Logistic Regression pilot model
{model_section}

## Caution for anyone reading the pilot's own output CSVs

As found in earlier work on this project, `genome_id` values (e.g. `"562.100475"`) are silently
misread as `float64` by a default `pandas.read_csv` call, which can merge distinct ids that differ
only in trailing zeros. `bvbrc_recon.csv_loader.load_antibiotic_csv` already guards against this for
the reconnaissance CSVs; the new pilot CSVs written by this script
(`pilot_{antibiotic}_genome_order.csv`, `_selected_cohort.csv`, `_cohort_decision_table.csv`,
`_retrieval_manifest.csv`, `_sequence_qc.csv`) do **not** have a dedicated loader yet - always pass
`dtype={{"genome_id": str}}` when reading them back (verified live while preparing this report: an
unguarded read produced 9 spurious id mismatches against the correctly-typed selected-cohort table;
re-reading with the dtype fix produced zero).

## Known failures / exclusions

- Genomes excluded from the pilot entirely by the label-curation rule (Intermediate, conflicting,
  or missing phenotype) are listed with their reason in the cohort decision table (Step 2) - never
  silently dropped.
- Genomes selected into the pilot but whose assembly retrieval did not verify as complete are
  listed with their status and any error text in the retrieval manifest (Step 3) and were excluded
  from Steps 4-6 - also never silently dropped, just not usable for feature extraction.

## How to reproduce

```
python src/run_pilot.py
```

Re-running is idempotent for already-retrieved genomes (existing FASTA files are never
overwritten) and deterministic for cohort selection and the train/test split (fixed seed {seed}).
"""
