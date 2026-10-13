# Ciprofloxacin Pilot Report (PRELIMINARY)

- Analysis date (UTC): 2026-10-09T18:37:24.241132+00:00
- Antibiotic: ciprofloxacin
- Random seed (cohort sampling + train/test split): 20261009

**This is a small, preliminary, non-lineage-aware pilot for ONE antibiotic. It does not represent
the full candidate dataset, does not select a final modelling cohort, and its model metrics are
not the three-antibiotic benchmark's expected performance.**

## Step 2 - Cohort selection

Label-curation rule: reused `conflict_analysis.apply_policy_2` verbatim ("require a single binary
interpretation" - a genome is eligible only if every recognized AST record for this antibiotic is
exactly Resistant, or exactly Susceptible; any Intermediate co-occurrence or any Resistant/
Susceptible disagreement excludes the genome). This is the same policy already defined and tested
in `reports/phenotype_conflict_analysis.md` - reused here, not reinvented.

- Eligible pool: 1840 Resistant, 5457 Susceptible
- Selected: 100 genomes - 25 Resistant, 75 Susceptible
- Sampling: a reproducible random sample (`random_state=20261009`) from each eligible class pool,
  proportional to the pool's own Resistant:Susceptible ratio - not forced to 50/50.
- Every genome in the full antibiotic cohort was decided (included or excluded) and recorded, with
  its reason, in `data/processed/pilot_ciprofloxacin_cohort_decision_table.csv`.

**Limitations (stated, not hidden):**

- Sampling is proportional to the eligible pool's natural class ratio (1840 Resistant : 5457 Susceptible), not rebalanced - the pilot is therefore imbalanced (25 Resistant : 75 Susceptible) by design, reflecting the underlying data rather than an invented balance.
- This pilot cohort is a small, non-lineage-aware random sample from genomes passing a strict single-interpretation label rule; it is NOT claimed to be representative of the full candidate dataset, and genomes were not selected based on ease of sequence download.

## Step 3 - Assembly retrieval

Used the already-verified `fetch_complete_assembly` route (batched `genome_sequence` FASTA
retrieval, `reports/bvbrc_assembly_retrieval_feasibility.md`), extended minimally to fetch EVERY
contig per genome rather than one probe contig, with completeness cross-checked against both the
live JSON-mode contig index and the already-known `genome_contigs` metadata value.

| Retrieval status | Genomes |
|---|---|
| `already_present_skipped` | 100 |

- **100 / 100** selected genomes have a verified-complete assembly available for
  feature extraction.
- Full per-genome retrieval status, contig counts, and any errors are recorded in
  `data/processed/pilot_ciprofloxacin_retrieval_manifest.csv`.
- FASTA files were written under `data/raw/ciprofloxacin_pilot/` (one file per genome_id); an
  already-present file was never overwritten - re-running this pilot will skip genomes already
  retrieved.

## Step 4 - Sequence QC (Biopython)

- Assemblies QC'd: 100
- Contig count range: 39-357
- Total length range: 4,090,023-5,825,951 bp
- Mean ambiguous-base proportion: 0.000006
- Assemblies with any duplicate contig ids: 0
- Assemblies with any empty contigs: 0
- Assemblies with parse errors: 0

No quality threshold was applied to exclude any assembly from the k-mer/model steps based on these
QC characteristics - they are reported for a later, informed decision, per the task constraints.


## Step 5 - k-mer feature extraction

- **k = 10.** The submitted proposal states k is approximately 10; instructor feedback separately
  raised a concern about k=31. **This discrepancy is not resolved here** - k=10 was used for this
  pilot because it is cheap to run on ~100 genomes and matches the literal proposal value, not
  because the k=31 question has been settled. k is a configurable parameter in
  `bvbrc_recon.kmer_features`, so the pilot can be rerun at k=31 (or any other value) on request.
- Canonical k-mers: a k-mer and its reverse complement are counted as one feature.
- Any k-mer window containing a non-ACGT character was skipped entirely (never counted under
  either orientation).
- Contigs were processed independently - no k-mer spans a contig boundary.
- Feature matrix: **100 genomes x 524346 observed canonical
  k-mers**, 49,668,801 nonzero entries. Stored as a SciPy CSR matrix
  (`data/processed/pilot_ciprofloxacin_kmer_matrix.npz`) - never densified, regardless of the
  observed density below.
- **Observed density is high at k=10, not low**: 94.7%
  of matrix entries are nonzero (only 5.3% are zero/"sparse").
  This is expected, not a bug: a ~5 Mbp *E. coli* genome contains far more k=10 windows than the
  ~524,800 possible canonical 10-mers, so by the
  pigeonhole principle almost every possible 10-mer appears in almost every genome - at this k, the
  feature space is nearly saturated per genome and carries correspondingly little genome-specific
  information per individual feature. This would look very different at a larger k (e.g. the
  proposal-vs-instructor k=31 question above) - the matrix would be enormous in nominal
  dimensionality and genuinely sparse, since no single genome could saturate a k=31 space. This
  density/dimensionality trade-off is itself relevant evidence for resolving the open k question,
  though this report does not resolve it.
- Genome-id-to-row mapping saved explicitly: `data/processed/pilot_ciprofloxacin_genome_order.csv`.
- k-mer-to-column mapping saved explicitly: `data/processed/pilot_ciprofloxacin_kmer_vocabulary.csv`.
- Full configuration (k, seed, input directory, dimensions): `data/processed/pilot_ciprofloxacin_kmer_config.json`.

## Step 6 - Preliminary Logistic Regression pilot model

**PRELIMINARY PILOT RESULT - not the final benchmark, and this is an ordinary random train/test
split, NOT a lineage-independent split.**

- Train: 70 genomes {'Susceptible': 52, 'Resistant': 18}
- Test: 30 genomes {'Susceptible': 23, 'Resistant': 7}
- Accuracy: 0.9
- ROC-AUC: 0.9627329192546584
- Average precision: 0.9200680272108843
- Recall per class: {'Susceptible': 1.0, 'Resistant': 0.5714285714285714}
- Confusion matrix `[[TN, FP], [FN, TP]]` (0=Susceptible, 1=Resistant): [[23, 0], [3, 4]]
- Warnings: none


## Caution for anyone reading the pilot's own output CSVs

As found in earlier work on this project, `genome_id` values (e.g. `"562.100475"`) are silently
misread as `float64` by a default `pandas.read_csv` call, which can merge distinct ids that differ
only in trailing zeros. `bvbrc_recon.csv_loader.load_antibiotic_csv` already guards against this for
the reconnaissance CSVs; the new pilot CSVs written by this script
(`pilot_ciprofloxacin_genome_order.csv`, `_selected_cohort.csv`, `_cohort_decision_table.csv`,
`_retrieval_manifest.csv`, `_sequence_qc.csv`) do **not** have a dedicated loader yet - always pass
`dtype={"genome_id": str}` when reading them back (verified live while preparing this report: an
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
overwritten) and deterministic for cohort selection and the train/test split (fixed seed 20261009).
