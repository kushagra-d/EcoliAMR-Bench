# Full-Dataset Feasibility Assessment and Pipeline Plan

- Analysis date (UTC): 2026-10-09T20:51:51.675694+00:00
- Scope: feasibility assessment and pipeline DESIGN only. No additional genome was downloaded, no
  full-scale k-mer extraction was run, no benchmark model was trained on the full dataset. The
  k=10-vs-k=31 comparison below uses ONLY the existing 100-genome ciprofloxacin pilot's already-
  downloaded assemblies.
- Every number below is labeled as either **measured** (computed directly from existing data or
  from the small pilot comparison run in this task) or **estimated/extrapolated** (projected from
  measured pilot values to full-dataset scale, with the extrapolation method stated).

## Step 2 - Policy A cohort size and resource estimates, per antibiotic

Cohort sizes and MLST coverage below reuse `final_policy_analysis.apply_policy_a` fresh against the
existing reconnaissance CSVs (read-only) - these are **measured**, not estimated, and match the
numbers already reported in `reports/final_phenotype_policy_analysis.md` exactly (same function,
same inputs).

### Ciprofloxacin

- Policy A eligible genomes: **7307** (Resistant 1846 [25.3%], Susceptible 5461 [74.7%])
- MLST available: 7274 / 7307 (783 distinct groups)
- Mean genome length (measured, eligible genomes): 5,069,783 bp
- Mean contig count (measured): 130.2
- NCBI assembly_accession populated: 430 / 7307 (5.9%)
- NCBI biosample_accession populated: 3033 / 7307 (41.5%)

### Ampicillin

- Policy A eligible genomes: **6490** (Resistant 3659 [56.4%], Susceptible 2831 [43.6%])
- MLST available: 6448 / 6490 (802 distinct groups)
- Mean genome length (measured, eligible genomes): 5,084,871 bp
- Mean contig count (measured): 138.3
- NCBI assembly_accession populated: 458 / 6490 (7.1%)
- NCBI biosample_accession populated: 3233 / 6490 (49.8%)

### Gentamicin

- Policy A eligible genomes: **7340** (Resistant 862 [11.7%], Susceptible 6478 [88.3%])
- MLST available: 7306 / 7340 (805 distinct groups)
- Mean genome length (measured, eligible genomes): 5,066,943 bp
- Mean contig count (measured): 128.9
- NCBI assembly_accession populated: 394 / 7340 (5.4%)
- NCBI biosample_accession populated: 3120 / 7340 (42.5%)


**Cross-antibiotic overlap (measured, not estimated)**: the same genome is frequently tested
against more than one antibiotic. The union of Policy-A-eligible genomes across all three
antibiotics is smaller than the sum of the three individual counts - a full pipeline should
retrieve each unique genome assembly ONCE and reuse it across all three antibiotics' analyses,
not redownload it per antibiotic.

### FASTA storage estimate

Measured from the existing 100-genome ciprofloxacin pilot: **4.95
MB/genome average** (495 MB / 100 genomes). This is a reasonable **linear** extrapolation basis -
genome length is a property of the organism, not of how many other genomes are in the collection.

| Antibiotic (or union) | n genomes | Estimated FASTA storage |
|---|---|---|
| ciprofloxacin | 7,307 | 36.2 GB |
| ampicillin | 6,490 | 32.1 GB |
| gentamicin | 7,340 | 36.3 GB |
| union (unique genomes, all 3 antibiotics) | 8,710 | 43.1 GB |

### k-mer feature-matrix dimension and memory estimates (full dataset)

**This is the central, non-obvious finding of this feasibility assessment**: the number of matrix
COLUMNS does NOT scale linearly with genome count the way FASTA storage does. At k=10, the pilot's
100 genomes already reach ~99.9% of the entire theoretically possible canonical-k-mer space - more
genomes will add almost no new columns, only more (increasingly dense) nonzero entries. At k=31,
the opposite problem appears: the space is so large that the pilot's own 100 genomes show no
saturation at all, so the true column count at full scale cannot be measured from this pilot and
can only be bounded - the upper bound below is a crude, explicitly labeled naive linear
extrapolation, not a prediction, and is almost certainly an overestimate because more genomes of
the same species share more sequence, not less.

| Antibiotic (or union) | n genomes | k | Regime | Est. n_features | Est. nnz | Est. CSR memory |
|---|---|---|---|---|---|---|
| ciprofloxacin | 7,307 | 10 | saturated | 524,346-524,800 | 3,629,299,289 | 43.6 GB |
| ampicillin | 6,490 | 10 | saturated | 524,346-524,800 | 3,223,505,184 | 38.7 GB |
| gentamicin | 7,340 | 10 | saturated | 524,346-524,800 | 3,645,689,993 | 43.7 GB |
| union (unique genomes, all 3 antibiotics) | 8,710 | 10 | saturated | 524,346-524,800 | 4,326,152,567 | 51.9 GB |
| ciprofloxacin | 7,307 | 31 | unsaturated | 32,648,927-2,385,657,095 | 37,069,096,761 | 444.8 GB |
| ampicillin | 6,490 | 31 | unsaturated | 32,648,927-2,118,915,362 | 32,924,379,086 | 395.1 GB |
| gentamicin | 7,340 | 31 | unsaturated | 32,648,927-2,396,431,241 | 37,236,508,859 | 446.8 GB |
| union (unique genomes, all 3 antibiotics) | 8,710 | 31 | unsaturated | 32,648,927-2,843,721,541 | 44,186,647,433 | 530.2 GB |

**Both configurations are estimated to produce an impractically large in-memory sparse matrix at
full dataset scale on commodity hardware** (tens of GB at k=10, potentially hundreds of GB at
k=31) - this is explicit motivation for the k-configuration discussion in Step 3/5 below, not a
decision made here.

## Step 3 - k=10 vs. k=31 pilot comparison (measured, 100 ciprofloxacin genomes)

**Memory note (discovered empirically while running this comparison, not assumed)**: a naive
implementation that holds all 100 genomes' full per-genome k-mer counts in memory simultaneously
before merging them OOM-killed this process at k=31 (confirmed: a single genome's k=31 count table
alone uses ~500-700MB RSS; 100 simultaneously exceeded this machine's 17GB RAM). The comparison
below used a corrected, memory-bounded implementation that processes one genome at a time -
documented in `src/run_kmer_k_comparison.py` - which completed both k values successfully with a
measured peak RSS of 2.8GB (k=10) and 7.6GB (k=31), safely under an 11GB abort threshold set with
~6GB of headroom on this 17GB machine.

| k | Genomes processed | n_features | nnz | Density | Extraction time | npz storage | In-memory CSR |
|---|---|---|---|---|---|---|---|
| 10 | 100/100 (complete) | 524,346 | 49,668,801 | 94.73% | 273.4s | 113.6 MB | 596.0 MB |
| 31 | 100/100 (complete) | 32,648,927 | 507,309,385 | 15.54% | 557.4s | 711.9 MB | 6087.7 MB |


**Observed pattern (measured on this one 100-genome pilot - not a general conclusion)**: k=31
shows somewhat higher accuracy/ROC-AUC than k=10 for most model/split combinations in this specific
run, consistent with the CARD interpretation task's earlier finding that k=10 produces very weakly
specific sequence matches (CARD's own reference set saturates ~87% of the k=10 space). A larger k
gives each feature more specific sequence information, which is plausible motivation - but this is
one measurement on 100 genomes with no cross-validation, and is not proof that k=31 generalizes
better. The feature-dimensionality risk (32.6M features for 100 genomes) is far more severe than
k=10's, and cannot be dismissed by one favorable-looking run.


### Exploratory baseline model comparison (same existing saved splits, no tuning)

| k | Split | Model | Accuracy | ROC-AUC |
|---|---|---|---|---|
| 10 | random | LogisticRegression | 0.9 | 0.9627329192546584 |
| 10 | random | RandomForest | 0.9 | 0.9472049689440994 |
| 10 | random | XGBoost | 0.9 | 0.9099378881987579 |
| 10 | lineage | LogisticRegression | 0.7931034482758621 | 0.8896103896103896 |
| 10 | lineage | RandomForest | 0.7586206896551724 | 0.9123376623376623 |
| 10 | lineage | XGBoost | 0.8620689655172413 | 0.8636363636363636 |
| 31 | random | LogisticRegression | 0.9 | 0.9316770186335404 |
| 31 | random | RandomForest | 0.9 | 0.9875776397515528 |
| 31 | random | XGBoost | 0.9666666666666667 | 0.9285714285714286 |
| 31 | lineage | LogisticRegression | 0.8275862068965517 | 0.8766233766233766 |
| 31 | lineage | RandomForest | 0.8620689655172413 | 0.9805194805194806 |
| 31 | lineage | XGBoost | 0.896551724137931 | 0.9318181818181819 |

**Limitations of this comparison, stated explicitly:**
- 100 genomes is small enough that any apparent k=31 advantage could reflect this specific sample,
  not a general property of k=31 for this task - no cross-validation or repeated sampling was
  performed.
- Both k values were evaluated on the EXACT SAME saved random and lineage splits already used for
  the k=10 pilot's own reported results - no new split was generated, and test-set rows/labels
  were never used to choose k, select features, or tune any model (only the already-fixed model
  configurations from the existing pilot/model-comparison work were reused).
- A higher k=31 metric here is not evidence that k=31 "works better" in general - it is one
  measurement, on one small, non-lineage-representative cohort, with two models (Random Forest,
  XGBoost) whose default regularization/feature-subsampling behavior interacts with extreme
  dimensionality in ways not characterized here.
- The much larger k=31 feature count compounds the existing p >> n risk already noted in
  `reports/pilot_model_comparison.md`; nothing here resolves that risk, it is simply larger at
  k=31.

## Step 4 - Reproducible full-dataset pipeline design


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


## Step 5 - Recommendation


**Recommended next step: do NOT commit to k=10 or k=31 for the full run yet.** Both are estimated,
from measured pilot characteristics, to produce an impractically large feature matrix at full
dataset scale (tens of GB at k=10 because the matrix becomes nearly dense; potentially hundreds of
GB at k=31 because the vocabulary has not saturated even at pilot scale). This is evidence, not a
final verdict - the k=31 upper-bound estimate in particular is a known overestimate (it assumes no
additional cross-genome sharing as more genomes are added, which is unlikely for genomes of the
same species).

**Concrete, bounded next step**: run the SAME memory-safe k-mer extraction pattern used in this
task, on an intermediate genome count (e.g. 500-1,000 genomes, still far below the full ~8,710-genome
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

