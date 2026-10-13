# Out-of-Core Hashed Feature Matrix Prototype

Exploratory pilot only. Existing 100-genome ciprofloxacin cohort, genome
order, and both saved splits (random, lineage) were reused unchanged.
No genomes were downloaded. No existing artifact was overwritten.

## What was built

- `src/bvbrc_recon/streaming_hashed_matrix.py` — hashes directly from each
  genome's sequence stream (`canonical_kmer` → `hash_kmer_to_bucket`, the
  same deterministic blake2b-based hash used by `feature_hashing.py`).
  No raw k-mer matrix or raw vocabulary is ever built as a prerequisite.
  Each genome's row is appended to disk immediately and discarded from
  memory. Row placement in the final matrix is governed by a `row_index`
  field (the genome's fixed position in `genome_order`), not by
  file-append order, so resumption cannot duplicate or misplace rows.
- `src/bvbrc_recon/incremental_model.py` — `SGDClassifier(loss="log_loss")`
  trained via `partial_fit` on bounded batches, since `LogisticRegression`
  has no incremental API and would require the full matrix in RAM.
- `src/run_out_of_core_prototype.py` — CLI that runs the above against the
  real pilot data and writes the artifacts below.
- `tests/test_streaming_hashed_matrix.py` (23 tests), `tests/test_incremental_model.py`
  (8 tests) — determinism, canonicalization, ambiguous-base/contig-boundary
  handling, resumption safety, failure tracking, resource-limit aborts,
  batch-only memory access.

Disk format: append-only JSONL, one line per genome
(`{"genome_id", "row_index", "cols", "values"}`). Documented
simplicity/throughput tradeoff: resumption checks existing genome IDs via
an O(rows-written) file scan rather than a separate index — acceptable at
pilot scale (100 genomes), not necessarily at full scale.

New artifacts (all under `data/processed/`, none overwrite existing
pilot files):
- `out_of_core_ciprofloxacin_rows_k15_d262144.jsonl` (disk-backed row store, 291 MB)
- `out_of_core_ciprofloxacin_config_k15_d262144.json` (measurements below)
- `out_of_core_ciprofloxacin_incremental_model_results_k15_d262144.json`
- `out_of_core_ciprofloxacin_k15_d262144_experiment.log`

No failure log file was produced — all 100 genomes extracted without error,
so `FailureLog` was never written to (verified separately by unit tests
that failures are logged, never zero-filled).

k=15 and n_buckets=2^18 (262,144) were chosen to match the existing
in-memory `pilot_ciprofloxacin_kmer_hashed_matrix_d262144.npz` pilot
exactly, so peak-RSS and matrix statistics are directly comparable to an
already-measured baseline.

## Is construction genuinely out-of-core?

Yes, architecturally: `contigs_loader` in `run_out_of_core_prototype.py`
reads one genome's FASTA file per call — unlike every prior pipeline in
this project (e.g. `run_minimizer_experiment.py`), no dict of all 100
genomes' contigs is ever built. `build_streaming_hashed_matrix` holds at
most one genome's sequence plus one ≤262,144-entry bucket-count dict at a
time, and appends immediately to disk. The completed store was reopened
in a fresh `DiskBackedRowStore` instance and consumed via `iter_batches`
and `read_rows_for_indices`, both of which cap in-memory batch size (16
rows here; `max_batch_rows_materialized_at_once = 16`, confirmed never
reaching the full 100-row count).

Resumption safety was demonstrated directly, not just unit-tested: the
build function was invoked a second time against the same row store and
failure log. Result: `n_rows_written_this_run=0`,
`n_genomes_skipped_already_present=100`, no duplicate rows
(`row_store.n_rows_written()` stayed at 100).

**Caveat on the resumption test performed**: this ran the build function
twice to completion, not a true mid-run kill/restart. A real process
interruption (SIGKILL mid-write) was not exercised against the live
pipeline; the JSONL-append format combined with genome-ID-based skip
logic should tolerate this (a half-written last line is simply the next
genome retried, since its ID wasn't recorded as present), but this exact
scenario is only covered by the unit tests, not by this real-data run.

## Resource measurements (real 100-genome pilot, k=15, n_buckets=262144)

| Metric | Value |
|---|---|
| Extraction-only peak RSS (run 1, 100 genomes) | 200 MB |
| Whole-process peak RSS (incl. resumption check, batched reopen, both model evaluations) | 1,419 MB |
| Existing in-memory raw k=15 matrix build peak RSS | 7,158 MB |
| Existing in-memory hashed d=262144 build peak RSS (raw matrix + post-hoc hash) | 7,837 MB |
| Extraction wall-clock (run 1, 100 genomes) | 368.2 s |
| Resumption-check wall-clock (run 2, 100 genomes, all skipped) | 2.3 s |
| Row-store disk usage | 303,233,399 bytes (≈291 MB) |
| Matrix dimensions (reopened, batch-read) | 100 × 262,144 |
| nnz | 26,214,400 |
| Density | 100% |

`ru_maxrss` (used for all peak-RSS figures here, as elsewhere in this
project) is the maximum RSS over the whole process lifetime, not a
point-in-time reading — the 1,419 MB whole-process figure includes
sklearn/pandas import overhead and the downstream model-training phase,
not just extraction. Reported both to avoid cherry-picking the smaller
number: even the whole-process peak is ~5x below either existing
in-memory pipeline's peak.

The row-store's 291 MB on disk is ~5x the existing `.npz`'s 59.9 MB for
the same (k, n_buckets) pair — the JSONL format trades storage compactness
for streaming-append simplicity; it was not optimized for size.

**Independent correctness check beyond the synthetic fixture**: nnz
(26,214,400) and density (100%) from this out-of-core run are *identical*
to the existing in-memory `pilot_ciprofloxacin_kmer_hashed_matrix_d262144.npz`
(same values recorded in `pilot_ciprofloxacin_kmer_hashed_config_d262144.json`),
despite the two being built by architecturally unrelated code paths
(post-hoc hashing of a materialized raw matrix vs. direct hash-during-walk
with no raw matrix). This was not a designed test — it fell out of reusing
the same (k, n_buckets) as the existing pilot — but it is a strong real-data
correctness signal on top of the synthetic fixture check (`fixture_check.passed=true`
in the config JSON).

## Downstream model results (SGDClassifier via partial_fit, both existing splits)

| Split | n_train | n_test | Accuracy | ROC-AUC | Avg. Precision | Resistant recall | Susceptible recall |
|---|---|---|---|---|---|---|---|
| random | 70 | 30 | 0.767 | 0.5 | 0.233 | 0.0 | 1.0 |
| lineage | 71 | 29 | 0.759 | 0.5 | 0.241 | 0.0 | 1.0 |

Both confusion matrices show the model predicting the majority class
(Susceptible) for every test genome. `train_class_counts`/`test_class_counts`
match the existing LogisticRegression baseline exactly for both splits,
confirming the split logic and label alignment are correct — the
degeneracy is in the SGD model's fit, not a data-pipeline bug.

**This is a genuine limitation, not an equivalence claim.** On the same
features, the existing full-batch `LogisticRegression` baseline achieves
accuracy=0.9/AUC=0.938 (random) and accuracy=0.759/AUC=0.857 (lineage)
(`pilot_ciprofloxacin_kmer_hashed_config_d262144.json`). `SGDClassifier`
here collapsed to constant-majority-class prediction under: no feature
scaling, no class weighting (matching the baseline's own lack of
class weighting), default learning-rate schedule, and only ~5 epochs over
~5 batches of 16 rows (≈25 `partial_fit` calls total, vs. the baseline's
up-to-1000-iteration full-batch solver). This prototype demonstrates that
downstream training and evaluation CAN run in strictly bounded batches
without ever loading the full matrix — it does not demonstrate that the
incremental model matches the quality of the batch baseline. Improving
that (more epochs, feature scaling, learning-rate tuning, or a different
incremental estimator) was not attempted — out of scope for this
prototype per the task's stop condition.

## Test results

355 tests passed (full suite, including the 23 new streaming-matrix tests
and 8 new incremental-model tests). No previously-passing test was broken.

## Final validation (Step 8)

- All existing raw k-mer (`k10,15,17,19,21,31`), hashed
  (`d65536,262144,1048576`), and minimizer (`k31_w5,10,15`) `.npz`
  artifacts: MD5-identical to their pre-run state.
- `genome_order.csv`, `random_split.csv`, `lineage_split.csv`,
  `selected_cohort.csv`: MD5-identical to their pre-run state.
- `data/raw/ciprofloxacin_pilot/`: exactly 100 `.fna` files, unchanged — no
  new genomes downloaded.
- Prototype does not require a full raw feature matrix in RAM: confirmed
  both architecturally (one-genome-at-a-time `contigs_loader`, no
  prerequisite raw-matrix step) and empirically (200 MB extraction-phase
  peak RSS vs. 7,158–7,837 MB for the existing in-memory pipelines on the
  identical k=15/d=262144 configuration).
- `git status`: only the pre-existing modified `.gitignore` plus new
  untracked files; nothing staged or committed.

## Remaining blockers before full-dataset execution

1. **Model quality at this batch/epoch configuration is not acceptable as-is.**
   The incremental model needs tuning (feature scaling, more epochs/passes,
   possibly a different incremental estimator) before it could replace the
   batch `LogisticRegression` baseline for real evaluation — currently it
   only proves out-of-core *mechanics*, not competitive accuracy.
2. **Resumption was only tested via two full completed runs**, not a true
   mid-run process kill; real interruption-recovery at full scale is
   unverified outside the unit tests.
3. **The JSONL row-store format is disk-inefficient** (≈5x the size of the
   equivalent `.npz` at pilot scale) and uses an O(rows-written) scan for
   duplicate-detection on resume — both would need revisiting before
   scaling past a few hundred genomes to the thousands expected in the
   full dataset.
4. **Only one representation (hashing, k=15, d=262144) and one antibiotic
   (ciprofloxacin) were prototyped.** This task was explicitly scoped to
   the existing 100-genome ciprofloxacin pilot only, per its stop
   condition; it says nothing about out-of-core behavior at full dataset
   size, for other k/d values, or for the other two antibiotics.
5. No full-dataset download, three-antibiotic benchmark, or Random
   Forest/XGBoost work was started, per the task's explicit stop
   condition.
