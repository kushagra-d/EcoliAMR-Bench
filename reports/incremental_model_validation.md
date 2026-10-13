# Incremental Model Fix: Diagnosis and Validation

Exploratory pilot only. Existing 100-genome ciprofloxacin cohort, genome
order, both saved splits, and the existing SQLite-backed hashed matrix
were reused unchanged. No genomes were downloaded. No existing matrix,
split, or saved result was overwritten.

## Root cause (Step 1)

Diagnosed empirically against the real pilot data (not guessed): the
hashed features are RAW k-mer counts, and each genome's total count
scales with genome length/sequencing depth - a confound unrelated to
resistance phenotype. `SGDClassifier`'s mini-batch updates are far more
sensitive to this scale variation than the batch `LogisticRegression`
baseline's `lbfgs` solver, which tolerates unscaled sparse counts fine
over up to 1000 full-batch iterations. A direct diagnostic sweep (ad hoc,
not shipped as code) on the real random-split train/test data found:

| Change from the original config (5 epochs, no scaling) | Test ROC-AUC |
|---|---|
| none (original) | 0.500 |
| 50 epochs only | 0.500 |
| `class_weight` balancing only | 0.500 |
| `StandardScaler(with_mean=False)` (column scaling) only | 0.500 |
| `learning_rate="constant", eta0=0.01` only | 0.500 |
| **L2 row-normalization only** | **0.795** |

Row normalization was the only single change that moved ROC-AUC off
exactly 0.5. Column-wise scaling did not help - the confound is in each
genome's total count (a row-wise property), not in any one feature's
variance.

A **second, unrelated bug** was found and fixed during this
investigation: `SqliteRowStore.read_rows_for_indices` (from the prior
task) silently returned zero rows when given `numpy.int64` indices
(e.g. from `sklearn.model_selection.train_test_split`) instead of plain
Python `int` - no exception, just an empty result. This did not cause
the original AUC=0.5 symptom (the existing callers used plain-`int`
`enumerate` indices), but it would have broken the internal-validation
procedure below, and any future caller using numpy-derived indices.
Fixed in `src/bvbrc_recon/sqlite_row_store.py` by coercing indices to
`int()` before binding; regression-tested in
`tests/test_incremental_model.py::test_sqlite_store_read_rows_for_indices_accepts_numpy_int_indices`.

## Baseline reproduction (Step 2)

`src/run_batch_baseline_reproduction.py` reruns `evaluate_with_precomputed_split`
(the project's existing batch-LR code, unmodified) on the existing
`pilot_ciprofloxacin_kmer_hashed_matrix_d262144.npz` and both existing
splits. Result: **bit-for-bit identical** to the original saved
`model_results` block in `pilot_ciprofloxacin_kmer_hashed_config_d262144.json`
(`matches_existing_saved_result_exactly: true`), confirming labels, row
order, and splits are consistent with what the incremental pipeline
also uses (cross-checked again below via exact `train_class_counts`/
`test_class_counts` equality). Written to a new path, never touching the
original.

## Configuration selection (Step 3) - decided before touching test data

Fixed-grid investigation, scoped deliberately (not an unrestricted
search): a stratified 80/20 internal validation split was carved from
the random split's TRAIN partition only (seed=7, 56 sub-train / 14
internal-val rows), never touching either real test set. With L2 row
normalization fixed a priori (justified above, independent of any
outcome metric), epochs ∈ {10, 25, 50} × class_weight ∈ {none,
train-count-derived "balanced" dict} were evaluated on this internal
validation split only:

| epochs | class_weight | internal val AUC | internal val Resistant recall |
|---|---|---|---|
| 10 | none | 0.55 | 0.0 |
| 10 | balanced | 0.55 | 0.0 |
| 25 | none | 0.55 | 0.0 |
| 25 | balanced | 0.55 | 0.0 |
| **50** | **none** | **0.625** | 0.0 |
| 50 | balanced | 0.55 | 0.0 |

**Selected: `normalize_rows=True`, `n_epochs=50`, no `class_weight`.**
`class_weight` showed no consistent benefit on this internal validation
split (only 14 held-out rows - noisy) and was dropped to avoid an
untested, unjustified knob. `learning_rate` was not revisited (the
earlier diagnostic already showed `constant`/`eta0=0.01` gave no benefit
without normalization, and adding it as a further axis after already
locking normalization+epochs was judged scope creep). This full grid and
decision is recorded machine-readably in
`data/processed/incremental_model_validation_ciprofloxacin_k15_d262144.json`'s
`configuration_selection` field.

## Results on both real test sets (Step 4) - reported honestly

| Split | Model | Accuracy | ROC-AUC | Avg. Precision | Resistant recall | Susceptible recall | Confusion matrix |
|---|---|---|---|---|---|---|---|
| random | batch LR (baseline) | 0.900 | 0.938 | — | 0.571 | 1.000 | [[23,0],[3,4]] |
| random | incremental (fixed) | 0.767 | 0.745 | 0.589 | **0.000** | 1.000 | [[23,0],[7,0]] |
| lineage | batch LR (baseline) | 0.759 | 0.857 | — | 0.714 | 0.773 | [[17,5],[2,5]] |
| lineage | incremental (fixed) | 0.759 | 0.539 | 0.266 | **0.000** | 1.000 | [[22,0],[7,0]] |

`train_class_counts`/`test_class_counts` were confirmed identical to the
batch baseline's on both splits (`class_counts_match_batch_baseline: true`
in the output JSON), so the comparison is on matched data, not an
artifact of different splits.

**Honest assessment - the fix is a real, partial improvement, not a
full fix:**
- Random split: ROC-AUC rose from 0.500 (original bug) to 0.745 - a
  genuine, substantial gain in ranking quality, though still well below
  the batch baseline's 0.938.
- Lineage split: ROC-AUC rose only from 0.500 to 0.539 - barely above
  chance, far below the batch baseline's 0.857.
- **Resistant recall remained exactly 0.0 on both splits** under the
  standard 0.5 probability threshold, despite the random split's
  meaningfully improved ranking. This means the model's calibrated
  decision boundary still defaults to predicting the majority class for
  every test genome, even where its underlying ranking has improved. No
  custom threshold was tuned to force a nonzero recall number - that
  would have been exactly the kind of test-set tuning this task
  prohibited.
- This is reported as a **legitimate, unresolved limitation of
  mini-batch SGD at this sample/feature-dimension ratio** (~70-90
  training genomes against 262,144 features - a severely underdetermined
  regime for any linear model), not as an implementation bug: the same
  configuration, the same features (confirmed row-exact in the prior
  task), and the same labels produce a far better-calibrated model under
  the batch `lbfgs` solver. The mini-batch SGD optimizer's convergence
  envelope, step-size schedule, and lack of a strong regularization path
  comparable to `lbfgs`'s appear to be the binding constraint, not a
  fixable scaling/weighting choice investigated here.

## Training time and peak RSS

| Split | fit_seconds (50 epochs) | wall_seconds | peak RSS before | peak RSS after |
|---|---|---|---|---|
| random | 155.9 s | 157.3 s | 153.6 MB | 2,245.5 MB |
| lineage | 158.0 s | 159.3 s | 2,245.5 MB | 2,314.2 MB |

(Peak RSS is `ru_maxrss`, a cumulative high-water mark for the whole
process - the lineage row's "before" value is the random split's
"after" value, both splits ran in one process.)

Both figures are substantially higher than the pure extraction-phase
measurements from the prior tasks (~200 MB) and higher even than the
earlier 5-epoch incremental run (~1.4 GB). **Root cause, diagnosed, not
guessed:** at k=15/n_buckets=262144 every row is 100% dense (established
in the prior task), so `SqliteRowStore._rows_to_csr` materializes a
Python list of up to 16 × 262,144 ≈ 4.2 million individual Python `int`
objects per batch (via `.tolist()`) to build each COO chunk. Reading the
same ~70-90 training rows fresh from SQLite on every one of 50 epochs
multiplies this cost by 50, which is the dominant cost in both the
155-158 second fit time and the ~2.3 GB peak RSS - NOT a violation of
the batch-size memory bound (`max_batch_rows_held_in_memory` stayed at
16 throughout, confirmed in the output JSON) but a real, measured
performance cost of combining a fully-dense feature configuration with
this store's (col, value)-pair Python-object representation, repeated
every epoch rather than cached once. This is a genuine limitation of the
current implementation at this specific (k, n_buckets) configuration,
not fixed in this task (out of scope - this task's job was the model's
learning behavior, not store performance), and is listed below as a
blocker.

## Test results (Step 5)

378 tests pass (372 before this task, plus 6 new: scale-invariant
predictions under row normalization, ROC-AUC cross-checked against an
independently-refit model, minority-class recall computation, training-
only preprocessing with extraneous untouched rows in the store, the
`numpy.int64` indexing regression, and artifact-path isolation). No
previously-passing test was broken.

## Final validation (Step 7)

- `pilot_ciprofloxacin_kmer_hashed_matrix_d262144.npz` and all other raw/
  hashed/minimizer `.npz` artifacts: unchanged.
- `genome_order.csv`, `random_split.csv`, `lineage_split.csv`,
  `selected_cohort.csv`: unchanged.
- The original saved batch-LR result
  (`pilot_ciprofloxacin_kmer_hashed_config_d262144.json`): unchanged -
  the reproduction in Step 2 wrote to a separate new path.
- Prior out-of-core/SQLite-store artifacts from the two previous tasks:
  unchanged.
- `data/raw/ciprofloxacin_pilot/`: still exactly 100 `.fna` files - no
  new downloads.
- `git status`: only the pre-existing modified `.gitignore` plus new
  untracked files; nothing staged or committed.

## Remaining blockers before full-dataset execution

1. **The incremental model is still not competitive with the batch
   baseline**, and Resistant recall is still exactly zero on both
   splits under the standard threshold. This is now understood to be a
   real optimizer/sample-size limitation (not a scaling or weighting
   bug), but it is unresolved - a different incremental algorithm,
   substantially more training data, or accepting a non-default decision
   threshold chosen via a principled (train/validation-only) procedure
   would need investigation before this path is usable for real
   evaluation.
2. **Training time scales with epochs × training-set re-reads from
   disk**, and at this fully-dense (k=15, d=262144) configuration each
   re-read is expensive in Python-object terms (~4.2M objects/batch).
   50 epochs took ~156-158 seconds per split; this would not scale
   gracefully to more genomes or more epochs without caching the
   training partition's batches once (trading some memory-boundedness
   for speed) or switching to a less dense configuration.
3. **The internal-validation grid used to pick epochs was tiny (14
   held-out rows)** and noisy - the selection is a reasonable, honest,
   non-test-tuned choice, but not a confident one. A larger pilot would
   allow a more reliable internal validation procedure.
4. **Only one representation (k=15, d=262144), one antibiotic, and the
   existing 100-genome pilot were used.** No full-dataset download,
   three-antibiotic benchmark, or Random Forest/XGBoost work was
   started, per this task's explicit stop condition.
