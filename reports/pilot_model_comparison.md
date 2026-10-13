# Pilot Model Comparison: Logistic Regression vs. Random Forest vs. XGBoost (Ciprofloxacin)

- Analysis date (UTC): 2026-10-09T19:09:48.305863+00:00
- Antibiotic: ciprofloxacin
- Random seed (all models, both splits): 20261009

**PRELIMINARY PILOT RESULTS from a single ~100-genome cohort for one antibiotic - not the final
three-antibiotic benchmark.** Reuses the EXISTING k-mer feature matrix and the EXISTING saved
lineage-aware split; no genome was downloaded, no k-mer was recomputed, and the original pilot
cohort/matrix/splits were not modified.

## Dataset and feature matrix

- Genomes: 100 (25 Resistant / 75 Susceptible, per the original pilot cohort)
- Feature matrix: 100 x 524346 canonical k=10 k-mers, 49,668,801 nonzero
  entries (94.7% dense) - loaded from the existing
  `data/processed/pilot_ciprofloxacin_kmer_matrix.npz`, never recomputed or densified by this task.

## Split methodology and saved assignments used

- **Random split**: the same stratified random split Logistic Regression's original pilot report
  used (same seed, same `test_size=0.3`). Row indices were regenerated via
  `compute_random_split_indices`, which is proven (by a dedicated unit test) to reproduce the
  *exact* same partition as the original pilot's `train_and_evaluate_pilot_model` call - not a new
  split. Saved for the first time as `data/processed/pilot_ciprofloxacin_random_split.csv` for
  auditability.
- **Lineage-aware split**: loaded directly from the EXISTING
  `data/processed/pilot_ciprofloxacin_lineage_split.csv` (MLST-group-held-out, produced by the prior
  task) - **not regenerated**. Genome-id-to-row alignment and the absence of any row overlap
  between partitions were verified before use.

## Model configurations

- **LogisticRegression**: `{'max_iter': 1000, 'random_state': seed}` (lbfgs solver, scikit-learn's default - unchanged from the original pilot)
- **RandomForest**: `{'n_estimators': 100, 'n_jobs': -1, 'random_state': 20261009}`
- **XGBoost**: `{'n_estimators': 100, 'n_jobs': -1, 'random_state': 20261009}`

All three models: no learned preprocessing (scaler, feature selection, etc.) was applied - raw
k-mer counts are fed directly to each model, exactly as for the original Logistic Regression
pilot, so "fit preprocessing on training data only" is trivially satisfied (there is none to leak).
No hyperparameter search was performed for any model; Random Forest and XGBoost use their
library's own defaults plus only `random_state` (reproducibility) and `n_jobs=-1` (parallelism,
does not change results). Random Forest's default `max_features="sqrt"` for classifiers was left
as-is deliberately: with 524,346 features and only 100 genomes (p >> n), this
default is the standard mitigation against each tree considering all features at every split, not
a tuning choice made here.

## Results - random split

| Model | Train/Test n | Train / Test class counts | Accuracy | ROC-AUC | Avg. precision | Recall (R) | Recall (S) | Confusion matrix [[TN,FP],[FN,TP]] | Fit/Predict time |
|---|---|---|---|---|---|---|---|---|---|
| LogisticRegression | 70/30 | {'Susceptible': 52, 'Resistant': 18} / {'Susceptible': 23, 'Resistant': 7} | 0.9 | 0.9627329192546584 | 0.9200680272108843 | 0.5714285714285714 | 1.0 | [[23, 0], [3, 4]] | n/a (not tracked for LogisticRegression) |
| RandomForest | 70/30 | {'Susceptible': 52, 'Resistant': 18} / {'Susceptible': 23, 'Resistant': 7} | 0.9 | 0.9472049689440994 | 0.8864468864468864 | 0.5714285714285714 | 1.0 | [[23, 0], [3, 4]] | 0.548s / 1.047s |
| XGBoost | 70/30 | {'Susceptible': 52, 'Resistant': 18} / {'Susceptible': 23, 'Resistant': 7} | 0.9 | 0.9099378881987579 | 0.6210884353741497 | 1.0 | 0.8695652173913043 | [[20, 3], [0, 7]] | 11.894s / 0.096s |

## Results - lineage-aware split

| Model | Train/Test n | Train / Test class counts | Accuracy | ROC-AUC | Avg. precision | Recall (R) | Recall (S) | Confusion matrix [[TN,FP],[FN,TP]] | Fit/Predict time |
|---|---|---|---|---|---|---|---|---|---|
| LogisticRegression | 71/29 | {'Susceptible': 53, 'Resistant': 18} / {'Susceptible': 22, 'Resistant': 7} | 0.7931034482758621 | 0.8896103896103896 | 0.7398268398268397 | 0.8571428571428571 | 0.7727272727272727 | [[17, 5], [1, 6]] | n/a (not tracked for LogisticRegression) |
| RandomForest | 71/29 | {'Susceptible': 53, 'Resistant': 18} / {'Susceptible': 22, 'Resistant': 7} | 0.7586206896551724 | 0.9123376623376623 | 0.6654761904761904 | 0.0 | 1.0 | [[22, 0], [7, 0]] | 0.490s / 0.852s |
| XGBoost | 71/29 | {'Susceptible': 53, 'Resistant': 18} / {'Susceptible': 22, 'Resistant': 7} | 0.8620689655172413 | 0.8636363636363636 | 0.7261009667024704 | 0.7142857142857143 | 0.9090909090909091 | [[20, 2], [2, 5]] | 12.181s / 0.094s |

## Runtime observations

Fit/predict times are wall-clock, single run, on this machine - indicative only, not benchmarked
across repetitions. At 524,346 features for only 100 genomes, Random Forest and
XGBoost both completed without exceeding available memory; neither required densifying the sparse
matrix (verified by tests that the sparse matrix's `.toarray()` is never invoked during
evaluation).

**XGBoost sparse-input caveat (documented, not resolved here)**: XGBoost's sparsity-aware split
finding treats an implicit zero in a sparse matrix as a "missing value" signal during tree
building, not necessarily as the literal count zero. For k-mer **count** data, a zero entry means
"this k-mer was not observed in this genome" - a meaningful value, not missing data. This is a
known characteristic of how XGBoost consumes sparse input in general, not a bug introduced here,
but it means XGBoost's internal treatment of the feature matrix is subtly different from Random
Forest's and Logistic Regression's (which treat the stored zero as an ordinary feature value) -
worth keeping in mind when comparing feature importances or fine-tuning later, though it is not
expected to invalidate the pilot comparison here.

## Comparison and interpretation

- **Random split vs. lineage-aware split is not a paired comparison**: the two splits use
  different genomes in the test set (different composition, different class balance, different
  group structure) - see `reports/pilot_lineage_evaluation.md` for the full discussion already
  established for Logistic Regression, which applies identically here for Random Forest and
  XGBoost.
- Any difference between a model's random-split and lineage-aware metrics is preliminary evidence
  of sensitivity to test-set composition, not proof that controlling for lineage alone caused the
  difference, and not evidence that one model is "better" in general from a single ~100-genome
  pilot.
- No model was tuned against either test set's results; the configurations above were fixed before
  any evaluation was run.

## Warnings raised during evaluation

None.

## Computational limitations and caveats

- A single train/test split per model per evaluation strategy - no cross-validation, so variance
  across seeds/splits is not characterized here (noted as a next step in the lineage evaluation
  report too).
- 524,346 raw k=10 k-mer features for 100 genomes is an extreme p >> n regime;
  all three models are documented as prone to overfitting noise in this regime (Logistic
  Regression's L2 regularization, Random Forest's `max_features="sqrt"`, and XGBoost's shallow
  default tree depth all provide some mitigation, but none of this was tuned or verified beyond
  what each library does by default).
- This pilot's k=10 choice (vs. the separately unresolved proposal-vs-instructor k=10-vs-31
  question) was not revisited here - see `reports/pilot_ciprofloxacin_report.md`.

## Pilot results vs. final benchmark - explicit distinction

These are preliminary results from ONE ~100-genome pilot cohort for ONE antibiotic
(ciprofloxacin), using ONE k-mer configuration (k=10) and ONE random seed per split. They do not
establish which model, if any, will perform best on the full three-antibiotic benchmark, do not
use lineage-aware splitting as the default evaluation method going forward (that decision is still
open), and are not a substitute for evaluating Ampicillin and Gentamicin or for a larger cohort.
