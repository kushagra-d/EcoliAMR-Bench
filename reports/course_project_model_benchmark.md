# Course-Project Model Benchmark: Random vs. Lineage-Aware Evaluation

Runs the primary three-antibiotic, three-model benchmark on the frozen
956-genome cohort (`reports/course_project_final_cohort.md`) and its
hashed feature matrix (`reports/course_project_feature_generation.md`).
18/18 intended evaluations (3 models x 3 antibiotics x 2 split
strategies) completed successfully. No genome was downloaded, no
alternative feature matrix was generated, and every pre-existing pilot
artifact is unchanged.

## 1. Experimental design

- **Feature matrix**: the frozen 956 x 262,144 hashed matrix (k=15,
  d=262,144), loaded ONCE as a single scipy sparse CSR matrix directly
  from the SQLite row store - never densified, never rebuilt.
- **Splits**: the already-validated, frozen random (stratified 80/20)
  and lineage-aware (MLST-group, target 80/20) splits for each
  antibiotic, reproduced deterministically (seed `20261009`) from the
  genome IDs and labels in `course_project_final_cohort_{antibiotic}_manifest.csv`.
  Before any model was fit, every reproduced split's size and test-set
  class counts were asserted against the exact numbers already
  documented in `reports/course_project_final_cohort.md` - **all six
  (3 antibiotics x 2 strategies) matched exactly**, confirming this
  task reused the frozen splits rather than generating new ones.
- **No split was selected or adjusted based on any model result** -
  every split was fixed and verified before the first model was fit.
- **Labels never touched the feature matrix**: antibiotic-specific
  labels came only from each antibiotic's own manifest; one genome's
  label for ciprofloxacin was never assumed to apply to ampicillin or
  gentamicin.

## 2. Model settings (reused unchanged from existing project code)

| Model | Configuration | Source |
|---|---|---|
| Logistic Regression | `max_iter=1000, random_state=20261009` (lbfgs solver, sklearn default) | `pilot_model.py`, used by every prior task in this project |
| Random Forest | `n_estimators=100, n_jobs=-1, random_state=20261009` (all other sklearn defaults, including `max_features="sqrt"`) | `model_comparison.RANDOM_FOREST_PARAMS` |
| XGBoost | `n_estimators=100, n_jobs=-1, random_state=20261009` (xgboost 3.2.0 library defaults otherwise) | `model_comparison.XGBOOST_PARAMS` |

**No class weighting or resampling was applied to any model.** This
was a deliberate choice, not an oversight: no prior task in this
project configured `class_weight` or any resampling for these models,
and the task's own instruction to "preserve real-world class
imbalance" (carried through every prior planning task) argues against
introducing a new, previously-unused rebalancing mechanism at this
stage. If a reviewer wants class weighting evaluated, that is a
follow-up decision, not a silent default introduced here.

The shared metrics function (`pilot_model.compute_classification_metrics`)
was extended, additively, with balanced accuracy, macro-F1, and
per-class precision (previously only recall and accuracy were
computed) - this is the only production-code change made for this
task; it is purely additive (new dict keys only, all existing fields
and call sites unchanged) and is covered by 4 new unit tests
(`tests/test_pilot_model.py`), in addition to the existing 546 tests,
which all still pass unmodified.

## 3. Computational practicality

| Measurement | Value |
|---|---|
| Feature matrix shape | 956 x 262,144 |
| SQLite database size on disk | 1.87 GiB |
| Time to load the full matrix into memory | 16.0 s |
| Peak RSS immediately after loading | 4,577 MB |
| Peak RSS across the entire 18-evaluation run | **6,545 MB** |
| Project's established memory-safety threshold | 11,000 MB |
| Matrix ever densified (`.toarray()`/`.todense()`) | **No** - verified by code inspection; every model's `.fit()`/`.predict()` call received the sparse CSR slice directly |
| Total fit time, all 18 evaluations | 650.0 s |
| Total predict time, all 18 evaluations | 15.5 s |
| Total wall-clock (incl. matrix load) | 681.5 s ≈ 11.4 min |
| Mean fit time per model | LR 58.9 s, Random Forest 3.2 s, XGBoost 46.2 s |

Peak memory stayed at **59% of the 11GB safety threshold** at its
highest point - the benchmark never came close to triggering the
memory-safety stop condition (checked before every single fit; it was
never triggered). No model was infeasible; no evaluation was skipped
for resource reasons.

## 4. Results (all 18 evaluations)

Full detail: `data/processed/course_project_model_benchmark_results.json`;
flat table: `data/processed/course_project_model_benchmark_summary.csv`.

### Ciprofloxacin (836 eligible genomes: 190 Resistant / 646 Susceptible)

| Model | Split | n_train/n_test | Test R/S | Balanced acc. | Macro-F1 | ROC-AUC | Avg. precision | R precision/recall | S precision/recall | Confusion matrix [[S,S→R],[R→S,R]] |
|---|---|---|---|---|---|---|---|---|---|---|
| LogisticRegression | random | 668/168 | 38/130 | 0.949 | 0.949 | 0.977 | 0.965 | 0.921 / 0.921 | 0.977 / 0.977 | [[127,3],[3,35]] |
| RandomForest | random | 668/168 | 38/130 | 0.887 | 0.910 | 0.978 | 0.933 | 0.938 / 0.789 | 0.941 / 0.985 | [[128,2],[8,30]] |
| XGBoost | random | 668/168 | 38/130 | 0.915 | 0.915 | 0.973 | 0.932 | 0.868 / 0.868 | 0.962 / 0.962 | [[125,5],[5,33]] |
| LogisticRegression | lineage | 631/205 | 91/114 | 0.890 | 0.891 | 0.946 | 0.920 | 0.888 / 0.868 | 0.897 / 0.912 | [[104,10],[12,79]] |
| RandomForest | lineage | 631/205 | 91/114 | **0.648** | 0.619 | 0.962 | 0.955 | **1.000 / 0.297** | 0.640 / 1.000 | [[114,0],[64,27]] |
| XGBoost | lineage | 631/205 | 91/114 | 0.859 | 0.864 | 0.944 | 0.934 | 0.910 / 0.780 | 0.843 / 0.939 | [[107,7],[20,71]] |

### Ampicillin (734 eligible genomes: 393 Resistant / 341 Susceptible)

| Model | Split | n_train/n_test | Test R/S | Balanced acc. | Macro-F1 | ROC-AUC | Avg. precision | R precision/recall | S precision/recall | Confusion matrix |
|---|---|---|---|---|---|---|---|---|---|---|
| LogisticRegression | random | 587/147 | 79/68 | 0.875 | 0.876 | 0.943 | 0.959 | 0.867 / 0.911 | 0.891 / 0.838 | [[57,11],[7,72]] |
| RandomForest | random | 587/147 | 79/68 | 0.789 | 0.791 | 0.915 | 0.934 | 0.769 / 0.886 | 0.839 / 0.691 | [[47,21],[9,70]] |
| XGBoost | random | 587/147 | 79/68 | 0.860 | 0.862 | 0.944 | 0.954 | 0.847 / 0.911 | 0.887 / 0.809 | [[55,13],[7,72]] |
| LogisticRegression | lineage | 539/195 | 137/58 | 0.773 | 0.780 | 0.876 | 0.942 | 0.859 / 0.891 | 0.717 / 0.655 | [[38,20],[15,122]] |
| RandomForest | lineage | 539/195 | 137/58 | 0.709 | 0.720 | 0.817 | 0.911 | 0.818 / 0.883 | 0.660 / 0.534 | [[31,27],[16,121]] |
| XGBoost | lineage | 539/195 | 137/58 | 0.790 | 0.780 | 0.866 | 0.930 | 0.885 / 0.839 | 0.662 / 0.741 | [[43,15],[22,115]] |

### Gentamicin (827 eligible genomes: 93 Resistant / 734 Susceptible)

| Model | Split | n_train/n_test | Test R/S | Balanced acc. | Macro-F1 | ROC-AUC | Avg. precision | R precision/recall | S precision/recall | Confusion matrix |
|---|---|---|---|---|---|---|---|---|---|---|
| LogisticRegression | random | 661/166 | 19/147 | 0.720 | 0.748 | 0.795 | 0.597 | 0.643 / 0.474 | 0.934 / 0.966 | [[142,5],[10,9]] |
| RandomForest | random | 661/166 | 19/147 | 0.621 | 0.657 | 0.784 | 0.479 | 0.625 / 0.263 | 0.911 / 0.980 | [[144,3],[14,5]] |
| XGBoost | random | 661/166 | 19/147 | 0.707 | 0.766 | 0.863 | 0.612 | 0.889 / 0.421 | 0.930 / 0.993 | [[146,1],[11,8]] |
| LogisticRegression | lineage | 741/86 | **6**/80 | 0.642 | 0.642 | 0.583 | 0.238 | 0.333 / 0.333 | 0.950 / 0.950 | [[76,4],[4,2]] |
| RandomForest | lineage | 741/86 | **6**/80 | **0.494** | 0.479 | 0.751 | 0.242 | **0.000 / 0.000** | 0.929 / 0.988 | [[79,1],[6,0]] |
| XGBoost | lineage | 741/86 | **6**/80 | 0.660 | 0.707 | 0.760 | 0.412 | 0.667 / 0.333 | 0.952 / 0.988 | [[79,1],[4,2]] |

## 5. Random vs. lineage-aware evaluation: what differs, and why raw accuracy would mislead

**Raw accuracy is not used to compare the two strategies anywhere in
this report**, per the task's instruction - here is why, with the
most striking example from this run's own numbers:

- Gentamicin, Random Forest, lineage-aware split: **raw accuracy =
  91.9%**, which sounds strong. The confusion matrix shows why that
  number is misleading: the model predicted "Susceptible" for every
  single genome except one, achieving **0/6 Resistant recall** -
  balanced accuracy for this exact same run is **0.494**, i.e. no
  better than chance on the classes, not 0.919. Raw accuracy here
  mostly reflects that Susceptible outnumbers Resistant 80:6 in this
  particular test fold, not genuine resistance-detection skill.
- This pattern - a model achieving high raw accuracy by defaulting
  toward the majority class on a severely imbalanced test fold - is
  exactly what balanced accuracy, macro-F1, and per-class
  precision/recall are reported to catch, and exactly why the task
  instructed against raw-accuracy-only comparison.

**The two split strategies also differ in test size and Resistant
prevalence for every antibiotic** (not just gentamicin), which is a
second, independent reason raw accuracy (or any single pooled number)
cannot be compared directly across strategies:

| Antibiotic | Random test (n, R%) | Lineage test (n, R%) |
|---|---|---|
| Ciprofloxacin | 168, 22.6% | 205, 44.4% |
| Ampicillin | 147, 53.7% | 195, 70.3% |
| Gentamicin | 166, 11.4% | 86, 7.0% |

The lineage-aware test fold's Resistant prevalence is **higher** than
the random fold's for ciprofloxacin and ampicillin (because, as
`reports/course_project_final_cohort.md` already documented, large
MLST groups happened to land on the test side) and **lower** for
gentamicin. A model's balanced accuracy or macro-F1 under lineage-aware
evaluation is therefore being measured against a different class
mixture than under random evaluation, for every antibiotic - any
"random vs. lineage-aware" comparison in this report is a comparison
of generalization behavior under two different, specific evaluation
conditions, not a controlled ablation holding everything else fixed.

**Qualitative pattern observed (not a general claim)**: for
ciprofloxacin and ampicillin, every model's balanced accuracy is lower
under lineage-aware evaluation than under random evaluation (e.g.
ciprofloxacin LR: 0.949 random vs. 0.890 lineage; ampicillin XGBoost:
0.860 random vs. 0.790 lineage) - consistent with (but not proof of)
the random split allowing some genomes from the same clonal lineage to
appear on both sides, which lineage-aware evaluation specifically
prevents. Gentamicin does not show this pattern cleanly (its
lineage-aware numbers are noisy for the reason in Section 6) and
Random Forest's ciprofloxacin lineage-aware score (0.648) is a clear
outlier within this one run, not a trend.

## 6. Gentamicin lineage-aware evaluation: explicit uncertainty discussion

The gentamicin lineage-aware test fold has **6 Resistant genomes** out
of 93 Resistant total in its 827-genome cohort - already flagged as a
limitation in `reports/course_project_final_cohort.md` before any
model was run. The three models' Resistant recall on this fold: LR
2/6 (33.3%), Random Forest 0/6 (0.0%), XGBoost 2/6 (33.3%).

**Why these numbers should not be read as precise rates**: with only 6
positive test examples, each single prediction changes Resistant
recall by exactly 1/6 ≈ 16.7 percentage points. A "33.3% recall" result
means literally "got 2 of 6 right" - one different prediction on one
genome would have produced 16.7% or 50.0% instead. This is not a
measurement precise enough to support a claim like "XGBoost recalls
Resistant gentamicin genomes better than Random Forest" from this
fold alone; it is consistent with, at most, "Random Forest predicted
Resistant for only 1 test genome overall at all three models combined,
missing some" captured literally rather than quantified more finely.
ROC-AUC and average precision on this same fold (0.583-0.760,
0.238-0.412) are similarly built from only 6 positive examples and
carry the same granularity problem, compounded by needing to rank only
6 positives against 80 negatives. None of this means the models are
"bad" at gentamicin resistance specifically - it means this particular
856-genome-cohort, seed-20261009, lineage-aware fold is too small on
its Resistant side to distinguish genuine model skill from this
specific split's sampling luck.

## 7. Limitations and unresolved issues

1. **Gentamicin lineage-aware evaluation (Section 6) is the clearest
   case, but every small test fold in this benchmark carries some
   version of the same granularity problem** - ampicillin's lineage
   fold (195 genomes) and gentamicin's random fold (166 genomes, 19
   Resistant) are larger but still modest; metric precision scales
   with test-fold size, which this benchmark did not control across
   split strategies (Section 5).
2. **Hashed features (k=15, d=262,144) have no per-feature biological
   interpretability** (already documented in
   `reports/course_project_feature_generation.md`) - this benchmark
   reports which model/split combination scores better by these
   metrics, not which genomic features drive any prediction.
3. **No hyperparameter search was performed**, per the task's own
   instruction - all three models use their stated fixed configuration
   (Section 2) for every antibiotic and split; a different
   configuration (e.g. regularization strength, tree depth, class
   weighting) could change any of these numbers, and this benchmark
   does not explore that space.
4. **One seed, one split per strategy per antibiotic** - as already
   flagged in the split-design and final-cohort reports, a different
   seed would plausibly place different MLST groups into the
   lineage-aware test fold, changing both its size and class balance,
   which would in turn change every metric in Section 4 for that
   antibiotic's lineage-aware row. No cross-validation or repeated
   resampling was performed.
5. **Statistical significance between models or between split
   strategies was not tested** - this report states what was measured
   (Section 4) and explains why comparing those numbers directly can
   mislead (Sections 5-6), but it does not compute confidence
   intervals or significance tests for any metric difference. That is
   explicitly left to a follow-up statistical-analysis task.
6. **No biological interpretation of which features drove any
   prediction is given here** - also explicitly left to a follow-up
   task, consistent with the hashed representation's known
   interpretability limitation (point 2).

## 8. Validation summary

| Check | Result |
|---|---|
| All 18 intended model x antibiotic x split combinations accounted for | **18/18 completed, 0 missing** |
| Benchmark stopped early for a memory or infeasibility reason | **No** |
| Every reproduced split matches the already-frozen, documented split exactly | **True** (6/6 checks: 3 antibiotics x 2 strategies) |
| No MLST group appears in both train and test, for any lineage-aware split | **True** (independently re-verified for all 3 antibiotics, same check as the split-design/freeze tasks) |
| Both classes present in every train and every test partition | **True** (18/18; smallest minority-class count is gentamicin lineage-aware test's 6 Resistant genomes, still present) |
| Feature matrix densified at any point | **No** |
| Peak RSS vs. 11GB safety threshold | 6,545 MB / 11,000 MB - never exceeded |
| Existing pilot/expansion/cohort-scale model results and feature artifacts unchanged | Verified (file list and checksums spot-checked; this task wrote only new, distinctly-named files) |
| Full test suite | 549/549 pass (546 pre-existing + 3 new for the additive metrics extension; 1 existing test file gained 2 more assertions in-place) |

## 9. Outputs of this task

- `reports/course_project_model_benchmark.md` (this file)
- `data/processed/course_project_model_benchmark_results.json` - full
  detail for all 18 evaluations (params, warnings, confusion matrices,
  every metric, timings)
- `data/processed/course_project_model_benchmark_summary.csv` - flat,
  one-row-per-evaluation consolidated table
- Additive extension to `bvbrc_recon/pilot_model.compute_classification_metrics`
  (balanced accuracy, macro-F1, per-class precision) and its two
  result dataclasses (`PilotModelResult`, `model_comparison.ModelEvalResult`),
  covered by 3 new unit tests plus 2 extended existing tests

No genome was downloaded, no alternative feature matrix was generated,
and the proposal/instructor spreadsheet and every pre-existing pilot
artifact are unchanged.
