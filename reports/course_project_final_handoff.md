# Final Handoff: EcoliAMR-Bench Course Project

Consolidates the completed study for final-report preparation. Every
number below was re-verified against the stored artifacts for this
audit (`course_project_final_cohort_manifest.csv` and its
per-antibiotic manifests, `course_project_final_cohort_hashed_build_config_k15_d262144.json`,
`course_project_model_benchmark_results.json`,
`course_project_model_benchmark_summary.csv`) - not copied from prior
report prose uncritically. No model was retrained, no data was
downloaded, no feature was regenerated, and no existing report, code,
proposal, or spreadsheet was changed to produce this file.

## 1. Final study design

- **Cohort**: 956 unique genomes with validated assemblies (300 reused
  from earlier retrieval rounds + 656 of 700 newly attempted; 44
  genomes failed retrieval/identity checks and are permanently
  excluded - `reports/course_project_cohort_design.md` Section 8).
- **Antibiotic-specific usable samples** (verified directly from
  `course_project_final_cohort_{antibiotic}_manifest.csv`):

  | Antibiotic | n | Resistant | Susceptible |
  |---|---|---|---|
  | Ciprofloxacin | 836 | 190 | 646 |
  | Ampicillin | 734 | 393 | 341 |
  | Gentamicin | 827 | 93 | 734 |

- **Feature representation**: direct k=15 feature hashing, 262,144
  dimensions, built out-of-core (`sqlite_row_store.build_sqlite_backed_matrix`
  - hashes during the per-genome sequence walk, never materializes a
  raw k-mer matrix). Verified matrix shape 956 x 262,144, 250,609,664
  nonzero entries, 100% density, 0 build failures
  (`course_project_final_cohort_hashed_build_config_k15_d262144.json`).
  This representation has no per-feature biological interpretability
  (Section 4).
- **Models**: Logistic Regression (`max_iter=1000`), Random Forest
  (`n_estimators=100, n_jobs=-1`), XGBoost (`n_estimators=100,
  n_jobs=-1`) - all three with `random_state=20261009`, all other
  settings at library defaults, no class weighting or resampling, no
  hyperparameter search.
- **Evaluation strategies**: stratified random 80/20 split, and
  MLST-group-aware split (target 80/20, `GroupShuffleSplit`), both
  seeded `20261009`, independently reproduced and verified for each
  antibiotic before any model was fit (`course_project_model_benchmark_results.json`
  confirms `n_train`/`n_test`/test-class-counts for all 6 antibiotic x
  strategy combinations match exactly what Section 2 of
  `reports/course_project_final_cohort.md` already documented).
- **18/18 planned evaluations completed** (3 models x 3 antibiotics x
  2 strategies), 0 missing, 0 stopped for resource reasons (peak RSS
  6,545 MB against an 11,000 MB safety threshold).

**One historical note, not an error**: `reports/course_project_split_design.md`
validated the splits on a *proposed* 1,000-genome, pre-retrieval cohort
(projected antibiotic subset sizes 876/776/860). That proposal is
superseded by the actual 956-genome cohort above (836/734/827) once 44
of the 700 additional genomes failed retrieval. All reports from
`reports/course_project_final_cohort.md` onward correctly use the
final 956-genome numbers; `course_project_split_design.md` should be
read as validating the split *methodology* (seed, test fraction,
grouping approach), not the final sample sizes.

## 2. Strongest findings (verified)

- **Lineage-aware evaluation scored lower than random evaluation in
  all 9 model x antibiotic combinations, on both ROC-AUC and balanced
  accuracy** - recomputed directly from the stored results for this
  audit (9/9 negative differences, both metrics), not merely repeated
  from the interpretation report.
- **Gentamicin, Random Forest, lineage-aware split**: 91.9% raw
  accuracy, 0.494 balanced accuracy, 0/6 Resistant recall - confusion
  matrix `[[79,1],[6,0]]`. A concrete, verified case where high raw
  accuracy coincided with below-chance balanced performance, driven by
  the test fold's 80:6 class skew.
- **Ciprofloxacin, Random Forest, lineage-aware split** shows the
  single largest balanced-accuracy drop in the whole study (−0.239),
  explained by its own confusion matrix: Resistant precision 1.000,
  recall 0.297 - the model became much more conservative about
  predicting "Resistant" on this specific fold.
- **Magnitude of the random-vs-lineage gap varies by antibiotic and
  model with no single clean driver**: ampicillin's three models drop
  by a similar amount (−0.070 to −0.102 balanced accuracy); gentamicin
  and ciprofloxacin each show one large outlier (Logistic Regression
  and Random Forest respectively) alongside otherwise modest drops.

## 3. Observed results vs. plausible explanations vs. what the study cannot establish

- **Observed** (counted directly from stored results): the consistent
  negative direction above; the specific metric values in
  `course_project_model_benchmark_summary.csv`; the exact split sizes
  and class counts for each antibiotic and strategy.
- **Plausible, not proven**: that the consistent drop reflects clonal
  genomes leaking across train/test in the random split, which
  lineage-aware evaluation prevents. An equally consistent alternative
  is that the lineage-aware test folds simply have different size and
  class composition (true for every antibiotic - Section 1's table in
  `reports/course_project_results_interpretation.md`) for reasons
  unrelated to leakage.
- **Not established by this study**: statistical significance of any
  difference (none was tested); that any observed model or antibiotic
  ranking would hold under a different seed, repeated resampling, or a
  larger cohort; causation of any kind.

## 4. Biological-interpretation audit

**Existing artifact found**: `reports/pilot_ciprofloxacin_biological_interpretation.md`
and its supporting CSV (`pilot_ciprofloxacin_card_candidate_matches.csv`,
39 rows, cross-checked against the report's own counts: 37 exact + 2
approximate matches, matching the report exactly).

**What that exercise actually covers (explicitly scoped, not the final
study)**: 39 top-ranked k-mer candidates from a Logistic
Regression/Random Forest refit on a **70-genome ciprofloxacin
training-only** subset, using **raw canonical k=10 k-mers** - a
completely different, non-hashed, single-antibiotic, small-pilot
representation from the one used in the final 956-genome, k=15/d=262,144
hashed benchmark. No equivalent exercise exists for ampicillin or
gentamicin, and none was ever run on the hashed representation or the
full 956-genome cohort.

**Its own findings (already a negative result, reported honestly in
the original task)**: of 39 candidates, only 5 matched exactly one
distinct CARD gene (the least ambiguous outcome available); **none of
those 5 correspond to a known fluoroquinolone-resistance mechanism**
(gyrA/gyrB/parC/parE/qnr/oqxA/oqxB/qepA/aac(6')-Ib-cr). 25
candidate/gene-hit pairs happened to include a known-mechanism gene
name, but in every one of those cases the same candidate also matched
unrelated genes - zero unambiguous hits to a known mechanism. The
report itself attributes this partly to k=10 nearly saturating CARD's
reference sequence space (87.2% coverage), making any k=10 "exact
match" weak evidence on its own.

**Conclusion for this audit**: the existing evidence does **not**
support any defensible gene- or mechanism-level biological claim for
this project, on either the pilot k=10 exercise (which found no
unambiguous link itself) or the final study (which used a
hashed representation that cannot be mapped back to individual k-mers
or genes at all, as already documented in
`reports/course_project_feature_generation.md`). **Nothing in this
project's hashed, full-cohort benchmark identifies, or should be
described as identifying, any specific resistance gene or mechanism.**
The pilot CARD exercise should be cited, if at all, only as a labeled,
preliminary, single-antibiotic, k=10-specific side-exploration that
itself returned a negative/inconclusive result - not as biological
support for the main benchmark's findings.

## 5. Final limitations checklist

- [ ] Only one random split and one lineage-aware split were evaluated
      per antibiotic (seed `20261009`) - no repeated resampling.
- [ ] No statistical significance testing was performed on any metric
      difference, model comparison, or split comparison.
- [ ] Random and lineage-aware test folds differ in both size and
      Resistant prevalence for every antibiotic - metric differences
      reflect both the evaluation strategy and this fold-composition
      difference, inseparably.
- [ ] The gentamicin lineage-aware test set contains only 6 Resistant
      genomes; each single prediction changes Resistant recall by
      ≈16.7 percentage points, making its recall/precision/ROC-AUC
      figures too coarse to treat as precise estimates.
- [ ] MLST (7-gene scheme) grouping is a practical relatedness proxy,
      not proof of genomic independence between groups.
- [ ] No hyperparameter search was performed for any model.
- [ ] The hashed feature representation (k=15, d=262,144) has no
      per-feature biological interpretability; the only existing
      biological-matching exercise (Section 4) used a different,
      smaller, single-antibiotic pilot representation and found no
      unambiguous link to a known resistance mechanism.
- [ ] 44 genomes selected for the intended cohort were never
      successfully retrieved and remain permanently excluded
      (root cause documented in
      `course_project_additional_700_execution_summary.json`).
- [ ] The cohort's lineage composition and sampling are specific to
      this BV-BRC-derived dataset; generalization to broader *E. coli*
      populations is not established.

## 6. What remains before final submission

**Essential (report/figure preparation only - no new experiments)**:
- Consolidate the six existing reports (`course_project_final_cohort.md`,
  `course_project_split_design.md`, `course_project_feature_generation.md`,
  `course_project_model_benchmark.md`, `course_project_results_interpretation.md`,
  this handoff) into whatever single document format the course
  submission requires.
- Produce any figures the final report needs (e.g. a bar or dot plot
  of the random-vs-lineage balanced-accuracy/ROC-AUC deltas in Section
  2's table) from the already-computed, stored values - no new
  computation is required, only visualization.
- Proofread the final write-up specifically for the observed/plausible/
  not-established distinction (Section 3) and the biological-claim
  boundary (Section 4), since these are the two places an overclaim is
  easiest to introduce unintentionally during final editing.

**Optional future scientific extensions (explicitly out of scope for
this handoff and not started)**:
- Statistical significance testing (e.g. McNemar's test or bootstrap
  confidence intervals) for the random-vs-lineage and model-vs-model
  differences.
- Repeated-seed or cross-validated evaluation to separate genuine
  lineage effects from single-split sampling noise, particularly for
  gentamicin.
- A hyperparameter search, if the course scope is extended to allow it.
- A biologically interpretable feature representation (e.g. the
  pilot's raw k-mer or minimizer approach, extended to the full cohort
  with appropriate memory handling) paired with CARD matching for
  ampicillin and gentamicin, if gene-level interpretation becomes a
  course requirement - this project's current hashed representation
  cannot support that without rebuilding features differently.
