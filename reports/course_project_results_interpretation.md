# Scientific Interpretation of the Primary Model Benchmark

Interprets the results already produced and validated in
`reports/course_project_model_benchmark.md` and
`data/processed/course_project_model_benchmark_results.json`. No model
was retrained, no split was changed, no genome was downloaded, and no
feature matrix was generated for this task - every number below was
read from the existing stored results, not recomputed from scratch.

## 1. Comparison table

**Sign convention**: Δ = lineage-aware value − random-split value, for
the same model and antibiotic. A **negative Δ means lineage-aware
evaluation scored lower** than random evaluation; a positive Δ would
mean the opposite.

| Antibiotic | Model | Random ROC-AUC | Lineage ROC-AUC | ΔROC-AUC | Random bal. acc. | Lineage bal. acc. | Δbal. acc. | Random test n (R/S) | Lineage test n (R/S) |
|---|---|---|---|---|---|---|---|---|---|
| Ciprofloxacin | Logistic Regression | 0.977 | 0.946 | −0.031 | 0.949 | 0.890 | −0.059 | 168 (38/130) | 205 (91/114) |
| Ciprofloxacin | Random Forest | 0.978 | 0.962 | −0.016 | 0.887 | 0.648 | **−0.239** | 168 (38/130) | 205 (91/114) |
| Ciprofloxacin | XGBoost | 0.973 | 0.944 | −0.029 | 0.915 | 0.859 | −0.056 | 168 (38/130) | 205 (91/114) |
| Ampicillin | Logistic Regression | 0.943 | 0.876 | −0.067 | 0.875 | 0.773 | −0.102 | 147 (79/68) | 195 (137/58) |
| Ampicillin | Random Forest | 0.915 | 0.817 | −0.098 | 0.789 | 0.709 | −0.080 | 147 (79/68) | 195 (137/58) |
| Ampicillin | XGBoost | 0.944 | 0.866 | −0.078 | 0.860 | 0.790 | −0.070 | 147 (79/68) | 195 (137/58) |
| Gentamicin | Logistic Regression | 0.795 | 0.583 | −0.212 | 0.720 | 0.642 | −0.078 | 166 (19/147) | 86 (6/80) |
| Gentamicin | Random Forest | 0.784 | 0.751 | −0.033 | 0.621 | 0.494 | −0.128 | 166 (19/147) | 86 (6/80) |
| Gentamicin | XGBoost | 0.863 | 0.760 | −0.102 | 0.707 | 0.660 | −0.047 | 166 (19/147) | 86 (6/80) |

Every one of the 18 underlying numbers (9 ROC-AUC pairs, 9 balanced-accuracy
pairs) was read directly from `course_project_model_benchmark_results.json`;
every Δ above was computed as lineage-aware minus random, with the
result double-checked programmatically for this task (not transcribed
by hand).

## 2. Interpretation of the findings

### Observation: the direction of the difference is fully consistent

**All 9 model x antibiotic combinations show a negative Δ for both
ROC-AUC and balanced accuracy** - lineage-aware evaluation scored
lower than random evaluation in every single case measured in this
benchmark, with no exception. This is a clean, countable observation
from the stored results, not an interpretation.

**Plausible interpretation** (not proven by this data alone): this is
consistent with the random split allowing genomes from the same MLST
group (clonal lineage) to appear on both sides of the train/test
boundary, which would let a model partly recognize a near-duplicate
genome it has already seen rather than genuinely generalize - lineage-
aware evaluation specifically closes that path. An equally consistent
alternative explanation is that the lineage-aware test folds simply
contain a harder or differently-distributed set of genomes for reasons
unrelated to clonal leakage (Section 2's class-prevalence differences,
discussed below, are one concrete candidate). This benchmark's design
- one split per strategy, no repeated resampling - cannot distinguish
between these explanations; it can only report that the direction is
consistent across every model and antibiotic tested.

### Does the pattern differ by antibiotic?

The *direction* does not differ (9/9 negative), but the *magnitude*
does, and not in a way that tracks any single obvious cause cleanly:

- **Ampicillin** shows the most uniform pattern: all three models drop
  by a similar amount under lineage-aware evaluation (ΔROC-AUC −0.067
  to −0.098; Δbalanced accuracy −0.070 to −0.102) - the most
  "model-independent" looking result in this table.
- **Ciprofloxacin** shows the opposite: all three models' ROC-AUC
  values stay close together and high under both strategies (0.973-0.978
  random, 0.944-0.962 lineage) - but Random Forest's balanced accuracy
  fell much further than the other two models (−0.239 vs. −0.056 to
  −0.059 for Δbalanced accuracy), a model-specific pattern discussed
  below, not an antibiotic-wide one.
- **Gentamicin** shows the largest single ROC-AUC drop in the entire
  table (Logistic Regression, −0.212) alongside one of the smallest
  (Random Forest, −0.033) - the most inconsistent *magnitude* across
  models for one antibiotic. Section 3 explains why gentamicin's
  lineage-aware numbers in particular should be treated as
  low-precision measurements, which plausibly explains some of this
  inconsistency rather than reflecting a genuine antibiotic-level
  effect.

**No claim is made here that any one antibiotic is "more affected" by
lineage structure than another** in a generalizable sense - the
magnitude differences above are specific to this one cohort, this one
seed, and (for gentamicin) a test fold too small to measure precisely.

### Differences between the three models

- **Random Forest showed the largest single balanced-accuracy drop in
  the entire table** (ciprofloxacin, −0.239) alongside the smallest
  gentamicin ROC-AUC drop (−0.033) - it is the most volatile of the
  three models across antibiotics in this benchmark, not uniformly
  "better" or "worse" under lineage-aware evaluation than Logistic
  Regression or XGBoost.
- **Logistic Regression and XGBoost behave more similarly to each
  other** than either does to Random Forest: their Δbalanced-accuracy
  values cluster more tightly (−0.047 to −0.102 across all 6 of their
  antibiotic combinations, vs. Random Forest's −0.080 to −0.239).
- The ciprofloxacin lineage-aware Random Forest result is explained by
  its own confusion matrix (already reported in the benchmark report):
  **Resistant precision 1.000, Resistant recall 0.297** - when it
  predicted "Resistant," it was always correct, but it missed 70% of
  actual Resistant genomes, defaulting to "Susceptible" far more than
  the other two models on this specific split. This is a concrete,
  measured behavior, not a general claim about Random Forest's
  suitability for this problem.

### Class imbalance and minority-class (Resistant) recall

Resistant recall under the **random** split already tracks each
antibiotic's native class balance before lineage-awareness is even
introduced: gentamicin (11.2% Resistant) has the lowest Resistant
recall of the three antibiotics for every model (0.263-0.474),
ciprofloxacin (22.7% Resistant) is intermediate (0.789-0.921), and
ampicillin (53.5% Resistant, the only antibiotic with Resistant as
close to a majority class) has the highest (0.886-0.911). This
ordering matches the three antibiotics' known class-imbalance ordering
exactly, consistent with - but not sole proof of - minority-class
recall being harder to achieve as a class becomes rarer in training
data, a well-known general pattern in imbalanced classification rather
than a finding specific to this dataset.

Lineage-aware evaluation layers a second, separate effect on top of
this: for gentamicin specifically, Resistant recall falls further
under the lineage split for every model (e.g. Random Forest: 0.263
random -> 0.000 lineage) - but, as Section 3 details, gentamicin's
lineage-aware test fold has only 6 Resistant genomes, so this specific
recall figure is measured too coarsely to separate a real
lineage-related effect from this particular fold's small-sample
behavior.

### The gentamicin Random Forest example

The benchmark report's headline example - **91.9% raw accuracy, 0.494
balanced accuracy, 0/6 Resistant recall**, all from the same
evaluation (gentamicin, Random Forest, lineage-aware split) -
illustrates a specific, measured failure mode: the model predicted
"Susceptible" for 79 of 80 true-Susceptible genomes and for all 6
true-Resistant genomes, achieving high raw accuracy purely because
Susceptible dominates this particular 86-genome test fold (80:6). Its
balanced accuracy of 0.494 is slightly below what a coin flip would
achieve in expectation (0.5) - this single model/antibiotic/split
combination learned a rule for this fold that performs no better than
chance once class imbalance is corrected for, despite its high raw
accuracy. This is reported as a specific, verified result of this one
evaluation, not generalized to "Random Forest cannot detect gentamicin
resistance."

### How differing test-set sizes and class prevalence limit direct comparison

Random and lineage-aware test folds differ in both size and Resistant
prevalence for every antibiotic (right two columns of Section 1's
table): ciprofloxacin's lineage fold is both larger (205 vs. 168) and
has much higher Resistant prevalence (44.4% vs. 22.6%) than its random
fold; ampicillin shows the same direction (195 vs. 147 genomes, 70.3%
vs. 53.7% Resistant); gentamicin's lineage fold is smaller (86 vs. 166)
with lower Resistant prevalence (7.0% vs. 11.4%). Because ROC-AUC,
balanced accuracy, and every other metric in this benchmark are
computed on these differently-sized, differently-balanced test folds,
**a Δ value in Section 1's table reflects the combined effect of
(a) whatever the lineage-aware split structurally changes and
(b) this specific fold's size and class mixture** - the two cannot be
separated with the data produced by this benchmark. A Δ of, say,
−0.212 (gentamicin, Logistic Regression) is a true, measured
difference between two specific evaluation conditions; it is not a
controlled estimate of "the effect of lineage-awareness" holding test
composition fixed.

## 3. Uncertainty and limitations

- **Only one split per strategy, per antibiotic, was evaluated** (seed
  `20261009`). Every number in Section 1 is a single realization, not
  an average over repeated splits - a different seed would plausibly
  change which genomes (and which MLST groups) land in each test fold,
  which would change the exact Δ values, though not necessarily their
  consistently-negative direction (Section 2).
- **No statistical significance test was performed** on any Δ value or
  on any model-vs-model comparison. Nothing in this report should be
  read as "the difference between X and Y is statistically
  significant" - that claim was not tested and is not made here.
- **Random and lineage-aware splits use different test genomes and
  different class distributions** for every antibiotic (Section 1's
  right two columns) - this is stated plainly here because it is the
  main reason Section 2's magnitude comparisons (as opposed to the
  direction-of-difference observation) must be read cautiously.
- **The gentamicin lineage-aware test set contains only 6 Resistant
  genomes.** Each individual correct or incorrect prediction on this
  fold changes Resistant recall by exactly 1/6 ≈ 16.7 percentage
  points - the measured recall values (0.0%, 33.3%, 33.3% for Random
  Forest, Logistic Regression, and XGBoost respectively) are too
  coarse-grained to distinguish genuine differences in model skill
  from this fold's specific sampling outcome. The same granularity
  problem applies, with less severity, to this fold's ROC-AUC and
  average precision, which are also built from only 6 positive
  examples.
- **MLST is a proxy for lineage relatedness, not proof of genomic
  independence between groups.** A 7-gene scheme can place two genomes
  in different groups despite recent shared horizontal transfer or
  other close relationships MLST does not capture (already documented
  in `lineage_split.py`'s own docstring) - "lineage-aware" in this
  project means "MLST-group-aware," which is a practical
  approximation, not a guarantee that no relatedness crosses the
  train/test boundary.
- **Feature hashing (k=15, d=262,144) prevents direct biological
  interpretation of individual hashed features** (already documented
  in `reports/course_project_feature_generation.md`) - this report
  interprets *performance* differences between splits and models; it
  makes no claim about which genomic sequences or resistance
  mechanisms drive any prediction, because that information is not
  recoverable from a hashed feature without an unhashing step this
  project does not implement.
- **This study does not establish generalization to *E. coli*
  populations beyond this specific 956-genome, BV-BRC-derived cohort.**
  The cohort's lineage composition, geographic/temporal sampling, and
  the specific 44 genomes excluded for retrieval/identity reasons
  (`reports/course_project_cohort_design.md` Section 8) are all
  properties of this particular dataset, not of *E. coli* resistance
  in general.

Nothing in this report claims statistical significance or causation
for any observed difference - Section 2 explicitly separates what was
counted/measured (the consistent negative direction, the specific
magnitudes) from what is a plausible but unproven explanation (clonal
leakage in the random split).

## 4. What this report does and does not conclude

**Established by the stored results (counted, not inferred)**: lineage-
aware evaluation scored lower than random evaluation for every one of
the 9 model/antibiotic combinations, on both ROC-AUC and balanced
accuracy, in this one benchmark run.

**Not established**: that this difference is statistically significant,
that it is caused specifically by clonal leakage in the random split
(rather than, for example, the test folds' differing class
composition), that it would hold under a different seed or a larger
cohort, or that any model/antibiotic ranking observed here generalizes
beyond this dataset.
