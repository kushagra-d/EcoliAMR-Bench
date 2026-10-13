# Incremental Model Quality: Investigation Without Compromising Evaluation Validity

Exploratory pilot only. Existing 100-genome ciprofloxacin cohort,
splits, and hybrid-reader artifacts were reused unchanged. No genomes
were downloaded. Labels, saved splits, and the feature mapping were NOT
modified.

## Root causes investigated (Step 1)

Confirmed from `incremental_model.py` and the earlier fix task: the
current path already applies L2 row-normalization (fixed the original
ROC-AUC=0.5 collapse) and uses 50 epochs. Remaining open questions going
into this task: (a) does `SGDClassifier`'s loss/optimization suit this
262,144-dim, ~70-row sparse problem; (b) does class imbalance (26%
Resistant) explain the persisting zero Resistant recall; (c) would a
structurally different incremental estimator do better on the SAME
features. All three were investigated; results below.

## Reference baseline (Step 2)

Reused the batch LogisticRegression reproduction from an earlier task
(`data/processed/batch_lr_reproduction_ciprofloxacin_d262144.json`),
already verified `matches_existing_saved_result_exactly: true` against
the original saved result - re-confirmed here, not recomputed, since it
was already an independent artifact at a new path, never overwriting
the original.

| Split | Accuracy | ROC-AUC | Avg. Precision | Resistant recall | Susceptible recall |
|---|---|---|---|---|---|
| random | 0.900 | 0.938 | 0.876 | 0.571 | 1.000 |
| lineage | 0.759 | 0.857 | 0.820 | 0.714 | 0.773 |

## Predefined candidate set (Step 3) - decided before any validation result was seen

| Config | Model | Preprocessing | Justification |
|---|---|---|---|
| `sgd_log_loss_baseline` | `SGDClassifier(loss="log_loss")` | L2-normalized | The established path from the prior fix task - the baseline this task tries to beat. |
| `sgd_log_loss_balanced` | + `class_weight` (computed once from real train label counts, not sklearn's per-batch `"balanced"` string) | L2-normalized | Standard imbalance handling for linear classifiers. |
| `sgd_modified_huber_baseline` | `SGDClassifier(loss="modified_huber")` | L2-normalized | scikit-learn documents this loss as more robust to outliers than log-loss, while still supporting `predict_proba`; untried before. |
| `sgd_modified_huber_balanced` | + the same explicit `class_weight` | L2-normalized | Combines the robust-loss and imbalance candidates. |
| `multinomial_nb_default_prior` | `MultinomialNB()` | **raw counts, never normalized** (normalizing would break the multinomial-count assumption) | A fundamentally different, generative estimator natively suited to non-negative count features - this project's hashed representation exactly - with no learning-rate/regularization tuning and native `partial_fit` support. |
| `multinomial_nb_balanced_prior` | `MultinomialNB(class_prior=[0.5, 0.5])` | raw counts | Forces the decision away from the majority-class base rate. |

**Explicitly excluded, not silently skipped:** alternative `learning_rate` schedules for SGD were investigated in an earlier task and showed no benefit once normalization was already in place; revisiting them was deprioritized in favor of testing different loss functions, explicit class weighting, and a structurally different estimator. This was a scope decision, not an oversight.

## Strict evaluation boundaries (Step 4)

Selection used a stratified 80/20 internal validation carve of the
**random split's TRAIN partition only** (seed=7; 56 sub-train rows, 14
held-out validation rows) - the same precedent set in the earlier
incremental-model-fix task. `class_weight` for each candidate was
computed from the relevant training partition's OWN real label counts
(sub-train's counts during selection; each real split's own train
counts during final evaluation) - never from validation or test data.
**Neither real test set was touched until after the configuration was
selected.** All six candidates were evaluated and recorded, not just
the winner - see the table below and
`data/processed/incremental_model_candidate_selection_ciprofloxacin_k15_d262144.json`
for the full machine-readable record.

### Internal validation results (all six candidates, train-only)

| Config | Accuracy | ROC-AUC | Avg. Precision | Resistant recall |
|---|---|---|---|---|
| sgd_log_loss_baseline | 0.714 | 0.625 | **0.526** | 0.0 |
| sgd_log_loss_balanced | 0.714 | 0.550 | 0.496 | 0.0 |
| sgd_modified_huber_baseline | 0.714 | 0.500 | 0.286 | 0.0 |
| sgd_modified_huber_balanced | 0.714 | 0.500 | 0.286 | 0.0 |
| multinomial_nb_default_prior | 0.571 | 0.700 | 0.400 | **1.0** |
| multinomial_nb_balanced_prior | 0.571 | 0.700 | 0.400 | **1.0** |

**Selection rule, declared before this table was produced:** rank by
average precision first, Resistant recall as a tie-break. Applied
faithfully, this rule selects `sgd_log_loss_baseline` - the SAME
configuration already in use. **This is reported honestly, not
reframed after the fact:** `class_weight` and `modified_huber` both
made things measurably WORSE on this internal validation fold (lower
average precision, lower or equal AUC, recall still 0.0) - genuine
negative results, not omitted. `MultinomialNB` achieves materially
better Resistant recall (1.0 vs. 0.0) and better ROC-AUC (0.70 vs.
0.625) but LOWER average precision (0.40 vs. 0.53) and lower raw
accuracy - the predefined, average-precision-first rule does not select
it, even though it directly addresses this task's named concern about
Resistant recall. `class_prior=[0.5, 0.5]` produced IDENTICAL results
to the default prior on this fold, for both metrics - no distinguishable
effect was measured.

## Fair evaluation on both real test sets (Step 5)

**Official selection (`sgd_log_loss_baseline`, chosen by the predefined
rule), applied once to each split's real test set:**

| Split | Accuracy | ROC-AUC | Avg. Precision | Resistant recall | Susceptible recall | Confusion matrix |
|---|---|---|---|---|---|---|
| random | 0.767 | 0.745 | 0.589 | **0.000** | 1.000 | [[23,0],[7,0]] |
| lineage | 0.759 | 0.539 | 0.266 | **0.000** | 1.000 | [[22,0],[7,0]] |

This is **not an improvement** over the status quo measured in the
prior task - the predefined, train-only selection procedure, applied
honestly, reconfirms it. Compared to the batch LR baseline (random:
acc 0.900/AUC 0.938/avg-prec 0.876/Resistant-recall 0.571; lineage: acc
0.759/AUC 0.857/avg-prec 0.820/Resistant-recall 0.714), the incremental
path remains clearly behind on every metric that matters for this
imbalanced problem, especially Resistant recall and average precision -
accuracy alone (0.759, matching the lineage baseline by coincidence)
would be a misleading read on its own, exactly as this task's
instructions warned.

**Reported for completeness, not as the selected result** (Step 4's
"report all tested configurations, including failures" - this is the
opposite of a failure, but it did not win under the declared rule, so
it is reported transparently rather than silently substituted in):
`multinomial_nb_default_prior` applied to both real test sets:

| Split | Accuracy | ROC-AUC | Avg. Precision | Resistant recall | Susceptible recall | Confusion matrix |
|---|---|---|---|---|---|---|
| random | 0.800 | 0.671 | 0.390 | **0.429** | 0.913 | [[21,2],[4,3]] |
| lineage | 0.517 | 0.682 | 0.333 | **1.000** | 0.364 | [[8,14],[0,7]] |

**This is a genuine, meaningful tradeoff, not a clean win:**
`MultinomialNB` achieves nonzero - on lineage, PERFECT - Resistant
recall on both real test sets, something the selected SGD configuration
never achieves. It also beats SGD's ROC-AUC on the lineage split (0.682
vs. 0.539). But it does so by over-predicting "Resistant" on the
lineage split (Susceptible recall collapses to 0.364, accuracy to
0.517) and has lower average precision than SGD on both splits. Neither
configuration is unambiguously better; which one is operationally
preferable depends on whether missing a Resistant genome (false
negative) or over-flagging Susceptible genomes (false positive) is the
more costly error for this screening task - a judgment call outside the
scope of a pure train-only-validation selection rule, and reported here
as an open question rather than resolved unilaterally.

## Training time and peak RSS (Step 5)

| Config | Split | fit_seconds | Peak RSS before | Peak RSS after |
|---|---|---|---|---|
| sgd_log_loss_baseline | random | 3.54 s | 607.5 MB | 855.2 MB |
| sgd_log_loss_baseline | lineage | 3.59 s | 855.3 MB | 855.5 MB |
| multinomial_nb_default_prior | random | 1.68 s | 153.1 MB | 537.4 MB |
| multinomial_nb_default_prior | lineage | 1.70 s | 537.5 MB | 551.2 MB |

(Peak RSS is `ru_maxrss`, the cumulative per-process high-water mark,
same convention as every prior task - the lineage row's "before" value
is the random split's "after" value within the same process run.)
`MultinomialNB` was consistently faster to fit than `SGDClassifier`
here - plausible given it has no gradient/learning-rate computation per
`partial_fit` call, just count accumulation - though this was not the
focus of the investigation and is reported as an observed side effect,
not a validated general claim.

## Test results (Step 6)

441 tests pass (428 before this task, plus 13 new in
`tests/test_incremental_model_candidates.py`: the `balanced_class_weight`
helper matching sklearn's own formula and correctly omitting zero-count
classes, mechanical confirmation that an explicit `class_weight` dict is
actually accepted and used, `predict_proba`-based ROC-AUC correctness on
perfectly separable data, `MultinomialNB` running correctly on raw
unnormalized counts and with an explicit balanced prior, stateless
normalization provably unaffected by unrelated rows elsewhere in the
store, minority-class recall/average-precision reporting, exact
train/test class-count alignment against the label dictionary, the
bounded-batch-size invariant, determinism for both `SGDClassifier` and
`MultinomialNB` under a fixed seed, and artifact-path isolation). No
previously-passing test was broken.

## Final validation (Step 8)

- All existing matrices, the hybrid cache artifacts (reused read-only,
  `is_complete()` confirmed true before any run - no rebuild occurred),
  the SQLite store, both saved splits, and every prior task's results:
  unchanged (checksums re-verified).
- `data/raw/ciprofloxacin_pilot/`: still exactly 100 `.fna` files - no
  new downloads.
- Bounded-memory access confirmed: `max_batch_rows_held_in_memory`
  stayed at `batch_size` (16) in every run reported above, for both
  `SGDClassifier` and `MultinomialNB` candidates alike - no estimator
  investigated here required the full matrix in RAM at any point.
- `git status`: only the pre-existing modified `.gitignore` plus new
  untracked files; nothing staged or committed.

## Is the incremental approach suitable for further scaling? (Summary)

**Not yet, on the evidence gathered across this and the prior task.**
The officially selected configuration (by a predefined, honestly-applied
rule) is statistically indistinguishable from the status quo and
remains clearly behind the batch baseline on every metric that matters
for this imbalanced problem. A genuinely different estimator
(`MultinomialNB`) was found that trades away average precision and
(on one split) accuracy for materially better Resistant recall - a real,
reportable alternative, but not a validated replacement, and the choice
between the two is a judgment call this task does not resolve. Both
remain far below the batch `LogisticRegression` baseline's
average-precision and Resistant-recall numbers on the random split.

## Remaining blockers before further scaling

1. **No incremental configuration tested here matches the batch
   baseline's quality.** The gap is largest on average precision and
   Resistant recall, the two metrics this task explicitly flagged as
   most important for this imbalanced problem.
2. **The MultinomialNB vs. SGD tradeoff is unresolved** - which
   error profile (missed Resistant genomes vs. over-flagged Susceptible
   genomes) is acceptable is a domain/operational decision, not a
   statistical one, and was correctly left open rather than decided
   unilaterally.
3. **The internal validation fold is tiny (14 rows)** - the same
   limitation flagged in the earlier fix task - so none of these
   rankings should be treated as precise; they are directionally
   informative at best.
4. **Only one representation (k=15, d=262144), one antibiotic, and the
   100-genome pilot were used.** No full-dataset download,
   three-antibiotic benchmark, or Random Forest/XGBoost work was
   started, per this task's explicit stop condition.
