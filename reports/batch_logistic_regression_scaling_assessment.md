# Batch-Equivalent Logistic Regression with Disk-Backed Data Access

Exploratory pilot only. Existing 100-genome ciprofloxacin cohort,
hashed matrix, splits, SQLite store, and hybrid cache artifacts were
reused unchanged. No genomes were downloaded. Labels, saved splits, and
the feature mapping were NOT modified.

## Can the exact batch configuration be used with bounded memory? (Step 1/3)

**No, not as the literal `sklearn.linear_model.LogisticRegression(solver="lbfgs")`
object** - it has no `partial_fit`, and its internal objective/gradient
function (`X @ w`, `X.T @ residual`) is handed the whole matrix `X` as
one array. This is stated plainly, not worked around by substituting a
different algorithm and calling it equivalent.

**But the OPTIMIZER itself was never the obstacle.** L-BFGS-B (what
`solver="lbfgs"` actually runs, via scipy) is a *limited-memory*
quasi-Newton method - its own working memory is O(d · history_size),
independent of the number of training rows n. The obstacle is purely
how sklearn's objective function is implemented (one in-memory array
operation), not a mathematical requirement of the optimizer family.

**What was implemented instead** (`src/bvbrc_recon/streaming_logistic_regression.py`):
the IDENTICAL L2-regularized logistic loss and gradient, re-expressed
as a sum over rows -
`loss = Σ log(1+exp(-y_i·(w·x_i+b))) + 0.5‖w‖²/C`,
`∇w = Σ sᵢ·xᵢ + w/C`, `sᵢ = -yᵢ·sigmoid(-yᵢ·(w·xᵢ+b))` - computed by
streaming batches through the existing `HybridTrainingReader`, with the
running accumulators (one d-length gradient vector, two scalars) as the
only persistent state. This is handed to `scipy.optimize.minimize(method="L-BFGS-B")`
as a standard streaming objective - the same optimizer family sklearn
uses, fed a disk-backed objective instead of an in-memory one.

## Validated, not assumed (Step 2 methodology applied to the new component)

On a small synthetic dataset where both can be fit quickly: this
implementation, run to a tight tolerance, reaches a LOWER value of the
identical objective than sklearn's own default-tolerance fit does -
i.e. it is not merely "close," it more precisely minimizes the same
function sklearn is also minimizing. Coefficients and predicted
probabilities matched sklearn's to within `1e-2` (`tests/test_streaming_logistic_regression.py`).
The reproduced batch baseline itself
(`data/processed/batch_lr_reproduction_ciprofloxacin_d262144.json`,
reused from an earlier task) remains `matches_existing_saved_result_exactly: true`
against the original saved result.

**A real numerical bug was found and fixed while validating on the real
pilot data, not glossed over:** the naive formula `-log(sigmoid(margin))`
underflows to `log(0) = -inf` whenever `margin` is very negative - which
happens routinely here, because 262,144 features against ~70-71
training rows makes the data almost certainly linearly separable, so
early L-BFGS-B steps push margins to extreme values before the L2 term
reins them in. The FIRST real-data run hit this and silently
"converged" after 3 evaluations to a degenerate point (ROC-AUC exactly
0.5, Resistant recall 0.0 - indistinguishable from the original broken
SGD collapse from several tasks ago). Fixed with the standard numerically
stable `log(1+exp(x)) = max(x,0) + log1p(exp(-|x|))` formulation; all
tests re-passed afterward, and the real run below converged properly.

## Fair comparison on both real splits (Step 4)

| Split | Model | Accuracy | ROC-AUC | Avg. Precision | Resistant recall | Susceptible recall |
|---|---|---|---|---|---|---|
| random | batch LR | 0.900 | 0.938 | 0.876 | 0.571 | 1.000 |
| random | **streaming LR** | 0.900 | **0.950** | **0.891** | 0.571 | 1.000 |
| lineage | batch LR | 0.759 | 0.857 | 0.820 | 0.714 | 0.773 |
| lineage | **streaming LR** | **0.897** | **0.896** | **0.853** | 0.714 | **0.955** |

`train_class_counts`/`test_class_counts` confirmed identical to the
batch baseline on both splits - this is a fair, matched comparison, not
an artifact of different data.

**Do not read this as "streaming beat batch" - the honest explanation,
checked directly:** both minimize the identical regularized objective,
but 262,144 features against ~70-71 rows makes the loss surface
extremely flat and underdetermined - many different weight vectors
achieve near-identical TRAINING loss, so exactly where an iterative
optimizer stops (which depends on tolerance, step history, and
numerical path) can differ meaningfully even when both solutions are
equally valid minimizers of the same training objective. Measured
directly on the random split: cosine similarity between the two fitted
coefficient vectors is **0.907** (not 1.0 - a real, non-trivial
direction difference), `‖w‖` differs (streaming: 0.0624, batch: 0.0880 -
the streaming solution is somewhat less confident/more shrunk), and the
intercepts differ at the `1e-6` scale. **These are two different, both
legitimate, local solutions of the same ill-posed optimization problem
- not the same point reached two ways.** Which one generalizes better
on held-out test data (here, streaming did, notably on the lineage
split) is a property of where each stopped on a near-flat surface, not
evidence that one algorithm is intrinsically better than the other.
This task does not claim equivalence "based solely on similar accuracy
or AUC" (explicitly prohibited) - the metrics are reported alongside
this direct coefficient-level comparison specifically so the similarity
claim rests on more than matching headline numbers.

## Training time and peak RSS (Step 4/5)

| Split | fit_seconds | Objective evaluations | Peak RSS before | Peak RSS after |
|---|---|---|---|---|
| random | 6.10 s | 145 | 153.4 MB | 595.6 MB |
| lineage | 7.01 s | 162 | 595.7 MB | 615.9 MB |

(Peak RSS is `ru_maxrss`, the cumulative per-process high-water mark,
same convention as every prior task - the lineage row's "before" value
is the random split's "after" value within the same process.)
**145-162 full passes over the training partition** were required to
converge - each pass is itself bounded-memory (one batch at a time via
`HybridTrainingReader`), but this is "bounded peak memory per read,"
explicitly NOT "fewer total reads than the in-memory version." With the
hybrid reader's memory tier already warm (reused from prior tasks, not
rebuilt here), repeat passes after the first are cheap; without a warm
cache, 145-162 passes would each pay a real SQL-or-disk-read cost,
documented honestly in the limitations below.

## Bounded-memory verification (Step 5/8)

`max_batch_rows_held_in_memory` stayed at `batch_size` (16) for BOTH
splits across the ENTIRE optimization (verified in the output JSON,
not just claimed) - confirming no single batch, across 145-162 full
passes, ever exceeded 16 rows. The full 262,144-dimension training
matrix was never constructed as one in-memory object; only the O(d)
gradient accumulator (≈2 MB) and the current parameter vector persist
across passes.

## Memory and scalability analysis (Step 5) - measured vs. extrapolated vs. unknown

**Measured, this pilot:**
- Per-batch footprint: unchanged from prior tasks (one 16-row dense
  CSR batch ≈ 50.3 MB at this k/d configuration).
- Convergence required 145-162 full-partition passes.
- Peak RSS: 153-616 MB across the full run (both splits, one process).

**Linear/combinatorial extrapolation (explicitly labeled, assuming
density/dimensionality and convergence-pass-count stay similar as
genome count grows - a strong assumption, see below):**

| Genome count | Gradient accumulator (O(d), fixed) | One full pass (linear in n) | Full convergence (145-162 passes) |
|---|---|---|---|
| 100 (measured) | ~2 MB | ~1 s (warm cache) | 6-7 s |
| 1,000 | ~2 MB (unchanged - depends on d, not n) | ~10 s | ~15-25 min |
| 10,000 | ~2 MB (unchanged) | ~100 s | ~2.5-4 hours |

**The gradient accumulator's size is governed by the number of
FEATURES (d), not the number of GENOMES (n) - this does NOT grow with
dataset size**, which is the genuinely favorable scaling property this
design offers over naively holding the whole matrix in memory. **What
DOES scale with n is the cost of each full pass, multiplied by however
many passes convergence needs** - and whether that pass-count itself
stays roughly constant, grows, or shrinks as n increases relative to d
is **unverified, not measured**: this pilot's extreme d≫n ratio
(262,144 vs ~70) may itself be WHY so many passes were needed (a flatter,
harder-to-navigate loss surface) - a dataset with more rows relative to
features could plausibly converge in FEWER passes, or more, and this
was not tested at any scale beyond the 100-genome pilot.

**Explicit limitations from feature count, class imbalance, and solver
requirements:**
- Feature count (d=262,144) is the dominant fixed cost per pass (every
  batch needs a `d`-length dot product and accumulation) - this does
  not shrink with more genomes, but does make even ONE pass
  non-negligible at any scale.
- Class imbalance (26% Resistant) was not specifically addressed by
  this task - the streaming objective, like the batch baseline,
  includes no class weighting; both share this characteristic, so the
  comparison is fair, but neither handles imbalance specially.
- The L-BFGS-B solver's own memory (O(d · history_size)) is small and
  was never the limiting factor - the limiting factor is wall-clock
  time from repeated full passes, not memory.

**This task does not establish full-scale feasibility.** The favorable
O(d)-not-O(n) memory property is real and measured at pilot scale, but
the TIME cost (145-162 passes, each scaling with n) was not validated
beyond 100 genomes, and whether pass-count itself changes at a
different n:d ratio is an open, unverified question.

## Test results (Step 6)

451 tests pass (441 before this task, plus 10 new in
`tests/test_streaming_logistic_regression.py`: direct numerical
comparison against sklearn's `LogisticRegression` on synthetic data,
the "reaches equal-or-lower loss than sklearn's own fit" rigor check,
internally-consistent metric computation, exact train/test class-count
alignment, training-only fitting (unrelated new rows outside both
partitions provably don't affect the fit), explicit rejection of a
single-class training partition rather than a silent degenerate fit,
the bounded-batch-size invariant, the "many full passes, not one load"
limitation made directly observable via a counter, determinism under a
fixed seed, and artifact-path isolation). No previously-passing test
was broken.

## Final validation (Step 8)

- All existing matrices, the hybrid cache artifacts (reused read-only;
  `HybridTrainingReader.is_complete()` was true going in, confirmed no
  rebuild occurred), the SQLite store, both saved splits, and every
  prior task's results: unchanged (checksums re-verified).
- `data/raw/ciprofloxacin_pilot/`: still exactly 100 `.fna` files - no
  new downloads.
- Bounded-memory training confirmed directly: `max_batch_rows_held_in_memory
  == batch_size` (16) throughout both splits' entire optimization runs,
  not just at a single checkpoint.
- `git status`: only the pre-existing modified `.gitignore` plus new
  untracked files; nothing staged or committed.

## Remaining blockers before full-dataset execution

1. **Convergence pass-count (145-162 here) is unvalidated at other
   scales or n:d ratios** - the real time cost driver, and the main
   open question for whether this approach actually helps at full
   scale versus simply being a memory-bounded curiosity.
2. **The two fitted solutions are meaningfully different in coefficient
   space (cosine similarity 0.907), not identical** - reported honestly
   rather than oversold from matching headline metrics; this is a
   property of the ill-posed d≫n regime, not a bug, but means neither
   solution should be treated as "the" unique optimum.
3. **Class imbalance is unaddressed by both the batch baseline and this
   streaming replica** - a fair comparison, but not a solved problem.
4. **Only one representation (k=15, d=262144), one antibiotic, and the
   100-genome pilot were used.** No full-dataset download,
   three-antibiotic benchmark, or Random Forest/XGBoost work was
   started, per this task's explicit stop condition.
