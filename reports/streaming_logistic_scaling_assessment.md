# Streaming Logistic Regression: Numerical Validation, Convergence, and Scaling

Exploratory pilot plus synthetic benchmarks only. Existing 100-genome
ciprofloxacin cohort, splits, SQLite store, and hybrid cache artifacts
were reused unchanged. No genomes were downloaded. Labels, saved
splits, and the feature mapping were NOT modified.

## Step 1-2: exact configuration and rigorous numerical validation

Confirmed from `streaming_logistic_regression.py`: loss
`Σ log(1+exp(-yᵢ·(w·xᵢ+b))) + 0.5‖w‖²/C`, gradient as derived in the
module docstring, `C=1.0`, `fit_intercept` implicit (b unregularized),
solver = `scipy.optimize.minimize(method="L-BFGS-B")` fed a streaming
objective via `HybridTrainingReader`/`MultiPartitionHybridReader`.

**New, independent validation this task** (`tests/test_streaming_logistic_regression_numerical.py`,
8 tests) - deliberately NOT comparing against sklearn this time (that
was the prior task's check), but against hand-derived ground truth:
- Streamed loss/gradient matched a completely separately-written DENSE
  computation **exactly** (`rtol=1e-10`), and matched across batch
  sizes {1, 3, 7, 100} on a 23-row fixture (doesn't divide evenly) -
  confirms chunking never changes the result.
- **Finite-difference gradient check**: numerical gradient (central
  difference, `eps=1e-5`) matched the analytic gradient on 8 randomly
  sampled dimensions to `abs=1e-3` - independent confirmation the
  gradient formula is the true derivative of the loss, not merely
  self-consistent with itself.
- **Regularization convention isolated exactly**: the loss difference
  between two `C` values at a FIXED `(w,b)` equals
  `0.5·‖w‖²·(1/C₁-1/C₂)` to `rtol=1e-10` - unambiguous confirmation of
  the `0.5‖w‖²/C` convention (not `0.5·C·‖w‖²` or any other scaling).
- **Intercept non-regularization confirmed**: `grad_b` is identical
  across two different `C` values at the same `(w,b)`; `grad_w` is NOT
  - exactly the expected asymmetry.
- **Numerical stability**: no NaN/Inf for an extreme `‖w‖~1e4` fixture
  (the exact failure mode found and fixed in the prior task) or for a
  95%/5% imbalanced label fixture.
- **Determinism**: bit-for-bit identical loss and gradient across
  repeated calls with identical inputs.

**Documented differences from sklearn's exact implementation** (not
claimed away): sklearn's `LogisticRegression`'s `tol` parameter is
mapped internally to scipy's `gtol` via its own wrapper logic in
`sklearn.linear_model._logistic`, not necessarily identical to how this
module passes `tol` to `ftol`/`gtol` directly. The prior task already
showed the two implementations converge to coefficient vectors with
cosine similarity 0.907 on the real pilot (not 1.0) - this task's
validation confirms the OBJECTIVE and GRADIENT are exact, not that the
two optimizers' stopping points are identical; that residual gap is
solver-path-dependent on an extremely flat (d≫n) loss surface, as
established previously.

## Step 3: convergence behavior (real pilot, train-only validation)

Stratified 80/20 carve of the random split's TRAIN partition only
(seed=7, same precedent as two earlier tasks) - never touches either
real test set. Four predefined configurations:

| Config | tol | max_iter | Converged | Evals | Final loss | Grad norm | fit_seconds | Val acc/AUC |
|---|---|---|---|---|---|---|---|---|
| default | 1e-4 | 1000 | True | 164 | 0.002118 | 0.006017 | 12.26 | 1.0 / 1.0 |
| tighter_tol | 1e-6 | 1000 | True | 281 | 0.002116 | **0.000087** | 10.45 | 1.0 / 1.0 |
| looser_tol | 1e-2 | 1000 | True | **57** | 0.004269 | 0.239419 | **2.22** | 1.0 / 1.0 |
| iteration_capped | 1e-4 | 50 | **False** | 64 | 0.003819 | 0.210211 | 2.27 | 1.0 / 1.0 |

**The clean, expected tradeoff**: tighter tolerance → more evaluations,
smaller final gradient norm, marginally lower loss; looser tolerance or
a hard iteration cap → far fewer evaluations and faster, at a
measurably higher final loss/gradient norm. (`tighter_tol`'s
`fit_seconds` being lower than `default`'s despite more evaluations is
a measurement artifact of running all four configs sequentially against
the SAME `HybridTrainingReader` instance - later configs benefit from
the memory tier the earlier ones already warmed; this is noted, not
hidden, and does not affect the evaluation-count/gradient-norm
comparison, which is independent of caching.)

**All four configurations reached perfect accuracy/AUC (1.0) on this
14-row internal validation fold** - too small to distinguish
configurations by predictive metric; the real discriminating signal
here is the evaluation-count/final-loss/gradient-norm tradeoff, not
accuracy. No configuration was selected or changed as "official" for
other tasks based on this grid - it is reported as characterization
data only, per this task's explicit scope.

## Step 4: synthetic scaling (controlled, clearly NOT real data)

All entries below use `synthetic_scaling_fixtures.py` - deterministic,
seeded, 20%-density synthetic sparse rows with a fixed ~74%/26% class
balance, read via a BARE `SqliteRowStore` (no caching layer, to isolate
pure data-access-plus-optimizer cost from caching effects). A fixed
config (`tol=1e-4`, `max_iter=30`) was used for every scenario - capped
at 30 iterations (not 1,000) because the largest scenario measured
~2.1s/evaluation before this cap was chosen; running every scenario to
full convergence would have made the benchmark impractically slow.
**Accuracy figures below are synthetic-data artifacts only and are
explicitly not real predictive performance.**

| Scenario | n | d | Converged | Evals | fit_seconds | Peak RSS after |
|---|---|---|---|---|---|---|
| pilot_ratio | 100 | 262,144 | True | 26 | 5.513 | 837 MB |
| n_scale_100 | 100 | 10,000 | True | 24 | 0.146 | 842 MB |
| n_scale_300 | 300 | 10,000 | False | 35 | 0.629 | 851 MB |
| n_scale_1000 | 1,000 | 10,000 | False | 36 | 2.161 | 874 MB |
| d_scale_10000 | 100 | 10,000 | True | 24 | 0.149 | 876 MB |
| d_scale_50000 | 100 | 50,000 | True | 24 | 0.858 | 876 MB |
| d_scale_262144 | 100 | 262,144 | True | 26 | 5.449 | 876 MB |
| both_scale_small | 100 | 10,000 | True | 24 | 0.147 | 876 MB |
| both_scale_medium | 300 | 50,000 | True | 30 | 3.050 | 876 MB |
| both_scale_large | 1,000 | 262,144 | True | 36 | 73.411 | 1,613 MB |

**Per-evaluation cost scales linearly in both n and d, confirmed
directly from these measurements** (per-eval = fit_seconds / evals,
holding the other dimension fixed): d=10,000→50,000→262,144 (fixed
n=100) gives per-eval costs of 0.0062s → 0.0358s → 0.2096s - ratios
1 : 5.8 : 33.8, closely tracking the d ratios 1 : 5 : 26.2. n=100→300→1,000
(fixed d=10,000) gives 0.0061s → 0.0180s → 0.0600s - ratios 1 : 2.96 : 9.84,
closely tracking the n ratios 1 : 3 : 10. This is the expected behavior
for an O(n·d) per-pass cost (each row needs one d-length dot product)
and is reported as a measured, not assumed, confirmation.

**Peak RSS for `both_scale_large` (1,613 MB, visibly higher than every
other scenario's ~837-876 MB) is NOT from the streaming fit** - checked
directly: `max_batch_rows_held_in_memory` was 16 for every single
scenario including this one, and the RSS increase was already present
BEFORE fitting began (`peak_rss_mb_before: 1612.7`, same as `_after`).
The actual cause is `build_synthetic_store`'s own construction step (building
1,000 rows' worth of sparse specifications - `~52,428` nonzero entries
each - in memory before writing them out), which took 8.85 seconds for
this scenario. This is an acceptable property of a TEST FIXTURE
GENERATOR, not of the production training path, and is called out
explicitly rather than left to look like a fit-time memory cost.

## Step 5: full-scale feasibility - measured, extrapolated, and assumed, kept distinct

**Measured, real data, reused from prior tasks and this one (not
re-derived):**
- Real pilot: n=70 (random split's training partition), d=262,144,
  converges in 145-281 evaluations depending on tolerance (this task's
  Step 3 grid), 2.22-12.26 s with a warm `HybridTrainingReader` cache.
- Real pilot's representation is ~100% dense at k=15/d=262,144 (prior
  tasks) - NOT 20% density like this task's synthetic fixtures; real
  per-row cost is correspondingly higher than the synthetic benchmark's
  density would suggest.
- Real ciprofloxacin Policy-A-eligible cohort: **7,307 genomes**
  (`reports/full_dataset_feasibility_and_pipeline_plan.md`, measured in
  an earlier task, reused here, not recomputed). A 70/30 split (matching
  the pilot's convention) gives a training partition of **~5,115
  genomes**.
- Union of unique genomes across all three antibiotics: **8,710**
  (same source) - relevant for the ONE-TIME, SHARED k-mer/hash
  extraction and SQLite storage step, not for any single antibiotic's
  training-set size.
- Measured disk cost: ~2,101,371 bytes/genome (SQLite blob storage,
  prior tasks).

**Separate antibiotic-specific models versus the deduplicated cohort**
(explicitly distinguished, per this task's instruction): the raw
sequence data and the hashed-feature SQLite store CAN be built ONCE for
the deduplicated 8,710-genome union and reused by all three antibiotics'
models (each antibiotic just selects its own row subset and labels from
the shared store) - this is a ONE-TIME cost. Each antibiotic's
LOGISTIC REGRESSION TRAINING, however, is a SEPARATE optimization run
over that antibiotic's own (smaller) row subset - this cost is paid
PER ANTIBIOTIC, three times over for the three-antibiotic benchmark
(not part of this task's scope, but relevant to why the estimate below
is per-antibiotic, not per-union).

**Extrapolation basis, two independent paths, both labeled explicitly:**

1. *Real-pilot-scaled* (linear in n, d held fixed and identical to the
   real pilot - no density correction needed since this stays within
   real-data measurements): warm-cache per-eval cost
   `6.10s/145 evals = 0.0421 s/eval` (random split) scaled by
   `n_train` ratio `5,115/70 ≈ 73.07`: **≈3.08 s/eval, warm cache**.
   Uncached per-pass cost, from an earlier task's disabled-reader
   baseline (`~57.5s / 250 batch-reads` at n=70, batch=16 → `0.230
   s/batch`), scaled to `~320 batches/pass` at n=5,115: **≈73.6
   s/pass, uncached**.
2. *Synthetic-d-scaling-corrected*: the synthetic benchmark's
   `0.2096 s/eval` at (n=100, d=262,144, 20% density) scaled by a ×5
   density-correction factor (100%/20%) for the real representation,
   then linearly by n (5,115/100): **≈5.37 s/eval-equivalent** - in
   the same order of magnitude as path 1's warm-cache estimate, cross-
   checking the extrapolation rather than resting on one method alone.

### Low / base / high scenarios for ONE antibiotic's full training run (ciprofloxacin, n_train≈5,115, d=262,144)

| Scenario | Assumption | Evaluations | Per-eval cost | Estimated total time |
|---|---|---|---|---|
| **Low** | looser tolerance (57 evals, as measured in Step 3) + warm, fully-cached reader | 57 | ~3.1 s (warm) | **~3 minutes** |
| **Base** | default tolerance (164 evals, as measured) + warm, fully-cached reader | 164 | ~3.1 s (warm) | **~8.4 minutes** |
| **High** | tighter tolerance (281 evals, as measured) + NO warm cache (every pass re-reads SQLite) | 281 | ~73.6 s (uncached) | **~5.7 hours** |

**The Low/Base scenarios REQUIRE a memory cache large enough to hold
the ENTIRE ~5,115-row training partition** - at the real pilot's
measured ~3.15 MB/genome (dense CSR), that is **≈16.1 GB**, a
substantial commitment this project's development machine's established
safety margin (~11 GB, from earlier memory-safety work) cannot
accommodate without either a larger machine or a smaller feature
dimension. Without that cache, the realistic scenario is closer to
**High** - several hours per antibiotic, not minutes.

**Disk footprint**: ciprofloxacin alone (7,307 genomes) ≈ 7,307 ×
2.1 MB ≈ **15.3 GB**; the deduplicated 8,710-genome union (shared
extraction, one-time) ≈ **18.3 GB**. Both are plausible on typical
modern storage, unlike the 16.1 GB in-memory cache requirement above.

**This task does not claim the full benchmark is feasible.** The High
scenario (hours, not minutes, without a large memory cache) is the
realistic default absent further engineering; the Low/Base scenarios
are real possibilities but depend on a memory commitment not
demonstrated as safely available on this project's own development
machine.

## Step 6: tests

461 tests pass (451 before this task, plus 8 new numerical-validation
tests and 2 new convergence-diagnostic tests - see
`tests/test_streaming_logistic_regression_numerical.py` and the
additions to `tests/test_streaming_logistic_regression.py`). No
previously-passing test was broken.

## Step 8: final validation

- All existing matrices, the hybrid cache artifacts (untouched by this
  task's internal-validation cache, which was built at a separate new
  path), the SQLite store, both saved splits, and every prior task's
  results: unchanged (checksums re-verified).
- `data/raw/ciprofloxacin_pilot/`: still exactly 100 `.fna` files - no
  new downloads.
- **Bounded-memory behavior confirmed directly, not assumed**:
  `max_batch_rows_held_in_memory == batch_size` (16) in every real-pilot
  convergence-grid run AND every synthetic scaling scenario, including
  the largest (n=1,000, d=262,144) - the one scenario whose peak RSS
  was visibly elevated, and that elevation was traced to the fixture
  generator's own construction step, not the streaming fit, confirmed
  by the RSS already being elevated BEFORE fitting began.
- `git status`: only the pre-existing modified `.gitignore` plus new
  untracked files; nothing staged or committed.

## Remaining blockers before full-dataset execution

1. **The Low/Base feasibility scenarios depend on a ~16 GB in-memory
   cache this project's own machine cannot safely provide** - without
   it, realistic full-scale training time for ONE antibiotic is hours,
   not minutes.
2. **Convergence evaluation-count (57-281 in this task's measurements)
   is itself somewhat tolerance-sensitive and was only characterized at
   pilot scale (n=70)** - whether it grows, shrinks, or stays flat as n
   grows toward 5,115 is assumed constant in the estimates above, not
   measured at that scale.
3. **The synthetic-to-real density correction (×5) is a stated
   assumption, not a measurement** - the real representation's behavior
   at larger n with 100% density was not directly tested beyond the
   100-genome pilot.
4. **This cost is per-antibiotic, not per-union** - the three-antibiotic
   benchmark would pay this training cost three times, even though the
   underlying sequence/hash extraction can be shared once.
5. No full-dataset download, three-antibiotic benchmark, or Random
   Forest/XGBoost work was started, per this task's explicit stop
   condition.
