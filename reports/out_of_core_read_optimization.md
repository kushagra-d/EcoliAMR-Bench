# Out-of-Core Read-Path Optimization

Exploratory pilot only. Existing 100-genome ciprofloxacin cohort, splits,
and the existing SQLite-backed hashed matrix were reused unchanged. No
genomes were downloaded. No existing artifact was overwritten. Model
behavior (feature representation, hash mapping, row normalization,
classifier configuration, split assignments) was NOT changed - this
task is a pure read-path performance fix, verified by exact-equivalence
checks below.

## Identified bottleneck (Step 1-2)

Two distinct costs, profiled separately on the real pilot (k=15,
n_buckets=262144, 100% dense rows at this configuration):

1. **Python-object reconstruction.** `SqliteRowStore._rows_to_csr`
   (prior task) converted each row's `cols`/`vals` numpy arrays to
   Python lists via `.tolist()`, then built COO coordinate lists with
   `list.extend`. At 262,144 entries/row this means ~4.2M individual
   Python `int` objects per 16-row batch. Profiled: reconstructing one
   full pass over the 70-row training partition (5 batches) took
   **3.05s** this way.
2. **Repeated SQL reads across epochs.** `evaluate_streaming_hashed_matrix`
   re-reads the entire training partition from SQLite on every epoch.
   At 50 epochs, the training workflow's prior end-to-end time (155.9s,
   measured in the previous task) is consistent with 50 x ~3.05s
   reconstruction + fetch cost.

SQL fetch itself is cheap (0.11s for the full training partition) -
the cost is almost entirely reconstruction, repeated unnecessarily.

## Optimization implemented (Step 3)

1. **`SqliteRowStore._rows_to_csr`, fixed in place** (same class, same
   public API, same on-disk schema): replaced Python-list construction
   with `np.concatenate`/`np.repeat` - no Python-level int objects, ever.
   The original implementation is kept as `_rows_to_csr_naive`, used only
   as the equivalence oracle and the "before" benchmark baseline, never
   called by production code.
2. **`src/bvbrc_recon/cached_row_reader.py` (new module): `CachedRowReader`.**
   Wraps a `SqliteRowStore`, caching each read batch (keyed by its exact
   `row_index` tuple) up to a configurable `max_cache_bytes` (default
   64 MB). Once the budget would be exceeded, new batches are served
   from SQLite and simply not cached - no eviction, a deliberate
   simplification. `read_rows_for_indices` has the identical
   signature/return shape as `SqliteRowStore`'s, so
   `evaluate_streaming_hashed_matrix` works with it completely
   unmodified - the model code was not touched by this task.
   `iter_batches` is an uncached passthrough (a single full pass never
   repeats, so caching it has no benefit).

**Cache invalidation/consistency:** rows are append-only and immutable
once written (PRIMARY KEY `row_index` / UNIQUE `genome_id`, no UPDATE
path anywhere in this codebase), so a cached batch for a fixed
`row_index` set cannot go stale. `clear_cache()` exists for explicit
invalidation anyway, not because staleness can occur today - tested in
`tests/test_cached_row_reader.py::test_cache_remains_valid_after_unrelated_rows_are_appended`.

**SQLite fallback:** `max_cache_bytes=0` disables caching entirely and
produces identical results, just without the speedup - tested in
`test_zero_budget_cache_still_produces_correct_results`. The cache is
never required for correctness, only for speed.

## Exact equivalence (Step 4)

`verify_exact_equivalence_all_genomes` (in the benchmark script) compared
the cached reader's output against the direct store reader for **all
100 pilot genomes** - genome IDs, row ordering, sparse column indices,
feature values, and L2-normalized rows:

```
n_genomes_checked: 100, n_mismatches: 0, n_normalized_row_mismatches: 0, passed: true
```

End-to-end: `evaluate_streaming_hashed_matrix` run with the raw
`SqliteRowStore` vs. with `CachedRowReader` produced **bit-for-bit
identical** accuracy, ROC-AUC, and confusion matrices on both the random
and lineage splits (`metrics_identical: true` for both, in the benchmark
JSON). The optimization changes speed only.

## Benchmark (Step 5) - medians over repeated runs

**Read-path only** (10 repeated passes over the 70-row random-split
training partition, 3 reps, median reported):

| Variant | Median seconds (10 passes) |
|---|---|
| naive reconstruction (forced) | 31.13 |
| fast reconstruction, uncached (default `SqliteRowStore` now) | 10.97 |
| fast reconstruction + default 64 MB cache | 8.73 |

**End-to-end training** (50 epochs, same settings as the prior task, 2
reps, median reported):

| Split | Variant | Median seconds | Peak RSS |
|---|---|---|---|
| random | uncached (fast reconstruction) | 58.33 | 2,273 MB |
| random | + default 64 MB cache | 47.42 | 2,273 MB |
| lineage | uncached (fast reconstruction) | 59.34 | 2,273 MB |
| lineage | + default 64 MB cache | 46.94 | 2,273 MB |

Reconstruction alone gives a real, substantial win: 155.9s (prior task,
naive) -> 58.3s (fast, uncached) end-to-end, a **2.7x** speedup just
from fixing `_rows_to_csr`. Peak RSS is unchanged from the prior task
(~2.27 GB, dominated by sklearn/pandas/sqlite3 process overhead already
documented there) - this task did not target memory, only speed, and
did not materialize the full matrix at any point (`max_batch_rows_held_in_memory`
stayed at 16 throughout, confirmed in the benchmark JSON).

**The default 64 MB cache budget under-delivers at this configuration,
and this was diagnosed, not hidden:** this pilot's training partition
(100% dense, k=15/d=262144) needs ~147-220 MB to cache in full (5
batches of up to ~33.6 MB each). The 64 MB default can only hold 1-2 of
those 5 batches, so most reads across epochs still miss the cache and
hit SQLite - explaining why the cache's measured end-to-end benefit
(~19-21%) is far below its theoretical ceiling. A supplementary,
single-run (not median-of-reps) check with `max_cache_bytes=250,000,000`
- sized to this pilot's actual training-partition size - confirms the
ceiling: **10 read-path passes dropped to 1.05s** (vs. 10.97s uncached,
a **10.4x** speedup), and the full 50-epoch end-to-end random-split
training run dropped to **4.76s** (vs. 58.33s uncached, a **12.3x**
speedup; vs. the original 155.9s naive implementation, a **~32.8x**
speedup) - with accuracy/ROC-AUC bit-for-bit identical to every other
configuration. This is recorded in
`data/processed/read_optimization_benchmark_ciprofloxacin_k15_d262144.json`'s
`supplementary_sized_cache_budget_demonstration` field. **Practical
implication: `max_cache_bytes` should be sized relative to the expected
training-partition size for real benefit** - the 64 MB default is a
conservative, safe-by-default choice, not a universally well-tuned one.

## Test results (Step 6)

392 tests pass (378 before this task, plus 14 new: vectorized-vs-naive
exact equivalence on dense and sparse synthetic rows, empty-batch
handling, cached-vs-direct exact equivalence, cache-hit behavior on
repeated reads, stable row/feature ordering, cache byte-budget
enforcement, cache invalidation via `clear_cache()`, cache validity
after unrelated appends, zero-budget SQLite fallback, uncached
`iter_batches` passthrough, deterministic outputs across fresh
instances, and drop-in compatibility with
`evaluate_streaming_hashed_matrix`). No previously-passing test was
broken.

## Final validation (Step 8)

- All existing raw/hashed/minimizer `.npz` artifacts, `genome_order.csv`,
  `random_split.csv`, `lineage_split.csv`, `selected_cohort.csv`, the
  batch-LR reproduction, and the prior JSONL/SQLite store artifacts:
  unchanged (checksums re-verified).
- `data/raw/ciprofloxacin_pilot/`: still exactly 100 `.fna` files - no
  new downloads.
- The optimized reader does not materialize the full matrix: confirmed
  by the equivalence check's batch-by-batch comparison (never more than
  `batch_size`=16 rows per yielded batch) and by `max_batch_rows_held_in_memory`
  staying at 16 in every end-to-end benchmark run.
- `git status`: only the pre-existing modified `.gitignore` plus new
  untracked files; nothing staged or committed.

## Remaining blockers before full-dataset execution

1. **The default cache budget (64 MB) is undersized for this pilot's
   own fully-dense configuration**, let alone a larger full-dataset run.
   A real deployment would need either a larger, explicitly-justified
   budget or a different caching granularity (e.g. per-genome rather
   than per-batch) to get the full benefit demonstrated in the
   supplementary run.
2. **Caching the entire training partition only works because this
   pilot's training set is small (~70-90 rows).** At full-dataset scale,
   a sufficient cache budget would itself become a significant memory
   commitment, reintroducing exactly the tension this task's design
   (a bounded, explicit budget with graceful SQLite fallback) was meant
   to manage - this remains untested beyond pilot scale.
3. **Peak RSS (~2.27 GB) is unchanged from the prior task** and was not
   addressed here - this task targeted speed, not memory; the earlier
   task's finding that this figure is dominated by process-level
   overhead (not the cache) still stands and is unresolved.
4. **Only one representation (k=15, d=262144) and one antibiotic were
   exercised.** No full-dataset download, three-antibiotic benchmark, or
   Random Forest/XGBoost work was started, and the classifier itself was
   not touched, per this task's explicit stop condition.
