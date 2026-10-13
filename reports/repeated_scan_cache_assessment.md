# Repeated-Scan-Aware Cache Strategies: A Fair, Measured Comparison

Exploratory pilot only. Existing 100-genome ciprofloxacin cohort,
splits, and SQLite store were reused unchanged. No genomes were
downloaded. Model, labels, splits, and feature mapping were NOT
changed - every strategy below produced bit-for-bit identical
accuracy/ROC-AUC/confusion-matrix on the same model configuration.

## Access pattern confirmed (Step 1)

`evaluate_streaming_hashed_matrix` calls `read_rows_for_indices(train_row_indices,
batch_size)` once per epoch, 50 times, with the IDENTICAL argument every
time, then calls it once more for the test partition. For the random
split's 70-row training partition (batch_size=16): 5 fixed batches,
scanned in the same order, every epoch. This is why the prior task's
LRU cache thrashed to zero hits at any budget below the full 220.2 MB
working set - LRU always evicts the batch about to be needed next.
Profiling confirmed (prior task, unchanged by this one): SQL fetch is
cheap (~0.11s/pass); vectorized reconstruction dominates (~1.03s/pass);
normalization is negligible (~0.0076s/batch).

## Four strategies implemented and compared (Step 2)

1. **`disabled`** - direct `SqliteRowStore`, no wrapper.
2. **`CachedRowReader`** (LRU, prior task) - deterministic eviction,
   thrashes on this access pattern below the full working-set size.
3. **`NonEvictingCachedRowReader`** (new, this task) - caches whatever
   fits on first use, NEVER evicts, falls back to SQLite permanently for
   anything that doesn't fit. Reinstates the very first (pre-LRU)
   design, now alongside LRU as an explicit, selectable alternative, not
   a silent regression.
4. **`PersistentBatchCache`/`MultiPartitionPersistentReader`** (new,
   this task) - materializes a FIXED partition's batches to disk once
   (`scipy.sparse.save_npz`, reusing this project's existing tested
   serialization), re-reads from disk on every subsequent call - zero
   SQL queries and zero raw-count reconstruction after the first build,
   but each read still pays disk I/O + zip-decompression cost.

All four share identical counters (`n_hits`/`n_misses`/`n_evictions`/
`n_fallbacks` for 2 and 3; `n_rebuilds`/`n_batches_read_from_artifact`
for 4) and the same two-budget semantics (`max_cache_bytes` vs.
`max_reader_memory_bytes`, not conflated) carried over from the prior
task. **Cache invalidation and replacement rules, by strategy:**
- LRU: evicts least-recently-used on overflow; invalidated only by
  `clear_cache()` (rows are immutable once written, so staleness cannot
  occur otherwise).
- Non-evicting: never evicts; a key that didn't fit once will never be
  retried for the life of the reader (deterministic, permanent fallback).
- Persistent: invalidated by comparing a stored fingerprint (source DB
  path/size/mtime, exact row-index tuple, batch size, `n_buckets`,
  `normalize_rows`) against a freshly computed one on every call -
  mechanically enforced (tested), not just documented.

## The result: do not assume sophistication wins (Step 2 instruction, honored)

| Strategy | Budget | Median 50-epoch seconds | Hits (of up to 252 reads) |
|---|---|---|---|
| disabled | n/a | 57.49 | n/a |
| LRU | 64 MB (insufficient) | 58.07 | **0** |
| LRU | 256 MB (sufficient) | 4.90 | 245 |
| non-evicting | 64 MB (insufficient) | **45.90** | 49 |
| non-evicting | 256 MB (sufficient) | 4.91 | 245 |
| persistent artifact | n/a (disk-backed) | cold 10.66 / warm 11.31 | n/a (always re-reads disk) |

**At a sufficient budget, LRU and non-evicting are statistically the
same** (4.90s vs. 4.91s) - eviction has nothing to do once everything
fits, as expected. **At an insufficient budget, non-evicting clearly
wins** (45.90s vs. 58.07s, confirming the prior task's finding that LRU
thrashing is a real regression for this workload) by giving up on the
batches it can't fit rather than repeatedly evicting and re-fetching
them.

**The persistent disk-artifact strategy did NOT beat a sufficiently-
sized in-memory cache, and this was measured, not assumed:** 11.31s
warm vs. 4.90-4.91s for LRU/non-evicting at 256 MB. The reason:
`scipy.sparse.load_npz` pays real per-call disk I/O and zip-decompression
overhead on every one of the ~350 batch loads across 50 epochs (7
batches x 50 epochs, train+test) - comparable to, not cheaper than, the
SQL-query-plus-vectorized-reconstruction cost it was meant to avoid. It
DID clearly beat the insufficient-budget in-memory cases (11.31s vs.
45.90-58.07s) and the ONE-TIME cold-build cost (10.66s) is cleanly
separated from the warm, repeated cost (11.31s median) as required -
notably, warm is barely faster than cold here, because nearly ALL of the
warm cost is the same per-call disk I/O the cold build also pays; there
is no "in-memory speedup" for this strategy by design.

**The persistent artifact's genuine advantage is disk compactness, not
speed:** 59,908,693 bytes (~59.9 MB) for both train and test partitions
combined - dramatically smaller than the 220.2 MB equivalent in-memory
CSR footprint or the ~210 MB SQLite blob storage for the same rows.
This is because `save_npz`'s zip compression is very effective on this
specific data: at 100% density, every row's `indices` array is the
identical sequence `[0, 1, ..., 262143]` repeated, which compresses
extremely well. **This strategy trades speed for both disk footprint and
zero in-memory cache growth** - a real, different tradeoff from 2 and 3,
not a strictly worse option.

**Peak RSS was identical across all six configurations: 1,180.25 MB**
(`ru_maxrss`, the same cumulative-high-water-mark convention as every
prior task in this project) - confirming yet again that cache/budget
choice and total process memory are unrelated and must be reported
separately, not conflated.

## Equivalence and correctness (Step 4)

Compared against the direct, uncached `SqliteRowStore` for the random
split's full training partition: **0 mismatches** across LRU,
non-evicting, and persistent-artifact batches (genome IDs, row order,
sparse values all exact). All six end-to-end configurations produced
**bit-for-bit identical** accuracy (0.7667) and ROC-AUC (0.7453) on the
random split. The lineage-split spot-check (disabled vs. persistent
artifact) also matched exactly
(`metrics_match: true`, `data/processed/repeated_scan_cache_benchmark_ciprofloxacin_k15_d262144.json`).
No strategy ever dropped a row or a feature - `max_batch_rows_held_in_memory`
stayed at `batch_size` (16) in every single run.

## Scalability assessment (Step 5) - distinguishing measured, extrapolated, and unknown

**Measured, this pilot (100 genomes, k=15, n_buckets=262144, 100%
dense):**
- In-memory CSR footprint: 3,145,732 bytes/genome (established, prior task).
- Persistent npz artifact: ~599,087 bytes/genome compressed (59,908,693 / 100).
- The full per-epoch working set (220.2 MB for 70 training genomes) must
  fit ENTIRELY in an in-memory cache's budget for that cache to help at
  all - partial budgets give zero benefit with LRU, modest benefit with
  non-evicting, never the full benefit.

**Linear extrapolation (explicitly labeled, not measured) assuming
density/dimensionality stay fixed as genome count grows:**

| Genome count | In-memory CSR (full working set) | Persistent npz artifact (disk) |
|---|---|---|
| 100 (measured) | 315 MB | 60 MB |
| 1,000 | ~3.1 GB | ~600 MB |
| 10,000 | ~31 GB | ~6 GB |

A cache sized to hold the FULL working set (the only budget shown to
help at all with LRU) would need multiple GB at 1,000 genomes and tens
of GB at 10,000 - likely **infeasible as an in-memory budget** on a
typical machine running this pipeline, especially alongside the ~1.2 GB
of process overhead already measured. The persistent disk artifact's
footprint scales far more favorably on paper (~6 GB disk at 10,000
genomes is plausible for a modern machine), but its SPEED at that scale
is **unverified, not extrapolated** here: whether `load_npz`'s per-call
overhead scales linearly with file count/size, or whether OS-level disk
caching changes the picture once the artifact no longer fits in page
cache, was not tested beyond this pilot's 100-genome scale.

**Unverified assumptions requiring real validation before relying on
any of this at full scale:**
- Density staying at 100% as genome diversity grows (not re-examined
  this task; flagged as a concern in the prior task too).
- SQL query and zip-decompression performance at thousands of rows -
  only measured at ~70-100.
- Whether a HYBRID strategy (in-memory cache up to whatever budget is
  actually available, falling back to persistent disk artifacts for the
  remainder) would outperform either pure strategy at full scale - a
  reasonable hypothesis given this task's results, but **not
  implemented or tested here** - speculation, not a finding.

**This task's results do not, by themselves, make the full benchmark
feasible.** They show which data-access strategy is best MATCHED to
this project's specific repeated-scan training pattern at pilot scale,
and that the best pilot-scale choice (a correctly-sized in-memory cache)
may not be available at full scale. The incremental classifier's own
predictive-quality limitations (documented in an earlier task - ROC-AUC
well below the batch baseline, zero Resistant recall) remain completely
unresolved and untouched by any of this read-path work.

## Test results (Step 6)

399 tests passed before this task (end of the prior task). This task
added 15 new tests (3 for `NonEvictingCachedRowReader`: never-evicts
behavior, stable partial hits under the exact workload that defeated
LRU, and budget/equivalence correctness; 12 for `PersistentBatchCache`/
`MultiPartitionPersistentReader`: exact equivalence, repeated-scan reuse
without rebuilding, stable ordering, strict scope enforcement raising on
mismatched partition/batch_size, fingerprint-based source invalidation
under three distinct triggers, pre-normalized-artifact correctness,
drop-in training-result equivalence, and artifact-path isolation).
**414 tests pass in total.** No previously-passing test was broken.

## Final validation (Step 8)

- All existing raw/hashed/minimizer `.npz` artifacts, both splits, the
  original batch-LR result, and every prior task's JSONL/SQLite/
  benchmark artifacts: unchanged (checksums re-verified).
- `data/raw/ciprofloxacin_pilot/`: still exactly 100 `.fna` files - no
  new downloads.
- Every strategy's cache occupancy stayed within its configured budget
  in every run (explicitly checked for LRU and non-evicting; the
  persistent artifact has no RAM budget to exceed by design - its
  footprint is bounded to one batch at a time during reads, confirmed
  by `max_batch_rows_held_in_memory` staying at `batch_size` throughout).
- `git status`: only the pre-existing modified `.gitignore` plus new
  untracked files; nothing staged or committed.

## Remaining blockers before full-dataset execution

1. **No strategy tested here is obviously right for full scale.** The
   fastest pilot-scale option (sufficiently-sized in-memory cache) may
   need an infeasible memory budget at 1,000+ genomes; the most
   disk-efficient option (persistent artifact) is the slowest at pilot
   scale and its scaling behavior is unverified.
2. **A hybrid strategy is a reasonable next step but was not
   implemented or tested** - this task measured four existing
   approaches, as scoped, rather than designing a fifth.
3. **Density/dimensionality stability at larger genome counts remains
   unverified**, inherited from the prior task's same caveat.
4. **The incremental classifier's predictive-quality limitations are
   completely unresolved and out of every read-path task's scope so
   far** - no amount of read-path optimization addresses this.
5. No full-dataset download, three-antibiotic benchmark, or Random
   Forest/XGBoost work was started, per this task's explicit stop
   condition.
