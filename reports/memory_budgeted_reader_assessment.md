# Memory-Budgeted Reader: Design, Benchmark, and an Honest Surprise

Exploratory pilot only. Existing 100-genome ciprofloxacin cohort,
splits, and SQLite store were reused unchanged. No genomes were
downloaded. No existing artifact overwritten.

## Why 250 MB worked on the pilot but isn't a real design (Step 1)

The prior task's 250 MB cache worked because it happened to exceed
this pilot's random-split training partition's exact total size
(measured precisely this task: **220,201,260 bytes**, 5 batches - four
16-row batches at 50,331,716 bytes each plus one 6-row batch). That
cache never evicted anything - once a batch didn't fit, it was
permanently skipped for the rest of the run. It worked for this pilot
by coincidence of sizing, not by design: there was no eviction policy,
no distinction between "cache budget" and "total reader memory," and
no mechanism that would behave sensibly if the training partition
didn't fit the chosen budget.

## What was implemted (Step 2)

`src/bvbrc_recon/cached_row_reader.py`, redesigned:

- **Deterministic LRU eviction** (`collections.OrderedDict`): a cache
  hit moves that entry to most-recently-used; eviction always removes
  the least-recently-used entry. Reproducible given the same read
  sequence (tested).
- **Two distinct, independently configurable budgets** (not conflated):
  `max_cache_bytes` bounds what may be *retained* between calls;
  `max_reader_memory_bytes` bounds `cache_bytes + the batch currently
  being returned`, checked and enforced via eviction *before* that
  batch is handed back, regardless of whether it ends up cached.
  Defaults equal when not given separately (reproduces simple
  single-budget behavior).
- **`max_batch_size`**: a safety ceiling on the effective batch size
  used per call, independent of the caller's requested `batch_size`.
- **Counters**: `n_hits`, `n_misses`, `n_evictions`, `n_fallbacks`
  (misses correctly served via SQLite but not cached - oversized
  relative to budget), plus live/peak occupancy (`n_bytes_cached`,
  `peak_bytes_cached`).
- Row reconstruction (vectorized, from the prior task) and
  `evaluate_streaming_hashed_matrix` are both unchanged.

## The honest finding this task surfaced (Steps 3-4)

Benchmarking across budgets {disabled, 16 MB, 64 MB, 128 MB, 256 MB}
on the real pilot (random split, 50 epochs, same settings as the prior
task) revealed something the prior task's simpler, non-evicting cache
did not:

| Budget | End-to-end 50-epoch median | Cache hits (of 252 reads) | Evictions | Within configured cache budget |
|---|---|---|---|---|
| disabled (no wrapper) | 56.61 s | n/a | n/a | n/a |
| 16 MB | 57.26 s | **0** | 0 | yes |
| 64 MB | 57.34 s | **0** | 251 | yes |
| 128 MB | 57.40 s | **0** | 249 | yes |
| 256 MB | 4.86 s | 245 | 2 | yes |

**16 MB, 64 MB, and 128 MB all deliver ZERO hits** despite 64 MB and
128 MB actively caching (1 and 3 of the 5 batches respectively,
confirmed by `n_batches_cached`/`peak_bytes_cached` in the raw JSON) -
and despite correctly respecting their configured byte budgets the
entire time. The reason: each training epoch scans the SAME 5 distinct
batches in the SAME order. When the cache can't hold all 5
simultaneously, inserting batch *k* evicts an earlier batch in the
*same* pass (LRU evicts whatever was used longest ago, which by the
time the scan reaches batch 5 is batch 1 or 2 - the very batches the
*next* epoch will ask for first). By the time any given batch is
requested again, LRU has always evicted it in the interim. This is the
well-known "sequential/cyclic scan defeats LRU" pathology: **partial
LRU caching of a repeated full-partition scan provides literally no
benefit until the budget covers the ENTIRE per-epoch working set** - at
which point (256 MB here, enough for all 220.2 MB) the benefit is
dramatic (245/252 hits, 56.6s -> 4.86s, an **11.6x** speedup).

**This is a real regression relative to the prior task's simpler cache**,
which never evicted - it cached whatever fit on the first attempt and
left it cached forever, giving a modest but real, stable partial benefit
(the prior task measured ~19-21% improvement at 64 MB). The new LRU
design satisfies this task's explicit "evict deterministically"
requirement, but for this project's specific access pattern (full
repeated scans per epoch), eviction *under*-performs no-eviction at
partial budgets. This is reported as a genuine, measured, non-obvious
finding, not glossed over - see "Remaining blockers" below.

**Within-budget confirmation:** every run's `n_bytes_cached` stayed at
or under its configured `max_cache_bytes` (explicitly checked, see
`within_cache_budget: true` in every entry of the benchmark JSON) - the
budget was never violated, even while delivering zero benefit.

**Peak RSS vs. cache budget - reported separately, not conflated:** all
five configurations show the SAME process peak RSS, **1,543 MB**
(`ru_maxrss`, a cumulative high-water mark across this one script's
entire run, same convention as every prior task). This number has
nothing to do with the cache budget (16 MB through 256 MB) - it reflects
sklearn/pandas/sqlite3/numpy import and working overhead, as already
established in the prior task. **The cache budget bounds only the
cache's own bytes; it does not bound, and should never be assumed to
approximate, total process memory.**

**Exact equivalence (Step 4):** compared against the direct,
uncached `SqliteRowStore` across all 100 real pilot genomes - genome
IDs, row order, sparse values, and normalized rows all identical
(`n_mismatches: 0`). The lineage split's spot-check (disabled vs. 256 MB)
also produced bit-for-bit identical accuracy and ROC-AUC. Budget
enforcement never dropped a row or a feature in any configuration -
every run processed all requested rows correctly, including the
zero-hit 16/64/128 MB runs (confirmed by `max_batch_rows_held_in_memory`
staying at `batch_size` throughout, and by the lineage spot-check's
metric equality).

Interestingly, even the FULL 100-genome equivalence check (not just the
70-row training partition) with a 256 MB budget showed **0 cache hits
on its second pass** - because all 100 genomes together need
~314 MB (100/16 batches x ~50.3 MB), which exceeds 256 MB. This is the
exact same thrashing phenomenon, confirming it generalizes beyond the
one split tested in the main benchmark.

## Full-scale feasibility estimate (Step 5)

**Measured, not extrapolated:** at k=15/n_buckets=262144 (100% dense at
this pilot), one genome's row costs **2,101,371 bytes on disk** (SQLite
blob storage: 210,137,088 bytes / 100 genomes, exact) and **3,145,732
bytes in reconstructed CSR form** (50,331,716 bytes / 16 rows per dense
batch, exact: 8 bytes/entry `data` + 4 bytes/entry `indices`).

**Linear extrapolation (explicitly labeled as such, not a measurement)**
from these exact per-genome costs, assuming density/dimensionality stay
unchanged as genome count grows (an assumption, not verified - see
below):

| Genome count | Disk (SQLite) | Full in-memory CSR (if it had to fit at once) |
|---|---|---|
| 100 (measured) | 210 MB | 315 MB |
| 1,000 | ~2.1 GB | ~3.1 GB |
| 10,000 | ~21 GB | ~31 GB |

**The practical implication of this task's own finding is sharper than
a smooth size estimate:** because partial-budget caching delivered
*zero* benefit on this pilot's repeated full-scan access pattern, the
same would be expected to hold at full scale - a cache budget smaller
than the FULL per-epoch working set (which the table above says could
be several GB to tens of GB) would likely provide no speedup at all,
only the overhead of bookkeeping. Caching at full scale would need a
budget sized to the ENTIRE training partition to help at all, which may
not be practical.

**What is genuinely unknown and would require larger-scale validation,
not assumed:**
- Whether SQL query performance (fetch, `WHERE row_index IN (...)`)
  degrades nonlinearly past a few hundred rows - only measured at
  70-100.
- Whether disk I/O becomes a bottleneck once the database no longer
  fits in OS page cache (210 MB fits easily in RAM here; a multi-GB
  database at full scale likely would not).
- Whether density stays at 100% as genome diversity grows - this
  project's earlier resource-estimation work (prior tasks, not this
  one) already documented saturation/collision effects for raw k-mer
  vocabularies at various k; whether the HASHED representation's
  density changes with genome count at fixed (k, n_buckets) was not
  re-examined here.
- Available disk space on any real full-scale run has not been checked
  against the ~2.1 GB/1,000-genome estimate above - this must be
  verified before attempting, not assumed sufficient.

**This reader optimization does not, by itself, make the full
benchmark feasible.** It fixes a bounded-memory read/cache mechanism;
it does not address: the feature representation's own scaling behavior
(dense hashed counts, already a documented concern from earlier tasks),
or the incremental classifier's own predictive-quality limitations
(ROC-AUC well below the batch baseline, zero Resistant recall on both
splits - established and NOT touched in this task, per its explicit
scope).

## Test results (Step 6)

399 tests pass (392 before this task, plus 7 new: strict-LRU eviction
order, eviction-order reproducibility across runs, oversized-batch
correctness without caching, the `max_cache_bytes` vs.
`max_reader_memory_bytes` distinction, the default-equality behavior
when the second budget isn't given, cold-start-after-reopening-the-database,
and artifact-path isolation). No previously-passing test was broken.

## Final validation (Step 8)

- All existing raw/hashed/minimizer `.npz` artifacts, both splits, the
  original batch-LR result, and every prior task's JSONL/SQLite/
  benchmark artifacts: unchanged (checksums re-verified).
- `data/raw/ciprofloxacin_pilot/`: still exactly 100 `.fna` files - no
  new downloads.
- The reader does not materialize the full matrix: confirmed by
  `max_batch_rows_held_in_memory` staying at `batch_size` (16) in every
  run, including the full-dataset 100-genome equivalence check, and by
  the cache's own occupancy never exceeding its configured budget in
  any configuration tested.
- `git status`: only the pre-existing modified `.gitignore` plus new
  untracked files; nothing staged or committed.

## Remaining blockers before full-dataset execution

1. **The eviction policy this task was required to add (deterministic
   LRU) measurably underperforms the prior task's simpler no-eviction
   cache for this project's actual access pattern** (repeated full-
   partition scans per epoch) at any budget below the full working-set
   size. A future task should reconsider: either revert to a
   no-eviction "cache what fits once" policy for this specific
   workload, or accept that caching here is effectively all-or-nothing
   and size budgets accordingly - partial budgets are not a safe
   middle ground.
2. **A full-scale cache budget sized to the entire training partition
   could itself require several GB to tens of GB**, reintroducing the
   memory pressure this design was meant to avoid - genuinely untested
   beyond pilot scale.
3. **SQL/disk-I/O behavior at thousands of rows, and whether density
   stays at 100% as genome count grows, are both unknown** and were not
   validated here - only linearly extrapolated.
4. **The incremental classifier's predictive-quality limitations are
   unresolved and out of this task's scope** - a faster, better-cached
   reader does not make the model itself competitive with the batch
   baseline.
5. No full-dataset download, three-antibiotic benchmark, or Random
   Forest/XGBoost work was started, per this task's explicit stop
   condition.
