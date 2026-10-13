# Hybrid Disk-Backed, Memory-Bounded Training Data Pipeline

Exploratory pilot only. Existing 100-genome ciprofloxacin cohort,
splits, and SQLite store were reused unchanged. No genomes were
downloaded. Model, labels, splits, and feature mapping were NOT
changed - every strategy produced bit-for-bit identical metrics.

## Why repeated `load_npz` stayed slow, and what the hybrid design fixes (Step 1)

The prior task's `PersistentBatchCache` re-decompresses every batch
from its `.npz` file on EVERY read, every epoch - there is no in-memory
tier at all, so "warm" still means "decompress again." Measured again
this task: 11.38s median for 50 epochs (unchanged from the prior
task's 11.31s, confirming this cost is structural, not noise).
`NonEvictingCachedRowReader` has the opposite problem: no PERSISTENT
state - a fresh process always starts from zero, paying the full
SQL-query-plus-vectorized-reconstruction cost on its first pass, every
time a new process starts.

`src/bvbrc_recon/hybrid_training_reader.py` combines both: a
non-evicting in-memory tier (fastest, bounded by `max_memory_cache_bytes`
/ `max_reader_memory_bytes` - same two-budget semantics as
`cached_row_reader.py`, not conflated) sits in front of a persistent,
per-batch-resumable `.npz` disk tier, which sits in front of the SQLite
source of truth. A memory miss costs one decompression (no SQL); a
disk-tier miss (only on first build, or after true invalidation) costs
one SQL query + reconstruction + compression + write.

## Partial-build recovery (mechanically enforced, Step 2)

Two marker files distinguish "building" from "committed":
`build_fingerprint.json` (written before any batch write) vs.
`fingerprint.json` (written only after every expected batch is
confirmed present AND loadable). Each batch's `.npz` is written to a
staging path that still ends in `.npz` (scipy's `save_npz` silently
appends `.npz` to any name that doesn't already end with it, which
would otherwise corrupt a naive `.tmp`-suffixed staging scheme) and
`Path.rename`d into place - atomic on the same filesystem.
`is_complete()` re-validates every batch by actually loading it
(catches corruption, not just absence) - this is deliberately expensive
(O(batches) real disk reads), so it is memoized per reader instance
(`_verified_complete`) after the first successful check, otherwise a
50-epoch loop would re-decompress every batch every epoch just to
confirm "yes, still complete" (a real bug found and fixed during this
task's own development - see below).

**Tested directly, not just documented** (`tests/test_hybrid_training_reader.py`):
a true cross-process kill (`tests/_hybrid_interrupt_subprocess.py`,
same `os._exit(1)` technique as the earlier SQLite crash-recovery test)
interrupts a 5-batch build after 2 batches commit; the parent process
confirms `is_complete()` is `False`, then resumes and rebuilds ONLY the
3 missing batches (`n_source_reads == 3`, not 5) - selective repair,
not a full rebuild. Separately, a missing or deliberately-corrupted
single batch file is detected and individually rebuilt without
re-fetching the others.

**A genuine bug found and fixed while developing this task:** the
first implementation called the expensive `is_complete()` check on
EVERY `read_rows_for_indices` call (every epoch), which re-decompressed
every batch just to confirm validity - completely negating the memory
tier's benefit (measured 21.25s instead of the eventual 3.80s on real
data). Fixed by memoizing the check per instance for its lifetime (a
fresh instance, e.g. a new process, still re-validates properly).
Reported here because it is exactly the kind of "assumed it would be
fast, measured it, it wasn't" finding this project's tasks have
consistently asked for.

## Fair benchmark (Steps 4-5) - real pilot, random split, 50 epochs, medians over reps

| Strategy | Cold/build seconds | Warm seconds (median) | Disk bytes | Peak RSS |
|---|---|---|---|---|
| disabled | n/a | 56.30 | n/a | 1,408.96 MB |
| non-evicting @256MB | n/a | 4.82 | n/a | 1,408.96 MB |
| persistent artifact | 10.57 | 11.38 | 59,908,693 | 1,408.96 MB |
| **hybrid** | 13.93 | **3.80** | 59,905,668 | 1,408.96 MB |

**"Warm" is explicitly defined per strategy and never compared
unfairly:** non-evicting's number is simply its only number (no
persistent state to warm). Persistent artifact's "warm" reuses the
SAME in-process object across reps. **Hybrid's "warm" is the strict
test of its actual design goal: a brand-NEW `HybridTrainingReader`
instance (simulating a fresh process) reusing a disk tier built by a
PRIOR run** - not merely a second call on the same object. That
instance's own cold-start cost is a handful of cheap decompressions
(5 batches, measured in `cold cache stats`/`warm cache stats` below),
not a full SQL+reconstruction+compression cycle.

**Result: the hybrid reader's cold-disk-tier-but-fresh-process run
(3.80s) was the FASTEST of all four strategies measured - faster even
than the purely-in-memory non-evicting cache (4.82s)**, because it
skips SQL query and vectorized reconstruction entirely on its first
pass (going straight to cheap `load_npz` decompression), something
non-evicting cannot do since it has no persistent state to reuse.
Confirmed on the lineage split too (3.88s, `metrics_match: true`).

**Cache-stats trace, cold run** (train partition): `n_source_reads=5,
n_disk_hits=5, n_memory_hits=245` - first epoch builds+decompresses+
caches all 5 batches, remaining 49 epochs are pure memory hits.
**Warm run, fresh instance** (train partition): `n_source_reads=0,
n_disk_hits=5, n_memory_hits=245` - the disk tier from the cold run was
reused with ZERO SQLite queries; only decompression (5 reads) plus
the same 245 memory hits.

**Disk footprint matches the prior task's persistent-artifact
measurement almost exactly** (59.9 MB vs. 59.9 MB) - same npz
compression mechanism, same highly-repetitive dense-row structure.

**Peak RSS was identical across all four strategies: 1,408.96 MB**
(`ru_maxrss`, same cumulative-process convention as every prior task) -
again confirming cache/budget choice and total process memory are
unrelated.

## Equivalence and correctness (Step 5)

Compared direct `SqliteRowStore` reads against non-evicting, persistent,
and hybrid (both its first AND second pass, to exercise the memory-hit
path specifically) for the full random-split training partition: **0
mismatches** across all four (genome IDs, row order, sparse values).
Normalized rows (L2) were also compared directly and matched exactly
(`n_normalized_row_mismatches: 0`). All strategies' end-to-end runs
produced bit-for-bit identical accuracy (0.7667) and ROC-AUC (0.7453) on
both the random split and the lineage-split spot-check.
`max_batch_rows_held_in_memory` stayed at `batch_size` (16) throughout
every run - no strategy ever materializes more than one batch's worth
of the full matrix.

## Full-scale feasibility (Step 6) - measured vs. extrapolated vs. unknown

**Measured, this pilot:**
- Hybrid disk-tier footprint: 59,905,668 bytes for both partitions
  combined (100 genomes) - consistent with the prior task's persistent-
  artifact measurement (~599,057 bytes/genome compressed).
- In-memory footprint when the full working set is cached: 220,201,260
  bytes (train partition, unchanged from prior tasks).
- Hybrid cold-build time: 13.93s for 100 genomes' worth of SQL read +
  reconstruct + compress + write (train + test combined).

**Linear extrapolation (explicitly labeled, assuming density/dimensionality
stay fixed as genome count grows):**

| Genome count | Disk (hybrid artifact) | In-memory (full working set, if budget allows) | Build time (SQL+reconstruct+compress) |
|---|---|---|---|
| 100 (measured) | 60 MB | 220 MB (train only) | 13.9 s |
| 1,000 | ~600 MB | ~2.2 GB | ~139 s (~2.3 min) |
| 10,000 | ~6 GB | ~22 GB | ~1,390 s (~23 min) |

**What this does NOT establish (explicitly, per this task's
instruction not to claim feasibility prematurely):**
- Whether `load_npz`'s decompression throughput holds up at thousands
  of batches, or whether disk I/O becomes the bottleneck once the
  artifact exceeds OS page-cache size (far more likely at 10,000
  genomes' ~6 GB than at this pilot's 60 MB) - **unverified, not
  extrapolated.**
- Whether the in-memory tier's budget could realistically cover even a
  SMALL fraction of a multi-GB working set at full scale, or whether
  the hybrid's advantage shrinks toward the persistent-only numbers
  once the memory tier can no longer hold the active batches - **unverified.**
- Whether density stays at 100% as genome diversity grows (same
  open question carried from the two prior tasks).
- The ~23-minute extrapolated build time for 10,000 genomes, even if
  the linear scaling held exactly, would be a ONE-TIME cost per
  partition/configuration - still needs to be weighed against actual
  available time budgets for a real run, which this task does not
  assess.

**This task's results do not make the full benchmark feasible by
themselves.** They show a specific, measured, cross-process-reuse
benefit for THIS pilot's access pattern - not a validated full-scale
design. The incremental classifier's own predictive-quality limitations
(documented in an earlier task, untouched by every read-path task since)
remain completely unresolved.

## Test results (Step 7)

414 tests passed before this task (end of the prior task). This task
added 14 new tests in `tests/test_hybrid_training_reader.py`: exact equivalence against the
direct store, three-tier counter correctness (memory/disk/source),
memory-budget enforcement with fallback-to-disk (not source), stable
batch ordering, strict partition/batch-size scope enforcement,
fingerprint-based invalidation on both database changes and different
row-index sets, missing-batch detection and selective rebuild,
corrupted-batch detection and rebuild, the "never marks incomplete as
complete" invariant checked directly, the true cross-process kill
recovery test, drop-in training-result equivalence via
`MultiPartitionHybridReader`, the adapter's error behavior for an
unconfigured partition, and artifact-path isolation.

**428 tests pass in total.** No previously-passing test was broken.

## Final validation (Step 9)

- All existing raw/hashed/minimizer `.npz` artifacts, both splits, the
  original batch-LR result, and every prior task's JSONL/SQLite/
  benchmark/persistent-artifact files: unchanged (checksums re-verified).
- `data/raw/ciprofloxacin_pilot/`: still exactly 100 `.fna` files - no
  new downloads.
- The hybrid pipeline does not materialize the full matrix: confirmed
  by `max_batch_rows_held_in_memory` staying at `batch_size` in every
  run, and by the disk tier being built and read strictly one batch at
  a time (`self._store.read_rows_for_indices(set(chunk), batch_size=len(chunk))`
  per expected chunk, never a full-partition query).
- `git status`: only the pre-existing modified `.gitignore` plus new
  untracked files; nothing staged or committed.

## Remaining blockers before full-dataset execution

1. **The hybrid design's scaling behavior beyond pilot size is
   unverified** - both the disk-tier decompression throughput and the
   in-memory tier's practical budget relative to a multi-GB working set
   are open questions, not measured here.
2. **The ~13.9s one-time cold-build cost (per partition, per
   configuration) extrapolates to tens of minutes at full scale** -
   still a real cost to budget for, even though it is paid once, not
   every epoch.
3. **Density/dimensionality stability at larger genome counts remains
   unverified**, carried from the two prior tasks' same open question.
4. **The incremental classifier's predictive-quality limitations are
   completely unresolved** and untouched by this or any prior read-path
   task.
5. No full-dataset download, three-antibiotic benchmark, or Random
   Forest/XGBoost work was started, per this task's explicit stop
   condition.
