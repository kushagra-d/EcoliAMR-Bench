# Disk-Backed Store Validation: SQLite Replacement for the JSONL Row Store

Exploratory pilot only. Existing 100-genome ciprofloxacin cohort, genome
order, and both saved splits were reused unchanged. No genomes were
downloaded. No existing artifact (including the prior task's JSONL
store) was overwritten or modified.

## Why the JSONL store was large and slow to duplicate-check

`DiskBackedRowStore` (prior task) wrote each row as one JSON line with
explicit decimal-text `cols`/`values` arrays. At k=15/n_buckets=262144
every row is fully dense (every bucket fires for every genome, per the
prior task's finding), so each line held 262,144 `(col, value)` pairs as
ASCII text - ~7+ characters/integer vs. 4 raw bytes in a packed binary
array. More importantly, `existing_genome_ids()` (the duplicate check
run once per build, and the one explicitly flagged by this task) parsed
the ENTIRE file to extract genome IDs - its cost was tied to total row
*payload* size (303 MB), not row *count* (100).

## New format

`src/bvbrc_recon/sqlite_row_store.py` — one SQLite file (stdlib
`sqlite3`, no new dependency), one table:

```sql
CREATE TABLE rows (
    row_index INTEGER PRIMARY KEY,
    genome_id TEXT NOT NULL UNIQUE,
    n_entries INTEGER NOT NULL,
    cols BLOB NOT NULL,   -- packed np.uint32 array, little-endian
    vals BLOB NOT NULL    -- packed np.uint32 array, same length as cols
)
```

- `row_index` is the PRIMARY KEY — ordered iteration and random access by
  position are both indexed, not linear scans.
- `genome_id` is `UNIQUE` — SQLite builds an index for this automatically,
  so a duplicate-genome check is one indexed point lookup
  (`row_exists`), not a scan.
- Hashing is unchanged: still direct-from-sequence
  (`hash_genome_to_bucket_counts`, reused as-is from
  `streaming_hashed_matrix.py`), still no prerequisite raw k-mer matrix,
  still one genome's contigs loaded/hashed/discarded at a time.
- `build_sqlite_backed_matrix` mirrors the prior `build_streaming_hashed_matrix`
  but calls `row_store.row_exists(genome_id)` per genome instead of
  pre-loading a full genome-ID set — duplicate detection no longer
  requires materializing anything in memory first.

**Crash safety.** Each `append_row` is one SQLite transaction
(`with self._conn: ...`): commits on success, rolls back on any
exception. A row's five columns are all `NOT NULL`, so there is no
partial-row state to represent — a row is either fully present
(transaction committed) or entirely absent. The default rollback-journal
mode (WAL was deliberately not used, to keep the recovery story to "one
journal file, replayed automatically on next open") means a process
killed mid-transaction leaves no trace once any connection reopens the
database.

**What was and wasn't tested.** Tested: (1) an in-process
`conn.rollback()` after an uncommitted INSERT — row is simply absent; (2)
a true cross-process kill — a subprocess helper
(`tests/_sqlite_interrupt_subprocess.py`) commits N rows normally, begins
an N+1th INSERT, then calls `os._exit(1)` (skips Python/SQLite cleanup
entirely). The parent test reopens the same file and confirms exactly N
rows present, none duplicated or reordered, and that resuming a build
safely retries the lost row. NOT tested or claimed: disk-hardware
corruption, truncated/partially-copied files, or multi-process
concurrent writers — none of those failure modes were exercised.

The prior JSONL artifacts (`out_of_core_ciprofloxacin_rows_k15_d262144.jsonl`
and its config/results) were left untouched; this task wrote only new
`out_of_core_v2_*` paths.

## Measured efficiency (real 100-genome pilot, k=15, n_buckets=262144)

| Metric | JSONL (prior task) | SQLite (this task) |
|---|---|---|
| Row-store size | 303,233,399 bytes (303.2 MB) | 210,137,088 bytes (210.1 MB) — **30.7% smaller** |
| Build wall-clock (100 genomes) | 368.2 s | 489.6 s — **33% slower** |
| Build peak RSS | 200 MB | 206 MB (comparable) |
| Duplicate-check: full existing-ID scan | 2,466.4 ms | 0.18 ms — **~13,700x faster** |
| 100 indexed point lookups (`row_exists`) | n/a (not supported) | 0.48 ms |

The size reduction is smaller than a naive "binary vs. ASCII" estimate
would suggest (~30%, not ~50%) because every row here is 100% dense —
storing explicit `(col, value)` pairs for a fully dense row is not far
from storing the data directly; a format that detected full density and
skipped storing column indices entirely would roughly halve this further,
but that optimization was not implemented (it would be a configuration-
specific special case, not a general improvement, and this task's actual
bottleneck — duplicate-check cost — is already solved without it).

The build is slower, not faster, with SQLite: each `append_row` commits
a transaction (durability guarantee has a cost), versus JSONL's simple
buffered file append. This is a real, documented tradeoff — durability
and indexed lookups were traded for ~33% more wall-clock time at write
time. The duplicate-detection win (the specific problem this task was
asked to fix) is overwhelming; the write-time cost is the price of that
guarantee.

## Exact row-level agreement with the existing hashed matrix

Every one of the 100 rows in the new SQLite store was compared against
the corresponding row of the existing
`pilot_ciprofloxacin_kmer_hashed_matrix_d262144.npz` (same k=15, same
n_buckets=262144; that matrix's row *i* is known to correspond to
`genome_order[i]` exactly, since its build completed with
`n_genomes_processed == n_requested`, confirmed in
`pilot_ciprofloxacin_kmer_hashed_config_d262144.json`). Comparison was
column-index-by-column-index (via each row's nonzero `(col, value)` set),
not just aggregate statistics.

**Result: 100/100 rows match exactly. Zero mismatches.** This holds
despite the two matrices being built by architecturally unrelated code
paths — one by post-hoc hashing of a materialized raw matrix
(`feature_hashing.hash_matrix`), the other by direct hash-during-walk
with no raw matrix at all (`hash_genome_to_bucket_counts`) — both resting
on the same `canonical_kmer` + `hash_kmer_to_bucket` functions. This is
the strongest correctness evidence produced so far in this project's
out-of-core work, stronger than the previous task's aggregate nnz/density
match.

## Interrupted-run recovery (real test, not just unit test)

The unit test `test_true_process_kill_mid_row_is_recovered` (in
`tests/test_sqlite_row_store.py`) exercises the exact recovery path this
task required: a subprocess commits 3 rows, begins a 4th, and is killed
via `os._exit(1)` before that 4th row's transaction commits. The parent
process then:
1. Reopens the database and confirms exactly 3 rows present, the killed
   genome's row completely absent, no corruption.
2. Resumes a build over the full 5-genome list — genomes 0-2 are skipped
   (already present, not reprocessed), genomes 3-4 are retried and
   written, final count is 5/5 with `complete=True`, and row ordering
   matches `row_index` throughout (not reopen/append order).

This demonstrates the SQLite format resolves this task's recovery
requirement cleanly: there was never a question of "rebuilding" a
partial row's bytes, because the transaction boundary means an
incomplete row simply never exists to rebuild — it is detected by
absence and retried, exactly as Step 3 of this task allows.

## Model evaluation (fixed, untuned settings — not this task's job to fix)

The new store's `read_rows_for_indices` method has the same interface as
the old `DiskBackedRowStore`'s, so `evaluate_streaming_hashed_matrix`
(from the prior task, `incremental_model.py`) worked against it
unmodified — confirming the new format genuinely supports bounded-batch
access for downstream model code without any adaptation.

| Split | n_train | n_test | Accuracy | ROC-AUC | Resistant recall | Susceptible recall |
|---|---|---|---|---|---|---|
| random | 70 | 30 | 0.767 | 0.5 | 0.0 | 1.0 |
| lineage | 71 | 29 | 0.759 | 0.5 | 0.0 | 1.0 |

These numbers are identical to the prior task's JSONL-backed run, as
expected: the underlying features are now confirmed row-for-row
identical, and the classifier (`SGDClassifier(loss="log_loss")`,
`batch_size=16`, `n_epochs=5`, `seed=20261009` — unchanged, not tuned per
this task's instruction) is the same algorithm. The model still collapses
to constant majority-class prediction on both splits. **This is not
claimed as equivalent to the batch LogisticRegression baseline** (which
achieves accuracy 0.9/AUC 0.938 on random and accuracy 0.759/AUC 0.857 on
lineage, on the same features) — it remains an open, unresolved modeling
limitation, not something this task attempted to fix.

## Test results

372 tests pass (355 before this task, plus 17 new in
`tests/test_sqlite_row_store.py`: persistent row indexing, indexed
duplicate detection, append/reopen, in-process rollback, true
cross-process kill recovery, resumption without duplication, exact
sparse-value preservation, batch reads bounded by `batch_size`,
resource-limit aborts (memory/disk/max-failed-rows), failure tracking
without zero-rows, incomplete-run `complete=False`, and artifact
isolation). No previously-passing test was broken.

## Final validation

- All existing raw k-mer, hashed, and minimizer `.npz` artifacts: unchanged.
- The prior task's JSONL row store and its config/results: unchanged.
- `genome_order.csv`, `random_split.csv`, `lineage_split.csv`,
  `selected_cohort.csv`: unchanged.
- `data/raw/ciprofloxacin_pilot/`: still exactly 100 `.fna` files — no new
  downloads.
- `git status`: only the pre-existing modified `.gitignore` plus new
  untracked files; nothing staged or committed.

## Remaining blockers before full-dataset execution

1. **Incremental model quality is still unresolved** — this task
   confirmed the collapse is not a data or storage-format bug (features
   are row-exact identical to the batch-trained baseline's), but did not
   attempt a fix. A different incremental estimator, feature scaling, or
   many more epochs would need investigation before this path is usable
   for real evaluation.
2. **Write throughput is now the bottleneck, not read/lookup.** SQLite's
   per-row commit cost made this build ~33% slower than JSONL. At full
   dataset scale this should be re-examined — e.g. batching several
   genomes per transaction (trading some crash-recovery granularity for
   speed) rather than one commit per genome.
3. **Only exercised at pilot scale (100 genomes, one k/d pair, one
   antibiotic).** No evidence here about behavior at full-dataset
   genome counts, other k/n_buckets configurations, or the other two
   antibiotics.
4. **Crash testing covered one failure mode** (a hard kill before a
   single row's commit). File corruption, concurrent-writer conflicts,
   and WAL-mode behavior were not exercised and are not claimed safe.
5. No full-dataset download, three-antibiotic benchmark, or Random
   Forest/XGBoost work was started, per this task's explicit stop
   condition.
