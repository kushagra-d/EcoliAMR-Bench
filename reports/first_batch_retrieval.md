# First Retrieval Batch: Execution and Validation

Executed ONLY `batch_id == 0` (100 genomes) of the full-cohort
retrieval plan built in the prior task. No other batch was started. No
model training was started. All original pilot/expansion artifacts
were reused read-only and are unchanged.

## Step 2: safety review found two real, concrete defects - both fixed

1. **Sequence-identity validation gap** (`sequence_retrieval.fetch_complete_assembly`):
   the function fetched FASTA for a genome's requested `sequence_id`s in
   batches, but never checked that each RETURNED record's `sequence_id`
   was actually one of the ones requested for that genome - only that
   the final COUNT matched. A server-side bug or ambiguous ID could, in
   principle, have folded another genome's contig into this one's
   assembly without detection. Fixed: every returned record is now
   checked against the requested batch's ID set; anything unrequested
   is discarded and logged as an explicit error, never silently
   included.
2. **Non-atomic FASTA write** (the `write_fasta` helper, duplicated in
   `run_pilot.py` and `run_cohort_expansion.py`): wrote directly to the
   final `.fna` path. A process killed mid-write (plausible across an
   86-batch, 8,510-genome campaign) would leave a truncated file at the
   FINAL path, which every resume check (`fasta_path.exists()`) would
   then treat as already-complete - silently corrupting a "validated"
   assembly. Fixed: `sequence_retrieval.write_assembly_fasta_atomic`
   writes to a `.part` staging file first and `Path.rename`s it into
   place only after the write succeeds completely; both call sites were
   updated to use the single shared, now-tested implementation.

Both fixes are covered by dedicated tests, including a true
process-kill simulation for the atomic-write fix
(`test_write_assembly_fasta_atomic_leaves_no_final_file_if_interrupted`)
and a fabricated-server-response test for the identity fix
(`test_fetch_complete_assembly_rejects_unrequested_sequence_id`).

**Concurrency**: kept strictly sequential (bounded at 1), matching
every prior successful retrieval run in this project (200/200 real
genomes, 0% failure across the pilot and expansion tasks). This was a
deliberate choice, not an oversight - introducing real concurrency
against a shared public endpoint was judged an unvalidated risk, not a
"defect fix" this task's safety review called for.

## Step 3: execution results (real network, real data)

| Metric | Value |
|---|---|
| Genomes in batch | 100 |
| Successful | **100** |
| QC-rejected | 0 |
| Failed | 0 |
| Genomes needing a retry | 1 (`562.100096`, succeeded on attempt 2) |
| Total elapsed time | 828.0 s (~13.8 min) |
| Per-genome time | 5.1-71.4 s (median ~7 s; the one retried genome took 71.4 s including its backoff wait) |

Assemblies were written to the new, shared, antibiotic-independent
directory `data/raw/full_cohort_assemblies/` (not an antibiotic-named
directory, since one assembly serves any antibiotic that genome is
eligible for).

## Step 4: validation

- **File count**: exactly 100 `.fna` files on disk, matching the
  manifest's 100 rows exactly (set equality checked programmatically,
  not just counted).
- **Identity**: every retrieved file was re-parsed with Biopython and
  its header checked against its filename's genome ID - **0 mismatches
  across all 100 files**.
- **QC**: `qc_passed` is `True` for all 100 (via the project's existing,
  unmodified `qc_assembly` - no threshold was relaxed or invented to
  inflate this number; it reflects the real QC fields:
  `has_valid_nucleotide_data`, `n_contigs > 0`, no duplicate sequence
  IDs within the file).
- **Duplicates**: 0 duplicate `genome_id` rows in the manifest.
- **Manifest accuracy**: `full_cohort_batch_0_retrieval_manifest.csv`'s
  100 rows all show `status=complete`, matching the retrieval plan's
  own updated rows for batch 0 (also all `complete`).
- **Resumability preserved for any future failure**: not exercised for
  a REAL failure this run (there were none), but the retry-count/
  last-error mechanism was exercised for the one transient failure that
  WAS retried, and is covered by dedicated tests for the failure case
  (`test_bounded_retries_then_gives_up`, `test_resume_skips_already_retrieved_genomes`).
- **Original pilot/expansion artifacts**: unchanged (checksums
  re-verified; both raw directories still exactly 100 `.fna` files
  each).
- **No subsequent batch started**: `full_cohort_retrieval_plan.csv`
  shows all 8,410 remaining (batch_id > 0) rows still `planned` -
  untouched.
- **No model training started**: this task ran only retrieval and QC
  code; no feature-extraction or model-fitting script was invoked.

## Step 5: tests

535 tests pass (525 before this task, plus 10 new in this task: 2 for
the sequence-identity fix, 3 for the atomic-write fix including the
real process-kill simulation, and 5 integration tests for the new
batch-execution script covering successful completion, per-genome
failure isolation (both a returned-failure-status case and an
unexpected-exception case), bounded retry exhaustion, retry-then-
succeed, resume-skipping of already-retrieved and already-resolved
genomes, QC-rejection cleanup, and the oversized-batch guard). No
previously-passing test was broken.

## Final validation (Step 7)

- No subsequent batch was started (verified: all batch_id>0 plan rows
  still `planned`).
- No model training was started.
- Original pilot artifacts unchanged (checksums re-verified).
- All 100 successful downloads pass QC (verified directly from the
  manifest, not assumed).
- `git status`: only the pre-existing modified `.gitignore` plus new
  untracked files; nothing staged or committed.

## Remaining blockers / observations before proceeding to further batches

1. **This run's 100% success rate (now 300/300 across all three real
   retrieval runs in this project) is encouraging but still a small
   sample relative to the 8,410 genomes remaining** - no evidence yet
   of how failure rates behave at the scale of dozens of consecutive
   batches.
2. **One genuine transient failure-then-retry occurred this batch**,
   confirming the retry mechanism works under real conditions, not
   just in tests - but also confirming transient failures DO happen at
   this real scale and should be expected in future batches.
3. **Sequential (bounded-at-1) retrieval means the remaining 8,410
   genomes, at ~8.3s/genome average observed here, would take roughly
   19-20 hours of continuous retrieval** - unchanged from the prior
   task's estimate, now grounded in one real batch's measured timing
   rather than the smaller pilot/expansion samples alone.
4. Per this task's explicit stop condition, no further batches were
   executed and no model training was started.
