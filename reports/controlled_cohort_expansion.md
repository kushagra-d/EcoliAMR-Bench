# Controlled Real-Data Cohort Expansion (100 -> 200 Genomes)

Real-data validation task: proves the existing retrieval/QC/storage/
feature pipeline works beyond the original 100-genome pilot, bounded to
100 ADDITIONAL genomes (not the full 8,710-genome cohort). The original
pilot's 100 genomes, splits, matrices, and SQLite store are all reused
unchanged.

## Policy difference, stated explicitly (Step 1)

The original 100-genome pilot was selected under "Policy 2"
(`conflict_analysis.apply_policy_2` - excludes a genome on ANY
Intermediate co-occurrence). This task's instruction to "use the
existing Policy A eligibility logic" means the expansion uses
`final_policy_analysis.apply_policy_a` instead - the policy officially
recommended for all future work (`reports/final_phenotype_policy_analysis.md`),
less strict (tolerates a co-occurring Intermediate call alongside a
unanimous R/S interpretation). **The combined 200-genome pool therefore
spans two different, both-documented eligibility policies** - the
original 100 are never retroactively re-checked or re-labeled under
Policy A, per this task's explicit instruction to preserve the pilot
unchanged.

## Selection (Step 2)

`src/bvbrc_recon/cohort_expansion.py` + `src/run_cohort_expansion.py`:
deterministic (seeded, sorted pools before sampling), excludes all 100
existing pilot genome IDs, resumable (persisted selection is reused
verbatim on any later run, not re-derived), proportional class sampling
(matches the eligible pool's natural ratio, same convention as
`pilot_cohort.py`). Genome IDs preserved as strings throughout
(verified by test).

**Result**: 100 genomes selected - 25 Resistant, 75 Susceptible
(matching the known ~25% Policy-A Resistant rate for ciprofloxacin).

## Retrieval and QC (Step 3) - real network, real data

`fetch_complete_assembly`/`qc_assembly` reused completely unmodified
from the original pilot's infrastructure. Resumable by FASTA-file
existence; failures tracked explicitly with a configurable
`--max-failures` budget (20, not reached); never wrote a FASTA file for
an incomplete assembly.

**Result: 100/100 genomes retrieved successfully, 0 failures.**

| Metric | Value |
|---|---|
| Genomes retrieved | 100/100 |
| Retrieval failures | 0 |
| Contig-count mismatches | 0 (every genome's retrieved contig count matched both the live index and existing metadata) |
| QC errors | 0 |
| Duplicate sequence IDs | 0 |
| Mean genome length | 5,078,921 bp |
| Mean ambiguous-base proportion | 0.0044% (max 0.40%) |
| Mean contig count | 141.7 (range 4-1,263) |

`data/processed/cohort_expansion_ciprofloxacin_retrieval_manifest.csv`
and `..._sequence_qc.csv` record every genome's individual outcome.

## Feature construction: minimizers k=31, w=15 (Step 4)

Built on the COMBINED 200-genome cohort (100 pilot + 100 expansion),
reusing `minimizer_extraction.build_minimizer_matrix_streaming`
unchanged, writing to new `cohort_expansion_`-prefixed paths - the
existing 100-genome pilot's own k=31/w=15 artifact was never touched.

| Metric | 100-genome pilot (existing) | 200-genome combined (this task) | Ratio |
|---|---|---|---|
| Vocabulary size | 4,815,833 | 6,260,899 | **1.30x** for 2x genomes - sub-linear, as expected for related bacterial genomes with substantial minimizer overlap |
| nnz | 73,469,351 | 146,578,102 | 2.00x |
| Density | 15.26% | 11.71% | lower (larger vocabulary, similar per-genome feature count) |
| Mean features/genome | 734,693.51 | 732,891 | **~unchanged** - per-genome minimizer count depends on k/w/genome length, not cohort size, confirmed directly |
| Extraction time | 392.6 s | 853.8 s | 2.18x |
| `.npz` storage | not re-measured here | 310.9 MB | — |
| Peak RSS | 5,280.4 MB | 3,458.0 MB | **lower**, not higher - see caveat below |

**Peak RSS caveat, stated honestly, not smoothed over**: the 200-genome
run's peak RSS (3,458 MB) was LOWER than the original 100-genome run's
(5,280.4 MB), despite processing twice as many genomes. This is not
read as "more genomes need less memory" - `ru_maxrss` is a per-process,
OS-level high-water mark measured in two SEPARATE process invocations
run at different times under different system memory conditions, not a
controlled, repeatable measurement of the algorithm's own memory need.
No safety abort occurred in either run; `aborted_for_memory` was `false`
and `complete: true` for the 200-genome build.

## Feature construction: hashed k=15/d=262,144 SQLite store (Step 5 prerequisite)

Built via the existing, unmodified `sqlite_row_store.build_sqlite_backed_matrix`
on the same combined 200-genome cohort, at a new path
(`cohort_expansion_ciprofloxacin_rows_k15_d262144.db`) - the pilot-only
store is untouched.

| Metric | Value |
|---|---|
| Genomes hashed | 200/200, 0 failures |
| Build time | 791.4 s |
| Peak RSS | **156 MB** - genuinely bounded, one genome at a time, unaffected by the doubled cohort size |

## Exploratory model scaling: streaming Logistic Regression (Step 5)

A NEW, deterministic 70/30 train/validation partition was created
(`cohort_expansion_ciprofloxacin_train_val_partition.csv`, seed=20261009,
stratified) - **separate from, and never overwriting, the saved pilot
`random_split.csv`/`lineage_split.csv`**. No preprocessing beyond the
streaming LR module's own (none - matches the batch baseline's
convention of using raw hashed counts, unmodified).

| Metric | Value |
|---|---|
| Train / validation | 140 / 60 (35R/105S train, 15R/45S validation) |
| Converged | True (`CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL`) |
| Iterations / evaluations | 222 / 277 |
| Final loss / gradient norm | 0.004501 / 0.004534 |
| Fit time | 626.94 s (~10.4 min; uncached `SqliteRowStore`, consistent with the ~2 s/full-pass cost measured for an uncached 200-genome-scale pass in the prior task) |
| Peak RSS | 153.7 -> 564.1 MB |
| **Accuracy** | **0.933** |
| **ROC-AUC** | **0.981** |
| **Average precision** | **0.959** |
| **Resistant recall** | **0.867** |
| **Susceptible recall** | **0.956** |
| Confusion matrix | [[43, 2], [2, 13]] |

**Reported as exploratory only, per this task's explicit instruction -
not a claim of generalization**: these numbers are notably stronger
than the original 100-genome pilot's batch-LR baseline (0.900 acc /
0.938 AUC on its own random split) and dramatically stronger than every
incremental-model result measured in earlier tasks (which never
exceeded zero Resistant recall). This is directionally consistent with
"more training data helps," but is NOT a controlled comparison: this
uses a DIFFERENT validation partition, a cohort spanning two label
policies, and only a modest 2x increase in genome count - it should not
be read as proof that doubling data caused the improvement, only as an
encouraging, honestly-labeled data point from a single exploratory run.
`max_batch_rows_held_in_memory` stayed at 16 throughout - genuinely
bounded-memory per batch, same as every prior streaming LR run.

## Test results (Step 6)

484 tests pass (461 before this task, plus 23 new: 8 for
`cohort_expansion.py`'s selection logic - determinism, pilot exclusion,
genome-ID string preservation, pool-size handling, both-classes-present
guarantee, and the Policy A/Policy 2 distinction being mechanically
documented; 8 for `run_cohort_expansion.py`'s CLI orchestration -
selection idempotency, retrieval resumability with zero re-fetches on a
second run, failure tracking with no FASTA ever written for a failure,
the `max_failures` budget stopping further attempts, incomplete
assemblies never QC'd as usable, genome-ID string preservation through
the manifest, and artifact isolation from the pilot; 3 for the
minimizer-expansion script - combined genome ordering excluding failed
expansion genomes, matrix row count matching the usable combined
cohort, and no write to any pilot-prefixed path; 4 for the streaming-LR
expansion script - the new partition being separate from the saved
pilot splits, partition determinism across runs, bounded-memory
confirmation, and the minority-class blocker path triggering instead of
a silent degenerate fit). No previously-passing test was broken.

## Final validation (Step 8)

- All existing pilot matrices (`k10,15,17,19,21,31` raw, `d65536,262144,1048576`
  hashed, `k31_w5,10,15` minimizer), `genome_order.csv`,
  `random_split.csv`, `lineage_split.csv`, `selected_cohort.csv`, the
  pilot-only SQLite store, and the batch-LR reproduction: all unchanged
  (checksums re-verified, identical to every prior task's baseline).
- `data/raw/ciprofloxacin_pilot/`: still exactly 100 `.fna` files.
- `data/raw/ciprofloxacin_expansion/`: exactly 100 `.fna` files, and
  that set is EXACTLY equal to the persisted selection's 100 genome IDs
  - no extras, nothing missing (verified programmatically, not just by
  count).
- The retrieval manifest shows `complete` for all 100 genomes and zero
  rows where a non-complete status has a non-empty `fasta_path` -
  confirming no failed or incomplete genome was ever treated as
  successful.
- `git status`: only the pre-existing modified `.gitignore` plus new
  untracked files; nothing staged or committed.

## Remaining blockers before full-dataset execution

1. **The combined 200-genome cohort spans two different eligibility
   policies** (Policy 2 for the original 100, Policy A for the new
   100) - a real-data benchmark built on the FULL cohort would need to
   decide and apply ONE policy consistently, not inherit this
   project's incremental, task-by-task policy evolution.
2. **The exploratory streaming-LR improvement (accuracy 0.933, AUC
   0.981) is a single run on a single new partition** - not a
   controlled before/after comparison, and should not be extrapolated
   as "doubling data reliably improves results by this much."
3. **Retrieval time (not separately summed here, but each genome took
   7-104 seconds) means the full ~7,307-genome ciprofloxacin cohort, at
   this same per-genome rate, would take on the order of days, not
   hours** - this was not the focus of this task's measurement, but is
   a visible, real constraint for actual full-scale retrieval.
4. **Only ciprofloxacin was expanded.** Ampicillin and gentamicin were
   not touched, per the explicit three-antibiotic-benchmark exclusion.
5. No full-dataset download (the remaining ~7,107 ciprofloxacin-eligible
   genomes, or any of the other two antibiotics' cohorts) was started,
   per this task's explicit stop condition.
