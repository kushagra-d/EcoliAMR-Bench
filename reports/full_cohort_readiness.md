# Full-Cohort Selection and Retrieval Plan

Planning-only task: no genomes were downloaded, no model training was
started. All numbers below were computed FRESH from the currently
saved metadata CSVs (not assumed from an earlier report), reusing
`final_policy_analysis.apply_policy_a` verbatim.

## Step 2: Policy A cohort manifests, per antibiotic

| Antibiotic | Total genomes seen | Eligible | Resistant | Susceptible | Duplicates |
|---|---|---|---|---|---|
| ciprofloxacin | 10,904 | **7,307** | 1,846 (25.3%) | 5,461 (74.7%) | 0 |
| ampicillin | 7,383 | **6,490** | 3,659 (56.4%) | 2,831 (43.6%) | 0 |
| gentamicin | 11,175 | **7,340** | 862 (11.7%) | 6,478 (88.3%) | 0 |

These freshly-computed eligible counts match the earlier feasibility
report's figures exactly - a useful independent confirmation that the
saved metadata hasn't changed and the policy logic is being applied
consistently, not a result taken on faith. Every excluded genome has an
explicit, machine-readable `exclusion_reason` (Policy A's own outcome
string, e.g. `excluded_both_classes_present`,
`excluded_no_binary_interpretation`) in each antibiotic's
`full_cohort_{antibiotic}_policy_a_manifest.csv` - zero genomes are
dropped silently. Zero duplicate genome IDs within any single
antibiotic's manifest (verified programmatically, and guarded
internally - `build_policy_a_manifest` raises rather than silently
including a duplicate, should the upstream policy logic ever produce
one).

## Step 3: cross-antibiotic overlap

| Metric | Value |
|---|---|
| Total antibiotic-specific eligible records (sum, NOT deduplicated) | 21,137 |
| Unique genomes (union across all three) | **8,710** |
| ampicillin & ciprofloxacin | 5,114 |
| ampicillin & gentamicin | 5,138 |
| ciprofloxacin & gentamicin | 7,030 |
| **Three-way overlap** | **4,855** |

The union of unique genomes (8,710) matches the earlier project
estimate exactly. Per-genome antibiotic membership and EACH antibiotic's
own label (never assumed consistent across antibiotics) are recorded
in `full_cohort_cross_antibiotic_per_genome_labels.csv` - a genome
eligible for two or three antibiotics keeps a separate `label_{antibiotic}`
column for each, so a genome that happens to be Resistant for
ciprofloxacin but Susceptible for ampicillin is represented correctly,
not collapsed into one label.

## Step 4: assembly availability (antibiotic-independent, as it must be)

**A real bug was found and fixed during this task, not before shipping
it**: assembly files are antibiotic-independent (one FASTA file, named
only by `genome_id`, serves every antibiotic that genome happens to be
eligible for) - but the first implementation checked each antibiotic's
availability against only that antibiotic's OWN (historically
ciprofloxacin-named) raw directories, so a genome already retrieved
under ciprofloxacin was wrongly reported "missing" when assessed for
ampicillin or gentamicin. Caught by a sanity check (the "needed" count
should equal `8,710 - 200 = 8,510`, but the first run produced `8,708`)
and fixed by checking EVERY antibiotic's availability against the FULL
set of existing raw directories/manifests, regardless of which task
originally retrieved them. A regression test
(`tests/test_run_full_cohort_planning.py::test_genome_validated_under_one_antibiotic_counts_as_available_for_another`)
now guards this specifically.

| Antibiotic | Eligible | Already validated | Previously failed | Missing (never attempted) | Unresolved |
|---|---|---|---|---|---|
| ciprofloxacin | 7,307 | 200 | 0 | 7,107 | 0 |
| ampicillin | 6,490 | 140 | 0 | 6,350 | 0 |
| gentamicin | 7,340 | 192 | 0 | 7,148 | 0 |

The 140/192 "already validated" counts for ampicillin/gentamicin are
NOT from any ampicillin- or gentamicin-specific retrieval (none has
happened) - they are the subset of the 200 already-retrieved
ciprofloxacin-cohort genomes that HAPPEN to also be eligible for that
antibiotic, correctly recognized via the shared assembly files. Zero
previously-failed or unresolved genomes exist, because every retrieval
attempted so far (the original 100-genome pilot and the 100-genome
expansion) succeeded completely.

**Deduplicated genomes needing retrieval across all three antibiotics:
8,510** (`8,710` unique eligible genomes minus the `200` already
validated) - confirmed arithmetically consistent after the fix, not
just plausible.

## Step 5: batched retrieval plan (planning only)

`full_cohort_retrieval_plan.csv`: 8,510 planned download jobs, batched
deterministically (sorted genome ID order, never input-order-dependent)
into 86 batches of up to 100 genomes each (85 full batches + 1 final
batch of 10). Each row carries `antibiotics_needed` (semicolon-joined -
a genome needed by two or three antibiotics gets exactly ONE planned
job, listing all of them, never duplicated) and resumable tracking
fields (`status="planned"`, `retry_count=0`, `max_retries=3`,
`last_error=""`) so a later execution command can safely resume without
re-planning. **Zero duplicate genome_id rows in the plan** (verified
programmatically). No network request was made to produce this plan.

## Step 6: tests

520 tests pass (503 before this task, plus 17 new: 14 for
`full_cohort_planning.py` - manifest correctness and counts, explicit
exclusion reasons, the internal duplicate guard, pairwise/three-way
overlap arithmetic, per-genome label independence across antibiotics,
all four assembly-availability categories including the corrected
`unresolved` branch ordering, deterministic and duplicate-free batch
assignment, and the resumable tracking fields; 3 for the CLI
integration - the antibiotic-independence regression specifically, no
double-planning for a shared genome, and manifest-to-source-metadata
consistency). No previously-passing test was broken.

## Step 8: final validation

- No genomes were downloaded: `data/raw/` still contains only the two
  pre-existing ciprofloxacin directories (`ciprofloxacin_pilot`,
  `ciprofloxacin_expansion`), each still exactly 100 `.fna` files.
- No model training was started - this task ran only metadata-level
  planning code (`apply_policy_a`, overlap arithmetic, file-existence
  checks, batch assignment), never touched `streaming_logistic_regression.py`
  or any model-fitting path.
- Original pilot artifacts (matrices, splits, selected cohort):
  unchanged (checksums re-verified).
- Every cohort label was produced by `final_policy_analysis.apply_policy_a`
  directly, reused unmodified - never reimplemented or approximated.
- Retrieval batches are deterministic (sorted genome ID order) and
  contain zero duplicate download jobs (verified programmatically).
- `git status`: only the pre-existing modified `.gitignore` plus new
  untracked files; nothing staged or committed.

## Remaining blockers before full-dataset execution

1. **8,510 genomes still need retrieval** - at the per-genome retrieval
   times observed in the cohort-expansion task (7-104 seconds each,
   roughly 20-30s median), this plan's full execution would plausibly
   take on the order of 2-3 days of continuous, sequential retrieval -
   a real constraint this task surfaces but does not attempt to solve
   (e.g. parallelization was not investigated).
2. **This plan has never been executed** - no batch has actually been
   run, so real failure rates, rate-limiting behavior, or edge cases at
   this scale remain unknown. The pilot/expansion's 0% failure rate
   (200/200) is encouraging but was only two small, sequential runs.
3. **Policy A's cross-antibiotic class imbalance varies sharply**
   (11.7% Resistant for gentamicin vs. 56.4% for ampicillin) - any
   shared modeling approach across antibiotics will need to handle very
   different base rates, not addressed by this planning task.
4. **The three-way overlap (4,855 genomes, ~56% of the union) means
   most genomes carry labels for multiple antibiotics** - a future
   benchmark design should decide deliberately whether/how to exploit
   this (e.g. multi-task learning) rather than treating the three
   antibiotics as fully independent by default.
5. No retrieval was executed and no model training was started, per
   this task's explicit stop condition.
