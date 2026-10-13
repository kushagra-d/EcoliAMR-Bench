# Course-Project Cohort Design

**Scope note.** This project has been re-scoped from a production-style
full-dataset buildout to a course project: investigate AMR prediction
in *E. coli* from k-mer profiles and compare random vs. lineage-aware
evaluation, on a cohort that is scientifically defensible and
computationally manageable - not necessarily every eligible genome.
Sections 1-7 below are the original planning/analysis (no download, no
feature matrix, no model). **Section 8 was added after the proposed
700-genome retrieval was actually executed** and reports the real
outcome, including the genomes that did not resolve - no feature
matrix or model was built even in this update. All counts below are
computed directly from existing artifacts by
`src/run_course_project_cohort_design.py` (seed `20261009`), not
invented or extrapolated beyond what is marked as an extrapolation.

## 1. What already exists

- **Policy A full-cohort union pool**: 8,710 unique genomes, each
  evaluated for Policy A eligibility against all three antibiotics
  (`data/processed/full_cohort_{antibiotic}_policy_a_manifest.csv`).
  Eligible pool sizes and native Resistant rates (whole pool, not a
  sample): ciprofloxacin 7,307 eligible (25.3% R), ampicillin 6,490
  eligible (56.4% R), gentamicin 7,340 eligible (11.7% R). 7,572
  genomes (86.9% of the union) are eligible for **two or more**
  antibiotics; 4,855 (55.7%) are eligible for all three
  (`full_cohort_cross_antibiotic_overlap.json`).
- **Already-validated assemblies**: 300 `.fna` files on disk across
  three prior retrieval runs - 100 pilot, 100 expansion, 100 "batch 0"
  of the full-cohort plan - all QC-passed, all within the Policy A
  union pool. These were **not** selected by one consistent procedure:
  pilot was stratified by ciprofloxacin Policy 2, expansion by
  ciprofloxacin Policy A, and batch 0 by alphabetically-sorted genome
  ID (i.e. "whichever genomes sort first," not a representative draw).
  Their measured composition is reported below rather than assumed.
- **MLST lineage grouping**: the existing, tested
  `lineage_split.assign_lineage_groups` (group key = `genome_mlst`;
  missing-MLST genomes get their own singleton group, never merged)
  was reused unchanged to assess lineage diversity for every cohort
  size below.
- **One real measurement of retrieval throughput**: batch 0 (100
  genomes) took 828.0s total, 8.28s/genome average, 100/100 succeeded,
  1 genome needed a retry. This is the only empirical timing available
  and is the sole basis for the extrapolations in Section 4.

## 2. Sampling options considered

| Option | Description | Verdict |
|---|---|---|
| Antibiotic-specific cohorts (3 separate pools) | Draw an independent sample per antibiotic | Rejected: 86.9% of eligible genomes already serve 2+ antibiotics, so disjoint pools would re-retrieve largely the same assemblies three times, with no benefit and no leakage concern it actually solves (each antibiotic's analysis is independent regardless of pooling) |
| Shared pool, stratified to force each antibiotic's native R:S ratio | Draw with per-antibiotic label-aware weighting | Rejected: a single shared draw cannot simultaneously match three different native ratios (25.3% / 56.4% / 11.7% R); forcing one antibiotic's ratio would distort the other two, and "choosing genomes to hit a target ratio" starts to resemble selecting on the outcome we're trying to measure |
| Shared pool, uniform random draw (no label stratification) | Draw genome IDs uniformly at random from the union pool, independent of any antibiotic's label | **Adopted**: preserves each antibiotic's own native ratio *in its eligible subset* in expectation, never looks at any label to decide inclusion, and naturally yields high multi-antibiotic coverage because most of the pool already is multi-eligible |

Only Policy A eligibility, antibiotic labels (for reporting composition
after the fact, not for selection), and MLST values were used anywhere
in this analysis - no model was trained and no performance/test-set
outcome influenced which genomes were simulated as candidates (Step 3
requirement).

## 3. Leakage-avoidance check

The additional-genome draw (Section 4) uses **only**: the Policy A
union-pool genome ID list, and a fixed seed. It does not condition on
any antibiotic's R/S label - labels are reported *after* the draw, to
verify (not to produce) a reasonable class balance. No model was run
to produce these numbers. Lineage-aware feasibility is assessed by
actually counting the resulting MLST groups and per-group sizes at
each candidate size (Section 4), not by asserting it works.

## 4. Comparison of practical cohort sizes

All rows reuse the same 300 already-validated genomes; "new assemblies
needed" is the only retrieval work implied by each option. Per-antibiotic
columns show the eligible count within that cohort, Resistant count,
and Resistant rate (compared to the antibiotic's native rate across the
full 8,710-genome pool).

| Target size | New assemblies needed | Cipro (n, R%) | Ampicillin (n, R%) | Gentamicin (n, R%) | MLST groups (≥2 genomes) | Largest group |
|---|---|---|---|---|---|---|
| 300 (existing only) | 0 | 298, 20.1% | 240, 47.1% | 290, 9.3% | 108 (26) | 40 |
| 600 | 300 | 546, 22.3% | 470, 50.9% | 542, 10.9% | 191 (59) | 68 |
| **1,000** | **700** | 876, 24.1% | 776, 54.6% | 860, 11.3% | 256 (91) | 129 |
| 1,500 | 1,200 | 1,291, 24.6% | 1,142, 55.8% | 1,280, 12.0% | 326 (115) | 191 |

Native (full-pool) R rates for reference: ciprofloxacin 25.3%,
ampicillin 56.4%, gentamicin 11.7%. At every size, the sampled R rate
tracks the native rate within 1-6 points, closest at 1,000-1,500 and
least close (as expected from smaller N) at 300.

**Retrieval-time extrapolation** (explicitly an extrapolation from the
single 8.28s/genome batch-0 average, not a new measurement):
- 600 target (+300 new): ≈ 2,484s ≈ 41 minutes
- 1,000 target (+700 new): ≈ 5,796s ≈ 1.6 hours
- 1,500 target (+1,200 new): ≈ 9,936s ≈ 2.8 hours

All of these are far below the full-remaining-cohort estimate (~19-20
hours for the other ~8,410 genomes) from the prior task, consistent
with the new, deliberately smaller scope.

**Important caveat on "new assemblies needed":** every genome beyond
the existing 300 falls into `full_cohort_assembly_availability.json`'s
`missing_never_attempted` category for its antibiotic - retrieval has
literally never been tried for these specific genomes. The only real
evidence of retrievability is the 100%-success track record (300/300)
across the three batches attempted so far, which is encouraging but
covers a small, non-random slice of the 8,410-genome remainder.

## 5. Recommendation

**Target ~1,000 unique genomes**, reusing the existing 300 and drawing
700 additional genome IDs by the deterministic, unstratified uniform
sample described in Section 2 (seed `20261009`; reproducible via
`src/run_course_project_cohort_design.py`).

- **Pooling**: one shared genome pool across all three antibiotics,
  not three separate cohorts. Each antibiotic's analysis uses its own
  Policy A eligible subset of this shared cohort (ciprofloxacin ≈876,
  ampicillin ≈776, gentamicin ≈860 genomes) - these subset sizes and
  class balances are not equalized across antibiotics, by design.
- **Class balance**: no artificial rebalancing. The uniform draw
  already reproduces each antibiotic's native R:S ratio within a few
  points (Section 4); imposing a forced 50/50 split would misrepresent
  the real clinical prevalence and was explicitly ruled out.
- **Random split**: reuse the pilot's existing seeded random-split
  convention, applied separately to each antibiotic's eligible subset
  of the 1,000-genome cohort.
- **Lineage-aware split**: reuse `assign_lineage_groups` +
  `GroupShuffleSplit`, run **independently per antibiotic** on that
  antibiotic's eligible subset (not globally across all three) - group
  membership is the same regardless of antibiotic, but which genomes
  are even in-scope differs per antibiotic, so group counts and
  per-group class balance must be checked per antibiotic before
  trusting the split.
- **Feasibility of lineage-aware evaluation at n≈1,000**: feasible but
  not risk-free. 256 MLST groups, 91 with ≥2 genomes - enough groups to
  shuffle-split meaningfully. However, the single largest group has
  129 genomes (~13% of the whole cohort) - large enough that a random
  group-to-fold assignment could swing it entirely into train or
  entirely into test, materially shifting the test fold's class
  balance and size. This must be checked empirically on the actual
  split produced (group counts and per-fold class counts inspected
  directly), not assumed valid from the group count alone.

## 6. Limitations

1. "Never attempted" is not "confirmed retrievable" - the 700
   additional genomes have no individual retrieval history; only the
   aggregate 300/300 success rate from differently-selected prior
   batches supports expecting a similar success rate here.
2. The 1.6-hour retrieval estimate for 700 genomes is a linear
   extrapolation from one 100-genome batch; it does not account for
   possible rate-limiting, transient-failure clustering, or throughput
   changes at larger scale (one retry was already observed in the only
   batch measured).
3. MLST (7-gene scheme) grouping is a practical relatedness proxy, not
   a guarantee of full genomic independence between groups - stated in
   `lineage_split.py`'s own docstring and repeated here because it
   bears directly on how strongly the lineage-aware results can be
   interpreted.
4. The existing 300 genomes were assembled from three historically
   different selection procedures (two stratified, one
   alphabetical-ID-sequential); their reasonably-native-looking
   composition (Section 4) is an empirical observation about this
   particular 300, not a guaranteed property of how they were chosen.
5. Because each antibiotic's eligible subset of the shared cohort has
   a different N and native class balance, cross-antibiotic
   *performance* comparisons must account for that difference rather
   than treating the three antibiotics as equal-sized experiments.

## 7. Outputs of the original planning task

- `reports/course_project_cohort_design.md` (this file)
- `data/processed/course_project_cohort_design_analysis.json` - full
  computed counts behind every number in Section 4, for 300/600/1,000/1,500
- `data/processed/course_project_cohort_design_spec.json` - small
  machine-readable summary of the recommended design (target size,
  pooling rule, selection rule/seed, split-construction approach)

No genome was downloaded, no feature matrix was built, no model was
trained, and no existing cohort manifest was modified by the original
planning task (Sections 1-7).

## 8. Actual retrieval results (executed after the design above was validated)

The 700-genome sample from Sections 4-5 was reconstructed deterministically
from saved artifacts, independently re-verified against the proposal's
claimed counts (all matched exactly - see
`reports/course_project_split_design.md`), persisted as a fixed plan
(`data/processed/course_project_additional_700_retrieval_plan.csv`), and
then retrieved with `src/run_course_project_additional_retrieval.py` -
reusing the existing, unmodified BV-BRC downloader, sequence-identity
check, QC gate, bounded retries (3/genome), and atomic-write
implementation, sequential concurrency (bounded at 1, same as every
prior retrieval run in this project). The original full-cohort
retrieval plan (`full_cohort_retrieval_plan.csv`) was not touched or
executed.

### Result: 656 / 700 succeeded (93.7%); 44 did not resolve

| Metric | Value |
|---|---|
| Genomes planned | 700 |
| Successful (QC-passed) | **656** |
| Failed (retries exhausted) | **44** |
| QC-rejected | 0 |
| Still in "planned" state (unattempted) | 0 |
| Total attempts | 791 (91 were retries) |
| Runtime | 6,104.7s ≈ **1.70 hours** |
| Disk used by the new assemblies' shared directory | 3.6 GiB (756 files, including the 100 from batch 0) |
| Free disk space at completion | 64.4 GiB |

This is **below the 100% success rate observed in the three earlier,
smaller batches** (pilot, expansion, batch 0 - 300/300). The failure
rate did not trigger the run's consecutive-failure circuit breaker
(limit: 10 in a row), because the 44 failures are scattered throughout
the run rather than clustered - consistent with a per-genome data
issue, not a transient outage.

### Root cause of all 44 failures (verified, not guessed)

- **43 genomes**: every sequence record BV-BRC returned for the
  requested `sequence_id` batch was rejected by the existing identity
  check in `fetch_complete_assembly` (added in the batch-0 task) as
  not matching any ID that was actually requested -
  `n_contigs_retrieved=0` for all 43, even though
  `n_contigs_in_index` matched `n_contigs_expected_from_metadata`
  exactly. This points to BV-BRC's live sequence IDs for these
  specific genomes having drifted from what this project's cached
  metadata CSV recorded - the identity check is working as designed
  (refusing to splice in unrequested sequences), not malfunctioning.
- **1 genome** (`562.144965`): retrieved exactly 2000 contigs,
  consistent with BV-BRC's current index, but the cached metadata's
  `genome_contigs` field says 2057 - a stale-metadata contig-count
  mismatch, same root-cause family as above.

**No substitute genomes were retrieved in place of the 44 failures.**
Per the task's explicit instruction, the selected cohort was preserved
exactly; the failures are reported here for review rather than
silently worked around.

Full list of the 44 unresolved genome IDs is in
`data/processed/course_project_additional_700_execution_summary.json`
(`unresolved_genome_ids`), with the same root-cause breakdown.

### Final validation performed

- **File-count / identity**: all 656 successful `.fna` files exist on
  disk; none of the 44 failed genomes left a file behind (QC-rejection
  cleanup and the identity check's own discard both verified to leave
  no partial file). `data/raw/full_cohort_assemblies/` now holds 756
  files (100 from batch 0 + 656 new) - counted directly, not assumed.
- **QC**: all 656 successes have `qc_passed=True`; zero
  `duplicate_sequence_ids`; `n_contigs_retrieved == n_contigs_in_index`
  for all 656 - verified from the manifest, not asserted.
- **No duplicate retrieval jobs**: 0 duplicate `genome_id` rows in the
  700-genome manifest.
- **Original 300 assemblies unchanged**: `ciprofloxacin_pilot/` and
  `ciprofloxacin_expansion/` still have exactly 100 files each; the new
  700's genome IDs have zero overlap with batch 0's 100 (checked
  directly) - the original batch-0 files could not have been touched.
- **Per-antibiotic mapping, using only genomes that actually have a
  validated assembly right now** (956 = 300 + 656, not the full 1,000):

| Antibiotic | Validated + eligible now | Resistant | Susceptible | Shortfall vs. the planned 1,000-pool subset |
|---|---|---|---|---|
| Ciprofloxacin | 836 | 190 | 646 | -40 (was 876) |
| Ampicillin | 734 | 393 | 341 | -42 (was 776) |
| Gentamicin | 827 | 93 | 734 | -33 (was 860) |

  The per-antibiotic shortfalls exceed the 44 total failed genomes
  because most failed genomes were eligible for more than one
  antibiotic (consistent with the high cross-antibiotic overlap
  documented in Section 1) - each failed genome reduces more than one
  antibiotic's usable subset.

### Unresolved issue requiring review before proceeding to feature generation

The 44 genomes are **not part of the usable cohort** until this is
resolved. Two options, neither executed here (out of this task's
scope - retrieval-only):
1. Accept the 956-genome cohort (dropping the 44) and proceed - the
   per-antibiotic subset sizes above would become the actual basis for
   feature generation and splitting, not the Section 4/5 projections.
2. Investigate whether the 44 genomes' metadata-recorded sequence IDs
   can be refreshed from BV-BRC's current index and re-attempted - not
   done here, since silently changing the identity check's behavior or
   substituting different genomes was explicitly out of scope for this
   retrieval task.

### Outputs of this retrieval task

- `data/processed/course_project_additional_700_retrieval_plan.csv` -
  the fixed, pre-verified 700-genome plan, with final status per genome
- `data/processed/course_project_additional_700_manifest.csv` -
  per-genome retrieval manifest (status, contigs, QC, timing, errors)
- `data/processed/course_project_additional_700_execution_summary.json` -
  run-level summary, including the full unresolved-genome list and root-cause breakdown
- `reports/course_project_cohort_design.md` (this section)

No feature matrix was built and no model was trained during this
retrieval task. The original 300 assemblies and all three Policy A
manifests are unchanged (verified above). 546/546 tests pass (535
pre-existing + 11 new, covering this retrieval script's resume,
retry, QC-rejection, duplicate/row-count guard, and
consecutive-failure circuit-breaker behavior).
