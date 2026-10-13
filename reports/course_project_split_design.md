# Course-Project Cohort and Split Design: Validation

Validates the proposed ~1,000-genome cohort (300 already-validated +
700 deterministically sampled, seed `20261009`) and the random /
lineage-aware split design, before any additional retrieval. This is
validation only: no genome was downloaded, no feature matrix was
built, no model was trained, and no existing manifest was modified.
Every number below is freshly computed by
`src/validate_course_project_split_design.py` from saved artifacts
only (the three Policy A manifests, the on-disk assembly listings,
and the antibiotic metadata CSVs for MLST) - it does not import or
trust the prior planning script's in-memory state, so this constitutes
an independent check of the proposal's claims, not a re-assertion of
them.

## 1. Sampling plan verification

| Check | Result |
|---|---|
| Genomes already validated on disk | 300 |
| All 300 within the Policy A union pool | 300 (100%) |
| All 300 included in the proposed sample | **True** |
| Additional genomes sampled | 700 |
| All 700 within the Policy A union pool (eligible for ≥1 antibiotic) | **True** |
| Additional genomes disjoint from the 300 already-validated | **True** |
| Duplicate genome IDs in the final 1,000-genome sample | **0** |
| Unique genomes in the final sample | **1,000** |
| Two independent reruns of the seeded draw produce identical samples | **True** |

The deterministic procedure (force-include the 300, then draw 700
more via one fixed `numpy.random.RandomState(20261009)` shuffle of the
sorted remaining union-pool IDs, documented in
`course_project_cohort_design_spec.json`) is fully reproducible from
saved artifacts alone. No discrepancy was found - the exact 1,000-genome
pool reconstructs identically every time.

**Eligible-pool claims (full 8,710-genome Policy A pool), checked against the proposal:**

| Antibiotic | Eligible | Resistant | Susceptible | Matches proposal |
|---|---|---|---|---|
| Ciprofloxacin | 7,307 | 1,846 | 5,461 | ✓ |
| Ampicillin | 6,490 | 3,659 | 2,831 | ✓ |
| Gentamicin | 7,340 | 862 | 6,478 | ✓ |

**Per-antibiotic subset of the proposed ~1,000-genome sample:**

| Antibiotic | n | Resistant | Susceptible | Sample R% | Native R% | Matches proposal's claimed size |
|---|---|---|---|---|---|---|
| Ciprofloxacin | 876 | 211 | 665 | 24.1% | 25.3% | ✓ |
| Ampicillin | 776 | 424 | 352 | 54.6% | 56.4% | ✓ |
| Gentamicin | 860 | 97 | 763 | 11.3% | 11.7% | ✓ |

**MLST lineage grouping over the full ~1,000-genome pool:** 256 groups,
91 with ≥2 genomes, largest group 129 genomes - matches the proposal
exactly.

## 2. Random split validation (stratified 80/20, seed 20261009)

Built with the project's existing, tested
`pilot_model.compute_random_split_indices` (the same stratified
`train_test_split` logic used throughout the pilot), applied
independently to each antibiotic's own eligible subset of the shared
cohort.

| Antibiotic | Train n | Test n | Train R/S | Test R/S | Test R% | Both classes in train | Both classes in test |
|---|---|---|---|---|---|---|---|
| Ciprofloxacin | 700 | 176 | 169/531 | 42/134 | 23.9% | ✓ | ✓ |
| Ampicillin | 620 | 156 | 339/281 | 85/71 | 54.5% | ✓ | ✓ |
| Gentamicin | 688 | 172 | 78/610 | 19/153 | 11.0% | ✓ | ✓ |

All three achieve the requested 80/20 split almost exactly (stratification
has no group constraint to fight against), and the test partition's
class balance stays within ~1.4 points of that antibiotic's native
rate in every case. Both classes are present in both partitions for
all three antibiotics - feasible with no caveats.

## 3. Lineage-aware split validation (target 80/20, seed 20261009)

Built with the project's existing, tested
`lineage_split.split_by_lineage_group` (`GroupShuffleSplit` with an
explicit, independently-re-verified no-overlap check), run separately
per antibiotic on that antibiotic's own eligible subset - group
membership (MLST value) is shared, but which genomes are in scope
differs per antibiotic, so this is not one global split reused three
times.

| Antibiotic | Groups in subset (≥2 genomes) | Train n / groups | Test n / groups | Actual test fraction | Train R/S | Test R/S | Both classes present |
|---|---|---|---|---|---|---|---|
| Ciprofloxacin | 190 (48 with ≥2) | 778 / 190 | 98 / 48 | 11.2% | 182/596 | 29/69 | ✓ |
| Ampicillin | 184 (47 with ≥2) | 609 / 184 | 167 / 47 | 21.5% | 310/299 | 114/53 | ✓ |
| Gentamicin | 190 (48 with ≥2) | 685 / 190 | 175 / 48 | 20.3% | 60/625 | 37/138 | ✓ |

**Group-overlap check**: for all three antibiotics, the train-group set
and test-group set were confirmed disjoint, both by
`split_by_lineage_group`'s own internal invariant check and by an
independent re-check in the validation script (`train_groups &
test_groups == {}`). No MLST group appears on both sides, for any
antibiotic.

**Largest groups and their traced effect on split size**, found by
directly checking which partition each of the top-3 largest groups
landed in:

| Antibiotic | Top 3 group sizes | Where they landed |
|---|---|---|
| Ciprofloxacin | 107, 102, 71 | **all three in train** |
| Ampicillin | 99, 90, 59 | 99 in test; 90, 59 in train |
| Gentamicin | 109, 98, 68 | 109 in test; 98, 68 in train |

This directly explains the observed test-fraction deviations: for
ciprofloxacin, the two largest groups (107 + 102 = 209 genomes, ~24%
of its 876-genome subset) both happened to fall on the train side of
the random group shuffle, pulling the actual test fraction down to
11.2% against a requested 20% - a real, verified consequence of
`GroupShuffleSplit`'s randomness interacting with a skewed group-size
distribution, not a bug. Ampicillin and gentamicin's single largest
group (99 and 109 genomes respectively) fell on the *test* side
instead, keeping their actual test fractions close to the 20% target
(21.5%, 20.3%) but shifting those test partitions' Resistant fraction
well above their own training partitions' (ampicillin: 68.3% test R
vs. 50.9% train R; gentamicin: 21.1% test R vs. 8.8% train R).

**Feasibility: all three antibiotics produce a usable lineage-aware
split** - no partition is empty, every partition contains both
classes, and `split_by_lineage_group`'s own warning system (triggered
only when a class is entirely absent from a partition, or the test
fraction deviates by more than 10 points) raised **zero warnings** for
any of the three. No alternative design is required to make the
lineage-aware split *feasible* - but see Section 5 for why its
*class-balance behavior* still needs to be handled deliberately.

## 4. Class-balance comparison: random vs. lineage-aware test partitions

| Antibiotic | Random-split test R% | Lineage-split test R% | Difference |
|---|---|---|---|
| Ciprofloxacin | 23.9% | 29.6% | +5.7 pts |
| Ampicillin | 54.5% | 68.3% | +13.8 pts |
| Gentamicin | 11.0% | 21.1% | +10.1 pts |

For every antibiotic, the lineage-aware test partition has a
meaningfully different (always higher, in this draw) Resistant rate
than the random-split test partition for the *same* antibiotic and
*same* underlying subset. This is an expected, structural consequence
of enforcing group integrity on a skewed group-size distribution, not
an error - but it means the two evaluation strategies are not being
scored against comparable class priors, which matters for Section 5's
recommendation.

## 5. Recommended final evaluation design

1. **Cohort**: proceed with the verified ~1,000-genome shared pool (300
   reused + 700 to retrieve) exactly as proposed - every claim in the
   proposal reproduced exactly from saved artifacts, with no discrepancy.
2. **Per-antibiotic, per-strategy splits**: keep random and
   lineage-aware splits fully separate per antibiotic (6 total split
   objects: 3 antibiotics × 2 strategies), each built from that
   antibiotic's own eligible subset of the shared cohort, with the
   same seed (`20261009`) and nominal `test_size=0.2` used everywhere
   for consistency, exactly as validated above.
3. **Report observed, not nominal, split sizes.** The random splits
   hit 20% test almost exactly; the lineage-aware splits do not (11.2%
   - 21.5% observed here) - always report the *actual* resulting
   train/test sizes and class counts for whichever split was used, not
   the requested 80/20 target.
4. **Use class-balance-aware metrics for cross-strategy comparison.**
   Because the lineage-aware test partitions have measurably different
   Resistant rates than the random-split test partitions for the same
   antibiotic (Section 4), raw accuracy is not a fair basis for
   comparing "random vs. lineage-aware" performance. Prefer balanced
   accuracy, macro-averaged F1, or AUROC/AUPRC, which are far less
   sensitive to the test partition's class prior, when comparing the
   two evaluation strategies.
5. **Trade-off to document explicitly in the write-up**: the random
   split gives tight, predictable control over both split size and
   class balance, but it can place near-identical genomes (same MLST
   group, i.e. the same clonal lineage) on both sides of the train/test
   boundary - optimistic generalization estimates are a known risk.
   The lineage-aware split closes that leakage path by construction
   (verified: zero group overlap for all three antibiotics) but, as a
   direct and unavoidable consequence of this specific group-size
   distribution, sacrifices control over both the exact test fraction
   and the test partition's class balance. Neither strategy dominates;
   they answer different questions ("how well does the model do on
   similar-but-unseen genomes" vs. "how well does it generalize beyond
   genomes from a known lineage"), which is why the project compares
   both rather than picking one.

## 6. Limitations and unresolved issues

1. This validation reflects one specific seeded draw (`20261009`) of
   the 700 additional genomes and one seeded split of each resulting
   subset. The qualitative finding that lineage-aware test partitions
   get a different effective class prior than random-split test
   partitions is expected to recur with other seeds (it follows from
   the group-size distribution itself), but the exact magnitude (5.7 -
   13.8 points observed here) is specific to this draw and has not
   been checked across multiple seeds.
2. `GroupShuffleSplit`'s placement of the largest 1-3 MLST groups is,
   by construction, the dominant factor in both the lineage-aware test
   fraction and its class balance (Section 3) - a different seed could
   plausibly place ciprofloxacin's two largest groups on opposite
   sides instead of both in train, changing these numbers substantially.
   This sensitivity itself is worth stating as a property of the
   design, not just of this one run.
3. MLST (7-gene scheme) grouping remains a practical relatedness proxy,
   not a guarantee of full genomic independence between groups - stated
   in `lineage_split.py`'s own docstring and still applicable here.
4. The 700 additional genomes have never been individually retrieved;
   this validation covers the *selection and split design* only, not
   retrieval success. (That retrieval-feasibility question was already
   addressed in `reports/course_project_cohort_design.md`.)
5. No single, universally agreed metric for comparing random vs.
   lineage-aware performance under differing class priors was chosen
   here - Section 5 recommends balanced accuracy / macro-F1 / AUROC as
   good candidates, but this is a recommendation for the next modeling
   task to confirm, not a decision this validation task was scoped to make.

## 7. Outputs of this task

- `reports/course_project_split_design.md` (this file)
- `data/processed/course_project_split_design_validation.json` - full
  computed values behind every number above
- `data/processed/course_project_cohort_design_spec.json` - updated in
  place with the now-validated `test_size=0.2` and a pointer to this
  report (previously listed as "not yet decided")

No genome was downloaded, no feature matrix was built, no model was
trained, and no existing cohort manifest was modified by this task.
