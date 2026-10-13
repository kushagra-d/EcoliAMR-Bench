# Course-Project Final Cohort: Validation and Freeze

Validates and freezes the **actual** analysis cohort - the 956
genomes that really have a validated assembly on disk, after the
700-genome retrieval run (656/700 succeeded; see
`reports/course_project_cohort_design.md` Section 8 and
`reports/course_project_split_design.md` for the two reports this one
builds directly on). This task downloads nothing, builds no feature
matrix, trains no model, and does not modify any existing manifest.
All numbers below come from `src/freeze_course_project_final_cohort.py`,
which reconstructs the intended 1,000-genome sample deterministically
and intersects it with what is actually on disk - it does not assume
the gap matches the known 44 failures; it checks.

## 1. Reconstruction and gap check

| Check | Result |
|---|---|
| Intended sample (reconstructed, seed 20261009) | 1,000 |
| Originally-validated 300 (pilot + expansion + batch 0) | 300 |
| Actually validated on disk right now | 956 |
| Actual cohort (intended ∩ on-disk) | **956** |
| Genomes missing from the intended 1,000 | 44 |
| Genomes on disk but NOT part of the intended 1,000 (should be 0) | **0** |
| Missing set == the 44 already-known retrieval failures | **True** |
| Any other silent gap (missing minus the known 44) | **none found** |

The 44-genome gap between the intended and actual cohort is exactly
the 44 genomes already recorded as failed in
`course_project_additional_700_retrieval_plan.csv` - no other genome
is missing for an unexplained reason, and nothing on disk falls
outside the intended sample. The actual cohort is the full
intersection, preserving every genome that could legitimately be
retrieved.

## 2. Per-antibiotic usable cohort (actual 956-genome intersection)

Policy A eligibility applied exactly as in the existing manifests
(`full_cohort_{antibiotic}_policy_a_manifest.csv`) - no change to that
logic, no cross-antibiotic label reuse.

| Antibiotic | n | Resistant | Susceptible | R% | MLST groups (≥2 genomes) | Largest 3 groups | Missing MLST |
|---|---|---|---|---|---|---|---|
| Ciprofloxacin | 836 | 190 | 646 | 22.7% | 224 (75) | 104, 102, 71 | 1 |
| Ampicillin | 734 | 393 | 341 | 53.5% | 214 (62) | 96, 90, 59 | 2 |
| Gentamicin | 827 | 93 | 734 | 11.2% | 223 (74) | 106, 98, 68 | 1 |

## 3. Comparison against the original 1,000-genome design

| Antibiotic | Design n → Actual n | Shortfall | Design R% → Actual R% |
|---|---|---|---|
| Ciprofloxacin | 876 → 836 | -40 | 24.1% → 22.7% |
| Ampicillin | 776 → 734 | -42 | 54.6% → 53.5% |
| Gentamicin | 860 → 827 | -33 | 11.3% → 11.2% |

Class balance barely moved (within 1.4 points for every antibiotic) -
the 44 lost genomes were not concentrated in either class. Lineage
diversity shrank proportionally to the genome-count drop (e.g.
ciprofloxacin: 256 groups at n=1,000 → 224 groups at n=836,
lost ~12.5% of groups against a ~4.6% genome-count drop, meaning the
lost genomes included a slightly above-average share of singleton/small
groups) but remains ample for group-aware splitting in every case.

## 4. Random split validation (stratified 80/20, seed 20261009) - actual cohort

| Antibiotic | Train | Test | Test R/S | Both classes in train | Both classes in test |
|---|---|---|---|---|---|
| Ciprofloxacin | 668 | 168 | 38/130 | ✓ | ✓ |
| Ampicillin | 587 | 147 | 79/68 | ✓ | ✓ |
| Gentamicin | 661 | 166 | 19/147 | ✓ | ✓ |

All three remain valid: stratification keeps the test fraction at
almost exactly 20% and both classes are present in both partitions for
every antibiotic, same as on the full 1,000-genome design.

## 5. Lineage-aware split validation (target 80/20, seed 20261009) - actual cohort

| Antibiotic | Train / groups | Test / groups | Actual test fraction | Test R/S | No group overlap | Both classes in test |
|---|---|---|---|---|---|---|
| Ciprofloxacin | 631 / 179 | 205 / 45 | 24.5% | 91/114 | ✓ | ✓ |
| Ampicillin | 539 / 171 | 195 / 43 | 26.6% | 137/58 | ✓ | ✓ |
| Gentamicin | 741 / 178 | 86 / 45 | 10.4% | **6**/80 | ✓ | ✓ |

**Group-overlap check**: independently re-verified for all three -
`train_groups ∩ test_groups = ∅` in every case, same as the design-time
validation.

**All three remain technically feasible** (every partition non-empty,
both classes present, `split_by_lineage_group`'s own warning system
raised none of its automatic warnings for any antibiotic - no
deviation here exceeded its internal 10-point threshold). However:

- **Gentamicin's lineage-aware test fold has only 6 Resistant genomes**
  (out of 93 Resistant total in its 827-genome cohort). Traced
  directly: these 6 come from 6 different, small MLST groups
  (`ecoli_achtman_4.23`, `.1444`, `.641`, `.538`, `.20`, `.648`), each
  contributing exactly one Resistant genome - not one dominant cluster
  that happened to land in test, but the native scarcity of gentamicin
  resistance (11.2%) spread thinly across many small lineages. The
  split is valid by this task's criteria (both classes present, no
  group overlap), but any Resistant-class metric computed on this
  specific test fold (precision, recall, etc.) will be extremely
  noisy with only 6 positive examples - this is a real, quantified
  limitation to carry into the modeling task, not a defect in the
  split construction.
- Ciprofloxacin and ampicillin's lineage-aware test fractions (24.5%,
  26.6%) landed closer to the nominal 20% than they did on the full
  1,000-genome design (11.2%, 21.5%) - a different group happened to
  fall on the test side after 44 genomes left the pool. This
  reinforces the previously-documented point that `GroupShuffleSplit`
  outcomes are sensitive to exactly which large groups are present,
  not a new concern.

## 6. Freeze status: **PASSED** - cohort frozen as of this task

All checks in Sections 1, 4, and 5 passed (no silent gaps, every split
feasible with both classes represented in both partitions for all
three antibiotics). The 956-genome cohort is frozen for feature
generation, with the gentamicin lineage-aware test-fold limitation
(Section 5) carried forward as a documented caveat rather than a
blocker.

## 7. Limitations

1. **Gentamicin lineage-aware test fold has only 6 Resistant genomes** -
   flagged above; any metric reported from this specific fold should
   note the small positive-class count.
2. **The 44 unresolved genomes remain unresolved** - they are not part
   of the frozen cohort and were not re-attempted or substituted in
   this task (out of scope; see `course_project_additional_700_execution_summary.json`
   for their root-cause breakdown).
3. **This freeze reflects one seed's outcome** (`20261009`) for both
   the original sampling draw and both split constructions - as
   previously documented, different seeds would plausibly place
   different large MLST groups on the test side, changing the exact
   split-size and class-count numbers in Sections 4-5 (not the
   qualitative feasibility conclusion).
4. **Per-antibiotic usable cohort sizes remain unequal by design**
   (836 / 734 / 827) and now carry slightly different shortfalls (-40
   / -42 / -33) - cross-antibiotic performance comparisons must still
   account for this, as in the original split-design report.
5. MLST (7-gene scheme) grouping remains a practical relatedness
   proxy, not a guarantee of full genomic independence between groups.

## 8. Outputs of this task

- `reports/course_project_final_cohort.md` (this file)
- `data/processed/course_project_final_cohort_manifest.csv` - the
  frozen 956-genome cohort (genome IDs only - antibiotic eligibility
  and labels are antibiotic-specific, see below)
- `data/processed/course_project_final_cohort_ciprofloxacin_manifest.csv`,
  `..._ampicillin_manifest.csv`, `..._gentamicin_manifest.csv` -
  per-antibiotic usable genome IDs with their Resistant/Susceptible label
- `data/processed/course_project_final_cohort_validation.json` - full
  computed values behind every number above (reconstruction check,
  per-antibiotic counts/lineage, comparison table, both split results)

All original manifests (`full_cohort_*_policy_a_manifest.csv`), the
pilot/expansion/batch-0 assemblies, and the proposal/instructor
spreadsheet are unchanged. 546/546 tests pass. No genome was
downloaded, no feature matrix was built, and no model was trained in
this task.
