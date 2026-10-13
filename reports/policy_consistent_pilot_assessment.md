# Policy-Consistent 200-Genome Validation Cohort

Real-data reconciliation task: formally verifies (rather than assumes)
whether the 200 genomes already retrieved across the original pilot
(Policy 2) and the expansion (Policy A) form a single, Policy-A-
consistent cohort, and rebuilds features under that reconciled manifest.
No genomes were retrieved in this task. The original pilot and
expansion artifacts were reused read-only and are unchanged.

## Step 1-2: the two policies, and why reconciliation might have mattered

`conflict_analysis.apply_policy_2` (used for the original 100-genome
pilot) requires a genome's ENTIRE recognized categorical set to be
exactly `{Resistant}` or exactly `{Susceptible}` - no Intermediate at
all. `final_policy_analysis.apply_policy_a` (used for the 100-genome
expansion, and the policy this task mandates for the reconciled
cohort) requires only that the Resistant/Susceptible SUBSET of the
categorical set agree on one side - a co-occurring Intermediate call
does not exclude the genome. **Policy 2 is therefore a strict subset of
Policy A with identical labels for any genome passing both** - a
mathematical fact, but this task verifies it empirically against the
real metadata rather than relying on that argument alone.

## Step 3: reconciliation result (measured, not assumed)

`src/bvbrc_recon/cohort_reconciliation.py` + `src/run_cohort_reconciliation.py`
re-evaluated all 200 already-retrieved genomes against Policy A
(`apply_policy_a`, reused verbatim, never reimplemented), using the real
`ciprofloxacin_amr_metadata.csv`.

| Metric | Result |
|---|---|
| Pilot genomes (100) Policy-A eligible | **100/100** |
| Expansion genomes (100) Policy-A eligible | **100/100** |
| Label mismatches (Policy A label vs. original label, among eligible genomes) | **0** |
| Duplicate genome IDs across pilot + expansion | **0** |
| **Final reconciled cohort** | **200 genomes** (no exclusions, no shortfall) |
| Resistant / Susceptible | **50 / 150** |

**The combined 200-genome pool was already Policy-A-consistent all
along** - the subset relationship held empirically, not just
theoretically. This is a materially different, more reassuring finding
than the prior task's flagged concern ("the model result is therefore
exploratory and not a controlled comparison") - the mixed-policy
caveat, while correctly raised at the time, turns out not to reflect
an actual inconsistency in this specific case. The full per-genome
reconciliation table (`policy_consistent_ciprofloxacin_reconciliation_table.csv`)
records every genome's original cohort, original policy, Policy A
eligibility, label, and (for the small number of theoretically-possible
exclusions, of which there were none here) its exclusion reason -
machine-readable and auditable.

## Step 4: cohort integrity verification

- Every included genome's `policy_a_eligible` is `True` (checked
  programmatically against the reconciliation table, not just asserted
  in prose).
- Zero duplicate genome IDs among the 200 included.
- `label_matches_original` is `True` for all 200 (Policy A's own label
  agrees with each genome's previously-recorded label).
- Every included genome has a `complete` retrieval status in its
  original manifest (pilot or expansion) - reused, not re-verified by
  re-fetching.
- Resistant/Susceptible counts: 50/150 (reported above).
- Original pilot and expansion manifests/artifacts: unchanged
  (checksums re-verified in Step 9 below).

## Step 5: feature rebuild under the reconciled manifest

Reused the EXISTING, unmodified `sqlite_row_store.build_sqlite_backed_matrix`
and `minimizer_extraction.build_minimizer_matrix_streaming` - only the
GENOME ORDER changed (the reconciled manifest's own canonical,
alphabetically-sorted order, not the prior task's pilot-then-expansion
concatenation order) and the FASTA files were reused from their
existing locations (`ciprofloxacin_pilot/` or `ciprofloxacin_expansion/`)
with zero new retrieval.

### Hashed k=15, d=262,144 (new path: `policy_consistent_ciprofloxacin_rows_k15_d262144.db`)

| Metric | Value |
|---|---|
| Genomes | 200/200, 0 failures |
| Build time | 761.2 s |
| Peak RSS | 139 MB |
| Row-alignment vs. reconciled manifest | **0 mismatches** (checked for every one of the 200 rows) |

### Minimizer k=31, w=15 (new path: `policy_consistent_ciprofloxacin_minimizer_matrix_k31_w15.npz`)

| Metric | Value |
|---|---|
| Genomes | 200/200, 0 failures |
| Vocabulary size | 6,260,899 |
| nnz | 146,578,102 |
| Density | 11.71% |
| Mean features/genome | 732,891 |
| Extraction time | 795.7 s |
| Peak RSS | 3,394 MB |
| Row-alignment vs. reconciled manifest | **confirmed exactly** |

**These vocabulary/nnz/density/mean-features figures are numerically
IDENTICAL to the prior (mixed-policy-labeled) task's 200-genome build** -
expected and reassuring, since the underlying 200 genomes and their
sequences are unchanged; only the row ORDER differs (alphabetical here
vs. retrieval-order there). This is itself a useful reproducibility
check: the same pipeline, given the same genome set in a different
order, produces the same aggregate feature statistics.

## Step 6: exploratory evaluation on a genuinely new partition

A fresh, deterministic 70/30 train/validation partition was created
specifically for the reconciled manifest (seed=20261009, stratified) -
**not a reuse of the prior task's 140/60 partition**, per this task's
explicit instruction, even though the underlying 200-genome membership
and labels turned out identical to before. (The resulting SPLIT still
differs from the prior task's, because `train_test_split`'s stratified
assignment depends on input ORDER, which changed - alphabetical vs.
concatenation order.)

| Metric | Value |
|---|---|
| Train / validation | 140 / 60 (35 Resistant / 105 Susceptible train, 15 Resistant / 45 Susceptible validation) |
| Converged | True (`CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL`) |
| Iterations / evaluations | 214 / 270 |
| Final loss / gradient norm | 0.004234 / 0.003907 |
| Fit time | 607.58 s |
| Peak RSS | 153.9 -> 564.0 MB |
| **Accuracy** | **0.933** |
| **ROC-AUC** | **0.970** |
| **Average precision** | **0.940** |
| **Resistant recall** | **0.733** |
| **Susceptible recall** | **1.000** |
| Confusion matrix | [[45, 0], [4, 11]] |

**Labeled exploratory, not compared as a controlled experiment against
the prior mixed-policy result** (per this task's explicit instruction):
the two runs use different partitions (different stratified draws, due
to the reordered input) and are not an apples-to-apples before/after
comparison. For reference only, not as a claim: the prior mixed-policy
run scored accuracy 0.933/AUC 0.981/avg-prec 0.959/Resistant-recall
0.867 on ITS OWN partition - the numbers here are broadly similar in
magnitude, consistent with the fact that the underlying 200-genome data
is identical, but the specific partition draw differs enough to move
individual metrics (especially Resistant recall, computed over only 15
validation genomes - a small enough n that a few individual
predictions swing the recall noticeably).

## Step 7: tests

503 tests pass (484 before this task, plus 19 new: 11 for
`cohort_reconciliation.py` - pure-class eligibility, the
Intermediate-plus-single-sided-class case Policy A tolerates and
Policy 2 does not, explicit exclusion reasons for conflicting and
metadata-missing genomes, determinism, target-size capping, the
fewer-than-target fallback, duplicate-ID detection, genome-ID string
preservation, label-mismatch detection, and Resistant/Susceptible count
correctness; 3 for the hashed-store rebuild script - genomes loaded
from both pilot and expansion directories, genome order matching the
reconciled manifest, and no write to either prior store's path; 3 for
the minimizer rebuild script - row-alignment confirmation, matrix shape
matching the reconciled cohort, and artifact isolation from both prior
minimizer builds; 2 for the streaming-LR script - new-partition
determinism and bounded-memory confirmation). No previously-passing
test was broken.

## Step 9: final validation

- All original pilot artifacts (`pilot_ciprofloxacin_*` matrices,
  splits, selected cohort) and all expansion artifacts
  (`cohort_expansion_ciprofloxacin_*` selection, manifest, QC):
  unchanged (checksums re-verified, identical to every prior task's
  baseline).
- No new genomes were downloaded - `data/raw/ciprofloxacin_pilot/` and
  `data/raw/ciprofloxacin_expansion/` both still contain exactly 100
  `.fna` files each.
- Every one of the 200 included genomes verified `policy_a_eligible ==
  True` directly from the reconciliation table (not merely asserted).
- Feature-row alignment with the reconciled manifest verified exactly
  for both the hashed store (0 mismatches across all 200 rows) and the
  minimizer matrix (row order confirmed to match the manifest exactly).
- `git status`: only the pre-existing modified `.gitignore` plus new
  untracked files; nothing staged or committed.

## Remaining blockers before full-dataset execution

1. **This reconciliation only proves consistency for these specific 200
   genomes** - it does not establish that Policy A and Policy 2 would
   agree this cleanly across the full ~7,307-genome ciprofloxacin
   cohort; that would need its own check if the project ever mixed
   policies again at larger scale.
2. **The exploratory streaming-LR result here is a single run on a
   single new partition**, like the prior task's - still not a
   controlled, repeated-trial comparison, and still far below the
   ~5,115-genome training-set scale a real ciprofloxacin model would
   use.
3. **Feature rebuild time (~761-796 s per representation) was paid
   twice** (once for the mixed-policy cohort in the prior task, once
   here) for numerically-equivalent underlying data - an accepted cost
   of this task's explicit request for a traceable, independently-
   verified rebuild, not a sign of a problem.
4. Only ciprofloxacin was touched; ampicillin and gentamicin were not
   reconciled or rebuilt.
5. No full-dataset download or three-antibiotic benchmark was started,
   per this task's explicit stop condition.
