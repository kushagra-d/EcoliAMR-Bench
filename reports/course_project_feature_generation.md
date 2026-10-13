# Course-Project Feature Generation: Representation Selection and Build

Selects and generates ONE primary feature representation for the
frozen 956-genome cohort (`reports/course_project_final_cohort.md`).
This task downloads no genome and trains no model. All comparison
numbers reuse existing pilot/cohort-scale measurements and the
existing `resource_estimation.py` extrapolation formulas, projected to
the actual 956-genome target - not a new benchmark, and not the
8,710-genome union already assessed in
`reports/minimizer_resource_scaling_assessment.md` (a much larger,
not-this-project's-current-scope target).

## 1. The three candidates compared

| Representation | Source of pilot evidence | Out-of-core implementation exists? |
|---|---|---|
| Raw canonical k=10 | `reports/intermediate_k_comparison.md` (100-genome pilot) | No - `streaming_kmer_build.py` is memory-SAFE (aborts cleanly) but still materializes one full in-memory CSR matrix |
| Direct feature hashing, k=15, d=262,144 | `reports/feature_hashing_comparison.md` (100-genome, in-memory variant) **and** the REAL 200-genome `policy_consistent`/`cohort_expansion` runs (genuinely out-of-core variant) | **Yes** - `sqlite_row_store.build_sqlite_backed_matrix`, already used twice at 200-genome scale |
| Canonical k=31 minimizers, w=15 | `reports/minimizer_resource_scaling_assessment.md` (100-genome pilot) **and** the REAL 200-genome `policy_consistent`/`cohort_expansion` runs | No - `minimizer_extraction.py` materializes a full in-memory CSR matrix |

**Important distinction carried over from prior tasks, not re-derived
here**: this project has TWO different feature-hashing implementations.
`feature_hashing.py` (used in the original 100-genome comparison)
transforms an ALREADY-BUILT raw k=15 matrix - its memory cost is
dominated by that prerequisite, as documented in that report.
`sqlite_row_store.build_sqlite_backed_matrix` (used for the 200-genome
`cohort_expansion`/`policy_consistent` cohorts, and used again below)
hashes directly during the per-genome sequence walk and never builds a
raw matrix at all. The task names "direct feature hashing with k=15
and dimension 262144" - this is the genuinely out-of-core
implementation, which is what this task actually uses to generate the
956-genome matrix (Section 3).

## 2. Resource projection at the actual target (956 genomes, not 8,710)

Computed by `src/run_course_project_feature_representation_assessment.py`,
reusing `resource_estimation.py`'s existing formulas unchanged.

| Representation | Projected vocabulary | Projected memory | Projected time | Fits the project's 11GB safety threshold? |
|---|---|---|---|---|
| Raw k=10 | 524,346-524,800 (saturated) | **27.1 GB** | ~43.6 min | **No** |
| Hashed k=15, d=262,144 (out-of-core) | 262,144 (fixed) | **~140-156 MB** (measured flat at 200 genomes, not scaled with count - see rationale below) | ~60.6 min | **Yes** |
| Minimizer k=31, w=15 | 4.8M-46.0M (ratio-scaled, unsaturated) | **16.2 GB** | ~63.4 min | **No** |

**Why hashing's memory figure isn't scaled like the others**: unlike
the raw-k-mer and minimizer builders, `build_sqlite_backed_matrix`
hashes directly during the per-genome sequence walk and holds at most
one genome's sequence plus a ≤262,144-entry bucket-count dict in
memory at a time - peak RSS is structurally independent of genome
count, not merely small at the measured scale. The 200-genome
measurement (139-156 MB across the `cohort_expansion` and
`policy_consistent` runs) is the right figure to expect at 956, not a
pilot extrapolation - and Section 3 confirms this with the real
956-genome run.

**Biological/analytical suitability** (qualitative, from existing
reports, not re-measured here):
- Raw k=10 is CARD-weakly-specific (documented in the earlier
  CARD-interpretation task) - its small, saturated vocabulary means
  individual columns are not sequence-specific.
- Minimizers (k=31, w=15) retain per-feature biological specificity
  (each selected feature is still a real canonical 31-mer) at a
  reduced, window-based vocabulary.
- Feature hashing (any dimension) is explicitly NOT per-feature
  interpretable - `feature_hashing_comparison.md` measured zero
  "lonely" (collision-free) buckets at d=262,144 even at 100-genome
  scale (every bucket mixes ~88 distinct raw k=15-mers on average).
  This loss of interpretability is real and is carried forward as a
  limitation (Section 5), not minimized.

## 3. Selection: direct feature hashing, k=15, d=262,144 (out-of-core)

**Recommended and used.** Rationale, in order of weight:

1. **It is the only candidate that fits this project's own established
   11GB memory-safety threshold** at the 956-genome scale - raw k=10
   projects to 27.1 GB and minimizer k=31/w=15 to 16.2 GB, both via
   full in-memory materialization with no existing out-of-core
   alternative in this codebase. Building one would be new
   infrastructure work, which this course-project task is explicitly
   scoped to avoid.
2. **The implementation already exists and is already validated at a
   non-trivial scale** (200 real genomes, twice, with 0 failures both
   times) - reusing it for 956 genomes is a direct scale-up of tested
   code, not new development.
3. **It is compatible with this project's existing streaming
   Logistic Regression pipeline**, which was built specifically to
   train against this SQLite row-store format without materializing
   the full matrix in memory - the natural next step (not performed in
   this task) for model evaluation.
4. Its biological-interpretability cost (collision-driven, measured in
   `feature_hashing_comparison.md`) is accepted as a known,
   documented trade-off for practical feasibility at this project's
   actual scope - not newly discovered here.

This selection was made entirely from the above resource/implementation
evidence; no test-set model performance was consulted (none exists at
this cohort size - no model has been trained on the 956-genome cohort
in this or any prior task).

## 4. Actual generation results (measured, 956 genomes, real run)

Built by `src/run_course_project_feature_generation.py`, reusing
`sqlite_row_store.build_sqlite_backed_matrix` unchanged, reading
assemblies from the three raw directories that together hold the
frozen cohort (`ciprofloxacin_pilot/`, `ciprofloxacin_expansion/`,
`full_cohort_assemblies/`).

| Setting | Value |
|---|---|
| k | 15 |
| n_buckets (dimension) | 262,144 |
| Genomes requested | 956 |
| Genomes processed (rows written) | **956** |
| Failures | **0** |
| Row-alignment mismatches | **0** |
| Total nonzero entries (nnz) | 250,609,664 |
| Density | **100.00%** (every bucket touched by every genome - consistent with the dense-collision finding already measured at 100-genome scale) |
| Build time (measured) | 3,872.9s ≈ **64.5 minutes** (close to the 60.6 min projection) |
| Peak RSS (measured) | **145.9 MB** - confirms the flat-memory prediction; not meaningfully different from the 200-genome run's 139-156 MB despite nearly 5x the genome count |
| SQLite database size on disk | 2,008,838,144 bytes ≈ **1.87 GiB** |

## 5. Validation (performed, not assumed)

| Check | Result |
|---|---|
| Rows in the database | 956 |
| Distinct genome IDs in the database | 956 (no duplicates) |
| Database genome-ID set == frozen cohort manifest set | **True** |
| `row_index → genome_id` independently re-verified against `course_project_final_cohort_genome_order.csv` | **0 mismatches** (956/956 match) |
| Failure log (`..._hashed_failures_k15_d262144.jsonl`) | does not exist - zero failures, confirmed (not merely reported as 0) |
| Ciprofloxacin label manifest (836 genomes) all present in the feature matrix | **True** |
| Ampicillin label manifest (734 genomes) all present in the feature matrix | **True** |
| Gentamicin label manifest (827 genomes) all present in the feature matrix | **True** |
| Pilot/expansion/cohort_expansion/policy_consistent artifacts unchanged | verified (file counts and checksums unchanged) |
| Full test suite | 546/546 pass (no regression; no new test needed - no defect was found in this task) |

Antibiotic labels were never written into the feature matrix or its
row store - they remain solely in the three separate
`course_project_final_cohort_{antibiotic}_manifest.csv` files (genome
IDs + label only), consistent with the task's requirement to keep
labels and the genomic feature matrix separate. A downstream step maps
each antibiotic's genome IDs to row indices via
`course_project_final_cohort_genome_order.csv` - never by position or
by assuming manifest order matches row order.

## 6. Limitations

1. **Density is 100%** at this dimension - the sparsity benefit
   sometimes associated with feature hashing does not materialize
   here, consistent with the same finding at 100-genome scale. This is
   a fixed-width lossy summarization, not a sparsity technique, at
   k=15/d=262,144.
2. **No per-feature biological interpretability** - a trained model's
   hashed-feature coefficient cannot be attributed to a specific
   k-mer (zero "lonely" buckets measured previously); CARD-style
   interpretation, used earlier in this project for a different
   representation, is not possible on this matrix without an
   unhashing step that does not exist.
3. **This selection was made on resource/implementation grounds, not
   model performance** - no model has been trained on this 956-genome
   matrix yet (out of this task's scope), so nothing here claims this
   representation will produce the best Resistant/Susceptible
   separation, only that it is the practically feasible one to
   generate and train against at this project's actual scale.
4. **Raw k=10 and minimizer k=31/w=15 remain unevaluated at 956-genome
   scale** - their resource projections (Section 2) are extrapolations
   from 100- and 200-genome measurements respectively, not real
   956-genome runs; if a reviewer wanted either as a secondary
   comparison, building an out-of-core variant for one of them would
   be new infrastructure work, deliberately out of scope here.
5. **The construction-overhead extrapolation behind raw k=10's and the
   minimizer's 27.1 GB / 16.2 GB figures rests on a single prior
   measurement each** (100- and 200-genome respectively) - exactly the
   same caveat already documented in `resource_estimation.py`'s own
   docstrings; they are presented as the basis for ruling those two
   candidates out at this scale, not as precise figures.

## 7. Outputs of this task

- `reports/course_project_feature_generation.md` (this file)
- `data/processed/course_project_final_cohort_rows_k15_d262144.db` -
  the 956-row hashed feature matrix (SQLite, genome_id-indexed)
- `data/processed/course_project_final_cohort_genome_order.csv` -
  authoritative `genome_id ↔ row_index` mapping
- `data/processed/course_project_final_cohort_hashed_build_config_k15_d262144.json` -
  build settings and measured results (Section 4)
- `data/processed/course_project_feature_representation_assessment.json` -
  the full resource projection behind Section 2's table

No genome was downloaded, no model was trained, and the proposal /
instructor spreadsheet and every pre-existing pilot/expansion/
cohort-scale artifact are unchanged.
