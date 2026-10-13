# Intermediate k-mer Configuration Comparison (k=10, 15, 17, 19, 21, 31)

- Analysis date (UTC): 2026-10-09T22:23:49.866139+00:00
- Scope: uses ONLY the existing 100-genome ciprofloxacin pilot assemblies already on disk. No
  additional genome was downloaded. No full-scale feature matrix was built. k=10 and k=31 results
  are the EXISTING artifacts from the prior feasibility task, reused here unmodified - not rerun.
- **No final k-mer configuration has been approved for the full three-antibiotic benchmark as a
  result of this task.** This remains an open decision.

## Methodology and configuration

- k=15, 17, 19, 21 were extracted using the existing memory-safe streaming builder
  (`bvbrc_recon.streaming_kmer_build.build_matrix_streaming`), run as a SEPARATE OS process per k
  value (`src/run_intermediate_k_experiment.py <k>`) so that memory is fully released between k
  values rather than accumulating within one long-lived process.
- Canonicalization, ambiguous-base skipping, and per-contig (never cross-contig) window counting
  are identical to every other k already computed in this project - same function, not a
  reimplementation.
- Logistic Regression only (`bvbrc_recon.pilot_model.evaluate_with_precomputed_split`), same fixed
  configuration already used for k=10/k=31 (`max_iter=1000, random_state=20261009`), evaluated on
  the EXISTING saved random and lineage splits - neither split was regenerated, and no model
  hyperparameter was chosen or adjusted based on any test-set result.
- Random Forest and XGBoost were intentionally NOT run for every intermediate k (by task
  instruction) - the existing k=10/k=31 RF/XGBoost results from the prior task are not reproduced
  for k=15-21.

## Resource and feature-space comparison (measured, all k values)

| k | Genomes | n_features | nnz | Density | Extraction time | Peak RSS | npz storage | In-memory CSR |
|---|---|---|---|---|---|---|---|---|
| 10 | 100/100 (complete) | 524,346 | 49,668,801 | 94.73% | 273.4s | 2835 MB | 113.6 MB | 596.0 MB |
| 15 | 100/100 (complete) | 23,171,787 | 496,633,938 | 21.43% | 488.9s | 7158 MB | 714.0 MB | 5959.6 MB |
| 17 | 100/100 (complete) | 25,436,868 | 504,680,021 | 19.84% | 457.0s | 7488 MB | 710.0 MB | 6056.2 MB |
| 19 | 100/100 (complete) | 26,626,636 | 505,806,534 | 19.00% | 450.9s | 7035 MB | 709.7 MB | 6069.7 MB |
| 21 | 100/100 (complete) | 27,690,097 | 506,232,329 | 18.28% | 481.1s | 7523 MB | 710.1 MB | 6074.8 MB |
| 31 | 100/100 (complete) | 32,648,927 | 507,309,385 | 15.54% | 557.4s | 7565 MB | 711.9 MB | 6087.7 MB |

**Observed pattern (measured, not assumed)**: feature count grows sharply from k=10 to k=15 (524K
to 23.2M) but far more gradually from k=15 through k=31 (23.2M to 32.6M) - consistent with k=10
being governed by the small, quickly-saturated theoretical k-mer space, while k>=15 is instead
governed by the actual sequence diversity across these 100 *E. coli* genomes, which is a much
larger but still bounded quantity. Peak memory and total nonzero count are similarly flat across
k=15-31 (~496-507M nonzeros, ~7.0-7.6GB peak RSS) - **the pilot-scale memory cost of k=15 is
already nearly as high as k=31's**, despite the large difference in nominal k.

No k value failed or triggered the memory-safety abort at this 100-genome pilot scale - all six
configurations (including the pre-existing k=10/k=31) completed with `complete: true`.

## Logistic Regression results on existing saved splits

| k | Split | Accuracy | ROC-AUC | Avg. precision | Recall (R) | Recall (S) | Confusion matrix |
|---|---|---|---|---|---|---|---|
| 10 | random | 0.9 | 0.9627329192546584 | None | None | None | see reports/pilot_model_comparison.md for full detail |
| 10 | lineage | 0.7931034482758621 | 0.8896103896103896 | None | None | None | see reports/pilot_model_comparison.md for full detail |
| 15 | random | 0.9 | 0.9440993788819876 | 0.8833333333333333 | 0.5714285714285714 | 1.0 | [[23, 0], [3, 4]] |
| 15 | lineage | 0.896551724137931 | 0.9155844155844155 | 0.876984126984127 | 0.7142857142857143 | 0.9545454545454546 | [[21, 1], [2, 5]] |
| 17 | random | 0.9 | 0.9503105590062112 | 0.8909090909090909 | 0.5714285714285714 | 1.0 | [[23, 0], [3, 4]] |
| 17 | lineage | 0.9310344827586207 | 0.9155844155844155 | 0.876984126984127 | 0.7142857142857143 | 1.0 | [[22, 0], [2, 5]] |
| 19 | random | 0.9 | 0.9503105590062112 | 0.8909090909090909 | 0.5714285714285714 | 1.0 | [[23, 0], [3, 4]] |
| 19 | lineage | 0.896551724137931 | 0.9090909090909091 | 0.8740601503759398 | 0.7142857142857143 | 0.9545454545454546 | [[21, 1], [2, 5]] |
| 21 | random | 0.9 | 0.9627329192546583 | 0.9095238095238094 | 0.5714285714285714 | 1.0 | [[23, 0], [3, 4]] |
| 21 | lineage | 0.896551724137931 | 0.9090909090909091 | 0.8655567490153955 | 0.5714285714285714 | 1.0 | [[22, 0], [3, 4]] |
| 31 | random | 0.9 | 0.9316770186335404 | None | None | None | see reports/pilot_model_comparison.md for full detail |
| 31 | lineage | 0.8275862068965517 | 0.8766233766233766 | None | None | None | see reports/pilot_model_comparison.md for full detail |

**These are exploratory results from the SAME 100-genome pilot reused across every k value - they
do not establish which k generalizes better to the full dataset, and no k here is declared "best"
on the strength of these numbers alone.**


**Notable, non-monotonic pattern (stated plainly, not smoothed over)**: on the random split,
Logistic Regression's accuracy is identical (0.900) across every single k value tested, and
Resistant recall is 0.571 for every k except k=31 (1.000) - the random-split result is almost
entirely insensitive to k in this pilot. On the lineage-aware split, accuracy and Resistant recall
vary non-monotonically with k (e.g. lineage accuracy: k=10 0.793, k=15 0.897, k=17 0.931, k=19
0.897, k=21 0.897, k=31 0.828) - k=17 happens to show the best lineage-split accuracy in this run,
but k=21 (immediately adjacent) does not continue any such trend, and k=31 is worse than several
smaller k values. This is exactly the kind of pattern that must NOT be read as "k=17 is the best
k" - it is as consistent with sampling noise from a 29-genome lineage test set as with any genuine
effect, and no cross-validation was performed to distinguish the two.


## Full-dataset scaling risk (projected from pilot measurements - NOT known exactly)

| k | Target cohort | Regime (pilot-measured) | Est. n_features | Est. CSR memory |
|---|---|---|---|---|
| 10 | ciprofloxacin (7,307 eligible) | saturated | 524,346-524,800 | 43.6 GB |
| 10 | union, all 3 antibiotics (8,710) | saturated | 524,346-524,800 | 51.9 GB |
| 15 | ciprofloxacin (7,307 eligible) | unsaturated | 23,171,787-536,870,912 | 435.5 GB |
| 15 | union, all 3 antibiotics (8,710) | unsaturated | 23,171,787-536,870,912 | 519.1 GB |
| 17 | ciprofloxacin (7,307 eligible) | unsaturated | 25,436,868-1,858,671,944 | 442.5 GB |
| 17 | union, all 3 antibiotics (8,710) | unsaturated | 25,436,868-2,215,551,202 | 527.5 GB |
| 19 | ciprofloxacin (7,307 eligible) | unsaturated | 26,626,636-1,945,608,292 | 443.5 GB |
| 19 | union, all 3 antibiotics (8,710) | unsaturated | 26,626,636-2,319,179,995 | 528.7 GB |
| 21 | ciprofloxacin (7,307 eligible) | unsaturated | 27,690,097-2,023,315,387 | 443.9 GB |
| 21 | union, all 3 antibiotics (8,710) | unsaturated | 27,690,097-2,411,807,448 | 529.1 GB |
| 31 | ciprofloxacin (7,307 eligible) | unsaturated | 32,648,927-2,385,657,095 | 444.8 GB |
| 31 | union, all 3 antibiotics (8,710) | unsaturated | 32,648,927-2,843,721,541 | 530.2 GB |

**Full-scale memory usage is NOT known exactly from a 100-genome pilot** - every number above is a
projection using `resource_estimation.py`'s explicit, documented extrapolation rules (measured
nnz-per-genome scaled linearly; column count bounded between the pilot's own count and a capped
naive linear extrapolation). The practically important, somewhat counterintuitive finding: because
final sparse-matrix memory is dominated by nonzero COUNT (which is nearly flat across k=15-31 at
pilot scale) rather than by column count, **k=15 through k=31 are all projected to require a
similarly large (~430-530GB) full-scale matrix** - there is no k in this tested range that is
clearly "cheaper" than another once k reaches approximately 15. Only k=10 is projected
meaningfully cheaper in absolute memory (~44-52GB), at the cost of being nearly fully dense and
(per the earlier CARD-interpretation task) offering weak per-feature sequence specificity.

## Failures or memory-safety aborts

None occurred for any of the six k values at this 100-genome pilot scale. The memory-safety abort
mechanism (`build_matrix_streaming`'s per-genome RSS check) was exercised and verified by unit
tests (see `tests/test_streaming_kmer_build.py`, `tests/test_run_intermediate_k_experiment.py`),
but was never triggered by a real run in this task - this pilot size did not approach the 11GB
safety threshold for any tested k (the closest, k=31, peaked at ~7.6GB).

## Recommendation for the next validation stage


**No k value is recommended as final here** - that decision remains open, consistent with this
task's instructions. What this task's evidence supports:

1. **The earlier k=10-vs-k=31 framing was incomplete**: k=15 is already nearly as memory-expensive
   as k=31 at both pilot and projected full scale, so "smaller k means cheaper" does not hold once
   k passes roughly 15 for this dataset - the real dividing line is k=10 (cheap, dense, weakly
   specific) versus "k>=15" (all similarly expensive, all genuinely sparse), not a smooth gradient.
2. **Concrete next step**: before choosing a final k, evaluate a genuinely different
   dimensionality-control strategy at a still-small, still-bounded scale - e.g. feature hashing
   (fixed-width hashed k-mer counts) or minimizer/sketch-based k-mer selection - since this task's
   evidence suggests that simply moving the k dial within 15-31 will not, by itself, produce a
   full-scale-feasible matrix.
3. **If a raw canonical-k-mer approach is still preferred**, k=10 remains the only tested
   configuration with a projected full-scale memory footprint under 100GB - any choice of k>=15
   should be paired with an explicit plan for out-of-core or distributed matrix construction, not
   assumed to run in memory on commodity hardware.

This recommendation is itself a judgment call balancing feasibility evidence against the
information-content concerns raised elsewhere in this project (k=10's weak CARD specificity) - a
reviewer could reasonably prioritize differently.


## Limitations

- 100 genomes, one antibiotic, one train/test split per evaluation strategy: none of the
  comparisons above establish generalization, and the same small-pilot overfitting risk already
  documented for k=10/k=31 applies identically to every intermediate k tested here.
- Full-scale memory projections assume nnz-per-genome and column-count growth rates measured from
  THIS 100-genome cohort continue to hold at ~70-90x the genome count - a real but untested
  assumption, explicitly flagged (not verified) in `resource_estimation.py`'s own rationale text.
- No feature-selection, hashing, or out-of-core strategy was evaluated in this task - only raw
  canonical k-mer counts at each k, exactly as already used for k=10/k=31.
