# Feature Hashing as a Memory-Controlled Alternative to Raw Canonical k-mers

- Analysis date (UTC): 2026-10-10T12:43:42.842999+00:00
- Scope: uses ONLY the EXISTING raw k=15 pilot matrix (100 ciprofloxacin genomes, already built and
  verified unchanged by this task) as input. No genome was re-processed, no additional genome was
  downloaded, and the raw k=15/k=10/k=31 matrices and the saved splits were not modified.
- **No final feature representation (raw or hashed) is approved for the full benchmark as a result
  of this task.** That decision remains open.

## Methodology

- Feature hashing is implemented as a DOWNSTREAM transformation of the existing, already-correct
  raw k=15 canonical k-mer matrix (`bvbrc_recon.feature_hashing.build_hashed_representation`) -
  not a reimplementation of k-mer counting. Canonicalization, reverse-complement collapsing,
  ambiguous-base skipping, and contig-boundary handling are therefore inherited unchanged from the
  already-tested `streaming_kmer_build.py` output used to build that raw matrix.
- Hash function: `hashlib.blake2b` (8-byte digest) modulo the target dimension - deterministic
  across processes (verified by a test that runs the hash function in two separate subprocesses),
  unlike Python's built-in `hash()`, which is randomized per-process by `PYTHONHASHSEED`.
- Bucket values are the UNSIGNED SUM of every raw k=15-mer's count that hashes into that bucket (no
  sign-flipping, unlike scikit-learn's signed `FeatureHasher`) - a documented design choice for
  interpretability, not a measured requirement.
- Three fixed dimensions evaluated: 2^16 (65,536), 2^18 (262,144), 2^20 (1,048,576) columns.
- Logistic Regression only, identical fixed configuration already used throughout this project
  (`max_iter=1000, random_state=20261009`), evaluated on the EXISTING saved random and lineage
  splits - neither regenerated, no hyperparameter tuned against any test result.

## Resource measurements (measured, 100 genomes)

| Dimension | Genomes | Columns | nnz | Density | Build time | Peak RSS | npz storage | In-memory CSR |
|---|---|---|---|---|---|---|---|---|
| 2^16=65,536 | 100 | 65,536 | 6,553,600 | 100.00% | 44.3s | 5670 MB | 16.2 MB | 78.6 MB |
| 2^18=262,144 | 100 | 262,144 | 26,214,400 | 100.00% | 51.8s | 7837 MB | 59.9 MB | 314.6 MB |
| 2^20=1,048,576 | 100 | 1,048,576 | 103,921,597 | 99.11% | 56.5s | 9292 MB | 217.2 MB | 1247.1 MB |


**Central finding (measured, not assumed)**: at d=2^16 and d=2^18, the hashed matrix is
**100% dense** - every one of the 65,536 (or 262,144) buckets is touched by every one of the 100
genomes. At d=2^20, density is still 99.1%. The sparsity benefit often associated with feature
hashing does NOT materialize at these dimensions for this dataset, because each genome already
contributes close to 5 million distinct raw k-mers - far more than even the largest tested bucket
count - so collisions guarantee near-complete bucket occupancy regardless of dimension. Hashing
here is better understood as a fixed-width LOSSY SUMMARIZATION of the raw matrix than as a sparsity
technique.


## Collision diagnostics (measured exactly from the raw k=15 vocabulary)

| Dimension | Raw k=15 features | Colliding buckets | Lonely buckets | Empty buckets | Max distinct k-mers/bucket | Mean distinct k-mers/occupied bucket |
|---|---|---|---|---|---|---|
| 2^16=65,536 | 23,171,787 | 65,536/65,536 | 0 | 0 | 437 | 353.57 |
| 2^18=262,144 | 23,171,787 | 262,144/262,144 | 0 | 0 | 132 | 88.39 |
| 2^20=1,048,576 | 23,171,787 | 1,048,576/1,048,576 | 0 | 0 | 46 | 22.10 |

**Every single bucket at every tested dimension contains more than one distinct raw k-mer** (zero
"lonely" buckets, i.e. zero buckets holding exactly one distinct raw k-mer, at any dimension
tested) - collisions are the norm here, not an edge case, given 23.17M raw k=15-mers mapped into
at most ~1M buckets. This is a measured hash-collision rate, distinct from (and not to be confused
with) the ordinary, expected biological similarity of two different genomes sharing the same raw
k-mer - that kind of overlap is already correctly represented in the raw matrix before hashing
ever happens, and is not what "collision" means in this section.

## Model results: hashed representations vs. existing raw k-mer results

| Representation | Split | Accuracy | ROC-AUC | Avg. precision | Recall (R) | Recall (S) | Confusion matrix |
|---|---|---|---|---|---|---|---|
| hashed d=2^16 | random | 0.9 | 0.9751552795031055 | 0.9367346938775509 | 0.5714285714285714 | 1.0 | [[23, 0], [3, 4]] |
| hashed d=2^16 | lineage | 0.6551724137931034 | 0.8311688311688311 | 0.6529238618524333 | 0.8571428571428571 | 0.5909090909090909 | [[13, 9], [1, 6]] |
| hashed d=2^18 | random | 0.9 | 0.937888198757764 | 0.8755411255411254 | 0.5714285714285714 | 1.0 | [[23, 0], [3, 4]] |
| hashed d=2^18 | lineage | 0.7586206896551724 | 0.8571428571428572 | 0.8202614379084967 | 0.7142857142857143 | 0.7727272727272727 | [[17, 5], [2, 5]] |
| hashed d=2^20 | random | 0.9 | 0.9503105590062112 | 0.8766233766233766 | 0.5714285714285714 | 1.0 | [[23, 0], [3, 4]] |
| hashed d=2^20 | lineage | 0.7586206896551724 | 0.9025974025974026 | 0.8278388278388278 | 0.7142857142857143 | 0.7727272727272727 | [[17, 5], [2, 5]] |
| | | | | | | | |
| raw k=10 | random | 0.9 | 0.9627329192546584 | - | - | - | - |
| raw k=10 | lineage | 0.7931034482758621 | 0.8896103896103896 | - | - | - | - |
| raw k=15 | random | 0.9 | 0.9440993788819876 | 0.8833333333333333 | 0.5714285714285714 | 1.0 | [[23, 0], [3, 4]] |
| raw k=15 | lineage | 0.896551724137931 | 0.9155844155844155 | 0.876984126984127 | 0.7142857142857143 | 0.9545454545454546 | [[21, 1], [2, 5]] |
| raw k=31 | random | 0.9 | 0.9316770186335404 | - | - | - | - |
| raw k=31 | lineage | 0.8275862068965517 | 0.8766233766233766 | - | - | - | - |

**These are exploratory results from the SAME 100-genome pilot reused across every representation
tested in this entire project so far - no representation (raw or hashed, any k, any dimension) is
declared superior on the strength of pilot test scores alone.**

## Comparison with raw k-mer representations

- **Storage**: hashing gives a real, substantial storage reduction versus raw k=15 (710MB) - 16MB
  (d=2^16), 60MB (d=2^18), 217MB (d=2^20) - because total nonzero count drops as columns are folded
  together, even though per-matrix density rises.
- **Build time**: hashing (45-56s, reusing the already-extracted raw matrix) is far faster than raw
  extraction at any k (450-560s) - because it is a pure column-remapping operation on data already
  in memory, not a re-walk of every genome's sequence.
- **Peak memory**: comparable to or higher than the already-loaded raw matrix itself (5.7-9.3GB,
  including the ~4.7GB already used just to load the raw k=15 matrix) - hashing does NOT reduce
  peak in-process memory versus the raw representation it is derived from, since the raw matrix
  must still be held in memory to build any hashed version of it from the existing output; it
  reduces the SERIALIZED and final-matrix size, not the memory needed to produce it from an
  already-built raw matrix this way.
- **Interpretability loss**: with zero lonely buckets at every dimension tested, a hashed column's
  signal is a specific, measured kind of ambiguous (an average of 22-354 distinct canonical
  k-mers, depending on dimension) - a trained model's hashed-feature coefficient cannot be
  attributed back to a single biological k-mer, unlike every raw k-mer representation already used
  in this project (including the CARD-matching work, which depended on exactly this
  attributability and would not be possible on a hashed representation without unhashing, which is
  not implemented here).

## Limitations

- 100 genomes, one antibiotic, one split per evaluation strategy, exactly as for every other
  representation compared in this project - none of these numbers establish generalization.
- Hashing was evaluated starting from the raw k=15 matrix only, per the task's instruction - k=10
  or k=31 as the pre-hashing raw representation was not evaluated, so this comparison does not
  show whether hashing a different base k would behave similarly.
- Feature hashing explicitly introduces information-destroying collisions (measured above, not
  estimated) and, on its own, establishes neither biological validity of any resulting feature nor
  lineage-independent generalization - the lineage-split numbers above are measured outcomes on
  this one split, not evidence of robustness to lineage effects in general.
- Peak RSS during this experiment (up to 9.3GB) is closer to the 11GB safety threshold already
  established in prior tasks than any previously-tested configuration except raw k=31 - full-scale
  behavior of this same approach (hashing a full-scale raw matrix, which does not yet exist) is not
  measured here and should not be assumed safe without its own resource check.

## Recommendation for the next validation stage


**No representation is approved as final here.** This task's evidence narrows the decision rather
than resolving it:

1. **Feature hashing at these dimensions trades information for a fixed, bounded size.** At
   d=2^16, every bucket collides and lineage-split accuracy (0.655) is the worst result measured
   anywhere in this project so far - a real cost, not a hypothetical one. d=2^20 recovers most of
   the lineage-split performance seen with raw k=15 (0.759 vs 0.897) while remaining nearly fully
   dense, suggesting the useful dimension (if hashing is pursued) is likely well above 2^20 for
   this k, not below it.
2. **Concrete next step**: if feature hashing is pursued further, test larger dimensions
   (e.g. 2^22-2^24) and/or a smaller raw k before hashing (not just k=15), to find where hashed
   density actually drops meaningfully below 100% - this task deliberately stayed within the
   dimensions specified and did not explore beyond them.
3. **Alternative worth comparing directly**: a minimizer- or sketch-based k-mer selection (reducing
   the RAW vocabulary before any matrix is built, rather than folding an already-large raw
   vocabulary into fewer columns after the fact) may avoid the "guaranteed collision" regime seen
   here entirely - not evaluated in this task.

This remains a judgment call weighing storage/build-time savings (real and significant) against
collision-driven information loss and loss of CARD-style interpretability (also real and
measured) - a reviewer could reasonably prioritize either differently.

