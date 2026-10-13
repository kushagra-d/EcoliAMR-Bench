"""Renders reports/feature_hashing_comparison.md."""

from __future__ import annotations


def _resource_table(hash_configs: dict[int, dict]) -> str:
    lines = [
        "| Dimension | Genomes | Columns | nnz | Density | Build time | Peak RSS | npz storage | In-memory CSR |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for d in sorted(hash_configs):
        c = hash_configs[d]
        lines.append(
            f"| 2^{d.bit_length()-1}={d:,} | {c['n_genomes']} | {c['n_features']:,} | {c['nnz']:,} | "
            f"{c['density']:.2%} | {c['build_seconds']:.1f}s | {c['peak_rss_mb']:.0f} MB | "
            f"{c['npz_storage_bytes']/1e6:.1f} MB | {c['in_memory_csr_bytes']/1e6:.1f} MB |"
        )
    return "\n".join(lines)


def _collision_table(hash_configs: dict[int, dict]) -> str:
    lines = [
        "| Dimension | Raw k=15 features | Colliding buckets | Lonely buckets | Empty buckets | "
        "Max distinct k-mers/bucket | Mean distinct k-mers/occupied bucket |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in sorted(hash_configs):
        col = hash_configs[d]["collisions"]
        lines.append(
            f"| 2^{d.bit_length()-1}={d:,} | {col['n_raw_features']:,} | "
            f"{col['n_colliding_buckets']:,}/{col['n_buckets']:,} | {col['n_lonely_buckets']:,} | "
            f"{col['n_empty_buckets']:,} | {col['max_distinct_kmers_in_one_bucket']:,} | "
            f"{col['mean_distinct_kmers_per_occupied_bucket']:.2f} |"
        )
    return "\n".join(lines)


def _model_table(hash_configs: dict[int, dict], raw_comparison: dict) -> str:
    lines = [
        "| Representation | Split | Accuracy | ROC-AUC | Avg. precision | Recall (R) | Recall (S) | Confusion matrix |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in sorted(hash_configs):
        for split_name in ("random", "lineage"):
            r = hash_configs[d]["model_results"][split_name]
            lines.append(
                f"| hashed d=2^{d.bit_length()-1} | {split_name} | {r['accuracy']} | {r['roc_auc']} | "
                f"{r['average_precision']} | {r['recall_per_class'].get('Resistant')} | "
                f"{r['recall_per_class'].get('Susceptible')} | {r['confusion_matrix']} |"
            )
    lines.append("| | | | | | | | |")
    for k in sorted(raw_comparison):
        for split_name in ("random", "lineage"):
            r = raw_comparison[k].get(split_name)
            if r is None:
                continue
            lines.append(
                f"| raw k={k} | {split_name} | {r['accuracy']} | {r['roc_auc']} | "
                f"{r.get('average_precision', '-')} | {r.get('recall_per_class', {}).get('Resistant', '-')} | "
                f"{r.get('recall_per_class', {}).get('Susceptible', '-')} | {r.get('confusion_matrix', '-')} |"
            )
    return "\n".join(lines)


def render_feature_hashing_report(
    analysis_date: str,
    hash_configs: dict[int, dict],
    raw_comparison: dict,
    narrative: str,
    recommendation: str,
) -> str:
    return f"""# Feature Hashing as a Memory-Controlled Alternative to Raw Canonical k-mers

- Analysis date (UTC): {analysis_date}
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

{_resource_table(hash_configs)}

{narrative}

## Collision diagnostics (measured exactly from the raw k=15 vocabulary)

{_collision_table(hash_configs)}

**Every single bucket at every tested dimension contains more than one distinct raw k-mer** (zero
"lonely" buckets, i.e. zero buckets holding exactly one distinct raw k-mer, at any dimension
tested) - collisions are the norm here, not an edge case, given 23.17M raw k=15-mers mapped into
at most ~1M buckets. This is a measured hash-collision rate, distinct from (and not to be confused
with) the ordinary, expected biological similarity of two different genomes sharing the same raw
k-mer - that kind of overlap is already correctly represented in the raw matrix before hashing
ever happens, and is not what "collision" means in this section.

## Model results: hashed representations vs. existing raw k-mer results

{_model_table(hash_configs, raw_comparison)}

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

{recommendation}
"""
