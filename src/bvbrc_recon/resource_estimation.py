"""Resource-estimation formulas for scaling the pilot's measured k-mer
feature-matrix characteristics up to a full-dataset genome count.

The central, non-obvious fact this module encodes (measured, not
assumed): the number of DISTINCT canonical k-mers (matrix COLUMNS) does
NOT scale with genome count the way the number of ROWS does. At small k,
the canonical k-mer space has a hard ceiling (`theoretical_max_canonical_kmers`),
and a single bacterial genome can already come close to saturating it -
adding more genomes mostly adds more nonzero entries to existing columns,
not new columns. At large k, the space is practically unbounded, so
column count grows roughly with total sequence data instead. This module
never assumes one behavior without checking the pilot's own measured
saturation fraction first.
"""

from __future__ import annotations

from dataclasses import dataclass

SATURATION_THRESHOLD = 0.95  # documented judgment call, not a measured constant


def theoretical_max_canonical_kmers(k: int) -> int:
    if k % 2 == 0:
        return (4**k + 4 ** (k // 2)) // 2
    return 4**k // 2


@dataclass
class VocabularyScalingEstimate:
    k: int
    regime: str  # "saturated" or "unsaturated"
    pilot_n_features: int
    theoretical_max: int
    saturation_fraction: float
    estimated_n_features_low: int
    estimated_n_features_high: int
    rationale: str


def estimate_vocabulary_scaling(
    k: int,
    pilot_n_features: int,
    pilot_n_genomes: int,
    target_n_genomes: int,
) -> VocabularyScalingEstimate:
    """Returns a [low, high] bound on full-dataset column count, never a
    single point estimate from a naive linear assumption. If the pilot has
    already saturated the theoretical k-mer space, both bounds equal the
    (fixed) theoretical max - more genomes cannot add more columns. If not
    saturated, the low bound assumes NO new columns appear (conservative)
    and the high bound assumes new genomes contribute new columns at the
    SAME per-genome rate observed in the pilot, capped at the theoretical
    max - a crude, explicitly-labeled upper bound, not a prediction.
    """
    theoretical_max = theoretical_max_canonical_kmers(k)
    saturation_fraction = pilot_n_features / theoretical_max

    if saturation_fraction >= SATURATION_THRESHOLD:
        return VocabularyScalingEstimate(
            k=k,
            regime="saturated",
            pilot_n_features=pilot_n_features,
            theoretical_max=theoretical_max,
            saturation_fraction=saturation_fraction,
            estimated_n_features_low=pilot_n_features,
            estimated_n_features_high=theoretical_max,
            rationale=(
                f"The {pilot_n_genomes}-genome pilot already reaches {saturation_fraction:.1%} of the "
                f"theoretical max canonical {k}-mer space ({pilot_n_features:,}/{theoretical_max:,}) - "
                f"at or above the {SATURATION_THRESHOLD:.0%} saturation threshold used here. Scaling to "
                f"{target_n_genomes:,} genomes is expected to add very few new columns; the pilot's own "
                f"count is used as the realistic estimate, with the fixed theoretical ceiling as the "
                f"absolute upper bound."
            ),
        )

    naive_linear = min(
        int(pilot_n_features * (target_n_genomes / pilot_n_genomes)), theoretical_max
    )
    return VocabularyScalingEstimate(
        k=k,
        regime="unsaturated",
        pilot_n_features=pilot_n_features,
        theoretical_max=theoretical_max,
        saturation_fraction=saturation_fraction,
        estimated_n_features_low=pilot_n_features,
        estimated_n_features_high=naive_linear,
        rationale=(
            f"The {pilot_n_genomes}-genome pilot reaches only {saturation_fraction:.1%} of the "
            f"theoretical max canonical {k}-mer space ({pilot_n_features:,}/{theoretical_max:,}) - "
            f"below the {SATURATION_THRESHOLD:.0%} saturation threshold. The true column count at "
            f"{target_n_genomes:,} genomes is genuinely uncertain from a {pilot_n_genomes}-genome "
            f"pilot alone: the low bound assumes no new genome contributes any new k-mer (almost "
            f"certainly an underestimate), and the high bound is a naive linear extrapolation at the "
            f"pilot's own per-genome rate, capped at the theoretical ceiling (almost certainly an "
            f"overestimate, since cross-genome redundancy increases with more genomes of the same "
            f"species). The real value lies somewhere between, and only measuring a larger cohort "
            f"can narrow this."
        ),
    )


@dataclass
class MatrixResourceEstimate:
    estimated_nnz: int
    estimated_csr_memory_bytes: int
    estimated_npz_storage_bytes_low: int
    estimated_npz_storage_bytes_high: int
    vocabulary: VocabularyScalingEstimate


def estimate_matrix_resources(
    vocabulary: VocabularyScalingEstimate,
    pilot_n_genomes: int,
    pilot_nnz: int,
    target_n_genomes: int,
    value_itemsize: int = 8,
    index_itemsize: int = 4,
) -> MatrixResourceEstimate:
    """nnz (nonzero COUNT) is assumed to scale linearly with genome count -
    unlike column count, this is a reasonable assumption because each
    genome independently contributes roughly its own characteristic
    nonzero count regardless of how many other genomes are present (this
    is NOT re-verified here against multiple pilot sizes - it rests on the
    single 100-genome measurement, stated as an assumption, not a fact
    established by this analysis).
    """
    mean_nnz_per_genome = pilot_nnz / pilot_n_genomes
    estimated_nnz = int(mean_nnz_per_genome * target_n_genomes)

    # CSR storage: nnz values + nnz column indices + (n_rows+1) row pointers.
    estimated_csr_memory_bytes = (
        estimated_nnz * (value_itemsize + index_itemsize) + (target_n_genomes + 1) * index_itemsize
    )

    return MatrixResourceEstimate(
        estimated_nnz=estimated_nnz,
        estimated_csr_memory_bytes=estimated_csr_memory_bytes,
        estimated_npz_storage_bytes_low=estimated_csr_memory_bytes // 3,  # .npz compresses; crude floor
        estimated_npz_storage_bytes_high=estimated_csr_memory_bytes,
        vocabulary=vocabulary,
    )


def estimate_fasta_storage_bytes(target_n_genomes: int, measured_mean_bytes_per_genome: float) -> int:
    """FASTA storage scales linearly with genome COUNT - genome length is
    a property of the organism, not of how many other genomes are in the
    collection, so this (unlike vocabulary size) is a reasonable linear
    assumption from the pilot's own measured mean."""
    return int(target_n_genomes * measured_mean_bytes_per_genome)


@dataclass
class RatioScaledEstimate:
    """A vocabulary-size projection for a DERIVED representation (e.g. a
    minimizer scheme), obtained by scaling an already-computed anchor
    representation's [low, high] bounds (e.g. raw k=31's own
    `VocabularyScalingEstimate`) by a measured ratio between the two
    representations' PILOT vocabulary sizes. Used when the derived
    representation has no independent theoretical ceiling of its own to
    reason about (minimizer selection is governed by window dynamics,
    not simply by k, so `theoretical_max_canonical_kmers` does not apply
    to it directly - unlike raw canonical k-mers, where that ceiling is
    the right anchor)."""

    label: str
    ratio: float
    anchor_regime: str
    anchor_low: int
    anchor_high: int
    estimated_low: int
    estimated_base: int
    estimated_high: int
    rationale: str


def scale_vocabulary_by_measured_ratio(
    anchor: VocabularyScalingEstimate, ratio: float, label: str
) -> RatioScaledEstimate:
    """`ratio` = derived_pilot_vocabulary / anchor_pilot_vocabulary, measured
    directly from the pilot (e.g. a minimizer scheme's vocabulary divided
    by the raw-k vocabulary it was derived from). Assumes this ratio
    itself holds at full scale - an explicit, stated assumption, not a
    fact verified across multiple cohort sizes; `estimated_base` is the
    geometric mean of low/high, a presentation convenience, not an
    independently derived midpoint."""
    low = int(anchor.estimated_n_features_low * ratio)
    high = int(anchor.estimated_n_features_high * ratio)
    base = int((low * high) ** 0.5) if low > 0 and high > 0 else low
    return RatioScaledEstimate(
        label=label,
        ratio=ratio,
        anchor_regime=anchor.regime,
        anchor_low=anchor.estimated_n_features_low,
        anchor_high=anchor.estimated_n_features_high,
        estimated_low=low,
        estimated_base=base,
        estimated_high=high,
        rationale=(
            f"Derived from the anchor representation's own [{anchor.estimated_n_features_low:,}, "
            f"{anchor.estimated_n_features_high:,}] projection ({anchor.regime} regime) by applying "
            f"the pilot-measured ratio {ratio:.4f} ({label} vocabulary / anchor vocabulary at "
            f"100 genomes). This assumes the ratio observed at pilot scale holds at full scale - "
            f"not verified, since only one pilot size (100 genomes) exists to measure it from."
        ),
    )


def estimate_construction_overhead_bytes(
    pilot_peak_rss_bytes: float, pilot_csr_bytes: float, target_csr_bytes: float
) -> float:
    """Projects the memory used DURING construction beyond the final CSR
    matrix's own size (e.g. the vocabulary dict, COO staging lists),
    assuming this overhead scales proportionally with the final matrix
    size - a simplifying assumption, not verified across multiple pilot
    sizes (only one pilot measurement exists)."""
    if pilot_csr_bytes <= 0:
        return 0.0
    overhead_ratio = max(pilot_peak_rss_bytes - pilot_csr_bytes, 0.0) / pilot_csr_bytes
    return overhead_ratio * target_csr_bytes


def estimate_processing_seconds(pilot_seconds: float, pilot_n_genomes: int, target_n_genomes: int) -> float:
    """Linear extrapolation of measured wall-clock time by genome count -
    reasonable because per-genome extraction cost is driven mainly by
    that genome's own sequence length, which does not depend on how many
    other genomes are processed alongside it."""
    if pilot_n_genomes <= 0:
        return 0.0
    return pilot_seconds / pilot_n_genomes * target_n_genomes
