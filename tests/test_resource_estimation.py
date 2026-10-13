from bvbrc_recon.resource_estimation import (
    estimate_fasta_storage_bytes,
    estimate_matrix_resources,
    estimate_vocabulary_scaling,
    theoretical_max_canonical_kmers,
)


def test_theoretical_max_even_k_includes_palindrome_correction():
    # k=10: (4**10 + 4**5) // 2 = (1048576 + 1024) // 2 = 524800
    assert theoretical_max_canonical_kmers(10) == 524_800


def test_theoretical_max_odd_k_no_palindrome_correction():
    assert theoretical_max_canonical_kmers(9) == 4**9 // 2


def test_vocabulary_scaling_detects_saturated_regime():
    # mirrors the actual measured pilot: k=10 reaches 524346/524800 = 99.9%
    estimate = estimate_vocabulary_scaling(
        k=10, pilot_n_features=524_346, pilot_n_genomes=100, target_n_genomes=7307
    )
    assert estimate.regime == "saturated"
    assert estimate.estimated_n_features_low == 524_346
    assert estimate.estimated_n_features_high == theoretical_max_canonical_kmers(10)
    assert estimate.saturation_fraction > 0.95


def test_vocabulary_scaling_detects_unsaturated_regime():
    # a deliberately tiny fraction of the k=31 space
    small_fraction_n_features = 10_000_000
    estimate = estimate_vocabulary_scaling(
        k=31, pilot_n_features=small_fraction_n_features, pilot_n_genomes=100, target_n_genomes=7307
    )
    assert estimate.regime == "unsaturated"
    assert estimate.estimated_n_features_low == small_fraction_n_features
    assert estimate.estimated_n_features_high > estimate.estimated_n_features_low


def test_vocabulary_scaling_unsaturated_high_bound_is_capped_at_theoretical_max():
    # construct a case where naive linear extrapolation would exceed the ceiling
    estimate = estimate_vocabulary_scaling(
        k=9, pilot_n_features=1000, pilot_n_genomes=1, target_n_genomes=10_000_000
    )
    assert estimate.estimated_n_features_high <= theoretical_max_canonical_kmers(9)


def test_vocabulary_scaling_saturation_fraction_computed_correctly():
    estimate = estimate_vocabulary_scaling(k=10, pilot_n_features=262_400, pilot_n_genomes=50, target_n_genomes=100)
    assert abs(estimate.saturation_fraction - 0.5) < 1e-9


def test_estimate_matrix_resources_nnz_scales_linearly_with_genome_count():
    vocab = estimate_vocabulary_scaling(k=10, pilot_n_features=524_346, pilot_n_genomes=100, target_n_genomes=7307)
    estimate = estimate_matrix_resources(
        vocabulary=vocab, pilot_n_genomes=100, pilot_nnz=49_668_801, target_n_genomes=7307
    )
    expected_nnz = int(49_668_801 / 100 * 7307)
    assert estimate.estimated_nnz == expected_nnz


def test_estimate_matrix_resources_memory_accounts_for_csr_overhead():
    vocab = estimate_vocabulary_scaling(k=10, pilot_n_features=524_346, pilot_n_genomes=100, target_n_genomes=200)
    estimate = estimate_matrix_resources(vocabulary=vocab, pilot_n_genomes=100, pilot_nnz=1000, target_n_genomes=200)
    # nnz doubles to 2000; memory = 2000*(8+4) + 201*4
    assert estimate.estimated_csr_memory_bytes == 2000 * 12 + 201 * 4


def test_estimate_matrix_resources_npz_bounds_are_ordered():
    vocab = estimate_vocabulary_scaling(k=10, pilot_n_features=524_346, pilot_n_genomes=100, target_n_genomes=7307)
    estimate = estimate_matrix_resources(vocabulary=vocab, pilot_n_genomes=100, pilot_nnz=49_668_801, target_n_genomes=7307)
    assert estimate.estimated_npz_storage_bytes_low <= estimate.estimated_npz_storage_bytes_high


def test_estimate_fasta_storage_scales_linearly():
    assert estimate_fasta_storage_bytes(7307, 4_950_000.0) == int(7307 * 4_950_000.0)


def test_estimate_fasta_storage_zero_genomes_is_zero():
    assert estimate_fasta_storage_bytes(0, 4_950_000.0) == 0


# --- scale_vocabulary_by_measured_ratio ------------------------------------


def test_scale_vocabulary_by_measured_ratio_applies_ratio_to_both_bounds():
    from bvbrc_recon.resource_estimation import scale_vocabulary_by_measured_ratio

    anchor = estimate_vocabulary_scaling(k=31, pilot_n_features=32_648_927, pilot_n_genomes=100, target_n_genomes=7307)
    result = scale_vocabulary_by_measured_ratio(anchor, ratio=0.1474, label="minimizer w=15")
    assert result.estimated_low == int(anchor.estimated_n_features_low * 0.1474)
    assert result.estimated_high == int(anchor.estimated_n_features_high * 0.1474)
    assert result.anchor_regime == anchor.regime


def test_scale_vocabulary_by_measured_ratio_base_is_geometric_mean():
    from bvbrc_recon.resource_estimation import scale_vocabulary_by_measured_ratio

    anchor = estimate_vocabulary_scaling(k=31, pilot_n_features=32_648_927, pilot_n_genomes=100, target_n_genomes=7307)
    result = scale_vocabulary_by_measured_ratio(anchor, ratio=0.5, label="test")
    assert result.estimated_base == int((result.estimated_low * result.estimated_high) ** 0.5)


def test_scale_vocabulary_by_measured_ratio_orders_low_base_high():
    from bvbrc_recon.resource_estimation import scale_vocabulary_by_measured_ratio

    anchor = estimate_vocabulary_scaling(k=31, pilot_n_features=32_648_927, pilot_n_genomes=100, target_n_genomes=7307)
    result = scale_vocabulary_by_measured_ratio(anchor, ratio=0.1474, label="minimizer w=15")
    assert result.estimated_low <= result.estimated_base <= result.estimated_high


def test_scale_vocabulary_by_measured_ratio_zero_ratio_gives_zero():
    from bvbrc_recon.resource_estimation import scale_vocabulary_by_measured_ratio

    anchor = estimate_vocabulary_scaling(k=31, pilot_n_features=32_648_927, pilot_n_genomes=100, target_n_genomes=7307)
    result = scale_vocabulary_by_measured_ratio(anchor, ratio=0.0, label="degenerate")
    assert result.estimated_low == 0
    assert result.estimated_high == 0
    assert result.estimated_base == 0


# --- estimate_construction_overhead_bytes ----------------------------------


def test_construction_overhead_scales_proportionally_with_target_size():
    from bvbrc_recon.resource_estimation import estimate_construction_overhead_bytes

    # pilot: peak RSS 2x the final CSR size -> overhead ratio = 1.0
    overhead = estimate_construction_overhead_bytes(
        pilot_peak_rss_bytes=200.0, pilot_csr_bytes=100.0, target_csr_bytes=1000.0
    )
    assert overhead == 1000.0  # ratio 1.0 applied to target


def test_construction_overhead_zero_when_peak_rss_below_csr_size():
    from bvbrc_recon.resource_estimation import estimate_construction_overhead_bytes

    overhead = estimate_construction_overhead_bytes(
        pilot_peak_rss_bytes=50.0, pilot_csr_bytes=100.0, target_csr_bytes=1000.0
    )
    assert overhead == 0.0


def test_construction_overhead_handles_zero_pilot_csr_bytes():
    from bvbrc_recon.resource_estimation import estimate_construction_overhead_bytes

    overhead = estimate_construction_overhead_bytes(
        pilot_peak_rss_bytes=100.0, pilot_csr_bytes=0.0, target_csr_bytes=1000.0
    )
    assert overhead == 0.0


# --- estimate_processing_seconds -------------------------------------------


def test_estimate_processing_seconds_scales_linearly():
    from bvbrc_recon.resource_estimation import estimate_processing_seconds

    seconds = estimate_processing_seconds(pilot_seconds=400.0, pilot_n_genomes=100, target_n_genomes=7307)
    assert seconds == 400.0 / 100 * 7307


def test_estimate_processing_seconds_zero_pilot_genomes_is_zero():
    from bvbrc_recon.resource_estimation import estimate_processing_seconds

    assert estimate_processing_seconds(pilot_seconds=400.0, pilot_n_genomes=0, target_n_genomes=7307) == 0.0
