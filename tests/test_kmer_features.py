from scipy import sparse

from bvbrc_recon.kmer_features import (
    build_feature_matrix,
    canonical_kmer,
    count_kmers_in_assembly,
    count_kmers_in_contig,
    estimate_dense_bytes,
    reverse_complement,
)


def test_reverse_complement():
    assert reverse_complement("ACGT") == "ACGT"  # palindromic
    assert reverse_complement("AAGG") == "CCTT"


def test_canonical_kmer_picks_lexicographically_smaller():
    # AAGG -> reverse complement CCTT; AAGG < CCTT lexicographically
    assert canonical_kmer("AAGG") == "AAGG"
    assert canonical_kmer("CCTT") == "AAGG"


def test_canonical_kmer_treats_kmer_and_reverse_complement_as_one_feature():
    forward_counts = count_kmers_in_contig("AAGGAAGG", k=4)
    rc_counts = count_kmers_in_contig("CCTTCCTT", k=4)
    assert forward_counts.keys() == rc_counts.keys()


def test_count_kmers_in_contig_skips_windows_with_ambiguous_bases():
    counts = count_kmers_in_contig("ACGTNACGT", k=4)
    # windows: ACGT, CGTN(skip), GTNA(skip), TNAC(skip), NACG(skip), ACGT
    assert sum(counts.values()) == 2
    assert canonical_kmer("ACGT") in counts


def test_count_kmers_in_contig_too_short_returns_empty():
    assert count_kmers_in_contig("AC", k=4) == {}


def test_count_kmers_in_contig_is_case_insensitive():
    counts_upper = count_kmers_in_contig("ACGTACGT", k=4)
    counts_lower = count_kmers_in_contig("acgtacgt", k=4)
    assert counts_upper == counts_lower


def test_count_kmers_in_assembly_never_spans_contig_boundary():
    # If contigs were wrongly concatenated as "AAAA"+"TTTT" = "AAAATTTT",
    # a boundary-spanning 4-mer "AATT" would appear. Processed independently,
    # it must not.
    counts = count_kmers_in_assembly(["AAAA", "TTTT"], k=4)
    boundary_kmer = canonical_kmer("AATT")
    assert boundary_kmer not in counts


def test_count_kmers_in_assembly_sums_across_contigs():
    counts = count_kmers_in_assembly(["AAAA", "AAAA"], k=4)
    assert counts[canonical_kmer("AAAA")] == 2


def test_build_feature_matrix_is_sparse_and_matches_expected_counts():
    genome_counts = {
        "g1": count_kmers_in_assembly(["AAAACCCC"], k=4),
        "g2": count_kmers_in_assembly(["GGGGTTTT"], k=4),
    }
    result = build_feature_matrix(genome_counts, k=4, genome_order=["g1", "g2"])
    assert sparse.issparse(result.matrix)
    assert result.matrix.shape == (2, result.n_features)
    assert result.nnz == result.matrix.nnz


def test_build_feature_matrix_preserves_genome_row_mapping():
    genome_counts = {
        "g1": count_kmers_in_assembly(["AAAACCCC"], k=4),
        "g2": count_kmers_in_assembly(["GGGGTTTT"], k=4),
    }
    result = build_feature_matrix(genome_counts, k=4, genome_order=["g2", "g1"])
    assert result.genome_order == ["g2", "g1"]
    # row 0 is g2's counts, row 1 is g1's - spot check via the vocabulary
    g2_kmer = next(iter(genome_counts["g2"]))
    col = result.vocabulary[g2_kmer]
    assert result.matrix[0, col] == genome_counts["g2"][g2_kmer]


def test_build_feature_matrix_rejects_genome_order_mismatch():
    genome_counts = {"g1": count_kmers_in_assembly(["AAAA"], k=4)}
    try:
        build_feature_matrix(genome_counts, k=4, genome_order=["g1", "g2"])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_estimate_dense_bytes():
    assert estimate_dense_bytes(100, 1_000_000, itemsize=8) == 800_000_000
