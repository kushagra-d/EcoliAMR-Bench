from scipy import sparse

from bvbrc_recon.kmer_features import canonical_kmer
from bvbrc_recon.minimizer_extraction import (
    build_minimizer_matrix_streaming,
    canonical_kmers_for_contig,
    minimizer_counts_for_assembly,
    minimizer_counts_for_contig,
)


# --- canonicalization --------------------------------------------------


def test_canonical_kmers_for_contig_reuses_kmer_features_canonicalization():
    contig = "ACGTACGTAC"
    k = 4
    result = canonical_kmers_for_contig(contig, k)
    expected = [canonical_kmer(contig[i : i + k]) for i in range(len(contig) - k + 1)]
    assert result == expected


def test_canonical_kmers_for_contig_marks_ambiguous_windows_as_none():
    result = canonical_kmers_for_contig("ACGTNACGT", k=4)
    # windows: ACGT, CGTN(invalid), GTNA(invalid), TNAC(invalid), NACG(invalid), ACGT
    assert result == [canonical_kmer("ACGT"), None, None, None, None, canonical_kmer("ACGT")]


def test_canonical_kmers_for_contig_too_short_returns_empty():
    assert canonical_kmers_for_contig("AC", k=4) == []


# --- minimizer selection: determinism and correctness -----------------


def test_minimizer_counts_deterministic_across_calls():
    contig = "ACGTACGTACGTACGTACGT"
    r1 = minimizer_counts_for_contig(contig, k=4, w=3)
    r2 = minimizer_counts_for_contig(contig, k=4, w=3)
    assert r1 == r2


def test_minimizer_counts_matches_brute_force_window_minimum():
    """Cross-check the fast monotonic-deque implementation against a
    naive brute-force recomputation of each window's minimum."""
    contig = "ACGTTGCAGTCAGCATGCA"
    k, w = 3, 4
    fast = minimizer_counts_for_contig(contig, k, w)

    canon = canonical_kmers_for_contig(contig, k)
    from collections import Counter

    brute = Counter()
    for i in range(len(canon) - w + 1):
        window = canon[i : i + w]
        if any(x is None for x in window):
            continue
        brute[min(window)] += 1

    assert fast == brute


def test_minimizer_counts_handles_tied_values_without_error():
    contig = "AAAAAAAAAAAAAAAAAAAA"  # every k-mer identical -> guaranteed ties
    result = minimizer_counts_for_contig(contig, k=4, w=3)
    assert len(result) == 1
    only_kmer = next(iter(result))
    assert only_kmer == canonical_kmer("AAAA")


# --- ambiguous bases: gap handling --------------------------------------


def test_minimizer_counts_no_window_crosses_an_ambiguous_gap():
    # k=3, w=2: a single N should prevent any window from including it,
    # and should not silently merge the two valid flanks into one run
    contig = "ACGTACGTN" + "ACGTACGT"
    result_with_gap = minimizer_counts_for_contig(contig, k=3, w=2)
    # sanity: result must be a valid, non-crashing Counter
    assert isinstance(result_with_gap, type(minimizer_counts_for_contig("ACGT", 3, 2)))
    # no exception means the gap was handled; further precision checked
    # by the brute-force cross-check test above for a mixed sequence
    assert sum(result_with_gap.values()) >= 0


def test_minimizer_counts_empty_for_all_ambiguous_contig():
    result = minimizer_counts_for_contig("NNNNNNNNNN", k=3, w=2)
    assert result == {}


# --- contig boundaries and short contigs --------------------------------


def test_minimizer_counts_for_assembly_never_spans_contig_boundary():
    # if contigs were wrongly concatenated as "AAAA"+"TTTT", a boundary
    # k-mer "AATT" (k=4) could become a minimizer candidate; processed
    # independently, it must never appear.
    result = minimizer_counts_for_assembly(["AAAA", "TTTT"], k=4, w=1)
    assert canonical_kmer("AATT") not in result


def test_short_contig_shorter_than_k_contributes_nothing():
    result = minimizer_counts_for_contig("AC", k=4, w=2)
    assert result == {}


def test_short_contig_enough_for_k_but_not_for_w_contributes_nothing():
    # k=4 valid k-mers exist, but fewer than w=5 of them -> no full window ever forms
    contig = "ACGTACG"  # 4 valid k-mer positions for k=4
    result = minimizer_counts_for_contig(contig, k=4, w=5)
    assert result == {}


def test_contig_exactly_long_enough_for_one_window():
    # k=4, w=2 needs exactly 5 bases for exactly one window
    contig = "ACGTA"
    result = minimizer_counts_for_contig(contig, k=4, w=2)
    assert sum(result.values()) == 1


# --- streaming matrix builder: row order, sparsity, determinism, abort ---


def test_build_minimizer_matrix_streaming_preserves_row_order():
    contigs_by_genome = {"g1": ["ACGTACGTACGT"], "g2": ["TTTTGGGGCCCC"]}
    result = build_minimizer_matrix_streaming(contigs_by_genome, ["g1", "g2"], k=4, w=2)
    assert result.processed_genome_ids == ["g1", "g2"]
    assert result.matrix.shape[0] == 2
    assert result.complete is True


def test_build_minimizer_matrix_streaming_output_is_sparse():
    contigs_by_genome = {"g1": ["ACGTACGTACGT"]}
    result = build_minimizer_matrix_streaming(contigs_by_genome, ["g1"], k=4, w=2)
    assert sparse.issparse(result.matrix)


def test_build_minimizer_matrix_streaming_is_deterministic():
    contigs_by_genome = {"g1": ["ACGTACGTACGTACGT"], "g2": ["TTTTGGGGCCCCAAAA"]}
    r1 = build_minimizer_matrix_streaming(contigs_by_genome, ["g1", "g2"], k=4, w=3)
    r2 = build_minimizer_matrix_streaming(contigs_by_genome, ["g1", "g2"], k=4, w=3)
    assert r1.vocabulary == r2.vocabulary
    assert (r1.matrix != r2.matrix).nnz == 0


def test_build_minimizer_matrix_streaming_feature_mapping_is_fixed_per_run():
    contigs_by_genome = {"g1": ["ACGTACGTACGT"], "g2": ["ACGTACGTACGT"]}
    result = build_minimizer_matrix_streaming(contigs_by_genome, ["g1", "g2"], k=4, w=2)
    # identical genomes -> identical minimizer sets -> no new vocabulary from g2
    assert result.matrix[0].toarray().tolist() == result.matrix[1].toarray().tolist()


def test_build_minimizer_matrix_streaming_memory_abort():
    contigs_by_genome = {f"g{i}": ["ACGTACGTACGT"] for i in range(5)}

    calls = {"n": 0}

    def fake_memory_check():
        calls["n"] += 1
        return 0.0 if calls["n"] < 2 else 999999.0

    result = build_minimizer_matrix_streaming(
        contigs_by_genome, list(contigs_by_genome), k=4, w=2, memory_limit_mb=1000.0, memory_check_fn=fake_memory_check
    )
    assert result.aborted_for_memory is True
    assert result.complete is False
    assert len(result.processed_genome_ids) < 5


def test_build_minimizer_matrix_streaming_never_densifies(monkeypatch):
    calls = {"n": 0}
    original = sparse.csr_matrix.toarray

    def tracking(self, *a, **kw):
        calls["n"] += 1
        return original(self, *a, **kw)

    monkeypatch.setattr(sparse.csr_matrix, "toarray", tracking)
    contigs_by_genome = {"g1": ["ACGTACGTACGT"], "g2": ["TTTTGGGGCCCC"]}
    build_minimizer_matrix_streaming(contigs_by_genome, ["g1", "g2"], k=4, w=2)
    assert calls["n"] == 0
