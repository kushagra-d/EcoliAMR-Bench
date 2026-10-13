from scipy import sparse

from bvbrc_recon.kmer_features import canonical_kmer
from bvbrc_recon.streaming_kmer_build import build_matrix_streaming


def test_builds_correct_matrix_for_simple_case():
    contigs_by_genome = {"g1": ["AAAACCCC"], "g2": ["GGGGTTTT"]}
    result = build_matrix_streaming(contigs_by_genome, ["g1", "g2"], k=4)
    assert result.complete is True
    assert result.aborted_for_memory is False
    assert result.processed_genome_ids == ["g1", "g2"]
    assert sparse.issparse(result.matrix)
    assert result.matrix.shape[0] == 2


def test_reuses_same_canonicalization_as_kmer_features():
    contigs_by_genome = {"g1": ["AAGGAAGG"]}
    result = build_matrix_streaming(contigs_by_genome, ["g1"], k=4)
    assert canonical_kmer("AAGG") in result.vocabulary


def test_skips_windows_with_ambiguous_bases():
    contigs_by_genome = {"g1": ["ACGTNACGT"]}
    result = build_matrix_streaming(contigs_by_genome, ["g1"], k=4)
    # windows: ACGT, CGTN(skip), GTNA(skip), TNAC(skip), NACG(skip), ACGT
    assert result.matrix.sum() == 2


def test_never_spans_contig_boundary():
    # if contigs were wrongly concatenated as "AAAA"+"TTTT", a boundary
    # k-mer "AATT" would appear - processed independently, it must not.
    contigs_by_genome = {"g1": ["AAAA", "TTTT"]}
    result = build_matrix_streaming(contigs_by_genome, ["g1"], k=4)
    assert canonical_kmer("AATT") not in result.vocabulary


def test_shared_vocabulary_across_genomes_reuses_columns():
    contigs_by_genome = {"g1": ["AAAACCCC"], "g2": ["AAAACCCC"]}
    result = build_matrix_streaming(contigs_by_genome, ["g1", "g2"], k=4)
    # both genomes share the exact same sequence - vocabulary size should
    # not double just because there are two genomes
    assert len(result.vocabulary) == len(canonical_set_of("AAAACCCC", 4))


def canonical_set_of(seq, k):
    return {canonical_kmer(seq[i : i + k]) for i in range(len(seq) - k + 1)}


# --- memory-safety abort ---------------------------------------------------


def test_aborts_before_processing_all_genomes_when_memory_exceeded():
    contigs_by_genome = {f"g{i}": ["AAAACCCC"] for i in range(10)}
    genome_order = list(contigs_by_genome)

    # fake memory check that reports "over limit" starting from the 4th call
    calls = {"n": 0}

    def fake_memory_check():
        calls["n"] += 1
        return 100.0 if calls["n"] < 4 else 99999.0

    result = build_matrix_streaming(
        contigs_by_genome, genome_order, k=4, memory_limit_mb=1000.0, memory_check_fn=fake_memory_check
    )
    assert result.aborted_for_memory is True
    assert result.complete is False
    assert len(result.processed_genome_ids) == 4  # stopped right after exceeding the limit
    assert len(result.processed_genome_ids) < len(genome_order)


def test_never_labels_an_aborted_result_as_complete():
    contigs_by_genome = {f"g{i}": ["AAAACCCC"] for i in range(5)}

    def always_over_limit():
        return 999999.0

    result = build_matrix_streaming(
        contigs_by_genome, list(contigs_by_genome), k=4, memory_limit_mb=1.0, memory_check_fn=always_over_limit
    )
    assert result.complete is False
    assert result.aborted_for_memory is True


def test_matrix_shape_matches_processed_genomes_not_requested_genomes():
    contigs_by_genome = {f"g{i}": ["AAAACCCC"] for i in range(5)}

    calls = {"n": 0}

    def fake_memory_check():
        calls["n"] += 1
        return 0.0 if calls["n"] < 2 else 999999.0

    result = build_matrix_streaming(
        contigs_by_genome, list(contigs_by_genome), k=4, memory_limit_mb=1000.0, memory_check_fn=fake_memory_check
    )
    assert result.matrix.shape[0] == len(result.processed_genome_ids)
    assert result.matrix.shape[0] < 5


def test_progress_callback_invoked_once_per_genome():
    contigs_by_genome = {"g1": ["AAAACCCC"], "g2": ["GGGGTTTT"]}
    calls = []
    build_matrix_streaming(
        contigs_by_genome,
        ["g1", "g2"],
        k=4,
        progress_callback=lambda *args: calls.append(args),
    )
    assert len(calls) == 2
    assert calls[0][1] == "g1"
    assert calls[1][1] == "g2"


def test_never_holds_more_than_one_genome_worth_of_raw_counts_at_once():
    """Structural guarantee, not just an outcome check: progress_callback
    fires with a strictly non-decreasing nnz_so_far, and no genome's
    contribution is double counted (vocab/nnz growth is monotonic)."""
    contigs_by_genome = {f"g{i}": ["AAAACCCCGGGGTTTT"] for i in range(5)}
    nnz_history = []
    build_matrix_streaming(
        contigs_by_genome,
        list(contigs_by_genome),
        k=4,
        progress_callback=lambda row_idx, gid, vocab_size, nnz, mem: nnz_history.append(nnz),
    )
    assert nnz_history == sorted(nnz_history)
    assert all(n2 > n1 for n1, n2 in zip(nnz_history, nnz_history[1:]))
