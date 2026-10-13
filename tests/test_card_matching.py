from bvbrc_recon.card_matching import APPROXIMATE, EXACT, UNMAPPED, match_candidate, match_candidates
from bvbrc_recon.card_reference import build_card_kmer_index
from bvbrc_recon.kmer_features import canonical_kmer


def _write_fasta(tmp_path, name, records):
    path = tmp_path / name
    with open(path, "w") as fh:
        for header, seq in records:
            fh.write(f">{header}\n{seq}\n")
    return path


def _index_from(tmp_path, records):
    _write_fasta(tmp_path, "nucleotide_fasta_protein_homolog_model.fasta", records)
    return build_card_kmer_index(tmp_path, k=10, filenames=["nucleotide_fasta_protein_homolog_model.fasta"])


def test_exact_match_found(tmp_path):
    header = "gb|ACC1|+|0-20|ARO:1000001|geneA [Escherichia coli]"
    index = _index_from(tmp_path, [(header, "ACGTACGTAA")])
    result = match_candidate("ACGTACGTAA", index)
    assert result.match_type == EXACT
    assert result.gene_hits[0].short_name == "geneA"
    assert result.is_ambiguous is False


def test_exact_match_found_via_reverse_complement_orientation(tmp_path):
    header = "gb|ACC1|+|0-20|ARO:1000001|geneA [Escherichia coli]"
    index = _index_from(tmp_path, [(header, "ACGTACGTAA")])
    # query with the reverse complement of the indexed sequence - canonicalization
    # must still find it, since a k-mer and its reverse complement are "the same feature"
    from bvbrc_recon.kmer_features import reverse_complement

    rc_query = reverse_complement("ACGTACGTAA")
    result = match_candidate(rc_query, index)
    assert result.match_type == EXACT
    assert result.gene_hits[0].short_name == "geneA"


def test_approximate_match_single_substitution(tmp_path):
    header = "gb|ACC1|+|0-20|ARO:1000001|geneA [Escherichia coli]"
    index = _index_from(tmp_path, [(header, "ACGTACGTAA")])
    # one base changed (A->G at the end) relative to the indexed sequence
    result = match_candidate("ACGTACGTAG", index)
    assert result.match_type == APPROXIMATE
    assert result.gene_hits[0].short_name == "geneA"


def test_exact_takes_priority_over_approximate(tmp_path):
    header_exact = "gb|ACC1|+|0-20|ARO:1000001|exactGene [Escherichia coli]"
    header_near = "gb|ACC2|+|0-20|ARO:1000002|nearGene [Escherichia coli]"
    # nearGene is one substitution away from the exact query
    index = _index_from(tmp_path, [(header_exact, "ACGTACGTAA"), (header_near, "ACGTACGTAG")])
    result = match_candidate("ACGTACGTAA", index)
    assert result.match_type == EXACT
    assert {h.short_name for h in result.gene_hits} == {"exactGene"}  # not nearGene


def test_unmapped_when_no_match_within_distance_one(tmp_path):
    header = "gb|ACC1|+|0-20|ARO:1000001|geneA [Escherichia coli]"
    index = _index_from(tmp_path, [(header, "ACGTACGTAA")])
    result = match_candidate("TTTTTTTTTT", index)  # far from the indexed sequence
    assert result.match_type == UNMAPPED
    assert result.gene_hits == []
    assert result.is_ambiguous is False


def test_multiple_sequences_same_gene_not_flagged_ambiguous(tmp_path):
    header1 = "gb|ACC1|+|0-20|ARO:1000001|geneA [Escherichia coli]"
    header2 = "gb|ACC2|+|0-20|ARO:1000001|geneA [Klebsiella pneumoniae]"
    index = _index_from(tmp_path, [(header1, "ACGTACGTAA"), (header2, "ACGTACGTAA")])
    result = match_candidate("ACGTACGTAA", index)
    assert result.match_type == EXACT
    assert len(result.gene_hits) == 1
    assert result.gene_hits[0].n_supporting_sequences == 2
    assert set(result.gene_hits[0].organisms) == {"Escherichia coli", "Klebsiella pneumoniae"}
    assert result.is_ambiguous is False


def test_matches_to_distinct_genes_flagged_ambiguous(tmp_path):
    header1 = "gb|ACC1|+|0-20|ARO:1000001|geneA [Escherichia coli]"
    header2 = "gb|ACC2|+|0-20|ARO:1000002|geneB [Escherichia coli]"
    index = _index_from(tmp_path, [(header1, "ACGTACGTAA"), (header2, "ACGTACGTAA")])
    result = match_candidate("ACGTACGTAA", index)
    assert result.match_type == EXACT
    assert len(result.gene_hits) == 2
    assert result.is_ambiguous is True


def test_match_candidates_processes_a_batch(tmp_path):
    header = "gb|ACC1|+|0-20|ARO:1000001|geneA [Escherichia coli]"
    index = _index_from(tmp_path, [(header, "ACGTACGTAA")])
    results = match_candidates(["ACGTACGTAA", "TTTTTTTTTT"], index)
    assert [r.match_type for r in results] == [EXACT, UNMAPPED]


def test_match_candidate_rejects_wrong_length(tmp_path):
    header = "gb|ACC1|+|0-20|ARO:1000001|geneA [Escherichia coli]"
    index = _index_from(tmp_path, [(header, "ACGTACGTAA")])
    try:
        match_candidate("ACGT", index)
        assert False, "expected ValueError"
    except ValueError:
        pass
