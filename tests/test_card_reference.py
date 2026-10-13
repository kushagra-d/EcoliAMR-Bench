from bvbrc_recon.card_reference import build_card_kmer_index, parse_card_header
from bvbrc_recon.kmer_features import canonical_kmer


def test_parse_card_header_valid():
    header = "gb|U00096.3|-|2336792-2339420|ARO:3003294|Ecol_gyrA_FLO [Escherichia coli str. K-12]"
    meta = parse_card_header(header, source_model="test.fasta")
    assert meta.accession == "U00096.3"
    assert meta.strand == "-"
    assert meta.start == 2336792
    assert meta.end == 2339420
    assert meta.aro_id == "ARO:3003294"
    assert meta.short_name == "Ecol_gyrA_FLO"
    assert meta.organism == "Escherichia coli str. K-12"


def test_parse_card_header_invalid_returns_none_not_raises():
    assert parse_card_header("not a valid card header at all", "test.fasta") is None


def _write_fasta(tmp_path, name, records):
    path = tmp_path / name
    with open(path, "w") as fh:
        for header, seq in records:
            fh.write(f">{header}\n{seq}\n")
    return path


def test_build_card_kmer_index_indexes_canonical_kmers(tmp_path):
    header = "gb|ACC1|+|0-20|ARO:1000001|testGene [Escherichia coli]"
    _write_fasta(tmp_path, "nucleotide_fasta_protein_homolog_model.fasta", [(header, "ACGTACGTAA")])

    index = build_card_kmer_index(tmp_path, k=10, filenames=["nucleotide_fasta_protein_homolog_model.fasta"])
    assert index.n_sequences_loaded == 1
    assert index.n_sequences_unparsed_header == 0
    key = canonical_kmer("ACGTACGTAA")
    assert key in index.index
    assert index.index[key][0].short_name == "testGene"


def test_build_card_kmer_index_skips_ambiguous_windows(tmp_path):
    header = "gb|ACC1|+|0-20|ARO:1000001|testGene [Escherichia coli]"
    _write_fasta(tmp_path, "nucleotide_fasta_protein_homolog_model.fasta", [(header, "ACGTNACGTA")])

    index = build_card_kmer_index(tmp_path, k=10, filenames=["nucleotide_fasta_protein_homolog_model.fasta"])
    # only one 10-mer window exists (seq length 10), and it contains an N -> skipped entirely
    assert index.n_windows_indexed == 0
    assert index.n_windows_skipped_ambiguous == 1


def test_build_card_kmer_index_reports_unparsed_headers_without_dropping_silently(tmp_path):
    _write_fasta(tmp_path, "nucleotide_fasta_protein_homolog_model.fasta", [("not_a_card_header", "ACGTACGTAA")])
    index = build_card_kmer_index(tmp_path, k=10, filenames=["nucleotide_fasta_protein_homolog_model.fasta"])
    assert index.n_sequences_unparsed_header == 1
    assert index.n_sequences_loaded == 0
    assert "not_a_card_header" in index.unparsed_headers


def test_build_card_kmer_index_collapses_reverse_complement_to_same_key(tmp_path):
    header1 = "gb|ACC1|+|0-20|ARO:1000001|geneA [Escherichia coli]"
    header2 = "gb|ACC2|+|0-20|ARO:1000002|geneB [Escherichia coli]"
    # second sequence is the reverse complement of the first
    _write_fasta(
        tmp_path,
        "nucleotide_fasta_protein_homolog_model.fasta",
        [(header1, "ACGTACGTAA"), (header2, "TTACGTACGT")],
    )
    index = build_card_kmer_index(tmp_path, k=10, filenames=["nucleotide_fasta_protein_homolog_model.fasta"])
    assert len(index.index) == 1  # both map to the same canonical key
    key = list(index.index)[0]
    short_names = {m.short_name for m in index.index[key]}
    assert short_names == {"geneA", "geneB"}
