from bvbrc_recon.sequence_qc import qc_assembly


def write_fasta(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    return path


def test_qc_assembly_reports_contig_count_and_length(tmp_path):
    path = write_fasta(tmp_path, "g1.fna", ">c1\nACGTACGT\n>c2\nACGT\n")
    qc = qc_assembly("g1", path)
    assert qc.n_contigs == 2
    assert qc.total_length == 12
    assert qc.is_nonempty is True
    assert qc.has_valid_nucleotide_data is True
    assert qc.error is None


def test_qc_assembly_computes_ambiguous_base_proportion(tmp_path):
    path = write_fasta(tmp_path, "g1.fna", ">c1\nACGTNNNN\n")
    qc = qc_assembly("g1", path)
    assert qc.ambiguous_base_count == 4
    assert qc.ambiguous_base_proportion == 0.5


def test_qc_assembly_detects_duplicate_sequence_ids(tmp_path):
    path = write_fasta(tmp_path, "g1.fna", ">c1\nACGT\n>c1\nTTTT\n")
    qc = qc_assembly("g1", path)
    assert "c1" in qc.duplicate_sequence_ids
    assert any("duplicate" in w for w in qc.warnings)


def test_qc_assembly_handles_zero_record_file(tmp_path):
    path = write_fasta(tmp_path, "empty.fna", "")
    qc = qc_assembly("g1", path)
    assert qc.n_contigs == 0
    assert qc.is_nonempty is False
    assert any("zero FASTA records" in w for w in qc.warnings)


def test_qc_assembly_handles_missing_file(tmp_path):
    qc = qc_assembly("g1", tmp_path / "does_not_exist.fna")
    assert qc.error is not None
    assert "does not exist" in qc.error


def test_qc_assembly_flags_empty_contig_without_dropping(tmp_path):
    path = write_fasta(tmp_path, "g1.fna", ">c1\n\n>c2\nACGT\n")
    qc = qc_assembly("g1", path)
    assert qc.n_contigs == 2  # the empty contig is still counted, not silently dropped
    assert qc.empty_contig_count == 1
    assert any("empty sequence" in w for w in qc.warnings)


def test_qc_assembly_all_ambiguous_means_not_valid_nucleotide_data(tmp_path):
    path = write_fasta(tmp_path, "g1.fna", ">c1\nNNNNNNNN\n")
    qc = qc_assembly("g1", path)
    assert qc.has_valid_nucleotide_data is False
    assert qc.ambiguous_base_proportion == 1.0
