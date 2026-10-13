import urllib.error

import pytest

from bvbrc_recon.sequence_retrieval import (
    ASSEMBLY_STATUS_COMPLETE,
    ASSEMBLY_STATUS_DUPLICATE_IDS_IN_INDEX,
    ASSEMBLY_STATUS_INCOMPLETE,
    ASSEMBLY_STATUS_INDEX_QUERY_FAILED,
    ASSEMBLY_STATUS_NO_SEQUENCE_INDEX,
    STATUS_ALL_CONTIGS_TOO_LARGE,
    STATUS_FETCHED,
    STATUS_FETCH_FAILED,
    STATUS_INDEX_QUERY_FAILED,
    STATUS_NO_SEQUENCE_INDEX,
    fetch_complete_assembly,
    fetch_sequence_fasta,
    fetch_sequence_index,
    parse_fasta_records,
    probe_genome,
)
from fakes import FakeClient

from bvbrc_recon.client import BVBRCQueryError


# --- fetch_sequence_index -------------------------------------------------


def test_fetch_sequence_index_returns_entries():
    client = FakeClient(
        {
            "genome_sequence": [
                (
                    [
                        {"sequence_id": "s1", "length": 1000, "accession": "A1"},
                        {"sequence_id": "s2", "length": 2000, "accession": None},
                    ],
                    2,
                )
            ]
        }
    )
    entries = fetch_sequence_index(client, "562.1", limit=5)
    assert [e.sequence_id for e in entries] == ["s1", "s2"]
    assert entries[0].length == 1000


def test_fetch_sequence_index_propagates_query_errors():
    client = FakeClient({"genome_sequence": [BVBRCQueryError("boom")]})
    try:
        fetch_sequence_index(client, "562.1")
        assert False, "expected BVBRCQueryError"
    except BVBRCQueryError:
        pass


# --- fetch_sequence_fasta: mocked urllib ----------------------------------


class FakeFastaResponse:
    def __init__(self, body: bytes, status: int = 200, content_type: str = "text/plain"):
        self._body = body
        self.status = status
        self._content_type = content_type

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body

    @property
    def headers(self):
        return self

    def get(self, name, default=None):
        return self._content_type if name == "Content-Type" else default


def test_fetch_sequence_fasta_valid_response(monkeypatch):
    body = b">accn|some_contig description\nACGTACGTACGT\nACGT\n"

    def fake_urlopen(request, timeout=None):
        return FakeFastaResponse(body, content_type="text/x-fasta")

    monkeypatch.setattr("bvbrc_recon.sequence_retrieval.urllib.request.urlopen", fake_urlopen)
    result = fetch_sequence_fasta("some_contig")

    assert result.is_valid_fasta is True
    assert result.header_line == ">accn|some_contig description"
    assert result.nonempty_sequence is True
    assert result.sequence_preview == "ACGTACGTACGTACGT"
    assert result.http_status == 200
    assert result.content_type == "text/x-fasta"
    assert result.response_bytes == len(body)


def test_fetch_sequence_fasta_truncates_preview_to_requested_length(monkeypatch):
    body = b">h\n" + (b"A" * 500) + b"\n"

    def fake_urlopen(request, timeout=None):
        return FakeFastaResponse(body)

    monkeypatch.setattr("bvbrc_recon.sequence_retrieval.urllib.request.urlopen", fake_urlopen)
    result = fetch_sequence_fasta("some_contig", preview_length=10)
    assert result.sequence_preview == "A" * 10
    assert len(result.sequence_preview) < 500  # full sequence is never retained


def test_fetch_sequence_fasta_handles_empty_response():
    def fake_urlopen(request, timeout=None):
        return FakeFastaResponse(b"")

    import bvbrc_recon.sequence_retrieval as sr

    original = sr.urllib.request.urlopen
    sr.urllib.request.urlopen = fake_urlopen
    try:
        result = fetch_sequence_fasta("missing_id")
    finally:
        sr.urllib.request.urlopen = original

    assert result.is_valid_fasta is False
    assert "empty response" in result.error


def test_fetch_sequence_fasta_handles_non_fasta_response(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return FakeFastaResponse(b"not a fasta file at all")

    monkeypatch.setattr("bvbrc_recon.sequence_retrieval.urllib.request.urlopen", fake_urlopen)
    result = fetch_sequence_fasta("some_id")
    assert result.is_valid_fasta is False
    assert "did not start with a FASTA header" in result.error


def test_fetch_sequence_fasta_handles_http_error(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError("url", 400, "Bad Request", {}, None)

    monkeypatch.setattr("bvbrc_recon.sequence_retrieval.urllib.request.urlopen", fake_urlopen)
    result = fetch_sequence_fasta("bad_id")
    assert result.http_status == 400
    assert "HTTPError 400" in result.error


def test_fetch_sequence_fasta_handles_url_error(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("connection reset")

    monkeypatch.setattr("bvbrc_recon.sequence_retrieval.urllib.request.urlopen", fake_urlopen)
    result = fetch_sequence_fasta("some_id")
    assert result.is_valid_fasta is False
    assert "URLError" in result.error


# --- probe_genome: end-to-end with mocks ----------------------------------


def test_probe_genome_picks_smallest_contig_under_bound(monkeypatch):
    client = FakeClient(
        {
            "genome_sequence": [
                (
                    [
                        {"sequence_id": "big", "length": 5_000_000, "accession": None},
                        {"sequence_id": "small", "length": 500, "accession": None},
                    ],
                    2,
                )
            ]
        }
    )
    captured = {}

    def fake_fetch(sequence_id, **kwargs):
        captured["sequence_id"] = sequence_id
        from bvbrc_recon.sequence_retrieval import FastaFetchResult

        return FastaFetchResult(
            sequence_id=sequence_id, is_valid_fasta=True, nonempty_sequence=True, header_line=">x"
        )

    monkeypatch.setattr("bvbrc_recon.sequence_retrieval.fetch_sequence_fasta", fake_fetch)
    result = probe_genome(client, "562.1", None, max_contig_length=200_000)

    assert captured["sequence_id"] == "small"
    assert result.status == STATUS_FETCHED


def test_probe_genome_reports_no_sequence_index():
    client = FakeClient({"genome_sequence": [([], 0)]})
    result = probe_genome(client, "562.1", None)
    assert result.status == STATUS_NO_SEQUENCE_INDEX


def test_probe_genome_reports_all_contigs_too_large():
    client = FakeClient(
        {"genome_sequence": [([{"sequence_id": "big", "length": 9_000_000, "accession": None}], 1)]}
    )
    result = probe_genome(client, "562.1", None, max_contig_length=200_000)
    assert result.status == STATUS_ALL_CONTIGS_TOO_LARGE


def test_probe_genome_reports_index_query_failure():
    client = FakeClient({"genome_sequence": [BVBRCQueryError("simulated failure")]})
    result = probe_genome(client, "562.1", None)
    assert result.status == STATUS_INDEX_QUERY_FAILED


def test_probe_genome_reports_fetch_failed_when_fasta_invalid(monkeypatch):
    client = FakeClient(
        {"genome_sequence": [([{"sequence_id": "s1", "length": 500, "accession": None}], 1)]}
    )

    def fake_fetch(sequence_id, **kwargs):
        from bvbrc_recon.sequence_retrieval import FastaFetchResult

        return FastaFetchResult(sequence_id=sequence_id, is_valid_fasta=False, error="boom")

    monkeypatch.setattr("bvbrc_recon.sequence_retrieval.fetch_sequence_fasta", fake_fetch)
    result = probe_genome(client, "562.1", None)
    assert result.status == STATUS_FETCH_FAILED


# --- parse_fasta_records ---------------------------------------------------


def test_parse_fasta_records_splits_multiple_records():
    text = ">accn|a desc one\nACGT\nACGT\n>accn|b desc two\nTTTT\n"
    records = parse_fasta_records(text)
    assert [r.sequence_id for r in records] == ["a", "b"]
    assert records[0].sequence == "ACGTACGT"
    assert records[1].sequence == "TTTT"


def test_parse_fasta_records_keeps_empty_sequence_records_visible():
    text = ">accn|empty desc\n>accn|nonempty desc\nACGT\n"
    records = parse_fasta_records(text)
    assert records[0].sequence_id == "empty"
    assert records[0].sequence == ""
    assert records[1].sequence == "ACGT"


# --- fetch_complete_assembly -------------------------------------------


def _index_entries_payload(sequence_ids):
    return [{"sequence_id": sid, "length": 100, "accession": None} for sid in sequence_ids]


def _fasta_blob(sequence_ids):
    return "".join(f">accn|{sid} desc [Escherichia coli | 562.1]\nACGTACGT\n" for sid in sequence_ids).encode()


def test_fetch_complete_assembly_success_single_batch(monkeypatch):
    ids = ["s1", "s2", "s3"]
    client = FakeClient({"genome_sequence": [(_index_entries_payload(ids), 3)]})

    def fake_fetch_batch(batch, **kwargs):
        assert batch == ids
        return _fasta_blob(batch)

    monkeypatch.setattr("bvbrc_recon.sequence_retrieval._fetch_fasta_batch_raw", fake_fetch_batch)
    result = fetch_complete_assembly(client, "562.1", expected_contigs=3, batch_size=100)

    assert result.status == ASSEMBLY_STATUS_COMPLETE
    assert result.n_contigs_retrieved == 3
    assert result.n_contigs_in_index == 3
    assert result.duplicate_sequence_ids == []
    assert result.total_length_retrieved == 3 * 8


def test_fetch_complete_assembly_batches_large_contig_counts(monkeypatch):
    ids = [f"s{i}" for i in range(250)]
    client = FakeClient({"genome_sequence": [(_index_entries_payload(ids), 250)]})
    calls = []

    def fake_fetch_batch(batch, **kwargs):
        calls.append(list(batch))
        return _fasta_blob(batch)

    monkeypatch.setattr("bvbrc_recon.sequence_retrieval._fetch_fasta_batch_raw", fake_fetch_batch)
    result = fetch_complete_assembly(client, "562.1", batch_size=100)

    assert len(calls) == 3  # 100, 100, 50 - never one request per contig, never one giant request
    assert result.n_contigs_retrieved == 250
    assert result.status == ASSEMBLY_STATUS_COMPLETE


def test_fetch_complete_assembly_detects_duplicate_ids_in_index():
    ids = ["s1", "s1", "s2"]  # s1 appears twice in the index itself
    client = FakeClient({"genome_sequence": [(_index_entries_payload(ids), 3)]})
    result = fetch_complete_assembly(client, "562.1")
    assert result.status == ASSEMBLY_STATUS_DUPLICATE_IDS_IN_INDEX
    assert "s1" in result.duplicate_sequence_ids


def test_fetch_complete_assembly_detects_duplicate_ids_across_batches(monkeypatch):
    ids = ["s1", "s2"]
    client = FakeClient({"genome_sequence": [(_index_entries_payload(ids), 2)]})

    def fake_fetch_batch(batch, **kwargs):
        # simulate a server returning s1 twice within one "batch" response
        return _fasta_blob(["s1", "s1", "s2"])

    monkeypatch.setattr("bvbrc_recon.sequence_retrieval._fetch_fasta_batch_raw", fake_fetch_batch)
    result = fetch_complete_assembly(client, "562.1", batch_size=100)
    assert "s1" in result.duplicate_sequence_ids
    assert result.n_contigs_retrieved == 2  # s1 counted once, not twice


def test_fetch_complete_assembly_marks_incomplete_on_retrieval_shortfall(monkeypatch):
    ids = ["s1", "s2", "s3"]
    client = FakeClient({"genome_sequence": [(_index_entries_payload(ids), 3)]})

    def fake_fetch_batch(batch, **kwargs):
        return _fasta_blob(["s1", "s2"])  # s3 silently missing from the response

    monkeypatch.setattr("bvbrc_recon.sequence_retrieval._fetch_fasta_batch_raw", fake_fetch_batch)
    result = fetch_complete_assembly(client, "562.1", batch_size=100)
    assert result.status == ASSEMBLY_STATUS_INCOMPLETE
    assert result.n_contigs_retrieved == 2
    assert result.n_contigs_in_index == 3


def test_fetch_complete_assembly_marks_incomplete_when_metadata_mismatches(monkeypatch):
    ids = ["s1", "s2"]
    client = FakeClient({"genome_sequence": [(_index_entries_payload(ids), 2)]})

    def fake_fetch_batch(batch, **kwargs):
        return _fasta_blob(batch)

    monkeypatch.setattr("bvbrc_recon.sequence_retrieval._fetch_fasta_batch_raw", fake_fetch_batch)
    # index and retrieval agree (2 contigs), but existing genome_contigs metadata says 5
    result = fetch_complete_assembly(client, "562.1", expected_contigs=5, batch_size=100)
    assert result.status == ASSEMBLY_STATUS_INCOMPLETE
    assert any("genome_contigs" in e for e in result.errors)


def test_fetch_complete_assembly_reports_no_sequence_index():
    client = FakeClient({"genome_sequence": [([], 0)]})
    result = fetch_complete_assembly(client, "562.1")
    assert result.status == ASSEMBLY_STATUS_NO_SEQUENCE_INDEX


def test_fetch_complete_assembly_reports_index_query_failure():
    client = FakeClient({"genome_sequence": [BVBRCQueryError("simulated failure")]})
    result = fetch_complete_assembly(client, "562.1")
    assert result.status == ASSEMBLY_STATUS_INDEX_QUERY_FAILED
    assert result.errors


def test_fetch_complete_assembly_records_batch_errors_without_crashing(monkeypatch):
    ids = [f"s{i}" for i in range(150)]
    client = FakeClient({"genome_sequence": [(_index_entries_payload(ids), 150)]})
    call_count = {"n": 0}

    def fake_fetch_batch(batch, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise BVBRCQueryError("simulated transient failure for batch 1")
        return _fasta_blob(batch)

    monkeypatch.setattr("bvbrc_recon.sequence_retrieval._fetch_fasta_batch_raw", fake_fetch_batch)
    result = fetch_complete_assembly(client, "562.1", batch_size=100)
    assert result.status == ASSEMBLY_STATUS_INCOMPLETE
    assert any("batch" in e for e in result.errors)
    assert call_count["n"] == 2  # second batch still attempted despite first failing


def test_fetch_complete_assembly_skips_empty_sequence_contigs(monkeypatch):
    ids = ["s1", "s2"]
    client = FakeClient({"genome_sequence": [(_index_entries_payload(ids), 2)]})

    def fake_fetch_batch(batch, **kwargs):
        return b">accn|s1 desc\n\n>accn|s2 desc\nACGT\n"  # s1 has no sequence lines

    monkeypatch.setattr("bvbrc_recon.sequence_retrieval._fetch_fasta_batch_raw", fake_fetch_batch)
    result = fetch_complete_assembly(client, "562.1", batch_size=100)
    assert result.n_contigs_retrieved == 1  # only s2 counted
    assert any("empty sequence" in e for e in result.errors)


# --- sequence identity validation (fix added this task) ---


def test_fetch_complete_assembly_rejects_unrequested_sequence_id(monkeypatch):
    """A server response containing a sequence_id that was never part of
    THIS genome's requested batch must be discarded, not folded into the
    assembly - even if accepting it would make the count look right."""
    ids = ["s1", "s2"]
    client = FakeClient({"genome_sequence": [(_index_entries_payload(ids), 2)]})

    def fake_fetch_batch(batch, **kwargs):
        # server returns s1 correctly, but substitutes an unrequested id
        # ("s_other", belonging to some other genome) instead of s2
        return _fasta_blob(["s1", "s_other"])

    monkeypatch.setattr("bvbrc_recon.sequence_retrieval._fetch_fasta_batch_raw", fake_fetch_batch)
    result = fetch_complete_assembly(client, "562.1", batch_size=100)

    assert "s_other" not in {c.sequence_id for c in result.contigs}
    assert result.n_contigs_retrieved == 1  # only s1 - s_other was rejected, not counted
    assert result.status == ASSEMBLY_STATUS_INCOMPLETE  # count no longer matches the index (2)
    assert any("unrequested sequence_id" in e and "s_other" in e for e in result.errors)


def test_fetch_complete_assembly_accepts_all_requested_ids_normally(monkeypatch):
    """Sanity check: the identity filter must not reject legitimate,
    requested sequence_ids - only ones outside the requested batch."""
    ids = ["s1", "s2", "s3"]
    client = FakeClient({"genome_sequence": [(_index_entries_payload(ids), 3)]})

    def fake_fetch_batch(batch, **kwargs):
        assert set(batch) == set(ids)
        return _fasta_blob(batch)

    monkeypatch.setattr("bvbrc_recon.sequence_retrieval._fetch_fasta_batch_raw", fake_fetch_batch)
    result = fetch_complete_assembly(client, "562.1", batch_size=100)
    assert result.status == ASSEMBLY_STATUS_COMPLETE
    assert result.n_contigs_retrieved == 3


# --- atomic write (fix added this task) ---


def test_write_assembly_fasta_atomic_produces_complete_parseable_file(tmp_path):
    from bvbrc_recon.sequence_retrieval import ContigRecord, write_assembly_fasta_atomic

    contigs = [
        ContigRecord(sequence_id="c1", header=">accn|c1 desc", sequence="ACGT" * 20),
        ContigRecord(sequence_id="c2", header=">accn|c2 desc", sequence="TTTT" * 5),
    ]
    path = tmp_path / "g1.fna"
    write_assembly_fasta_atomic(path, contigs)

    assert path.exists()
    text = path.read_text()
    assert text.count(">") == 2
    assert "ACGT" in text and "TTTT" in text
    # no leftover staging file
    assert not (tmp_path / "g1.fna.part").exists()


def test_write_assembly_fasta_atomic_leaves_no_final_file_if_interrupted(tmp_path, monkeypatch):
    """Simulates a process killed mid-write: the staging write raises
    before completing, and the FINAL path must never come to exist -
    a resumed run's `fasta_path.exists()` check must correctly see this
    genome as not-yet-retrieved, never as a valid-looking partial file."""
    from bvbrc_recon import sequence_retrieval as module
    from bvbrc_recon.sequence_retrieval import ContigRecord

    contigs = [ContigRecord(sequence_id="c1", header=">accn|c1 desc", sequence="ACGT" * 20)]
    path = tmp_path / "g1.fna"

    real_open = open

    def failing_open(file, *args, **kwargs):
        fh = real_open(file, *args, **kwargs)
        original_write = fh.write

        def write_then_fail(data):
            original_write(data)
            raise OSError("simulated crash mid-write")

        fh.write = write_then_fail
        return fh

    monkeypatch.setattr("builtins.open", failing_open)
    with pytest.raises(OSError):
        module.write_assembly_fasta_atomic(path, contigs)

    assert not path.exists()  # the final path was never created
    # the leftover .part file is harmless - a later successful write
    # truncates/overwrites it, and it is never mistaken for `path` itself
    assert (tmp_path / "g1.fna.part").exists()


def test_write_assembly_fasta_atomic_overwrites_stale_partial_file(tmp_path):
    """A leftover `.part` file from a prior crash must not interfere
    with a later, successful retry for the same genome."""
    from bvbrc_recon.sequence_retrieval import ContigRecord, write_assembly_fasta_atomic

    stale_part = tmp_path / "g1.fna.part"
    stale_part.write_text(">accn|truncated\nAC")  # looks like a half-written file

    contigs = [ContigRecord(sequence_id="c1", header=">accn|c1 desc", sequence="ACGTACGT")]
    path = tmp_path / "g1.fna"
    write_assembly_fasta_atomic(path, contigs)

    assert path.read_text().count(">") == 1
    assert "ACGTACGT" in path.read_text()
    assert not stale_part.exists()  # renamed away, not left dangling
