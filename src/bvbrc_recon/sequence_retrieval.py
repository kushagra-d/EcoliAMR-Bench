"""BV-BRC genome_sequence FASTA retrieval (https://www.bv-brc.org/api/genome_sequence/,
Accept: application/dna+fasta - documented in the official BV-BRC-API repo README).

Two tiers, by design:

  - `probe_genome` / `fetch_sequence_fasta`: the original bounded feasibility probe -
    fetches ONE small contig per genome, used to verify the route works at all
    (see reports/bvbrc_assembly_retrieval_feasibility.md). Not sufficient on its
    own to claim a complete assembly was retrieved.

  - `fetch_complete_assembly`: fetches EVERY contig for one genome_id, batched
    (a single `in(sequence_id,(...))` request over-long URLs returns HTTP 414 -
    verified live - so requests are chunked, matching the batching pattern
    already used in `genome_join.py`), and cross-checks the retrieved contig
    count against both the JSON-mode index count and, optionally, the already-
    known `genome_contigs` metadata field, flagging anything less than an exact
    match as incomplete rather than assuming success.

Neither tier writes anything to disk - callers (e.g. the pilot retrieval
script) decide whether and where to persist results.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .client import (
    BASE_URL,
    REQUEST_HEADERS,
    RETRYABLE_HTTP_STATUSES,
    BVBRCClient,
    BVBRCQueryError,
)


@dataclass
class SequenceIndexEntry:
    sequence_id: str | None
    length: int | None
    accession: str | None


@dataclass
class FastaFetchResult:
    sequence_id: str
    http_status: int | None = None
    content_type: str | None = None
    response_bytes: int = 0
    header_line: str | None = None
    sequence_preview: str = ""  # first N bases only - never the full sequence
    is_valid_fasta: bool = False
    nonempty_sequence: bool = False
    error: str | None = None


@dataclass
class GenomeProbeResult:
    genome_id: str
    assembly_accession: str | None
    status: str  # see STATUS_* constants below
    index_entries: list[SequenceIndexEntry] = field(default_factory=list)
    fetch: FastaFetchResult | None = None
    note: str = ""


STATUS_FETCHED = "fetched"
STATUS_NO_SEQUENCE_INDEX = "no_sequence_records_found"
STATUS_ALL_CONTIGS_TOO_LARGE = "all_contigs_exceeded_size_bound"
STATUS_INDEX_QUERY_FAILED = "index_query_failed"
STATUS_FETCH_FAILED = "fetch_failed"


def fetch_sequence_index(
    client: BVBRCClient, genome_id: str, limit: int = 5
) -> list[SequenceIndexEntry]:
    """Small JSON-only query: which contigs exist for this genome, and how
    long are they. Fetches no sequence data."""
    quoted = urllib.parse.quote(genome_id, safe="")
    rql = f"eq(genome_id,{quoted})&select(sequence_id,length,accession)&limit({limit})"
    records, _ = client.query("genome_sequence", rql)
    return [
        SequenceIndexEntry(
            sequence_id=r.get("sequence_id"), length=r.get("length"), accession=r.get("accession")
        )
        for r in records
    ]


def fetch_sequence_fasta(
    sequence_id: str,
    base_url: str = BASE_URL,
    timeout: float = 30.0,
    preview_length: int = 60,
) -> FastaFetchResult:
    """Fetches the FASTA for exactly ONE sequence_id (one contig), validates
    it, and returns only a short preview of the sequence - never the full
    content, and nothing is written to disk by this function."""
    result = FastaFetchResult(sequence_id=sequence_id)
    quoted = urllib.parse.quote(sequence_id, safe="")
    url = f"{base_url}/genome_sequence/?eq(sequence_id,{quoted})&limit(1)"
    headers = dict(REQUEST_HEADERS)
    headers["Accept"] = "application/dna+fasta"
    request = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            result.http_status = response.status
            result.content_type = response.headers.get("Content-Type")
            result.response_bytes = len(body)
    except urllib.error.HTTPError as exc:
        result.http_status = exc.code
        result.error = f"HTTPError {exc.code}: {exc.reason}"
        return result
    except urllib.error.URLError as exc:
        result.error = f"URLError: {exc.reason}"
        return result

    text = body.decode("utf-8", errors="replace").strip()
    if not text:
        result.error = "empty response body (no matching sequence_id, or empty FASTA)"
        return result

    lines = text.splitlines()
    if not lines[0].startswith(">"):
        result.error = "response did not start with a FASTA header ('>') - unexpected format"
        return result

    result.is_valid_fasta = True
    result.header_line = lines[0]
    sequence_only = "".join(lines[1:])
    result.nonempty_sequence = len(sequence_only) > 0
    result.sequence_preview = sequence_only[:preview_length]
    return result


def probe_genome(
    client: BVBRCClient,
    genome_id: str,
    assembly_accession: str | None,
    max_contig_length: int = 200_000,
    index_limit: int = 5,
) -> GenomeProbeResult:
    """End-to-end bounded probe for one genome: look up a small set of its
    contigs (JSON metadata only), pick the shortest one under
    `max_contig_length`, and fetch only that one contig's FASTA."""
    try:
        index_entries = fetch_sequence_index(client, genome_id, limit=index_limit)
    except BVBRCQueryError as exc:
        return GenomeProbeResult(
            genome_id=genome_id,
            assembly_accession=assembly_accession,
            status=STATUS_INDEX_QUERY_FAILED,
            note=str(exc),
        )

    if not index_entries:
        return GenomeProbeResult(
            genome_id=genome_id,
            assembly_accession=assembly_accession,
            status=STATUS_NO_SEQUENCE_INDEX,
            note="genome_sequence query returned zero contigs for this genome_id",
        )

    candidates = [e for e in index_entries if e.length is not None and e.length <= max_contig_length]
    if not candidates:
        return GenomeProbeResult(
            genome_id=genome_id,
            assembly_accession=assembly_accession,
            status=STATUS_ALL_CONTIGS_TOO_LARGE,
            index_entries=index_entries,
            note=f"all {len(index_entries)} sampled contigs exceeded the {max_contig_length} bp bound",
        )

    smallest = min(candidates, key=lambda e: e.length)
    fetch_result = fetch_sequence_fasta(smallest.sequence_id)
    status = STATUS_FETCHED if fetch_result.is_valid_fasta else STATUS_FETCH_FAILED
    return GenomeProbeResult(
        genome_id=genome_id,
        assembly_accession=assembly_accession,
        status=status,
        index_entries=index_entries,
        fetch=fetch_result,
    )


# --- Complete-assembly retrieval (extends the single-contig probe above) --


@dataclass
class ContigRecord:
    sequence_id: str
    header: str
    sequence: str


@dataclass
class CompleteAssemblyResult:
    genome_id: str
    status: str  # see ASSEMBLY_STATUS_* constants below
    contigs: list[ContigRecord] = field(default_factory=list)
    n_contigs_in_index: int = 0
    n_contigs_retrieved: int = 0
    n_contigs_expected_from_metadata: int | None = None
    duplicate_sequence_ids: list[str] = field(default_factory=list)
    total_length_retrieved: int = 0
    errors: list[str] = field(default_factory=list)


ASSEMBLY_STATUS_COMPLETE = "complete"
ASSEMBLY_STATUS_INCOMPLETE = "incomplete"
ASSEMBLY_STATUS_NO_SEQUENCE_INDEX = "no_sequence_records_found"
ASSEMBLY_STATUS_INDEX_QUERY_FAILED = "index_query_failed"
ASSEMBLY_STATUS_DUPLICATE_IDS_IN_INDEX = "duplicate_sequence_ids_in_index"


def parse_fasta_records(text: str) -> list[ContigRecord]:
    """Splits a (possibly multi-record) FASTA response into individual
    records. A malformed record (header with no following sequence lines,
    i.e. an empty sequence) is still returned - callers decide how to
    treat an empty sequence; this parser does not silently drop it."""
    records: list[ContigRecord] = []
    header: str | None = None
    sequence_lines: list[str] = []

    def flush():
        if header is not None:
            seq = "".join(sequence_lines)
            first_token = header[1:].split()[0] if len(header) > 1 else ""
            # BV-BRC FASTA headers look like ">accn|<sequence_id>   <description>"
            seq_id = first_token.split("|", 1)[1] if "|" in first_token else first_token
            records.append(ContigRecord(sequence_id=seq_id, header=header, sequence=seq))

    for line in text.splitlines():
        if line.startswith(">"):
            flush()
            header = line
            sequence_lines = []
        elif header is not None:
            sequence_lines.append(line.strip())
    flush()
    return records


def _fetch_fasta_batch_raw(
    sequence_ids: list[str],
    base_url: str = BASE_URL,
    timeout: float = 60.0,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    sleep_fn=time.sleep,
) -> bytes:
    """Fetches one batch of contigs' FASTA in a single request, with the
    same retry semantics as BVBRCClient.query (retry 429/5xx/timeout/
    connection errors; raise immediately on a non-retryable HTTP status)."""
    quoted = ",".join(urllib.parse.quote(sid, safe="") for sid in sequence_ids)
    url = f"{base_url}/genome_sequence/?in(sequence_id,({quoted}))&limit({len(sequence_ids)})"
    headers = dict(REQUEST_HEADERS)
    headers["Accept"] = "application/dna+fasta"

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRYABLE_HTTP_STATUSES:
                raise
            last_error = exc
        except urllib.error.URLError as exc:
            last_error = exc
        if attempt < max_retries:
            sleep_fn(backoff_seconds * attempt)
    raise BVBRCQueryError(f"FASTA batch fetch failed after {max_retries} attempts: {last_error}")


def fetch_complete_assembly(
    client: BVBRCClient,
    genome_id: str,
    expected_contigs: int | None = None,
    batch_size: int = 100,
    index_limit: int = 2000,
    max_retries: int = 3,
) -> CompleteAssemblyResult:
    """Fetches every contig for `genome_id`, batched to stay under the
    server's URL-length limit (HTTP 414 was observed live for ~470 ids in
    one request; 100 is a safe, verified batch size). Marks the result
    COMPLETE only when the retrieved contig count exactly matches the
    JSON-mode index count AND (if provided) the already-known
    `genome_contigs` metadata value - never assumed from a partial fetch.
    """
    result = CompleteAssemblyResult(genome_id=genome_id, status=ASSEMBLY_STATUS_INDEX_QUERY_FAILED)
    result.n_contigs_expected_from_metadata = expected_contigs

    try:
        index_entries = fetch_sequence_index(client, genome_id, limit=index_limit)
    except BVBRCQueryError as exc:
        result.errors.append(f"index query failed: {exc}")
        return result

    result.n_contigs_in_index = len(index_entries)
    if not index_entries:
        result.status = ASSEMBLY_STATUS_NO_SEQUENCE_INDEX
        return result

    index_ids = [e.sequence_id for e in index_entries if e.sequence_id]
    index_duplicate_ids = [sid for sid in set(index_ids) if index_ids.count(sid) > 1]
    if index_duplicate_ids:
        result.duplicate_sequence_ids.extend(index_duplicate_ids)
        result.status = ASSEMBLY_STATUS_DUPLICATE_IDS_IN_INDEX
        result.errors.append(
            f"genome_sequence index itself contains duplicate sequence_id values for this "
            f"genome_id: {index_duplicate_ids} - refusing to treat this as a reliable count"
        )
        return result

    seen_ids: set[str] = set()
    for batch_start in range(0, len(index_ids), batch_size):
        batch = index_ids[batch_start : batch_start + batch_size]
        try:
            body = _fetch_fasta_batch_raw(batch)
        except (BVBRCQueryError, urllib.error.HTTPError) as exc:
            result.errors.append(f"FASTA batch starting at index {batch_start} failed: {exc}")
            continue

        text = body.decode("utf-8", errors="replace")
        batch_records = parse_fasta_records(text)
        requested_ids = set(batch)
        for record in batch_records:
            if record.sequence_id not in requested_ids:
                # Identity check: a sequence_id the server returned but that
                # was never requested for THIS genome must never be folded
                # into this assembly, even if it would make the count look
                # right - silently accepting it risks cross-genome
                # contamination from a server-side bug or ambiguous ID.
                result.errors.append(
                    f"server returned unrequested sequence_id {record.sequence_id!r} while "
                    f"fetching genome_id {genome_id!r} - discarded, not included in this assembly"
                )
                continue
            if record.sequence_id in seen_ids:
                result.duplicate_sequence_ids.append(record.sequence_id)
                continue
            if not record.sequence:
                result.errors.append(f"contig {record.sequence_id!r} had an empty sequence")
                continue
            seen_ids.add(record.sequence_id)
            result.contigs.append(record)
            result.total_length_retrieved += len(record.sequence)

    result.n_contigs_retrieved = len(result.contigs)

    matches_index = result.n_contigs_retrieved == result.n_contigs_in_index
    matches_metadata = expected_contigs is None or result.n_contigs_retrieved == expected_contigs
    if matches_index and matches_metadata and not result.errors:
        result.status = ASSEMBLY_STATUS_COMPLETE
    else:
        result.status = ASSEMBLY_STATUS_INCOMPLETE
        if not matches_index:
            result.errors.append(
                f"retrieved {result.n_contigs_retrieved} contigs but the genome_sequence index "
                f"reported {result.n_contigs_in_index} for this genome_id"
            )
        if expected_contigs is not None and not matches_metadata:
            result.errors.append(
                f"retrieved {result.n_contigs_retrieved} contigs but existing genome metadata "
                f"(genome_contigs) reported {expected_contigs}"
            )

    return result


def write_assembly_fasta_atomic(path, contigs: list[ContigRecord]) -> None:
    """Writes `contigs` to `path` as FASTA, atomically: the full content
    is written to a staging file first (`path` with a `.part` suffix
    appended - never a bare `.tmp` replacement of the real extension, so
    it can never collide with or be mistaken for a real `.fna` file),
    then `Path.rename`d into place only after the write completes
    without error. A process killed mid-write leaves only the `.part`
    file behind - `path` itself either does not exist yet (correctly
    treated as "not yet retrieved" by every caller's resume-by-
    file-existence check) or is the complete file. This replaces an
    earlier pattern (duplicated in `run_pilot.py`/`run_cohort_expansion.py`)
    that wrote directly to the final path - found, during this task's
    safety review, to be able to leave a valid-looking but truncated
    assembly behind if interrupted mid-write, which a resumed run would
    then silently treat as already complete.
    """
    path = Path(path)
    staging_path = path.with_name(path.name + ".part")
    with open(staging_path, "w", encoding="utf-8") as fh:
        for contig in contigs:
            fh.write(contig.header + "\n")
            seq = contig.sequence
            for i in range(0, len(seq), 70):
                fh.write(seq[i : i + 70] + "\n")
    staging_path.rename(path)
