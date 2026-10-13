# BV-BRC Genome Assembly Retrieval - Feasibility Investigation

- Investigation date (UTC): 2026-10-09
- Scope: feasibility only. No bulk download was performed. No sequence data was written to
  `data/raw/`, `data/processed/`, or anywhere else in the repository.
- This report assigns no final phenotype labels and selects no modelling cohort.

## 1. Official sources consulted

- **BV-BRC-API GitHub repository README** (`https://github.com/BV-BRC/BV-BRC-API`, fetched via
  `raw.githubusercontent.com` - the actual source backing the public API, not a secondary
  description): documents the `application/dna+fasta` response format and that it is
  "supported for the `genome_feature` and `genome_sequence` collections."
- **BV-BRC-Solr schema repository** (`https://github.com/BV-BRC/BV-BRC-Solr`,
  `genome_sequence/managed-schema`, fetched directly): confirms the exact field list for the
  `genome_sequence` collection, including `genome_id`, `sequence_id`, `accession`, `length`, and
  a stored `sequence` field.
- **Web search results citing BV-BRC's own FTP documentation**
  (`bv-brc.org/docs/quick_references/ftp.html`) and the `p3-genome-fasta` CLI tool page: both
  describe a second, independent route - `ftps://ftp.bv-brc.org/genomes/$genome_id/$genome_id.fna`.
  Direct `WebFetch`/`curl` access to `bv-brc.org/docs/*` pages returned HTTP 403 in this session
  (the site's bot-protection blocks this differently per path, even with a browser User-Agent that
  worked for `bv-brc.org/api/*`), so **this specific FTP route is documented but not independently
  re-verified against the live FTPS server in this session** - it is reported as documented, not
  as tested.
- **Live, controlled test** against `https://www.bv-brc.org/api/genome_sequence/` (see Section 3) -
  this is the evidence this report relies on most, since it is directly verified rather than
  quoted from documentation.

## 2. Recommended retrieval route

**`https://www.bv-brc.org/api/genome_sequence/?eq(sequence_id,<SEQUENCE_ID>)` with
`Accept: application/dna+fasta`** (or `&http_accept=application/dna+fasta` in the URL when a
custom header cannot be set).

- **Identifier required:** `genome_id` (BV-BRC's own identifier, e.g. `562.56783`) to discover a
  genome's contigs via the JSON-mode `genome_sequence` index query, then `sequence_id` (e.g.
  `562.56783.con.0444`) to fetch one specific contig's FASTA. **Neither step requires an NCBI
  `assembly_accession`, `biosample_accession`, or any other NCBI cross-reference** - confirmed
  empirically in Section 3, not merely assumed from the schema.
- **Authentication:** none required for public genomes. No API key, token, or login was supplied
  anywhere in this session's requests, and all three live test genomes (all public BV-BRC data)
  returned data successfully. This matches the documented general behavior of the BV-BRC Data API
  ("Users are able to view public data... " per the API README) and this project's prior
  reconnaissance work, which also queried public data without authentication.
- **Batching:** the `genome_sequence` collection supports the same RQL operators as the rest of the
  API (`in()`, `select()`, `limit()`), so multiple `sequence_id`s or `genome_id`s can be queried in
  one request - this was not stress-tested at scale in this session (out of scope: "do not download
  all candidate genome sequences"), but the mechanism is the same one already verified for batched
  genome-metadata lookups in `src/bvbrc_recon/genome_join.py`.
- **Documented limits:** the same 25,000-record default cap and `&http_download=true` +
  `sort(+UNIQUE_KEY)` bulk-download exception documented for the rest of the API (see the earlier
  reconnaissance report) presumably applies here too, but was not re-tested in this session.

A second, independent route exists on paper (FTPS: `ftps://ftp.bv-brc.org/genomes/$genome_id/$genome_id.fna`),
also keyed by `genome_id` directly. It was **not** live-tested in this session (no FTPS client was
available/exercised here), so it is listed as a documented fallback, not a verified one.

## 3. Controlled test: method and results

**Method:** selected 3 genome IDs from the existing `ciprofloxacin_amr_metadata.csv` (read-only;
not modified), deliberately mixing genomes with and without a populated NCBI
`genome_assembly_accession`:

| genome_id | NCBI assembly_accession present? |
|---|---|
| `562.7738` | Yes (`GCA_001285565.1`) |
| `562.56783` | No |
| `562.145027` | No |

For each genome: (1) a small JSON query against `genome_sequence` for up to 5 contigs
(`sequence_id`, `length`, `accession` only - no sequence data), then (2) one FASTA fetch for the
single *smallest* contig among those 5 (bounded to avoid pulling a large chromosome-scale
sequence). This logic lives in `src/bvbrc_recon/sequence_retrieval.py::probe_genome`.

**Results** (all three succeeded):

| genome_id | assembly_accession | sequence_id fetched | HTTP status | Content-Type | Response bytes | Valid FASTA header | Nonempty sequence |
|---|---|---|---|---|---|---|---|
| `562.7738` | `GCA_001285565.1` | `562.7738.con.0071` | 200 | `application/dna+fasta` | 603 | Yes | Yes |
| `562.56783` | *(none)* | `562.56783.con.0444` | 200 | `application/dna+fasta` | 348 | Yes | Yes |
| `562.145027` | *(none)* | `562.145027.con.0038` | 200 | `application/dna+fasta` | 50,974 | Yes | Yes |

Each response's first line was a well-formed FASTA header starting with `>` and including the
BV-BRC accession-style identifier, a description, and the source genome name/id in brackets
(e.g. `>accn|562.56783.con.0444   NODE_444_length_235_cov_289.583333   [Escherichia coli strain
372-13 | 562.56783]`), followed by nonempty nucleotide sequence (spot-checked: a lowercase
`acgt`-only alphabet in every sampled preview). Only a 60-base preview of each sequence was kept
for this report - **no full sequence or FASTA file was written to the repository.**

**Negative-control checks** (to characterize error behavior per Step 2):

| Scenario | Result |
|---|---|
| Well-formed but nonexistent `sequence_id` | HTTP 200, **empty response body** (0 bytes) |
| Well-formed but nonexistent `genome_id` (index query) | HTTP 200 implicit, **empty JSON array** (`[]`) |
| Malformed `genome_id` (contains invalid characters) | **HTTP 400 Bad Request** |

This gives a verified, practical rule for later error handling: **a missing record is not an
HTTP error at all** - it is a 200 response with an empty body/list, indistinguishable at the
status-code level from "found but empty," so code must check payload emptiness explicitly, not
rely on catching an exception. A genuinely invalid identifier format, by contrast, does produce a
4xx. **Deprecated-genome and transient-server-error behavior were not empirically tested** - no
deprecated genome_id with a confirmed sequence record was available to test against in this
session, and a live 5xx could not be deliberately triggered; these remain open per Section 6.

## 4. Does retrieval depend on an NCBI accession?

**No - confirmed, not assumed.** Two of the three test genomes had no `assembly_accession` at all
(consistent with the prior reconnaissance finding that only ~6-7% of genomes in this dataset have
one), and both retrieved a valid FASTA contig successfully via `genome_id` -> `sequence_id` alone.
The NCBI accession fields (`assembly_accession`, `biosample_accession`, etc.) appear to be metadata
convenience cross-references in the `genome` collection, not a precondition for sequence retrieval
through `genome_sequence`.

## 5. Batching, rate-limit, reliability, and error-handling concerns

- **Batching**: `in(genome_id,(...))` or `in(sequence_id,(...))` should work for the JSON-mode index
  query (same mechanism already used in `genome_join.py`), reducing request count for bulk work.
  Whether the FASTA-mode (`application/dna+fasta`) response concatenates multiple requested
  sequences cleanly (with one header per contig) was not tested in this session - a reasonable
  follow-up before bulk retrieval.
- **Rate limits**: none encountered in this session's small number of requests; no documented
  explicit rate limit was found (same conclusion as the prior reconnaissance report).
- **Reliability**: the existing `BVBRCClient` retry/backoff logic (for 429/5xx, timeouts, and
  malformed JSON) only applies to the **JSON-mode** index query, not to the raw FASTA fetch, which
  this probe implemented separately with its own, simpler error handling (HTTP error -> recorded,
  not retried). Before any bulk use, the FASTA fetch path should get the same retry treatment as
  the rest of the client.
- **Error handling**: as found in Section 3 - treat an empty body as "not found," not as success or
  failure; treat 4xx as a genuine request problem (bad identifier/query); treat 5xx/timeout as
  transient and retry, per the existing client's pattern.
- **Size risk**: a genome reported as `genome_status: "Complete"` (a finished, few-contig or
  single-chromosome assembly) could return a multi-megabyte single-contig FASTA response if
  queried without the "pick the smallest contig" bound used here. Any future bulk work must budget
  for this explicitly (e.g. by streaming to disk rather than loading a full response into memory).

## 6. Verifying completeness and file integrity (for a later stage)

Not implemented in this feasibility pass, but identified as necessary before bulk retrieval:

- Compare the number of contigs retrieved for a genome against the `contigs` count already present
  in the joined genome metadata (`genome_contigs` in the existing CSVs) - a mismatch would indicate
  an incomplete pull.
- Compare total retrieved sequence length against `genome_length` from the same metadata.
- The `genome_sequence` schema includes a `sequence_md5` field (confirmed present in the Solr
  schema, not used in this probe) - a checksum comparison against that field, where populated,
  would be the strongest available integrity check, but was not exercised here.
- Confirm `genome_status` for every genome retrieved in bulk is not `"Deprecated"` (a handful of
  such records were observed in the earlier reconnaissance pass) before including it in a cohort.

## 7. What remains uncertain

- Whether the FTPS route (`ftps://ftp.bv-brc.org/genomes/...`) actually works as documented - not
  independently tested in this session.
- Whether `application/dna+fasta` behaves identically for `in(sequence_id,(...))` batched requests
  (multiple contigs in one call) as it does for the single-id requests tested here.
- Deprecated-genome and transient-server-error response behavior for this specific route - not
  observed live.
- Whether BV-BRC enforces any undocumented rate limit under sustained bulk use (this session's
  request volume was far too small to surface one).
- Whether `sequence_md5` is populated consistently enough to serve as a reliable integrity check
  across the full candidate genome set - not checked in this session.

None of the above has been resolved by invented rules or assumptions; each is listed here as an
open question for validation before any bulk retrieval is attempted.
