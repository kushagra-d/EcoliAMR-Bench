"""Entry point: controlled, bounded feasibility probe for BV-BRC genome
sequence FASTA retrieval (genome_sequence collection, Accept:
application/dna+fasta).

Tests at most a handful of genome IDs (a mix with and without an NCBI
assembly_accession), fetching exactly one small contig's FASTA per
genome. Prints a result summary; does not write any sequence data to
disk and does not touch the existing reconnaissance CSVs or reports.

Usage (from the project root, with the project venv active):

    python src/run_sequence_retrieval_probe.py
"""

from __future__ import annotations

from bvbrc_recon.client import BVBRCClient
from bvbrc_recon.sequence_retrieval import probe_genome

# Selected from data/processed/ciprofloxacin_amr_metadata.csv: a mix of
# genomes with and without a populated NCBI assembly_accession, so the
# probe does not assume the route only works for one or the other.
CANDIDATE_GENOMES = [
    ("562.7738", "GCA_001285565.1"),  # has an NCBI assembly_accession
    ("562.56783", None),  # no NCBI assembly_accession
    ("562.145027", None),  # no NCBI assembly_accession
]


def main() -> None:
    client = BVBRCClient()
    for genome_id, assembly_accession in CANDIDATE_GENOMES:
        result = probe_genome(client, genome_id, assembly_accession)
        print(f"\n=== genome_id={result.genome_id} assembly_accession={result.assembly_accession} ===")
        print(f"status: {result.status}")
        if result.note:
            print(f"note: {result.note}")
        print(f"contigs sampled from index: {len(result.index_entries)}")
        if result.fetch is not None:
            f = result.fetch
            print(f"sequence_id fetched: {f.sequence_id}")
            print(f"HTTP status: {f.http_status}, content-type: {f.content_type}, bytes: {f.response_bytes}")
            print(f"valid FASTA header: {f.is_valid_fasta} ({f.header_line!r})")
            print(f"nonempty sequence: {f.nonempty_sequence}")
            print(f"sequence preview (first 60 bases only): {f.sequence_preview!r}")
            if f.error:
                print(f"error: {f.error}")


if __name__ == "__main__":
    main()
