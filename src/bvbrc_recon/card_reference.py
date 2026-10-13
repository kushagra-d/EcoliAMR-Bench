"""Loads the official CARD (Comprehensive Antibiotic Resistance Database)
reference nucleotide sequences and builds a canonical k-mer index over
them, for matching against the pilot's own canonical k-mer features.

Source: https://card.mcmaster.ca/latest/data (the official CARD data
download endpoint). Retrieved live and extracted to
data/external/card/extracted/ - see reports/pilot_ciprofloxacin_biological_interpretation.md
for the exact retrieval command, CARD version, and license terms. This
module reads those already-extracted files; it does not re-download them
on every run (the download step is a one-time, documented, reproducible
command, not part of the routine pipeline).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from Bio import SeqIO

from .kmer_features import canonical_kmer

# All five CARD model types that ship nucleotide reference sequences, per
# CARD-Download-README.txt. Included for completeness/reproducibility -
# not pre-filtered to any particular drug class, so matches to AMR genes
# unrelated to fluoroquinolones are reported too (see report for why that
# matters at k=10).
DEFAULT_CARD_FASTA_FILENAMES = [
    "nucleotide_fasta_protein_homolog_model.fasta",
    "nucleotide_fasta_protein_variant_model.fasta",
    "nucleotide_fasta_protein_knockout_model.fasta",
    "nucleotide_fasta_protein_overexpression_model.fasta",
    "nucleotide_fasta_rRNA_gene_variant_model.fasta",
]

# ">gb|ACCESSION|STRAND|START-END|ARO:ID|SHORT_NAME [ORGANISM]" - verified
# against every header in all five files (6,415 sequences, 100% matched
# this exact shape) before relying on it.
_HEADER_RE = re.compile(
    r"^gb\|(?P<accession>[^|]+)\|(?P<strand>[+-])\|(?P<start>\d+)-(?P<end>\d+)\|"
    r"(?P<aro_id>ARO:\d+)\|(?P<short_name>\S+)\s*\[(?P<organism>[^\]]*)\]"
)

VALID_BASES = set("ACGT")


@dataclass(frozen=True)
class CardSequenceMeta:
    accession: str
    strand: str
    start: int
    end: int
    aro_id: str
    short_name: str
    organism: str
    source_model: str  # which CARD FASTA file this came from


def parse_card_header(header: str, source_model: str) -> CardSequenceMeta | None:
    """Returns None (not raises) for a header that doesn't match the
    expected shape - callers decide whether an unparsed header is a
    blocker; this never fabricates values for a header it can't parse."""
    match = _HEADER_RE.match(header)
    if match is None:
        return None
    return CardSequenceMeta(
        accession=match["accession"],
        strand=match["strand"],
        start=int(match["start"]),
        end=int(match["end"]),
        aro_id=match["aro_id"],
        short_name=match["short_name"],
        organism=match["organism"],
        source_model=source_model,
    )


@dataclass
class CardIndex:
    k: int
    index: dict[str, list[CardSequenceMeta]] = field(default_factory=dict)
    n_sequences_loaded: int = 0
    n_sequences_unparsed_header: int = 0
    n_windows_indexed: int = 0
    n_windows_skipped_ambiguous: int = 0
    unparsed_headers: list[str] = field(default_factory=list)


def build_card_kmer_index(fasta_dir: Path, k: int, filenames: list[str] = DEFAULT_CARD_FASTA_FILENAMES) -> CardIndex:
    card_index = CardIndex(k=k)

    for filename in filenames:
        path = fasta_dir / filename
        for record in SeqIO.parse(str(path), "fasta"):
            meta = parse_card_header(record.description, source_model=filename)
            if meta is None:
                card_index.n_sequences_unparsed_header += 1
                card_index.unparsed_headers.append(record.description)
                continue
            card_index.n_sequences_loaded += 1

            sequence = str(record.seq).upper()
            for i in range(len(sequence) - k + 1):
                window = sequence[i : i + k]
                if not VALID_BASES.issuperset(window):
                    card_index.n_windows_skipped_ambiguous += 1
                    continue
                key = canonical_kmer(window)
                card_index.index.setdefault(key, []).append(meta)
                card_index.n_windows_indexed += 1

    return card_index
