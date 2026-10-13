"""Sequence-level QC for retrieved pilot assemblies, using Biopython
(already a project dependency) to parse FASTA.

Reports observed characteristics only - no quality threshold is applied
or invented here. A "suspicious" assembly is still reported, not dropped;
deciding what to do about it is left to a later, explicit step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from Bio import SeqIO

AMBIGUOUS_BASES = set("NRYSWKMBDHVU")  # non-ACGT IUPAC nucleotide codes (case-insensitive)


@dataclass
class AssemblyQC:
    genome_id: str
    fasta_path: str
    n_contigs: int = 0
    total_length: int = 0
    ambiguous_base_count: int = 0
    ambiguous_base_proportion: float | None = None
    duplicate_sequence_ids: list[str] = field(default_factory=list)
    empty_contig_count: int = 0
    is_nonempty: bool = False
    has_valid_nucleotide_data: bool = False
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


def qc_assembly(genome_id: str, fasta_path: Path) -> AssemblyQC:
    qc = AssemblyQC(genome_id=genome_id, fasta_path=str(fasta_path))

    if not fasta_path.exists():
        qc.error = f"FASTA file does not exist: {fasta_path}"
        return qc

    try:
        contig_records = list(SeqIO.parse(str(fasta_path), "fasta"))
    except Exception as exc:  # noqa: BLE001 - any parse failure is reported, not swallowed silently
        qc.error = f"Biopython failed to parse this FASTA file: {exc}"
        return qc

    qc.n_contigs = len(contig_records)
    if qc.n_contigs == 0:
        qc.warnings.append("file parsed but contained zero FASTA records")
        return qc

    seen_ids: set[str] = set()
    for record in contig_records:
        seq_str = str(record.seq)
        length = len(seq_str)
        qc.total_length += length
        if length == 0:
            qc.empty_contig_count += 1
        if record.id in seen_ids:
            qc.duplicate_sequence_ids.append(record.id)
        else:
            seen_ids.add(record.id)
        for base in seq_str.upper():
            if base in AMBIGUOUS_BASES:
                qc.ambiguous_base_count += 1
            elif base not in "ACGT":
                qc.warnings.append(f"unexpected character {base!r} in contig {record.id!r}")

    qc.is_nonempty = qc.total_length > 0
    qc.has_valid_nucleotide_data = qc.is_nonempty and qc.ambiguous_base_count < qc.total_length
    qc.ambiguous_base_proportion = (
        qc.ambiguous_base_count / qc.total_length if qc.total_length > 0 else None
    )

    if qc.empty_contig_count:
        qc.warnings.append(f"{qc.empty_contig_count} contig(s) had an empty sequence")
    if qc.duplicate_sequence_ids:
        qc.warnings.append(f"duplicate contig identifiers within the file: {qc.duplicate_sequence_ids}")

    return qc
