"""Matches candidate canonical k-mers against a CardIndex (see
card_reference.py), classifying each as exact, approximate (single
nucleotide substitution), or unmapped - never claiming an approximate
match as a definitive gene identification, and explicitly flagging
ambiguity when a k-mer matches more than one distinct CARD gene/short_name.

Efficient by construction: because `CardIndex` is keyed by CANONICAL
k-mer (same canonicalization as the pilot's own feature matrix), and
reverse-complementing preserves Hamming distance, an exact or
approximate match in either strand orientation is found by generating
single-substitution neighbors of the candidate's own canonical string
and looking each one up directly - no need to re-scan the reference
sequences or separately check the reverse complement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .card_reference import CardIndex, CardSequenceMeta
from .kmer_features import canonical_kmer

BASES = "ACGT"

EXACT = "exact"
APPROXIMATE = "approximate"
UNMAPPED = "unmapped"


@dataclass(frozen=True)
class GeneHit:
    aro_id: str
    short_name: str
    organisms: tuple  # distinct organisms this gene/short_name was observed in among the hits
    n_supporting_sequences: int  # how many individual CARD reference sequences support this hit
    source_models: tuple  # which CARD FASTA file(s) this came from


@dataclass
class CandidateMatch:
    kmer: str
    match_type: str  # EXACT, APPROXIMATE, or UNMAPPED
    gene_hits: list[GeneHit] = field(default_factory=list)
    is_ambiguous: bool = False  # True if gene_hits spans more than one distinct (aro_id, short_name)


def _single_substitution_neighbors(kmer: str) -> set[str]:
    neighbors = set()
    for i, original_base in enumerate(kmer):
        for base in BASES:
            if base != original_base:
                neighbors.add(kmer[:i] + base + kmer[i + 1 :])
    return neighbors


def _summarize_hits(metas: list[CardSequenceMeta]) -> list[GeneHit]:
    by_gene: dict[tuple, list[CardSequenceMeta]] = {}
    for meta in metas:
        key = (meta.aro_id, meta.short_name)
        by_gene.setdefault(key, []).append(meta)

    hits = []
    for (aro_id, short_name), group in by_gene.items():
        hits.append(
            GeneHit(
                aro_id=aro_id,
                short_name=short_name,
                organisms=tuple(sorted({m.organism for m in group})),
                n_supporting_sequences=len(group),
                source_models=tuple(sorted({m.source_model for m in group})),
            )
        )
    return sorted(hits, key=lambda h: -h.n_supporting_sequences)


def match_candidate(kmer: str, card_index: CardIndex) -> CandidateMatch:
    if len(kmer) != card_index.k:
        raise ValueError(f"candidate k-mer length {len(kmer)} does not match CardIndex k={card_index.k}")

    canon = canonical_kmer(kmer)

    exact_metas = card_index.index.get(canon)
    if exact_metas:
        hits = _summarize_hits(exact_metas)
        return CandidateMatch(
            kmer=kmer, match_type=EXACT, gene_hits=hits, is_ambiguous=len(hits) > 1
        )

    approx_metas: list[CardSequenceMeta] = []
    for neighbor in _single_substitution_neighbors(canon):
        neighbor_canon = canonical_kmer(neighbor)
        metas = card_index.index.get(neighbor_canon)
        if metas:
            approx_metas.extend(metas)

    if approx_metas:
        hits = _summarize_hits(approx_metas)
        return CandidateMatch(
            kmer=kmer, match_type=APPROXIMATE, gene_hits=hits, is_ambiguous=len(hits) > 1
        )

    return CandidateMatch(kmer=kmer, match_type=UNMAPPED, gene_hits=[], is_ambiguous=False)


def match_candidates(kmers: list[str], card_index: CardIndex) -> list[CandidateMatch]:
    return [match_candidate(kmer, card_index) for kmer in kmers]
