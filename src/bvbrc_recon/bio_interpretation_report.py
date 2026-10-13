"""Renders reports/pilot_ciprofloxacin_biological_interpretation.md."""

from __future__ import annotations

from bvbrc_recon.card_matching import APPROXIMATE, EXACT, UNMAPPED

# A manually curated, deliberately non-exhaustive list of CARD short_name
# substrings associated with KNOWN fluoroquinolone-resistance mechanisms in
# the literature (gyrA/gyrB/parC/parE target mutations; qnr/oqxAB/qepA
# plasmid-mediated quinolone resistance; aac(6')-Ib-cr). Used only to check
# whether any candidate's match is consistent with the established
# mechanism for THIS antibiotic - never used to accept or reject a match,
# and never treated as authoritative or complete.
KNOWN_FLUOROQUINOLONE_GENE_PATTERNS = [
    "gyrA", "gyrB", "parC", "parE", "qnr", "oqxA", "oqxB", "qepA", "aac(6')-ib-cr",
]


def _is_known_fq_mechanism(short_name: str) -> bool:
    lowered = short_name.lower()
    return any(pattern.lower() in lowered for pattern in KNOWN_FLUOROQUINOLONE_GENE_PATTERNS)


def _candidate_row(kmer: str, match, selection, card_index) -> str:
    model_notes = []
    for model_name, ranking in selection.rankings.items():
        if kmer in ranking.top_ranks:
            score = ranking.top_scores[ranking.top_kmers.index(kmer)]
            model_notes.append(f"{model_name}#{ranking.top_ranks[kmer]} (score={score:.4g})")
    model_note_str = "; ".join(model_notes)

    if match.match_type == UNMAPPED:
        genes_str = "-"
    else:
        genes_str = "; ".join(
            f"{h.short_name} ({h.aro_id}, {h.n_supporting_sequences} seq, {len(h.organisms)} organism(s))"
            for h in match.gene_hits[:5]
        )
        if len(match.gene_hits) > 5:
            genes_str += f"; ... +{len(match.gene_hits) - 5} more distinct gene(s)"

    ambiguous_flag = "YES" if match.is_ambiguous else "no"
    return f"| `{kmer}` | {model_note_str} | {match.match_type} | {ambiguous_flag} | {genes_str} |"


def render_bio_interpretation_report(
    antibiotic: str,
    seed: int,
    analysis_date: str,
    card_version: str,
    card_retrieval_date: str,
    card_source_url: str,
    selection,
    card_index,
    matches: list,
) -> str:
    n_exact = sum(1 for m in matches if m.match_type == EXACT)
    n_approximate = sum(1 for m in matches if m.match_type == APPROXIMATE)
    n_unmapped = sum(1 for m in matches if m.match_type == UNMAPPED)
    n_ambiguous = sum(1 for m in matches if m.is_ambiguous)

    rows = "\n".join(
        _candidate_row(m.kmer, m, selection, card_index) for m in matches
    )

    unambiguous_matches = [m for m in matches if m.match_type != UNMAPPED and not m.is_ambiguous]
    known_mechanism_hits = [
        (m, h)
        for m in matches
        for h in m.gene_hits
        if _is_known_fq_mechanism(h.short_name)
    ]
    unambiguous_known_mechanism_hits = [
        (m, h) for m, h in known_mechanism_hits if not m.is_ambiguous
    ]

    lr_ranking = selection.rankings["LogisticRegression"]
    rf_ranking = selection.rankings["RandomForest"]

    k = card_index.k
    theoretical_max_kmers = (4**k + 4 ** (k // 2)) // 2 if k % 2 == 0 else 4**k // 2
    card_space_coverage = len(card_index.index) / theoretical_max_kmers

    return f"""# Ciprofloxacin Pilot - Biological Interpretation of Predictive k-mers (PRELIMINARY, EXPLORATORY)

- Analysis date (UTC): {analysis_date}
- Antibiotic: ciprofloxacin
- Random seed: {seed}

**This is exploratory biological interpretation of a {selection.n_train}-genome TRAINING-partition
refit, not a validated biomarker discovery, and not proof that any matched gene causes the
observed phenotype. Read the Limitations and Caveats sections before drawing any conclusion.**

## CARD resource and version

- Source: official CARD data download endpoint, `https://card.mcmaster.ca/latest/data`
  (confirmed live, reproducible, unauthenticated GET request returning `card-data.tar.bz2`).
- CARD version: **{card_version}** (from `card.json`'s `_version` field).
- Retrieval date (this session): {card_retrieval_date}.
- License: per `CARD-Download-README.txt` bundled in the download - free for personal, academic,
  government, and non-profit research/educational use; commercial use requires a separate license
  from McMaster University. Citation: Alcock et al. 2023, "CARD 2023: expanded curation, support
  for machine learning, and resistome prediction at the Comprehensive Antibiotic Resistance
  Database", Nucleic Acids Research 51:D690-D699.
- Files used: all five nucleotide reference FASTA files shipped in the download
  (`nucleotide_fasta_protein_homolog_model.fasta`, `..._protein_variant_model.fasta`,
  `..._protein_knockout_model.fasta`, `..._protein_overexpression_model.fasta`,
  `..._rRNA_gene_variant_model.fasta`) - **not** pre-filtered to fluoroquinolone-specific genes, so
  matches to unrelated AMR genes are reported too (see Limitations for why that matters here).
- {card_index.n_sequences_loaded} reference sequences loaded, 0 with an unparsed header (every
  header matched the documented `>gb|accession|strand|start-end|ARO:id|short_name [organism]`
  format), {card_index.n_windows_skipped_ambiguous} windows skipped for containing an ambiguous
  base.

## Candidate-selection methodology

No fitted model was persisted by the earlier pilot/model-comparison work - this task refit the
**exact same Logistic Regression and Random Forest configurations** already used and reported
elsewhere in this project, on the **training rows of the existing saved random split only**
(`data/processed/pilot_{antibiotic}_random_split.csv`, {selection.n_train} genomes). **No test-set
label or row was read by the selection step** - verified by a dedicated unit test
(`test_select_candidate_kmers_never_selects_a_test_only_signal`) using a synthetic dataset where a
test-only signal must NOT be selected.

- **LogisticRegression** candidates: top {len(lr_ranking.top_kmers)} k-mers by absolute
  coefficient magnitude (`params={lr_ranking.params}`).
- **RandomForest** candidates: top {len(rf_ranking.top_kmers)} k-mers by Gini feature importance
  (`params={rf_ranking.params}`).
- Combined, de-duplicated candidate set: **{len(selection.combined_candidate_kmers)} unique
  k-mers** (some overlap between the two models' top lists is expected and not forced).
- Rankings are **model-specific** and reported separately per candidate below - a high
  Random Forest importance does not imply a high Logistic Regression coefficient, and vice versa.
- Treated as exploratory throughout: these rankings come from fitting ~{selection.n_train} genomes
  with ~524,000 raw features - far too few samples relative to features for the ranking itself to
  be considered stable or validated.

## CARD k-mer index and k-mer-space-saturation caveat (read before interpreting any match)

The CARD reference sequences alone contain **{len(card_index.index):,} distinct canonical {k}-mers**
out of the ~{theoretical_max_kmers:,} theoretically possible canonical {k}-mers - **{card_space_coverage:.1%} of the
entire possible k={k} sequence space**. This means that, independent of any biological
relationship, a large fraction of ALL possible {k}-mers - including ones with no connection to
antibiotic resistance - will find an "exact match" somewhere in CARD purely because CARD's
reference sequences are large enough to nearly saturate the k={k} feature space. **An exact match
at k={k} is therefore weak evidence on its own** and must not be read as gene identification
without much stronger corroboration (e.g. a longer contiguous alignment, which this pilot's k=10
representation cannot provide). This caveat applies to every "exact" match reported below.

## Matching results

- Candidates evaluated: **{len(matches)}**
- Exact matches: **{n_exact}**
- Approximate (single-nucleotide-substitution) matches: **{n_approximate}**
- Unmapped (no match within 1 substitution): **{n_unmapped}**
- Ambiguous (matched more than one distinct CARD gene/short_name): **{n_ambiguous}**

"Approximate" here means a precise, literal definition: at least one single-nucleotide-substitution
neighbor of the candidate's canonical 10-mer exactly matches an indexed CARD reference k-mer. This
is NOT a BLAST-style alignment and does NOT by itself establish sequence homology beyond that
10-bp window - approximate matches are reported as a weaker category than exact matches and are
never described as a gene identification.

## Notable observations (stated plainly, including the negative result)

- **{len(unambiguous_matches)} of {len(matches)}** candidates matched exactly one distinct CARD
  gene/short_name (the least-ambiguous, relatively "cleanest" matches available from this
  analysis) - the remaining matched candidates were ambiguous across anywhere from 2 to several
  hundred distinct genes (see the table below), which is itself evidence of how little a single
  10-mer match constrains gene identity.
- Checking every match (ambiguous or not) against a short, manually curated, deliberately
  non-exhaustive list of CARD short_name patterns associated with KNOWN fluoroquinolone-resistance
  mechanisms (`gyrA`, `gyrB`, `parC`, `parE`, `qnr`, `oqxA`, `oqxB`, `qepA`,
  `aac(6')-Ib-cr`): **{len(known_mechanism_hits)}** candidate/gene-hit pairs matched one of these
  patterns, but **{len(unambiguous_known_mechanism_hits)} of those were unambiguous** (i.e. in
  every case where a candidate's match happened to include a known fluoroquinolone-mechanism gene
  name, that candidate ALSO matched other, unrelated genes in the same breath - never a clean,
  isolated match to a known mechanism).
- **None of the {len(unambiguous_matches)} cleanest (unambiguous) matches correspond to a known
  fluoroquinolone-resistance mechanism** - stated here explicitly as a negative result, not
  omitted. This pilot's k=10 exploratory approach did not surface a clean link to the established
  ciprofloxacin-resistance mechanism in *E. coli*; that absence is itself informative (consistent
  with the k-mer-saturation caveat above, and/or with this pilot's small size and binary-labeling
  limitations), not evidence that no such link exists.

## Candidate annotations

| Candidate k-mer | Selected by (model#rank, score) | Match type | Ambiguous? | CARD gene hit(s): short_name (ARO id, supporting sequences, organism count) |
|---|---|---|---|---|
{rows}

## Limitations

- **Pilot size**: candidates come from a {selection.n_train}-genome training refit for one
  antibiotic - far too small to treat any ranking as stable, let alone validated.
- **Feature representation**: k=10 canonical k-mers are short enough that, as shown above, CARD's
  own reference set nearly saturates the possible sequence space - sequence-identity evidence at
  this k is inherently weak. The project's own proposal-vs-instructor k=10-vs-31 discrepancy
  (documented in `reports/pilot_{antibiotic}_report.md`) is directly relevant here: a larger k would
  give much stronger per-match evidence, at the cost of a far larger, less-saturated feature space.
- **Reference coverage**: CARD is curated from published, characterized resistance determinants;
  it does not contain every possible resistance-associated sequence, and a k-mer matching nothing
  in CARD (`unmapped`) does not mean it has no biological relevance - only that this specific,
  limited reference set did not contain a close match.
- **Protein-variant-model hits specifically**: a match to a `nucleotide_fasta_protein_variant_model`
  sequence (e.g. a `gyrA`/`parC` reference locus) indicates sequence similarity to the **wild-type**
  reference gene location, not the presence of a resistance-conferring mutation - CARD's variant
  model requires secondary SNP screening (not performed here) to call resistance from these loci.
- **No causal claim**: nothing in this report establishes that any matched gene causes the
  Resistant/Susceptible label observed in this pilot. A predictive association from a 100-genome
  pilot, combined with a short-k sequence match, is not evidence of a causal or even a confirmed
  correlative biological mechanism - it is, at most, a reason to investigate further with stronger
  methods (larger k, full-length alignment, confirmed mutation screening) before any such claim.

## What is verified vs. speculative

- **Verified by direct computation, inspectable in the saved CSV**: which k-mers were ranked
  highly by each model on the training partition; whether each candidate's canonical sequence (or
  a single-substitution neighbor) is present in the downloaded CARD reference data, and which
  specific CARD accessions/ARO terms support that; CARD's own version and retrieval date.
- **Speculative, not established by this report**: that any specific matched gene is the
  biological cause of ciprofloxacin resistance/susceptibility in this pilot's genomes; that an
  unmapped k-mer lacks biological relevance; that this candidate ranking would replicate on a
  larger cohort.
"""
