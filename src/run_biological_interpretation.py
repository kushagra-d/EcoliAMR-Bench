"""Biological interpretation of the ciprofloxacin pilot's predictive
k-mers against the official CARD (Comprehensive Antibiotic Resistance
Database) reference sequences.

Reuses the EXISTING pilot k-mer matrix, vocabulary, genome order, labels,
and saved random-split assignment - no genome is downloaded, no k-mer is
recomputed, and no existing artifact is modified. Candidate k-mers are
selected via a training-partition-only refit (no fitted model was
persisted by earlier tasks), then matched against a local copy of CARD's
reference sequences already downloaded to data/external/card/extracted/
(see the report for the exact, reproducible retrieval command).

Usage (from the project root, with the project venv active):

    python src/run_biological_interpretation.py

Writes reports/pilot_ciprofloxacin_biological_interpretation.md and
data/processed/pilot_ciprofloxacin_card_candidate_matches.csv (new files
only).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from bvbrc_recon.bio_interpretation_report import render_bio_interpretation_report
from bvbrc_recon.card_matching import match_candidates
from bvbrc_recon.card_reference import build_card_kmer_index
from bvbrc_recon.kmer_candidates import select_candidate_kmers

ANTIBIOTIC = "ciprofloxacin"
SEED = 20261009
TOP_N_PER_MODEL = 20

DATA_PROCESSED = Path("data/processed")
CARD_DIR = Path("data/external/card/extracted")
REPORT_PATH = Path("reports/pilot_ciprofloxacin_biological_interpretation.md")


def load_pilot_assets():
    genome_order = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_genome_order.csv", dtype={"genome_id": str}
    )["genome_id"].tolist()
    selected = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_selected_cohort.csv", dtype={"genome_id": str}
    )
    label_by_genome = dict(zip(selected["genome_id"], selected["label"]))
    labels = [label_by_genome[g] for g in genome_order]

    matrix = sparse.load_npz(DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_matrix.npz")
    vocab_df = pd.read_csv(DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_vocabulary.csv")
    vocabulary = dict(zip(vocab_df["kmer"], vocab_df["column_index"]))

    random_split = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_random_split.csv", dtype={"genome_id": str}
    )
    genome_to_row = {gid: i for i, gid in enumerate(genome_order)}
    train_idx = np.array(
        [genome_to_row[g] for g in random_split.loc[random_split["partition"] == "train", "genome_id"]]
    )

    return genome_order, labels, matrix, vocabulary, train_idx


def get_card_version() -> str:
    card_json_path = CARD_DIR / "card.json"
    with open(card_json_path) as fh:
        data = json.load(fh)
    return data.get("_version", "unknown")


def main() -> None:
    genome_order, labels, matrix, vocabulary, train_idx = load_pilot_assets()
    print(f"Loaded pilot: {len(genome_order)} genomes, matrix shape {matrix.shape}, train n={len(train_idx)}")

    selection = select_candidate_kmers(
        matrix, labels, vocabulary, train_idx, seed=SEED, top_n_per_model=TOP_N_PER_MODEL
    )
    print(f"Selected {len(selection.combined_candidate_kmers)} unique candidate k-mers (union of both models' top {TOP_N_PER_MODEL})")

    card_version = get_card_version()
    print(f"CARD version: {card_version}")

    print("Building CARD k-mer index (k=10) from data/external/card/extracted/ ...")
    card_index = build_card_kmer_index(CARD_DIR, k=10)
    print(
        f"CARD index: {card_index.n_sequences_loaded} sequences, {len(card_index.index):,} distinct canonical k-mers"
    )

    matches = match_candidates(selection.combined_candidate_kmers, card_index)
    n_exact = sum(1 for m in matches if m.match_type == "exact")
    n_approx = sum(1 for m in matches if m.match_type == "approximate")
    n_unmapped = sum(1 for m in matches if m.match_type == "unmapped")
    print(f"Matches: {n_exact} exact, {n_approx} approximate, {n_unmapped} unmapped")

    rows = []
    for match in matches:
        model_hits = {
            model_name: ranking.top_ranks[match.kmer]
            for model_name, ranking in selection.rankings.items()
            if match.kmer in ranking.top_ranks
        }
        rows.append(
            {
                "kmer": match.kmer,
                "selected_by_models": ";".join(model_hits),
                "model_ranks": ";".join(f"{m}:{r}" for m, r in model_hits.items()),
                "match_type": match.match_type,
                "is_ambiguous": match.is_ambiguous,
                "n_gene_hits": len(match.gene_hits),
                "gene_hits": "; ".join(
                    f"{h.short_name}|{h.aro_id}|n_seq={h.n_supporting_sequences}|organisms={','.join(h.organisms)}"
                    for h in match.gene_hits
                ),
            }
        )
    matches_df = pd.DataFrame(rows)
    matches_csv_path = DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_card_candidate_matches.csv"
    matches_df.to_csv(matches_csv_path, index=False)
    print(f"Candidate match details written -> {matches_csv_path}")

    report_text = render_bio_interpretation_report(
        antibiotic=ANTIBIOTIC,
        seed=SEED,
        analysis_date=datetime.now(timezone.utc).isoformat(),
        card_version=card_version,
        card_retrieval_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        card_source_url="https://card.mcmaster.ca/latest/data",
        selection=selection,
        card_index=card_index,
        matches=matches,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"Biological interpretation report written -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
