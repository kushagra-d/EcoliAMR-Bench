"""Step 2 of the incremental-model-fix task: reproduces the existing
batch LogisticRegression pilot result from scratch, using only existing,
already-saved artifacts (the hashed matrix, genome order, cohort labels,
both splits) - confirming labels/row-ordering/splits are identical to
what the incremental pipeline uses, before trusting any comparison
between the two. Writes to a NEW path; never touches the original saved
result (`pilot_ciprofloxacin_kmer_hashed_config_d262144.json`, which
already contains its own `model_results` block from the original task).

Usage (from the project root, with the project venv active):

    python src/run_batch_baseline_reproduction.py

Writes:
    data/processed/batch_lr_reproduction_ciprofloxacin_d262144.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from bvbrc_recon.pilot_model import evaluate_with_precomputed_split

ANTIBIOTIC = "ciprofloxacin"
SEED = 20261009
N_BUCKETS = 2**18

DATA_PROCESSED = Path("data/processed")
OUTPUT_PATH = DATA_PROCESSED / f"batch_lr_reproduction_{ANTIBIOTIC}_d{N_BUCKETS}.json"
EXISTING_RESULT_PATH = DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_hashed_config_d{N_BUCKETS}.json"


def main() -> None:
    genome_order = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_genome_order.csv", dtype={"genome_id": str}
    )["genome_id"].tolist()
    selected = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_selected_cohort.csv", dtype={"genome_id": str}
    )
    label_by_genome = dict(zip(selected["genome_id"], selected["label"]))
    labels = [label_by_genome[gid] for gid in genome_order]

    matrix = sparse.load_npz(DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_hashed_matrix_d{N_BUCKETS}.npz")
    if matrix.shape[0] != len(genome_order):
        raise RuntimeError("ALIGNMENT BLOCKER: hashed matrix row count does not match genome_order length")

    random_split = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_random_split.csv", dtype={"genome_id": str}
    )
    lineage_split = pd.read_csv(
        DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_lineage_split.csv", dtype={"genome_id": str}
    )
    genome_to_row = {gid: i for i, gid in enumerate(genome_order)}

    def idx_for(split_df, partition):
        return np.array(
            [genome_to_row[g] for g in split_df.loc[split_df["partition"] == partition, "genome_id"]]
        )

    model_results = {}
    for split_name, split_df in [("random", random_split), ("lineage", lineage_split)]:
        train_idx = idx_for(split_df, "train")
        test_idx = idx_for(split_df, "test")
        result = evaluate_with_precomputed_split(matrix, labels, train_idx, test_idx, seed=SEED)
        model_results[split_name] = {
            "n_train": result.n_train,
            "n_test": result.n_test,
            "train_class_counts": result.train_class_counts,
            "test_class_counts": result.test_class_counts,
            "accuracy": result.accuracy,
            "roc_auc": result.roc_auc,
            "average_precision": result.average_precision,
            "recall_per_class": result.recall_per_class,
            "confusion_matrix": result.confusion_matrix,
            "warnings": result.warnings,
        }

    # Confirm this reproduction is bit-for-bit identical to the original
    # saved result, if it exists - the real point of this step.
    matches_existing = None
    if EXISTING_RESULT_PATH.exists():
        existing = json.loads(EXISTING_RESULT_PATH.read_text())
        existing_model_results = existing.get("model_results", {})
        matches_existing = existing_model_results == model_results

    output = {
        "model": "LogisticRegression(max_iter=1000)",
        "feature_source": f"pilot_{ANTIBIOTIC}_kmer_hashed_matrix_d{N_BUCKETS}.npz (read-only)",
        "seed": SEED,
        "n_buckets": N_BUCKETS,
        "model_results": model_results,
        "matches_existing_saved_result_exactly": matches_existing,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"wrote {OUTPUT_PATH}")
    print(f"matches_existing_saved_result_exactly={matches_existing}")
    for split_name, r in model_results.items():
        print(f"[{split_name}] acc={r['accuracy']} auc={r['roc_auc']} recall={r['recall_per_class']}")


if __name__ == "__main__":
    main()
