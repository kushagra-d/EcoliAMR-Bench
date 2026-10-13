"""Consolidates all feature-representation experiment results (raw
canonical k-mers, feature hashing, minimizers) into one machine-readable
JSON for downstream use (e.g. by a teammate writing the narrative
report). Reads only already-computed artifacts; writes one new file.

Usage:
    python src/run_feature_representation_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

ANTIBIOTIC = "ciprofloxacin"
DATA_PROCESSED = Path("data/processed")
OUT_PATH = DATA_PROCESSED / "pilot_ciprofloxacin_feature_representation_comparison_summary.json"

RAW_K_VALUES = [10, 15, 17, 19, 21, 31]
HASH_DIMENSIONS = [2**16, 2**18, 2**20]
MINIMIZER_K = 31
MINIMIZER_W_VALUES = [5, 10, 15]


def _parse_lr_from_combined_log(log_text: str, k: int) -> dict:
    lines = log_text.splitlines()
    in_section = False
    results = {}
    for line in lines:
        stripped = line.strip()
        if stripped == f"=== k={k} ===":
            in_section = True
            continue
        if stripped.startswith("=== k=") and in_section:
            break
        if not in_section:
            continue
        for split_name in ("random", "lineage"):
            if stripped.startswith(f"[{split_name}]"):
                acc = float(stripped.split("LR acc=")[1].split(" ")[0])
                auc = float(stripped.split("auc=")[1].split(" ")[0].rstrip("|"))
                results[split_name] = {"accuracy": acc, "roc_auc": auc}
    return results


def main() -> None:
    summary: dict = {"antibiotic": ANTIBIOTIC, "pilot_n_genomes": 100, "representations": {}}

    # raw canonical k-mers
    raw: dict = {}
    combined_log = (DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_k_comparison.log").read_text()
    for k in RAW_K_VALUES:
        config = json.loads(
            (DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_config_k{k}_comparison.json").read_text()
        )
        if k in (10, 31):
            lr = _parse_lr_from_combined_log(combined_log, k)
        else:
            results_path = DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_k{k}_lr_results.json"
            lr = json.loads(results_path.read_text()) if results_path.exists() else {}
        raw[str(k)] = {"config": config, "logistic_regression": lr}
    summary["representations"]["raw_canonical_kmers"] = raw

    # feature hashing
    hashed: dict = {}
    for d in HASH_DIMENSIONS:
        full = json.loads(
            (DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_hashed_config_d{d}.json").read_text()
        )
        model_results = full.pop("model_results")
        hashed[str(d)] = {"config": full, "logistic_regression": model_results}
    summary["representations"]["feature_hashing"] = {"raw_k_used_before_hashing": 15, "dimensions": hashed}

    # minimizers
    minimizer: dict = {}
    for w in MINIMIZER_W_VALUES:
        config = json.loads(
            (DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_minimizer_config_k{MINIMIZER_K}_w{w}.json").read_text()
        )
        lr_path = DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_minimizer_lr_results_k{MINIMIZER_K}_w{w}.json"
        lr = json.loads(lr_path.read_text()) if lr_path.exists() else {}
        minimizer[str(w)] = {"config": config, "logistic_regression": lr}
    summary["representations"]["minimizers"] = {"k": MINIMIZER_K, "window_sizes": minimizer}

    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Consolidated summary written -> {OUT_PATH}")


if __name__ == "__main__":
    main()
