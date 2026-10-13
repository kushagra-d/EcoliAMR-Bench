"""Builds reports/feature_hashing_comparison.md from already-computed
artifacts (does not run any hashing or model fit itself - run
src/run_feature_hashing_experiment.py first).

Usage (from the project root, with the project venv active):

    python src/run_feature_hashing_report.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from bvbrc_recon.feature_hashing_report import render_feature_hashing_report

ANTIBIOTIC = "ciprofloxacin"
HASH_DIMENSIONS = [2**16, 2**18, 2**20]
RAW_K_VALUES = [10, 15, 31]
DATA_PROCESSED = Path("data/processed")
REPORT_PATH = Path("reports/feature_hashing_comparison.md")


def _parse_lr_from_combined_log(log_text: str, k: int) -> dict:
    """Same parser already used for the intermediate-k report - reused
    here rather than reimplemented, to pull k=10/k=31's LR numbers from
    the one place they exist (the prior comparison's plain-text log)."""
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


def load_raw_comparison() -> dict:
    raw_comparison = {}
    combined_log = (DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_k_comparison.log").read_text()
    for k in RAW_K_VALUES:
        if k == 15:
            results_path = DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_k15_lr_results.json"
            raw_comparison[k] = json.loads(results_path.read_text())
        else:
            raw_comparison[k] = _parse_lr_from_combined_log(combined_log, k)
    return raw_comparison


def main() -> None:
    hash_configs = {}
    for d in HASH_DIMENSIONS:
        path = DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_hashed_config_d{d}.json"
        hash_configs[d] = json.loads(path.read_text())

    raw_comparison = load_raw_comparison()

    narrative = """
**Central finding (measured, not assumed)**: at d=2^16 and d=2^18, the hashed matrix is
**100% dense** - every one of the 65,536 (or 262,144) buckets is touched by every one of the 100
genomes. At d=2^20, density is still 99.1%. The sparsity benefit often associated with feature
hashing does NOT materialize at these dimensions for this dataset, because each genome already
contributes close to 5 million distinct raw k-mers - far more than even the largest tested bucket
count - so collisions guarantee near-complete bucket occupancy regardless of dimension. Hashing
here is better understood as a fixed-width LOSSY SUMMARIZATION of the raw matrix than as a sparsity
technique.
"""

    recommendation = """
**No representation is approved as final here.** This task's evidence narrows the decision rather
than resolving it:

1. **Feature hashing at these dimensions trades information for a fixed, bounded size.** At
   d=2^16, every bucket collides and lineage-split accuracy (0.655) is the worst result measured
   anywhere in this project so far - a real cost, not a hypothetical one. d=2^20 recovers most of
   the lineage-split performance seen with raw k=15 (0.759 vs 0.897) while remaining nearly fully
   dense, suggesting the useful dimension (if hashing is pursued) is likely well above 2^20 for
   this k, not below it.
2. **Concrete next step**: if feature hashing is pursued further, test larger dimensions
   (e.g. 2^22-2^24) and/or a smaller raw k before hashing (not just k=15), to find where hashed
   density actually drops meaningfully below 100% - this task deliberately stayed within the
   dimensions specified and did not explore beyond them.
3. **Alternative worth comparing directly**: a minimizer- or sketch-based k-mer selection (reducing
   the RAW vocabulary before any matrix is built, rather than folding an already-large raw
   vocabulary into fewer columns after the fact) may avoid the "guaranteed collision" regime seen
   here entirely - not evaluated in this task.

This remains a judgment call weighing storage/build-time savings (real and significant) against
collision-driven information loss and loss of CARD-style interpretability (also real and
measured) - a reviewer could reasonably prioritize either differently.
"""

    report_text = render_feature_hashing_report(
        analysis_date=datetime.now(timezone.utc).isoformat(),
        hash_configs=hash_configs,
        raw_comparison=raw_comparison,
        narrative=narrative,
        recommendation=recommendation,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"Report written -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
