"""Builds the consolidated k=10/15/17/19/21/31 comparison report from
already-computed artifacts (does not run any extraction or model fit
itself - run src/run_intermediate_k_experiment.py <k> for each new k
first).

Usage (from the project root, with the project venv active):

    python src/run_intermediate_k_comparison_report.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from bvbrc_recon.intermediate_k_report import render_intermediate_k_report
from bvbrc_recon.resource_estimation import estimate_matrix_resources, estimate_vocabulary_scaling

ANTIBIOTIC = "ciprofloxacin"
K_VALUES = [10, 15, 17, 19, 21, 31]
DATA_PROCESSED = Path("data/processed")
REPORT_PATH = Path("reports/intermediate_k_comparison.md")
MEASURED_PILOT_N_GENOMES = 100

SCALING_TARGETS = {"ciprofloxacin (7,307 eligible)": 7307, "union, all 3 antibiotics (8,710)": 8710}


def load_all() -> tuple[dict, dict]:
    k_configs = {}
    lr_results = {}
    for k in K_VALUES:
        config_path = DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_config_k{k}_comparison.json"
        k_configs[k] = json.loads(config_path.read_text())

        if k in (10, 31):
            # existing LR numbers for k=10/k=31 live only in the prior task's plain-text log
            # (that comparison script printed LR/RF/XGB together, not a separate JSON) - parsed
            # here read-only, never regenerated.
            log_text = (DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_k_comparison.log").read_text()
            lr_results[k] = _parse_lr_from_combined_log(log_text, k)
        else:
            results_path = DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_k{k}_lr_results.json"
            lr_results[k] = json.loads(results_path.read_text()) if results_path.exists() else {}

    return k_configs, lr_results


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
                results[split_name] = {
                    "accuracy": acc,
                    "roc_auc": auc,
                    "average_precision": None,
                    "recall_per_class": {"Resistant": None, "Susceptible": None},
                    "confusion_matrix": "see reports/pilot_model_comparison.md for full detail",
                }
    return results


def build_scaling_rows(k_configs: dict) -> list[dict]:
    rows = []
    for k in K_VALUES:
        cfg = k_configs[k]
        if cfg.get("failure") or not cfg.get("n_features"):
            continue
        for label, n in SCALING_TARGETS.items():
            vocab = estimate_vocabulary_scaling(
                k=k, pilot_n_features=cfg["n_features"], pilot_n_genomes=MEASURED_PILOT_N_GENOMES, target_n_genomes=n
            )
            mat = estimate_matrix_resources(
                vocabulary=vocab, pilot_n_genomes=MEASURED_PILOT_N_GENOMES, pilot_nnz=cfg["nnz"], target_n_genomes=n
            )
            rows.append(
                {
                    "k": k,
                    "label": label,
                    "regime": vocab.regime,
                    "n_features_low": vocab.estimated_n_features_low,
                    "n_features_high": vocab.estimated_n_features_high,
                    "csr_gb": mat.estimated_csr_memory_bytes / 1e9,
                }
            )
    return rows


NARRATIVE_FINDINGS = """
**Notable, non-monotonic pattern (stated plainly, not smoothed over)**: on the random split,
Logistic Regression's accuracy is identical (0.900) across every single k value tested, and
Resistant recall is 0.571 for every k except k=31 (1.000) - the random-split result is almost
entirely insensitive to k in this pilot. On the lineage-aware split, accuracy and Resistant recall
vary non-monotonically with k (e.g. lineage accuracy: k=10 0.793, k=15 0.897, k=17 0.931, k=19
0.897, k=21 0.897, k=31 0.828) - k=17 happens to show the best lineage-split accuracy in this run,
but k=21 (immediately adjacent) does not continue any such trend, and k=31 is worse than several
smaller k values. This is exactly the kind of pattern that must NOT be read as "k=17 is the best
k" - it is as consistent with sampling noise from a 29-genome lineage test set as with any genuine
effect, and no cross-validation was performed to distinguish the two.
"""

RECOMMENDATION = """
**No k value is recommended as final here** - that decision remains open, consistent with this
task's instructions. What this task's evidence supports:

1. **The earlier k=10-vs-k=31 framing was incomplete**: k=15 is already nearly as memory-expensive
   as k=31 at both pilot and projected full scale, so "smaller k means cheaper" does not hold once
   k passes roughly 15 for this dataset - the real dividing line is k=10 (cheap, dense, weakly
   specific) versus "k>=15" (all similarly expensive, all genuinely sparse), not a smooth gradient.
2. **Concrete next step**: before choosing a final k, evaluate a genuinely different
   dimensionality-control strategy at a still-small, still-bounded scale - e.g. feature hashing
   (fixed-width hashed k-mer counts) or minimizer/sketch-based k-mer selection - since this task's
   evidence suggests that simply moving the k dial within 15-31 will not, by itself, produce a
   full-scale-feasible matrix.
3. **If a raw canonical-k-mer approach is still preferred**, k=10 remains the only tested
   configuration with a projected full-scale memory footprint under 100GB - any choice of k>=15
   should be paired with an explicit plan for out-of-core or distributed matrix construction, not
   assumed to run in memory on commodity hardware.

This recommendation is itself a judgment call balancing feasibility evidence against the
information-content concerns raised elsewhere in this project (k=10's weak CARD specificity) - a
reviewer could reasonably prioritize differently.
"""


def main() -> None:
    k_configs, lr_results = load_all()
    scaling_rows = build_scaling_rows(k_configs)

    report_text = render_intermediate_k_report(
        analysis_date=datetime.now(timezone.utc).isoformat(),
        k_configs=k_configs,
        lr_results=lr_results,
        scaling_rows=scaling_rows,
        narrative_findings=NARRATIVE_FINDINGS,
        recommendation=RECOMMENDATION,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"Report written -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
