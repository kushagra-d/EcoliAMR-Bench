"""Quantitative resource-scaling assessment for the minimizer feature
representation (k=31, w=5/10/15), extending to a comparison against
every other representation already measured in this project (raw k=10/
15/17/19/21/31, feature hashing d=2^16/18/20).

Reads only already-computed artifacts (cross-checked against the saved
matrices/vocabularies where feasible); writes two new files only.

Usage:
    python src/run_minimizer_resource_scaling.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from bvbrc_recon.resource_estimation import (
    estimate_construction_overhead_bytes,
    estimate_fasta_storage_bytes,
    estimate_processing_seconds,
    estimate_vocabulary_scaling,
    scale_vocabulary_by_measured_ratio,
)

ANTIBIOTIC = "ciprofloxacin"
DATA_PROCESSED = Path("data/processed")
JSON_OUT_PATH = DATA_PROCESSED / "minimizer_resource_scaling_assessment.json"
REPORT_OUT_PATH = Path("reports/minimizer_resource_scaling_assessment.md")

PILOT_N_GENOMES = 100
TARGETS = {
    "ciprofloxacin": 7307,
    "ampicillin": 6490,
    "gentamicin": 7340,
    "union_unique": 8710,
}
MEASURED_FASTA_MB_PER_GENOME = 495.0 / 100.0  # from the full-dataset feasibility task
MINIMIZER_K = 31
MINIMIZER_W_VALUES = [5, 10, 15]
RAW_K_VALUES = [10, 15, 17, 19, 21, 31]
HASH_DIMENSIONS = [2**16, 2**18, 2**20]
SIXTEEN_GB_BYTES = 16 * 1024**3


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def cross_check_minimizer_artifacts() -> dict:
    """Direct verification against saved matrices/vocabularies, not just
    trust in the recorded config JSON - returns per-w pass/fail detail."""
    checks = {}
    for w in MINIMIZER_W_VALUES:
        config = load_json(DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_minimizer_config_k{MINIMIZER_K}_w{w}.json")
        matrix = sparse.load_npz(DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_minimizer_matrix_k{MINIMIZER_K}_w{w}.npz")
        vocab = pd.read_csv(DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_minimizer_vocabulary_k{MINIMIZER_K}_w{w}.csv")
        per_genome_nnz = np.diff(matrix.indptr)
        checks[w] = {
            "config": config,
            "shape_matches": tuple(matrix.shape) == (config["n_genomes_processed"], config["n_features"]),
            "nnz_matches": int(matrix.nnz) == config["nnz"],
            "vocab_rows_match_n_features": len(vocab) == config["n_features"],
            "mean_features_per_genome_matches": abs(
                float(per_genome_nnz.mean()) - config["mean_selected_features_per_genome"]
            )
            < 1e-6,
            "measured_per_genome_feature_distribution": {
                "min": int(per_genome_nnz.min()),
                "mean": float(per_genome_nnz.mean()),
                "median": float(np.median(per_genome_nnz)),
                "max": int(per_genome_nnz.max()),
                "std": float(per_genome_nnz.std()),
            },
        }
    return checks


def full_scale_estimate(
    *,
    vocab_low: int,
    vocab_base: int,
    vocab_high: int,
    vocab_regime: str,
    pilot_nnz: int,
    pilot_peak_rss_mb: float,
    pilot_in_memory_csr_bytes: int,
    pilot_seconds: float,
    target_n_genomes: int,
    overhead_measurement_is_contaminated: bool = False,
    prerequisite_total_memory_bytes: float = 0.0,
    prerequisite_note: str = "",
) -> dict:
    """Uniform downstream math shared by every representation type -
    nnz and processing time scale linearly with genome count (stated
    assumption, see resource_estimation.py); vocabulary bounds are
    resolved by the CALLER using whichever method fits that
    representation (fixed for hashing, saturation-aware for raw k-mers,
    ratio-based for minimizers).

    `overhead_measurement_is_contaminated`: set True when the pilot's own
    `pilot_peak_rss_mb` does NOT cleanly measure this representation's own
    construction overhead (e.g. feature hashing's pilot run also loaded a
    large pre-existing raw matrix in the SAME process before hashing it -
    confirmed in the feature-hashing task's own report). When True, the
    overhead-ratio extrapolation is skipped entirely (never silently
    applied to a measurement known not to represent what it would be
    extrapolating) and `prerequisite_total_memory_bytes` is used as an
    explicit floor instead, with `prerequisite_note` explaining why.
    """
    estimated_nnz = int(pilot_nnz / PILOT_N_GENOMES * target_n_genomes)
    csr_bytes = estimated_nnz * (8 + 4) + (target_n_genomes + 1) * 4

    if overhead_measurement_is_contaminated:
        overhead_bytes = None
        total_memory_bytes = max(csr_bytes, prerequisite_total_memory_bytes)
        memory_note = (
            "Construction-overhead measurement is CONTAMINATED for this representation (its pilot "
            "peak RSS includes loading an unrelated prerequisite matrix in the same process) and was "
            "NOT extrapolated. Total memory shown is the max of this representation's own final CSR "
            f"size and its prerequisite's own full-scale estimate. {prerequisite_note}"
        )
    else:
        overhead_bytes = estimate_construction_overhead_bytes(
            pilot_peak_rss_bytes=pilot_peak_rss_mb * 1e6,
            pilot_csr_bytes=pilot_in_memory_csr_bytes,
            target_csr_bytes=csr_bytes,
        )
        total_memory_bytes = csr_bytes + overhead_bytes
        memory_note = ""

    seconds = estimate_processing_seconds(pilot_seconds, PILOT_N_GENOMES, target_n_genomes)
    fasta_bytes = estimate_fasta_storage_bytes(target_n_genomes, MEASURED_FASTA_MB_PER_GENOME * 1e6)

    return {
        "target_n_genomes": target_n_genomes,
        "vocabulary": {
            "low": vocab_low,
            "base": vocab_base,
            "high": vocab_high,
            "regime": vocab_regime,
        },
        "estimated_nnz": estimated_nnz,
        "estimated_csr_bytes": csr_bytes,
        "estimated_construction_overhead_bytes": overhead_bytes,
        "estimated_total_memory_bytes": total_memory_bytes,
        "memory_note": memory_note,
        "fits_under_16gb": total_memory_bytes < SIXTEEN_GB_BYTES,
        "estimated_processing_seconds": seconds,
        "estimated_sequence_storage_bytes": fasta_bytes,
    }


def main() -> None:
    minimizer_checks = cross_check_minimizer_artifacts()
    raw_configs = {k: load_json(DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_config_k{k}_comparison.json") for k in RAW_K_VALUES}
    hash_configs = {d: load_json(DATA_PROCESSED / f"pilot_{ANTIBIOTIC}_kmer_hashed_config_d{d}.json") for d in HASH_DIMENSIONS}
    raw_k31_config = raw_configs[31]

    assessment: dict = {
        "pilot_n_genomes": PILOT_N_GENOMES,
        "targets": TARGETS,
        "cross_checks": {
            str(w): {k: v for k, v in c.items() if k != "config"} for w, c in minimizer_checks.items()
        },
        "minimizer": {},
        "raw_kmer": {},
        "feature_hashing": {},
    }

    # --- minimizer scaling, per w, per target ---
    for w in MINIMIZER_W_VALUES:
        cfg = minimizer_checks[w]["config"]
        ratio = cfg["n_features"] / raw_k31_config["n_features"]
        assessment["minimizer"][str(w)] = {"pilot_measured": cfg, "measured_ratio_to_raw_k31": ratio, "by_target": {}}
        for target_label, target_n in TARGETS.items():
            anchor = estimate_vocabulary_scaling(
                k=31, pilot_n_features=raw_k31_config["n_features"], pilot_n_genomes=PILOT_N_GENOMES, target_n_genomes=target_n
            )
            vocab = scale_vocabulary_by_measured_ratio(anchor, ratio, label=f"minimizer w={w}")
            est = full_scale_estimate(
                vocab_low=vocab.estimated_low,
                vocab_base=vocab.estimated_base,
                vocab_high=vocab.estimated_high,
                vocab_regime=f"ratio-scaled from raw k=31 ({vocab.anchor_regime})",
                pilot_nnz=cfg["nnz"],
                pilot_peak_rss_mb=cfg["peak_rss_mb"],
                pilot_in_memory_csr_bytes=cfg["in_memory_csr_bytes"],
                pilot_seconds=cfg["extraction_seconds"],
                target_n_genomes=target_n,
            )
            est["vocabulary_rationale"] = vocab.rationale
            assessment["minimizer"][str(w)]["by_target"][target_label] = est

    # --- raw k-mer scaling, per k, per target (reused from prior tasks' method) ---
    for k in RAW_K_VALUES:
        cfg = raw_configs[k]
        assessment["raw_kmer"][str(k)] = {"pilot_measured": cfg, "by_target": {}}
        for target_label, target_n in TARGETS.items():
            vocab = estimate_vocabulary_scaling(
                k=k, pilot_n_features=cfg["n_features"], pilot_n_genomes=PILOT_N_GENOMES, target_n_genomes=target_n
            )
            est = full_scale_estimate(
                vocab_low=vocab.estimated_n_features_low,
                vocab_base=int((vocab.estimated_n_features_low * vocab.estimated_n_features_high) ** 0.5),
                vocab_high=vocab.estimated_n_features_high,
                vocab_regime=vocab.regime,
                pilot_nnz=cfg["nnz"],
                pilot_peak_rss_mb=cfg["peak_rss_mb"],
                pilot_in_memory_csr_bytes=cfg["in_memory_csr_bytes"],
                pilot_seconds=cfg["extraction_seconds"],
                target_n_genomes=target_n,
            )
            assessment["raw_kmer"][str(k)]["by_target"][target_label] = est

    # --- feature hashing: column count is FIXED by construction, no uncertainty there.
    # IMPORTANT (discovered, not assumed): the pilot's `peak_rss_mb` for every hashing
    # dimension was measured in a process that ALSO loaded the pre-hashing raw k=15
    # matrix first (same finding already documented in the feature-hashing task's own
    # report) - that overhead ratio is NOT a clean measurement of hashing's own
    # construction cost and must not be extrapolated as if it were. Instead, hashing's
    # true full-scale memory floor is the max of (a) its own small final CSR size and
    # (b) its prerequisite raw matrix's (here, raw k=15's) own full-scale estimate,
    # since the current implementation hashes an already-fully-built raw matrix rather
    # than hashing during extraction. ---
    for d in HASH_DIMENSIONS:
        cfg = hash_configs[d]
        pre_hash_raw_k = cfg["raw_k"]
        assessment["feature_hashing"][str(d)] = {"pilot_measured": cfg, "by_target": {}}
        for target_label, target_n in TARGETS.items():
            prerequisite_total = assessment["raw_kmer"][str(pre_hash_raw_k)]["by_target"][target_label][
                "estimated_total_memory_bytes"
            ]
            est = full_scale_estimate(
                vocab_low=d,
                vocab_base=d,
                vocab_high=d,
                vocab_regime="fixed_by_construction",
                pilot_nnz=cfg["nnz"],
                pilot_peak_rss_mb=cfg["peak_rss_mb"],
                pilot_in_memory_csr_bytes=cfg["in_memory_csr_bytes"],
                pilot_seconds=cfg["build_seconds"],
                target_n_genomes=target_n,
                overhead_measurement_is_contaminated=True,
                prerequisite_total_memory_bytes=prerequisite_total,
                prerequisite_note=(
                    f"Prerequisite: raw k={pre_hash_raw_k} must be fully built in memory "
                    f"before this hashing step can run, as currently implemented."
                ),
            )
            assessment["feature_hashing"][str(d)]["by_target"][target_label] = est

    JSON_OUT_PATH.write_text(json.dumps(assessment, indent=2))
    print(f"Machine-readable assessment written -> {JSON_OUT_PATH}")

    _write_brief_report(assessment)
    print(f"Brief report written -> {REPORT_OUT_PATH}")


def _write_brief_report(assessment: dict) -> None:
    """Deliberately NOT a polished narrative - tables and short notes
    only, per the task's own instruction (teammate writes the final
    report manually)."""
    lines = [
        "# Minimizer Resource-Scaling Assessment (brief, machine-generated)",
        "",
        "See `data/processed/minimizer_resource_scaling_assessment.json` for full detail; this file "
        "is a brief, unpolished summary only - not a final narrative report.",
        "",
        "## Cross-checks (measured config vs. saved artifacts)",
        "",
    ]
    for w, checks in assessment["cross_checks"].items():
        lines.append(f"- w={w}: shape_matches={checks['shape_matches']}, nnz_matches={checks['nnz_matches']}, "
                     f"vocab_rows_match={checks['vocab_rows_match_n_features']}, "
                     f"feature_dist={checks['measured_per_genome_feature_distribution']}")
    lines.append("")
    lines.append("## Union-cohort (8,710 genomes) full-scale estimates, all representations")
    lines.append("")
    lines.append("| Representation | Config | Vocab [low,base,high] | Est. nnz | Est. total memory | Fits <16GB? | Est. time |")
    lines.append("|---|---|---|---|---|---|---|")
    for w, block in assessment["minimizer"].items():
        e = block["by_target"]["union_unique"]
        v = e["vocabulary"]
        lines.append(
            f"| minimizer | k=31,w={w} | [{v['low']:,}, {v['base']:,}, {v['high']:,}] | {e['estimated_nnz']:,} | "
            f"{e['estimated_total_memory_bytes']/1e9:.1f} GB | {e['fits_under_16gb']} | {e['estimated_processing_seconds']/60:.1f} min |"
        )
    for k, block in assessment["raw_kmer"].items():
        e = block["by_target"]["union_unique"]
        v = e["vocabulary"]
        lines.append(
            f"| raw k-mer | k={k} | [{v['low']:,}, {v['base']:,}, {v['high']:,}] | {e['estimated_nnz']:,} | "
            f"{e['estimated_total_memory_bytes']/1e9:.1f} GB | {e['fits_under_16gb']} | {e['estimated_processing_seconds']/60:.1f} min |"
        )
    for d, block in assessment["feature_hashing"].items():
        e = block["by_target"]["union_unique"]
        v = e["vocabulary"]
        lines.append(
            f"| feature hashing | d={d} | [{v['low']:,}, {v['base']:,}, {v['high']:,}] | {e['estimated_nnz']:,} | "
            f"{e['estimated_total_memory_bytes']/1e9:.1f} GB | {e['fits_under_16gb']} | {e['estimated_processing_seconds']/60:.1f} min |"
        )
    lines.append("")
    lines.append(
        "Notes: vocabulary [low,base,high] for raw k-mers uses the saturation-aware method already "
        "established in `resource_estimation.py`; for minimizers, [low,base,high] is derived by "
        "scaling raw k=31's own projection by the pilot-measured minimizer/raw-k31 vocabulary ratio "
        "(assumption: this ratio holds at full scale - unverified beyond the single 100-genome "
        "pilot); for feature hashing, vocabulary is fixed by construction (no uncertainty). nnz and "
        "processing time are extrapolated linearly from the pilot's own measured per-genome rate for "
        "every representation. Total memory for raw k-mers and minimizers = estimated CSR matrix "
        "size + a construction-overhead term scaled from that representation's OWN pilot-measured "
        "(peak_rss - final_CSR_size) ratio. **Feature hashing is different**: its pilot `peak_rss` was "
        "measured in a process that also loaded the pre-hashing raw matrix first (same finding "
        "already documented in the feature-hashing task's own report), so that ratio was NOT "
        "extrapolated for hashing; instead hashing's total memory is the max of its own tiny final "
        "CSR size and its prerequisite raw representation's (raw k=15's) own full-scale estimate - "
        "which is why all three hashing dimensions show the SAME ~623GB figure here, driven entirely "
        "by the prerequisite, not by the hashed matrix itself. This is a real, if non-obvious, "
        "finding: hashing AS CURRENTLY IMPLEMENTED (a post-hoc transform of an already-built raw "
        "matrix) does not reduce full-scale memory below its prerequisite's own requirement - only a "
        "streaming hash-during-extraction implementation (not built here) could. None of this has "
        "been measured at full scale; see the JSON's `rationale`/`memory_note` fields for the exact "
        "assumption behind every number."
    )

    REPORT_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
