"""Projects resource requirements for the THREE candidate feature
representations named in the course-project feature-generation task, at
the ACTUAL frozen cohort size (956 genomes) - not the full 8,710-genome
union already assessed in minimizer_resource_scaling_assessment.md,
which used a much larger target and does not reflect this project's
real course-project scope.

Reuses the existing, unmodified `resource_estimation.py` formulas and
the existing measured pilot/cohort-scale configs already on disk - no
new benchmark is run; this is pure extrapolation from measurements that
already exist, same methodology as `run_minimizer_resource_scaling.py`.

For feature hashing specifically, this uses the REAL 200-genome
measurement from the already-built, genuinely out-of-core SQLite row
store (`policy_consistent_ciprofloxacin_hashed_build_config_k15_d262144.json`)
as the extrapolation anchor - not the earlier, now-superseded in-memory
`feature_hashing.py` pilot measurement, which required first
materializing a full raw k=15 matrix and is not the implementation this
project would actually use to generate the course-project matrix.

Usage:
    python src/run_course_project_feature_representation_assessment.py

Writes:
    data/processed/course_project_feature_representation_assessment.json
"""

from __future__ import annotations

import json
from pathlib import Path

from bvbrc_recon.resource_estimation import (
    estimate_construction_overhead_bytes,
    estimate_processing_seconds,
    estimate_vocabulary_scaling,
    scale_vocabulary_by_measured_ratio,
)

DATA_PROCESSED = Path("data/processed")
OUT_PATH = DATA_PROCESSED / "course_project_feature_representation_assessment.json"
TARGET_N_GENOMES = 956


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    assessment: dict = {"target_n_genomes": TARGET_N_GENOMES, "candidates": {}}

    # --- Candidate 1: raw canonical k=10 (saturated vocabulary, in-memory, no out-of-core variant) ---
    raw10 = load_json(DATA_PROCESSED / "pilot_ciprofloxacin_kmer_config_k10_comparison.json")
    pilot_n = 100
    vocab = estimate_vocabulary_scaling(
        k=10, pilot_n_features=raw10["n_features"], pilot_n_genomes=pilot_n, target_n_genomes=TARGET_N_GENOMES
    )
    estimated_nnz = int(raw10["nnz"] / pilot_n * TARGET_N_GENOMES)
    csr_bytes = estimated_nnz * (8 + 4) + (TARGET_N_GENOMES + 1) * 4
    overhead_bytes = estimate_construction_overhead_bytes(
        pilot_peak_rss_bytes=raw10["peak_rss_mb"] * 1e6,
        pilot_csr_bytes=raw10["in_memory_csr_bytes"],
        target_csr_bytes=csr_bytes,
    )
    seconds = estimate_processing_seconds(raw10["extraction_seconds"], pilot_n, TARGET_N_GENOMES)
    assessment["candidates"]["raw_k10"] = {
        "pilot_measured_100_genomes": raw10,
        "projected_at_956": {
            "vocabulary_regime": vocab.regime,
            "vocabulary_low": vocab.estimated_n_features_low,
            "vocabulary_high": vocab.estimated_n_features_high,
            "vocabulary_rationale": vocab.rationale,
            "estimated_nnz": estimated_nnz,
            "estimated_csr_bytes": csr_bytes,
            "estimated_construction_overhead_bytes": overhead_bytes,
            "estimated_total_memory_bytes": csr_bytes + overhead_bytes,
            "estimated_processing_seconds": seconds,
            "requires_full_in_memory_materialization": True,
            "out_of_core_implementation_exists": False,
        },
    }

    # --- Candidate 2: feature hashing k=15, d=262144 - REAL 200-genome measurement, out-of-core ---
    hashed200 = load_json(DATA_PROCESSED / "policy_consistent_ciprofloxacin_hashed_build_config_k15_d262144.json")
    pilot_n_200 = 200
    seconds_hashed = estimate_processing_seconds(hashed200["elapsed_seconds"], pilot_n_200, TARGET_N_GENOMES)
    db_bytes_per_genome = hashed200["db_size_bytes"] / pilot_n_200
    assessment["candidates"]["hashed_k15_d262144"] = {
        "measured_200_genomes_real_run": hashed200,
        "projected_at_956": {
            "vocabulary_regime": "fixed_by_construction",
            "vocabulary": 262144,
            "estimated_processing_seconds": seconds_hashed,
            "estimated_db_storage_bytes": int(db_bytes_per_genome * TARGET_N_GENOMES),
            "peak_rss_mb_measured_at_200_is_the_relevant_figure": hashed200["peak_rss_mb"],
            "peak_rss_rationale": (
                "This implementation (sqlite_row_store.build_sqlite_backed_matrix) hashes "
                "directly during the per-genome sequence walk - at most one genome's own "
                "sequence and a <=262144-entry bucket-count dict are held in memory at a time. "
                "Peak RSS is therefore NOT expected to scale with genome count the way the raw "
                "k-mer or minimizer approaches do; the 200-genome measurement (not a 100-genome "
                "pilot extrapolation) is used directly as the expected figure at 956, pending "
                "the real 956-genome run's own measurement in this task."
            ),
            "requires_full_in_memory_materialization": False,
            "out_of_core_implementation_exists": True,
        },
    }

    # --- Candidate 3: canonical k=31 minimizers, w=15 - REAL 200-genome measurement, in-memory ---
    minimizer200 = load_json(DATA_PROCESSED / "policy_consistent_ciprofloxacin_minimizer_config_k31_w15.json")
    minimizer100 = load_json(DATA_PROCESSED / "pilot_ciprofloxacin_minimizer_config_k31_w15.json")
    raw_k31_100 = load_json(DATA_PROCESSED / "pilot_ciprofloxacin_kmer_config_k31_comparison.json")
    ratio_100 = minimizer100["n_features"] / raw_k31_100["n_features"]
    anchor = estimate_vocabulary_scaling(
        k=31, pilot_n_features=raw_k31_100["n_features"], pilot_n_genomes=100, target_n_genomes=TARGET_N_GENOMES
    )
    vocab_scaled = scale_vocabulary_by_measured_ratio(anchor, ratio_100, label="minimizer w=15")

    # nnz/time extrapolated from the REAL 200-genome measurement (closer anchor than the 100-genome pilot)
    estimated_nnz_mm = int(minimizer200["nnz"] / 200 * TARGET_N_GENOMES)
    csr_bytes_mm = estimated_nnz_mm * (8 + 4) + (TARGET_N_GENOMES + 1) * 4
    overhead_bytes_mm = estimate_construction_overhead_bytes(
        pilot_peak_rss_bytes=minimizer200["peak_rss_mb"] * 1e6,
        pilot_csr_bytes=minimizer200["in_memory_csr_bytes"],
        target_csr_bytes=csr_bytes_mm,
    )
    seconds_mm = estimate_processing_seconds(minimizer200["extraction_seconds"], 200, TARGET_N_GENOMES)
    assessment["candidates"]["minimizer_k31_w15"] = {
        "measured_200_genomes_real_run": minimizer200,
        "projected_at_956": {
            "vocabulary_regime": f"ratio-scaled from raw k=31 ({vocab_scaled.anchor_regime})",
            "vocabulary_low": vocab_scaled.estimated_low,
            "vocabulary_base": vocab_scaled.estimated_base,
            "vocabulary_high": vocab_scaled.estimated_high,
            "estimated_nnz": estimated_nnz_mm,
            "estimated_csr_bytes": csr_bytes_mm,
            "estimated_construction_overhead_bytes": overhead_bytes_mm,
            "estimated_total_memory_bytes": csr_bytes_mm + overhead_bytes_mm,
            "estimated_processing_seconds": seconds_mm,
            "requires_full_in_memory_materialization": True,
            "out_of_core_implementation_exists": False,
        },
    }

    OUT_PATH.write_text(json.dumps(assessment, indent=2, default=str))
    print(f"wrote {OUT_PATH}")
    for name, c in assessment["candidates"].items():
        p = c["projected_at_956"]
        mem = p.get("estimated_total_memory_bytes")
        mem_str = f"{mem/1e9:.2f} GB" if mem is not None else "n/a (see peak_rss figure)"
        secs = p.get("estimated_processing_seconds", 0)
        print(f"  {name}: projected_memory={mem_str} time={secs/60:.1f} min out_of_core={p['out_of_core_implementation_exists']}")


if __name__ == "__main__":
    main()
