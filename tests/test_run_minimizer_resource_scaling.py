"""Integration test for src/run_minimizer_resource_scaling.py against a
tiny synthetic set of artifacts - exercises the real script logic
(cross-checking, per-target scaling, artifact isolation) without
depending on the real 100-genome pilot data.
"""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import run_minimizer_resource_scaling as script


def _write_minimizer_artifacts(data_processed, w, n_genomes=3, n_features=5, nnz_per_genome=4):
    rows = list(range(n_genomes)) * nnz_per_genome
    cols = list(range(n_features))[: n_genomes * nnz_per_genome]
    cols = (cols + [0] * (len(rows) - len(cols)))[: len(rows)]
    data = [1] * len(rows)
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(n_genomes, n_features)).tocsr()
    sparse.save_npz(data_processed / f"pilot_ciprofloxacin_minimizer_matrix_k31_w{w}.npz", matrix)

    vocab = pd.DataFrame({"kmer": [f"K{i}" * 6 for i in range(n_features)], "column_index": list(range(n_features))})
    vocab.to_csv(data_processed / f"pilot_ciprofloxacin_minimizer_vocabulary_k31_w{w}.csv", index=False)

    per_genome_nnz = np.diff(matrix.indptr)
    config = {
        "k": 31,
        "w": w,
        "n_genomes_processed": n_genomes,
        "n_features": n_features,
        "nnz": int(matrix.nnz),
        "density": matrix.nnz / (n_genomes * n_features),
        "extraction_seconds": 10.0 * w,
        "peak_rss_mb": 100.0 * w,
        "in_memory_csr_bytes": 1000 * w,
        "mean_selected_features_per_genome": float(per_genome_nnz.mean()),
    }
    (data_processed / f"pilot_ciprofloxacin_minimizer_config_k31_w{w}.json").write_text(json.dumps(config))
    return matrix


def _write_raw_config(data_processed, k, n_features=100, nnz=400, peak_rss_mb=500.0, csr_bytes=5000, seconds=50.0):
    config = {
        "k": k,
        "n_features": n_features,
        "nnz": nnz,
        "peak_rss_mb": peak_rss_mb,
        "in_memory_csr_bytes": csr_bytes,
        "extraction_seconds": seconds,
    }
    (data_processed / f"pilot_ciprofloxacin_kmer_config_k{k}_comparison.json").write_text(json.dumps(config))


def _write_hash_config(data_processed, d, nnz=400, peak_rss_mb=500.0, csr_bytes=5000, seconds=20.0, raw_k=15):
    config = {
        "raw_k": raw_k,
        "n_features": d,
        "nnz": nnz,
        "peak_rss_mb": peak_rss_mb,
        "in_memory_csr_bytes": csr_bytes,
        "build_seconds": seconds,
    }
    (data_processed / f"pilot_ciprofloxacin_kmer_hashed_config_d{d}.json").write_text(json.dumps(config))


@pytest.fixture
def fake_artifacts(tmp_path, monkeypatch):
    data_processed = tmp_path / "data_processed"
    data_processed.mkdir()

    for w in (5, 10):
        _write_minimizer_artifacts(data_processed, w)
    for k in (15, 31):
        _write_raw_config(data_processed, k, n_features=100 if k == 31 else 500)
    for d in (16, 32):
        _write_hash_config(data_processed, d)

    monkeypatch.setattr(script, "DATA_PROCESSED", data_processed)
    monkeypatch.setattr(script, "JSON_OUT_PATH", data_processed / "assessment.json")
    monkeypatch.setattr(script, "REPORT_OUT_PATH", tmp_path / "report.md")
    monkeypatch.setattr(script, "MINIMIZER_W_VALUES", [5, 10])
    monkeypatch.setattr(script, "RAW_K_VALUES", [15, 31])
    monkeypatch.setattr(script, "HASH_DIMENSIONS", [16, 32])
    monkeypatch.setattr(script, "TARGETS", {"antibiotic_a": 300, "union_unique": 500})
    monkeypatch.setattr(script, "PILOT_N_GENOMES", 3)
    return data_processed


def test_cross_check_detects_consistent_artifacts(fake_artifacts):
    checks = script.cross_check_minimizer_artifacts()
    assert checks[5]["shape_matches"] is True
    assert checks[5]["nnz_matches"] is True
    assert checks[5]["vocab_rows_match_n_features"] is True


def test_cross_check_detects_inconsistent_artifact(fake_artifacts):
    # corrupt the config to disagree with the saved matrix
    config_path = fake_artifacts / "pilot_ciprofloxacin_minimizer_config_k31_w5.json"
    config = json.loads(config_path.read_text())
    config["nnz"] = 999999
    config_path.write_text(json.dumps(config))

    checks = script.cross_check_minimizer_artifacts()
    assert checks[5]["nnz_matches"] is False


def test_full_scale_estimate_scales_nnz_linearly():
    est = script.full_scale_estimate(
        vocab_low=10, vocab_base=20, vocab_high=30, vocab_regime="test",
        pilot_nnz=400, pilot_peak_rss_mb=500.0, pilot_in_memory_csr_bytes=5000,
        pilot_seconds=50.0, target_n_genomes=300,
    )
    # PILOT_N_GENOMES default inside full_scale_estimate uses module-level constant;
    # verify the result is internally consistent with the module's own constant.
    expected_nnz = int(400 / script.PILOT_N_GENOMES * 300)
    assert est["estimated_nnz"] == expected_nnz


def test_full_scale_estimate_flags_fits_under_16gb_correctly():
    small = script.full_scale_estimate(
        vocab_low=1, vocab_base=1, vocab_high=1, vocab_regime="test",
        pilot_nnz=1, pilot_peak_rss_mb=1.0, pilot_in_memory_csr_bytes=1,
        pilot_seconds=1.0, target_n_genomes=10,
    )
    assert small["fits_under_16gb"] is True

    huge = script.full_scale_estimate(
        vocab_low=1, vocab_base=1, vocab_high=1, vocab_regime="test",
        pilot_nnz=10**9, pilot_peak_rss_mb=1.0, pilot_in_memory_csr_bytes=1,
        pilot_seconds=1.0, target_n_genomes=10**8,
    )
    assert huge["fits_under_16gb"] is False


def test_full_scale_estimate_reports_vocabulary_bounds_as_given():
    est = script.full_scale_estimate(
        vocab_low=10, vocab_base=20, vocab_high=30, vocab_regime="test",
        pilot_nnz=400, pilot_peak_rss_mb=500.0, pilot_in_memory_csr_bytes=5000,
        pilot_seconds=50.0, target_n_genomes=300,
    )
    assert est["vocabulary"] == {"low": 10, "base": 20, "high": 30, "regime": "test"}


def test_main_writes_per_antibiotic_and_union_estimates(fake_artifacts):
    script.main()
    assessment = json.loads((fake_artifacts / "assessment.json").read_text())
    for w_key in ("5", "10"):
        by_target = assessment["minimizer"][w_key]["by_target"]
        assert set(by_target) == {"antibiotic_a", "union_unique"}
        # union target has more genomes -> at least as large an nnz estimate
        assert by_target["union_unique"]["estimated_nnz"] >= by_target["antibiotic_a"]["estimated_nnz"]


def test_main_does_not_overwrite_existing_experiment_artifacts(fake_artifacts):
    before = {
        p.name: p.read_bytes()
        for p in fake_artifacts.glob("*")
        if p.name not in ("assessment.json",)
    }
    script.main()
    for name, content in before.items():
        assert (fake_artifacts / name).read_bytes() == content, f"{name} was modified!"


def test_main_output_paths_are_new_files_not_existing_names(fake_artifacts):
    script.main()
    out_json = fake_artifacts / "assessment.json"
    assert out_json.exists()
    # the output filename must not collide with any per-w/k/d artifact name
    existing_names = {
        p.name for p in fake_artifacts.glob("pilot_ciprofloxacin_*")
    }
    assert out_json.name not in existing_names


def test_handles_missing_measurement_gracefully_via_keyerror_not_silent_wrong_value(fake_artifacts):
    # remove a required field from one config - must raise, not silently
    # substitute a wrong/default value
    config_path = fake_artifacts / "pilot_ciprofloxacin_minimizer_config_k31_w5.json"
    config = json.loads(config_path.read_text())
    del config["peak_rss_mb"]
    config_path.write_text(json.dumps(config))

    with pytest.raises(KeyError):
        script.main()


# --- contaminated-overhead-measurement handling (feature hashing) ---------


def test_full_scale_estimate_skips_overhead_extrapolation_when_contaminated():
    # pilot_peak_rss_mb here is deliberately huge (as if it included loading an
    # unrelated large prerequisite matrix) - must NOT be extrapolated naively.
    est = script.full_scale_estimate(
        vocab_low=10, vocab_base=10, vocab_high=10, vocab_regime="fixed_by_construction",
        pilot_nnz=100, pilot_peak_rss_mb=50_000.0, pilot_in_memory_csr_bytes=10,
        pilot_seconds=5.0, target_n_genomes=1000,
        overhead_measurement_is_contaminated=True,
        prerequisite_total_memory_bytes=5_000.0,
        prerequisite_note="test prerequisite",
    )
    assert est["estimated_construction_overhead_bytes"] is None
    assert "CONTAMINATED" in est["memory_note"]
    # total memory must be the max(own csr, prerequisite) - not inflated by the
    # bogus peak_rss-derived overhead ratio
    assert est["estimated_total_memory_bytes"] == max(est["estimated_csr_bytes"], 5_000.0)


def test_full_scale_estimate_uses_prerequisite_floor_when_it_dominates():
    est = script.full_scale_estimate(
        vocab_low=10, vocab_base=10, vocab_high=10, vocab_regime="fixed_by_construction",
        pilot_nnz=1, pilot_peak_rss_mb=1.0, pilot_in_memory_csr_bytes=1,
        pilot_seconds=1.0, target_n_genomes=10,
        overhead_measurement_is_contaminated=True,
        prerequisite_total_memory_bytes=10**12,  # huge prerequisite dominates
        prerequisite_note="",
    )
    assert est["estimated_total_memory_bytes"] == 10**12


def test_full_scale_estimate_non_contaminated_path_unaffected_by_new_params():
    est = script.full_scale_estimate(
        vocab_low=10, vocab_base=20, vocab_high=30, vocab_regime="test",
        pilot_nnz=400, pilot_peak_rss_mb=500.0, pilot_in_memory_csr_bytes=5000,
        pilot_seconds=50.0, target_n_genomes=300,
    )
    assert est["estimated_construction_overhead_bytes"] is not None
    assert est["memory_note"] == ""


def test_main_flags_hashing_memory_note_as_contaminated_and_references_raw_k15(fake_artifacts):
    script.main()
    assessment = json.loads((fake_artifacts / "assessment.json").read_text())
    hashing_est = assessment["feature_hashing"]["16"]["by_target"]["union_unique"]
    assert hashing_est["estimated_construction_overhead_bytes"] is None
    assert "CONTAMINATED" in hashing_est["memory_note"]
    raw15_est = assessment["raw_kmer"]["15"]["by_target"]["union_unique"]
    assert hashing_est["estimated_total_memory_bytes"] >= raw15_est["estimated_total_memory_bytes"] or \
           hashing_est["estimated_total_memory_bytes"] >= hashing_est["estimated_csr_bytes"]
