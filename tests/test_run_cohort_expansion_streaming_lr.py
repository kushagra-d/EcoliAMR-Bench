"""Integration test for src/run_cohort_expansion_streaming_lr.py -
verifies the new train/validation partition is separate from (and does
not reuse) the saved pilot splits, is deterministic, and that the
minority-class blocker path is honored rather than silently producing
a degenerate fit. Uses a tiny synthetic hashed SQLite store - no real
network or real genome dependency.
"""

import json

import pandas as pd
import pytest

import run_cohort_expansion_streaming_lr as script
from bvbrc_recon.sqlite_row_store import SqliteRowStore


def _build_tiny_store(path, n_rows, n_features=20):
    store = SqliteRowStore(path, n_buckets=n_features)
    for i in range(n_rows):
        store.append_row(f"g{i}", i, {c: (i + c) % 5 + 1 for c in range(n_features)})
    return store


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    data_processed = tmp_path / "data_processed"
    data_processed.mkdir()

    genome_ids = [f"g{i}" for i in range(30)]
    labels = ["Resistant" if i % 4 == 0 else "Susceptible" for i in range(30)]  # 25% Resistant

    pd.DataFrame({"genome_id": genome_ids}).to_csv(
        data_processed / "cohort_expansion_ciprofloxacin_combined_genome_order.csv", index=False
    )
    pd.DataFrame({"genome_id": genome_ids[:15], "label": labels[:15]}).to_csv(
        data_processed / "pilot_ciprofloxacin_selected_cohort.csv", index=False
    )
    pd.DataFrame({"genome_id": genome_ids[15:], "label": labels[15:]}).to_csv(
        data_processed / "cohort_expansion_ciprofloxacin_selected.csv", index=False
    )

    db_path = data_processed / "cohort_expansion_ciprofloxacin_rows_k15_d262144.db"
    _build_tiny_store(db_path, n_rows=30, n_features=20)

    monkeypatch.setattr(script, "DATA_PROCESSED", data_processed)
    monkeypatch.setattr(script, "DB_PATH", db_path)
    monkeypatch.setattr(script, "GENOME_ORDER_PATH", data_processed / "cohort_expansion_ciprofloxacin_combined_genome_order.csv")
    monkeypatch.setattr(script, "PARTITION_PATH", data_processed / "cohort_expansion_ciprofloxacin_train_val_partition.csv")
    monkeypatch.setattr(script, "RESULTS_PATH", data_processed / "cohort_expansion_ciprofloxacin_streaming_lr_results_k15_d262144.json")
    monkeypatch.setattr(script, "N_BUCKETS", 20)
    monkeypatch.setattr(script, "MAX_ITER", 50)

    return data_processed


def test_new_partition_is_separate_from_saved_pilot_splits(fake_env):
    script.main()
    partition_df = pd.read_csv(fake_env / "cohort_expansion_ciprofloxacin_train_val_partition.csv")
    assert set(partition_df["partition"]) <= {"train", "validation"}
    assert not (fake_env / "pilot_ciprofloxacin_random_split.csv").exists()
    assert not (fake_env / "pilot_ciprofloxacin_lineage_split.csv").exists()


def test_partition_is_deterministic_across_runs(fake_env):
    script.main()
    first = pd.read_csv(fake_env / "cohort_expansion_ciprofloxacin_train_val_partition.csv")

    (fake_env / "cohort_expansion_ciprofloxacin_train_val_partition.csv").unlink()
    script.main()
    second = pd.read_csv(fake_env / "cohort_expansion_ciprofloxacin_train_val_partition.csv")

    assert first["partition"].tolist() == second["partition"].tolist()


def test_records_bounded_memory_access(fake_env):
    script.main()
    results = json.loads((fake_env / "cohort_expansion_ciprofloxacin_streaming_lr_results_k15_d262144.json").read_text())
    assert results["max_batch_rows_held_in_memory"] <= 16


def test_insufficient_minority_class_is_blocked_not_silently_fit(tmp_path, monkeypatch):
    data_processed = tmp_path / "data_processed"
    data_processed.mkdir()
    genome_ids = [f"g{i}" for i in range(10)]
    labels = ["Resistant"] + ["Susceptible"] * 9  # only 1 Resistant - below the minimum

    pd.DataFrame({"genome_id": genome_ids}).to_csv(
        data_processed / "cohort_expansion_ciprofloxacin_combined_genome_order.csv", index=False
    )
    pd.DataFrame({"genome_id": genome_ids[:5], "label": labels[:5]}).to_csv(
        data_processed / "pilot_ciprofloxacin_selected_cohort.csv", index=False
    )
    pd.DataFrame({"genome_id": genome_ids[5:], "label": labels[5:]}).to_csv(
        data_processed / "cohort_expansion_ciprofloxacin_selected.csv", index=False
    )
    db_path = data_processed / "cohort_expansion_ciprofloxacin_rows_k15_d262144.db"
    _build_tiny_store(db_path, n_rows=10, n_features=20)

    monkeypatch.setattr(script, "DATA_PROCESSED", data_processed)
    monkeypatch.setattr(script, "DB_PATH", db_path)
    monkeypatch.setattr(script, "GENOME_ORDER_PATH", data_processed / "cohort_expansion_ciprofloxacin_combined_genome_order.csv")
    monkeypatch.setattr(script, "PARTITION_PATH", data_processed / "cohort_expansion_ciprofloxacin_train_val_partition.csv")
    results_path = data_processed / "cohort_expansion_ciprofloxacin_streaming_lr_results_k15_d262144.json"
    monkeypatch.setattr(script, "RESULTS_PATH", results_path)
    monkeypatch.setattr(script, "N_BUCKETS", 20)

    script.main()
    results = json.loads(results_path.read_text())
    assert results["blocked"] is True
    assert not (data_processed / "cohort_expansion_ciprofloxacin_train_val_partition.csv").exists()
