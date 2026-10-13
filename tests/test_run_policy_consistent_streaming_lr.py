import json

import pandas as pd
import pytest

import run_policy_consistent_streaming_lr as script
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
    labels = ["Resistant" if i % 4 == 0 else "Susceptible" for i in range(30)]

    pd.DataFrame({"genome_id": genome_ids}).to_csv(
        data_processed / "policy_consistent_ciprofloxacin_genome_order.csv", index=False
    )
    pd.DataFrame({"genome_id": genome_ids, "label": labels}).to_csv(
        data_processed / "policy_consistent_ciprofloxacin_selected_cohort.csv", index=False
    )

    db_path = data_processed / "policy_consistent_ciprofloxacin_rows_k15_d262144.db"
    _build_tiny_store(db_path, n_rows=30, n_features=20)

    monkeypatch.setattr(script, "DATA_PROCESSED", data_processed)
    monkeypatch.setattr(script, "DB_PATH", db_path)
    monkeypatch.setattr(script, "GENOME_ORDER_PATH", data_processed / "policy_consistent_ciprofloxacin_genome_order.csv")
    monkeypatch.setattr(script, "PARTITION_PATH", data_processed / "policy_consistent_ciprofloxacin_train_val_partition.csv")
    monkeypatch.setattr(script, "RESULTS_PATH", data_processed / "policy_consistent_ciprofloxacin_streaming_lr_results_k15_d262144.json")
    monkeypatch.setattr(script, "N_BUCKETS", 20)
    monkeypatch.setattr(script, "MAX_ITER", 50)

    return data_processed


def test_partition_is_new_and_deterministic(fake_env):
    script.main()
    first = pd.read_csv(fake_env / "policy_consistent_ciprofloxacin_train_val_partition.csv")
    (fake_env / "policy_consistent_ciprofloxacin_train_val_partition.csv").unlink()
    script.main()
    second = pd.read_csv(fake_env / "policy_consistent_ciprofloxacin_train_val_partition.csv")
    assert first["partition"].tolist() == second["partition"].tolist()


def test_bounded_memory_access_recorded(fake_env):
    script.main()
    results = json.loads((fake_env / "policy_consistent_ciprofloxacin_streaming_lr_results_k15_d262144.json").read_text())
    assert results["max_batch_rows_held_in_memory"] <= 16
