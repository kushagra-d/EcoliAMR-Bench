import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from bvbrc_recon.sqlite_row_store import (
    FailureLog,
    ResourceLimits,
    SqliteRowStore,
    build_sqlite_backed_matrix,
)

HELPER_SCRIPT = Path(__file__).parent / "_sqlite_interrupt_subprocess.py"


# --- persistent row indexing, append/reopen, ordering, exact values ---


def test_row_index_governs_placement_not_insertion_order(tmp_path):
    store = SqliteRowStore(tmp_path / "rows.db", n_buckets=10)
    store.append_row("g_last", 2, {0: 1})
    store.append_row("g_first", 0, {1: 2})
    store.append_row("g_mid", 1, {2: 3})

    genome_ids, row_indices, matrix = next(store.iter_batches(batch_size=10))
    by_row_index = dict(zip(row_indices, genome_ids))
    assert by_row_index == {0: "g_first", 1: "g_mid", 2: "g_last"}


def test_append_and_reopen_in_a_fresh_instance(tmp_path):
    path = tmp_path / "rows.db"
    store = SqliteRowStore(path, n_buckets=10)
    store.append_row("g0", 0, {1: 5, 3: 2})
    store.close()

    reopened = SqliteRowStore(path, n_buckets=10)
    assert reopened.existing_genome_ids() == {"g0"}
    assert reopened.n_rows_written() == 1


def test_exact_sparse_value_preservation(tmp_path):
    store = SqliteRowStore(tmp_path / "rows.db", n_buckets=20)
    bucket_counts = {5: 3, 7: 1, 19: 42}
    store.append_row("g0", 0, bucket_counts)

    genome_id, cols, vals = store.get_row(0)
    assert genome_id == "g0"
    recovered = dict(zip(cols.tolist(), vals.tolist()))
    assert recovered == bucket_counts


def test_get_row_returns_none_for_missing_row_index(tmp_path):
    store = SqliteRowStore(tmp_path / "rows.db", n_buckets=10)
    assert store.get_row(0) is None


# --- efficient duplicate detection ---


def test_row_exists_is_true_only_after_append(tmp_path):
    store = SqliteRowStore(tmp_path / "rows.db", n_buckets=10)
    assert store.row_exists("g0") is False
    store.append_row("g0", 0, {1: 1})
    assert store.row_exists("g0") is True


def test_duplicate_genome_id_raises_instead_of_silently_overwriting(tmp_path):
    store = SqliteRowStore(tmp_path / "rows.db", n_buckets=10)
    store.append_row("g0", 0, {1: 1})
    with pytest.raises(sqlite3.IntegrityError):
        store.append_row("g0", 1, {2: 2})


def test_build_sqlite_backed_matrix_skips_already_present_without_duplicating(tmp_path):
    row_store = SqliteRowStore(tmp_path / "rows.db", n_buckets=10)
    failure_log = FailureLog(tmp_path / "failures.jsonl")
    genome_order = ["g0", "g1", "g2"]
    calls = []

    def loader(genome_id):
        calls.append(genome_id)
        return ["ACGTACGT"]

    status1 = build_sqlite_backed_matrix(loader, genome_order, k=4, n_buckets=10, row_store=row_store, failure_log=failure_log)
    assert status1.n_rows_written_this_run == 3
    assert calls == ["g0", "g1", "g2"]

    status2 = build_sqlite_backed_matrix(loader, genome_order, k=4, n_buckets=10, row_store=row_store, failure_log=failure_log)
    assert status2.n_rows_written_this_run == 0
    assert status2.n_genomes_skipped_already_present == 3
    assert calls == ["g0", "g1", "g2"]  # loader was never called again
    assert row_store.n_rows_written() == 3


# --- partial-row / interrupted-write detection ---


def test_in_process_rollback_leaves_no_trace_of_the_row(tmp_path):
    store = SqliteRowStore(tmp_path / "rows.db", n_buckets=10)
    store.append_row("g0", 0, {1: 1})

    store._conn.execute(
        "INSERT INTO rows (row_index, genome_id, n_entries, cols, vals) VALUES (?, ?, ?, ?, ?)",
        (1, "g1", 1, (0).to_bytes(4, "little"), (9).to_bytes(4, "little")),
    )
    store._conn.rollback()

    assert store.row_exists("g1") is False
    assert store.n_rows_written() == 1


def test_true_process_kill_mid_row_is_recovered(tmp_path):
    db_path = tmp_path / "rows.db"
    result = subprocess.run(
        [sys.executable, str(HELPER_SCRIPT), str(db_path), "3", "10"],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 1  # the helper's deliberate os._exit(1)

    reopened = SqliteRowStore(db_path, n_buckets=10)
    assert reopened.existing_genome_ids() == {"genome0", "genome1", "genome2"}
    assert reopened.n_rows_written() == 3
    assert reopened.row_exists("genome3") is False  # the killed, uncommitted row

    # resuming a build must retry genome3 (and beyond) without duplicating 0-2
    failure_log = FailureLog(db_path.parent / "failures.jsonl")
    genome_order = [f"genome{i}" for i in range(5)]
    status = build_sqlite_backed_matrix(
        lambda gid: ["ACGTACGT"], genome_order, k=4, n_buckets=10, row_store=reopened, failure_log=failure_log
    )
    assert status.n_genomes_skipped_already_present == 3
    assert status.n_rows_written_this_run == 2  # genome3, genome4
    assert status.complete is True
    assert reopened.n_rows_written() == 5
    for i in range(5):
        row = reopened.get_row(i)
        assert row is not None
        assert row[0] == f"genome{i}"


# --- resource limits: memory, disk, failures (reused ResourceLimits dataclass) ---


def test_aborts_after_max_failed_rows(tmp_path):
    row_store = SqliteRowStore(tmp_path / "rows.db", n_buckets=10)
    failure_log = FailureLog(tmp_path / "failures.jsonl")

    def always_fails(genome_id):
        raise ValueError("boom")

    status = build_sqlite_backed_matrix(
        always_fails,
        [f"g{i}" for i in range(10)],
        k=4,
        n_buckets=10,
        row_store=row_store,
        failure_log=failure_log,
        limits=ResourceLimits(max_failed_rows=2),
    )
    assert status.aborted_reason is not None
    assert "max_failed_rows" in status.aborted_reason
    assert status.n_rows_written_this_run == 0
    assert row_store.n_rows_written() == 0


def test_aborts_on_memory_limit(tmp_path, monkeypatch):
    import bvbrc_recon.sqlite_row_store as mod

    row_store = SqliteRowStore(tmp_path / "rows.db", n_buckets=10)
    failure_log = FailureLog(tmp_path / "failures.jsonl")
    monkeypatch.setattr(mod, "peak_memory_mb", lambda: 99_999.0)

    status = build_sqlite_backed_matrix(
        lambda gid: ["ACGTACGT"],
        ["g0", "g1"],
        k=4,
        n_buckets=10,
        row_store=row_store,
        failure_log=failure_log,
        limits=ResourceLimits(memory_limit_mb=1.0),
    )
    assert status.aborted_reason is not None
    assert "peak RSS" in status.aborted_reason
    assert row_store.n_rows_written() == 0


def test_aborts_on_insufficient_disk_space(tmp_path, monkeypatch):
    import bvbrc_recon.sqlite_row_store as mod

    row_store = SqliteRowStore(tmp_path / "rows.db", n_buckets=10)
    failure_log = FailureLog(tmp_path / "failures.jsonl")
    monkeypatch.setattr(mod, "check_disk_space", lambda path, min_bytes: False)

    status = build_sqlite_backed_matrix(
        lambda gid: ["ACGTACGT"],
        ["g0"],
        k=4,
        n_buckets=10,
        row_store=row_store,
        failure_log=failure_log,
    )
    assert status.aborted_reason == "insufficient disk space before starting"
    assert row_store.n_rows_written() == 0


# --- failure tracking: never a zero row for a failed genome ---


def test_failed_genome_never_gets_a_row(tmp_path):
    row_store = SqliteRowStore(tmp_path / "rows.db", n_buckets=10)
    failure_log = FailureLog(tmp_path / "failures.jsonl")

    def loader(genome_id):
        if genome_id == "bad":
            raise RuntimeError("extraction failed")
        return ["ACGTACGT"]

    status = build_sqlite_backed_matrix(
        loader, ["good1", "bad", "good2"], k=4, n_buckets=10, row_store=row_store, failure_log=failure_log
    )
    assert row_store.row_exists("bad") is False
    assert "bad" in failure_log.genome_ids()
    assert status.n_failures_this_run == 1
    assert row_store.n_rows_written() == 2


# --- batch reads: bounded memory, correct filtering ---


def test_iter_batches_respects_batch_size(tmp_path):
    row_store = SqliteRowStore(tmp_path / "rows.db", n_buckets=10)
    for i in range(7):
        row_store.append_row(f"g{i}", i, {0: i})

    batch_sizes = []
    total_rows = 0
    for genome_ids, row_indices, matrix in row_store.iter_batches(batch_size=3):
        batch_sizes.append(matrix.shape[0])
        total_rows += matrix.shape[0]
    assert batch_sizes == [3, 3, 1]
    assert total_rows == 7


def test_read_rows_for_indices_filters_and_batches(tmp_path):
    row_store = SqliteRowStore(tmp_path / "rows.db", n_buckets=10)
    for i in range(10):
        row_store.append_row(f"g{i}", i, {0: i + 1})

    wanted = {1, 3, 5, 7, 9}
    seen_row_indices = set()
    for genome_ids, row_indices, matrix in row_store.read_rows_for_indices(wanted, batch_size=2):
        assert matrix.shape[0] <= 2
        seen_row_indices.update(row_indices)
    assert seen_row_indices == wanted


# --- never mark an incomplete store as complete ---


def test_status_not_complete_when_aborted(tmp_path):
    row_store = SqliteRowStore(tmp_path / "rows.db", n_buckets=10)
    failure_log = FailureLog(tmp_path / "failures.jsonl")

    status = build_sqlite_backed_matrix(
        lambda gid: ["ACGTACGT"],
        ["g0", "g1", "g2"],
        k=4,
        n_buckets=10,
        row_store=row_store,
        failure_log=failure_log,
        limits=ResourceLimits(memory_limit_mb=-1.0),  # guaranteed to trip immediately
    )
    assert status.complete is False
    assert status.aborted_reason is not None


# --- artifact isolation ---


def test_new_store_path_is_independent_of_other_stores(tmp_path):
    path_a = tmp_path / "a.db"
    path_b = tmp_path / "b.db"
    store_a = SqliteRowStore(path_a, n_buckets=10)
    store_b = SqliteRowStore(path_b, n_buckets=10)
    store_a.append_row("g0", 0, {1: 1})
    assert store_b.n_rows_written() == 0
    assert path_a.exists() and path_b.exists()
    assert path_a != path_b


# --- read-path optimization: exact equivalence between naive and vectorized reconstruction ---


def test_vectorized_reconstruction_matches_naive_exactly_dense_rows(tmp_path):
    store = SqliteRowStore(tmp_path / "rows.db", n_buckets=20)
    for i in range(5):
        # fully dense rows, like the real pilot at k=15/d=262144
        store.append_row(f"g{i}", i, {c: (i + 1) * (c + 1) for c in range(20)})

    cur = store._conn.execute("SELECT row_index, genome_id, n_entries, cols, vals FROM rows ORDER BY row_index")
    rows = cur.fetchall()

    genome_ids_fast, row_indices_fast, matrix_fast = store._rows_to_csr(rows)
    genome_ids_naive, row_indices_naive, matrix_naive = store._rows_to_csr_naive(rows)

    assert genome_ids_fast == genome_ids_naive
    assert row_indices_fast == row_indices_naive
    assert (matrix_fast != matrix_naive).nnz == 0  # exact elementwise equality
    assert matrix_fast.shape == matrix_naive.shape
    assert matrix_fast.nnz == matrix_naive.nnz


def test_vectorized_reconstruction_matches_naive_exactly_sparse_rows(tmp_path):
    store = SqliteRowStore(tmp_path / "rows.db", n_buckets=1000)
    patterns = [{3: 7, 500: 2}, {0: 1}, {999: 42, 1: 1, 2: 1}, {}]
    for i, pattern in enumerate(patterns):
        if not pattern:
            continue  # a row needs at least one entry per append_row's np.array(list(...))
        store.append_row(f"g{i}", i, pattern)

    cur = store._conn.execute("SELECT row_index, genome_id, n_entries, cols, vals FROM rows ORDER BY row_index")
    rows = cur.fetchall()

    fast = store._rows_to_csr(rows)
    naive = store._rows_to_csr_naive(rows)
    assert fast[0] == naive[0]
    assert fast[1] == naive[1]
    assert (fast[2] != naive[2]).nnz == 0


def test_vectorized_reconstruction_handles_empty_batch(tmp_path):
    store = SqliteRowStore(tmp_path / "rows.db", n_buckets=10)
    genome_ids, row_indices, matrix = store._rows_to_csr([])
    assert genome_ids == []
    assert row_indices == []
    assert matrix.shape == (0, 10)
