"""Helper invoked as a SEPARATE process by
`test_sqlite_row_store.py::test_true_process_kill_mid_row_is_recovered`.

Commits `n_committed` rows normally (each a complete, committed
transaction), then begins writing one more row and calls `os._exit(1)`
before committing it - a hard process kill that skips Python and SQLite
cleanup entirely, the closest reproducible stand-in for a real crash
(SIGKILL) available in a test. Not a test module itself (leading
underscore keeps pytest from collecting it); run directly:

    python tests/_sqlite_interrupt_subprocess.py <db_path> <n_committed> <n_buckets>
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bvbrc_recon.sqlite_row_store import SqliteRowStore  # noqa: E402


def main(db_path: str, n_committed: int, n_buckets: int) -> None:
    store = SqliteRowStore(Path(db_path), n_buckets=n_buckets)
    for i in range(n_committed):
        store.append_row(f"genome{i}", i, {0: i + 1})

    # Begin (but never commit) one more row, then kill the process outright.
    store._conn.execute(
        "INSERT INTO rows (row_index, genome_id, n_entries, cols, vals) VALUES (?, ?, ?, ?, ?)",
        (n_committed, f"genome{n_committed}", 1, (0).to_bytes(4, "little"), (999).to_bytes(4, "little")),
    )
    os._exit(1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
