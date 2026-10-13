"""Helper invoked as a SEPARATE process by
`test_hybrid_training_reader.py::test_true_process_kill_mid_build_is_recovered`.

Builds a `HybridTrainingReader`'s disk tier for `n_batches_before_kill`
batches normally (each batch's npz is written, renamed into place, and
`meta.json` updated - a real completed build step), then calls
`os._exit(1)` partway through writing the NEXT batch's npz (after
`save_npz` to the staging path, but BEFORE the rename into its final
name) - a hard process kill that skips Python cleanup entirely, the
same technique used for the SQLite crash-recovery test in an earlier
task. Not a test module itself (leading underscore keeps pytest from
collecting it); run directly:

    python tests/_hybrid_interrupt_subprocess.py <db_path> <cache_dir> <n_rows> <batch_size> <n_batches_before_kill>
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scipy import sparse  # noqa: E402

from bvbrc_recon.hybrid_training_reader import HybridTrainingReader  # noqa: E402
from bvbrc_recon.sqlite_row_store import SqliteRowStore  # noqa: E402


def main(db_path: str, cache_dir: str, n_rows: int, batch_size: int, n_batches_before_kill: int) -> None:
    store = SqliteRowStore(Path(db_path), n_buckets=20)
    for i in range(n_rows):
        store.append_row(f"g{i}", i, {c: i + c + 1 for c in range(20)})

    reader = HybridTrainingReader(
        Path(cache_dir), store, row_indices=set(range(n_rows)), batch_size=batch_size, max_memory_cache_bytes=0
    )
    # Drive the build loop manually, same logic as ensure_disk_materialized,
    # but stopping mid-write of the batch AFTER n_batches_before_kill.
    reader.cache_dir.mkdir(parents=True, exist_ok=True)
    reader._write_json_atomic(reader.build_fingerprint_path, reader._compute_fingerprint())
    reader._write_json_atomic(reader.meta_path, {"batches": []})
    staging_dir = reader.cache_dir / ".staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    expected = reader._expected_chunks()
    meta = {"batches": []}
    for idx, chunk in enumerate(expected):
        if idx >= n_batches_before_kill:
            # write the npz to staging but DO NOT rename it into place,
            # then kill the process - simulates a crash mid-batch-write.
            batch = None
            for genome_ids, row_indices, matrix in store.read_rows_for_indices(set(chunk), batch_size=len(chunk)):
                batch = (genome_ids, row_indices, matrix)
            staging_path = staging_dir / f"batch_{idx:05d}.npz"
            sparse.save_npz(staging_path, batch[2])
            os._exit(1)

        batch = None
        for genome_ids, row_indices, matrix in store.read_rows_for_indices(set(chunk), batch_size=len(chunk)):
            batch = (genome_ids, row_indices, matrix)
        final_path = reader.cache_dir / f"batch_{idx:05d}.npz"
        staging_path = staging_dir / f"batch_{idx:05d}.npz"
        sparse.save_npz(staging_path, batch[2])
        staging_path.rename(final_path)
        meta["batches"].append(
            {"batch_index": idx, "npz_file": final_path.name, "genome_ids": batch[0], "row_indices": batch[1]}
        )
        reader._write_json_atomic(reader.meta_path, meta)

    os._exit(1)  # only reached if n_batches_before_kill >= len(expected)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
