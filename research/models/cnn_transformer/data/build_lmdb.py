"""
Build a single LMDB archive from ASL parquet files.

On network-attached storage (RunPod) individual .pt files incur a per-open()
syscall for every sample every epoch.  A single LMDB archive eliminates that
overhead: one file open per process, then random-access reads.

Run once before training (from the project root):

    PYTHONPATH=research/models uv run python -m cnn_transformer.data.build_lmdb \\
        --data-dir  data/Isolated_ASL_Recognition \\
        --lmdb-path data/cache/cnn_transformer/asl.lmdb

Resume safety: existing keys are skipped, so an interrupted run can be
restarted without re-processing completed samples.

Keys are versioned (_CACHE_VERSION prefix) so rebuilding after a config
change (e.g. new face landmark set) produces new keys and never silently
reuses stale tensors.
"""
import argparse
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lmdb
import pandas as pd
import torch
from tqdm import tqdm

from ._cache_keys import lmdb_key as _lmdb_key, lmdb_length_key as _lmdb_length_key
from .preprocessing import frame_stacked_data

# Commit to LMDB every this many successful writes to cap dirty-page memory.
_WRITE_BATCH = 2_000


def _process_sample(args: tuple) -> tuple:
    """Worker: parse one parquet file and return serialised tensor bytes.

    Self-contained imports make this safe under spawn/forkserver (Python 3.14
    default on macOS/Linux). PYTHONPATH is inherited from the parent process.
    """
    import io
    import torch
    from cnn_transformer.data.preprocessing import frame_stacked_data

    path, full_path = args
    try:
        coords = torch.tensor(frame_stacked_data(full_path), dtype=torch.float32)
        buf = io.BytesIO()
        torch.save(coords, buf)
        return path, buf.getvalue(), len(coords), None
    except Exception as e:
        return path, None, None, str(e)


def build_lmdb(
    data_dir: str,
    lmdb_path: str,
    map_size_gb: int = 100,
    allow_errors: bool = False,
    num_workers: int | None = None,
) -> None:
    data_dir = Path(data_dir)
    lmdb_path = Path(lmdb_path)
    lmdb_path.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = os.cpu_count() or 4

    df = pd.read_csv(data_dir / "train.csv")
    n = len(df)
    paths = df["path"].tolist()
    print(f"Building LMDB: {n} samples → {lmdb_path}  (workers={num_workers})")

    env = lmdb.open(str(lmdb_path), map_size=map_size_gb * 1024**3)

    written = skipped = errors = 0

    # Pre-scan: find which paths still need processing, and backfill any
    # missing length keys left by older builds (sequential, rare).
    pending_args: list[tuple[str, str]] = []
    backfill: list[str] = []
    with env.begin() as txn:
        for path in paths:
            if txn.get(_lmdb_key(path)) is not None:
                if txn.get(_lmdb_length_key(path)) is None:
                    backfill.append(path)
                skipped += 1
            else:
                pending_args.append((path, str(data_dir / path)))

    if backfill:
        print(f"Backfilling length keys for {len(backfill)} existing entries...")
        with env.begin(write=True) as txn:
            for path in backfill:
                raw = txn.get(_lmdb_key(path))
                coords = torch.load(io.BytesIO(bytes(raw)), weights_only=True)
                txn.put(_lmdb_length_key(path), str(len(coords)).encode())

    # Parallel parse → single-threaded write.
    # Workers parse parquets concurrently; the main thread collects results and
    # writes to LMDB in batches (LMDB does not allow concurrent writes).
    write_buf: list[tuple[str, bytes, int]] = []

    def _flush(buf: list) -> None:
        with env.begin(write=True) as txn:
            for p, data, length in buf:
                txn.put(_lmdb_key(p), data)
                txn.put(_lmdb_length_key(p), str(length).encode())
        buf.clear()

    with tqdm(total=n, desc="Building LMDB", unit="sample") as pbar:
        pbar.update(skipped)
        pbar.set_postfix(written=written, skipped=skipped, errors=errors)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for path, data, length, error in executor.map(
                _process_sample, pending_args, chunksize=8
            ):
                if error:
                    errors += 1
                    tqdm.write(f"  ERROR: skipping {path}: {error}", file=sys.stderr)
                else:
                    write_buf.append((path, data, length))
                    written += 1
                    if len(write_buf) >= _WRITE_BATCH:
                        _flush(write_buf)

                pbar.update(1)
                pbar.set_postfix(written=written, skipped=skipped, errors=errors)

        if write_buf:
            _flush(write_buf)

    env.close()
    print(
        f"\nDone. LMDB at {lmdb_path}\n"
        f"  written={written}  skipped={skipped}  errors={errors}"
    )

    if errors > 0 and not allow_errors:
        print(
            f"\nFATAL: {errors} sample(s) failed. Re-run to retry, or pass "
            "--allow-errors to ignore and proceed.",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Build LMDB cache for ASL dataset")
    parser.add_argument("--data-dir", required=True, help="Directory containing train.csv")
    parser.add_argument("--lmdb-path", required=True, help="Output LMDB directory path")
    parser.add_argument(
        "--map-size-gb",
        type=int,
        default=100,
        help="LMDB virtual address space ceiling in GB (default 100)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel workers for parquet parsing (default: os.cpu_count())",
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Exit 0 even if some samples fail (useful for known-bad parquets)",
    )
    args = parser.parse_args()
    build_lmdb(
        args.data_dir, args.lmdb_path, args.map_size_gb, args.allow_errors, args.num_workers
    )


if __name__ == "__main__":
    main()
