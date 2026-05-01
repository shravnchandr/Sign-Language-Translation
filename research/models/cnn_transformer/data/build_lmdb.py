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
"""
import argparse
import io
from pathlib import Path

import lmdb
import pandas as pd
import torch

from .preprocessing import frame_stacked_data

# Write this many samples per LMDB transaction to cap in-memory dirty pages.
_WRITE_BATCH = 2_000


def build_lmdb(data_dir: str, lmdb_path: str, map_size_gb: int = 100) -> None:
    data_dir = Path(data_dir)
    lmdb_path = Path(lmdb_path)
    lmdb_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_dir / "train.csv")
    n = len(df)
    print(f"Building LMDB: {n} samples → {lmdb_path}")

    env = lmdb.open(str(lmdb_path), map_size=map_size_gb * 1024**3)

    written = skipped = errors = 0

    for batch_start in range(0, n, _WRITE_BATCH):
        batch = df.iloc[batch_start : batch_start + _WRITE_BATCH]
        with env.begin(write=True) as txn:
            for _, row in batch.iterrows():
                key = row["path"].encode()
                if txn.get(key) is not None:
                    skipped += 1
                    continue
                full_path = str(data_dir / row["path"])
                try:
                    coords = torch.tensor(
                        frame_stacked_data(full_path), dtype=torch.float32
                    )
                    buf = io.BytesIO()
                    torch.save(coords, buf)
                    txn.put(key, buf.getvalue())
                    written += 1
                except Exception as e:
                    errors += 1
                    print(f"  WARNING: skipping {row['path']}: {e}")

        done = batch_start + len(batch)
        print(
            f"  {done}/{n}  written={written}  skipped={skipped}  errors={errors}",
            flush=True,
        )

    env.close()
    print(
        f"\nDone. LMDB at {lmdb_path}\n"
        f"  written={written}  skipped={skipped}  errors={errors}"
    )


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
    args = parser.parse_args()
    build_lmdb(args.data_dir, args.lmdb_path, args.map_size_gb)


if __name__ == "__main__":
    main()
