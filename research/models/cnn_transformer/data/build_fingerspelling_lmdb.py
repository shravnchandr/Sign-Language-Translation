"""
Build an LMDB cache from the ASL Fingerspelling Recognition dataset.

The Fingerspelling parquets are in wide format (one large file per file_id,
multiple sequences indexed by sequence_id).  This script processes them one
parquet at a time, extracts per-sequence tensors in our ALL_COLUMNS layout,
and writes them to LMDB for fast random access during pre-training.

Output
------
  <lmdb-path>/          — LMDB directory (data.mdb + lock.mdb)
  <out-csv>             — CSV with columns: sequence_id, participant_id, phrase

Key format: b"fs:{sequence_id}"   (simple, no version hash needed)
Value:      torch-serialised (T, COORD_FEAT) float32 tensor

Run (from project root):
  PYTHONPATH=research/models python -m cnn_transformer.data.build_fingerspelling_lmdb \\
      --data-dir  data/ASL_Fingerspelling_Recognition \\
      --lmdb-path data/cache/fingerspelling/fs.lmdb \\
      --out-csv   data/cache/fingerspelling/train.csv
"""
import argparse
import io
import sys
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..config import ALL_COLUMNS, COORD_FEAT

# ---------------------------------------------------------------------------
# Column mapping: ALL_COLUMNS  →  fingerspelling wide-format column names
# ALL_COLUMNS fmt:  "{type}_{idx}_{axis}"  e.g. "left_hand_0_x"
# Fingerspelling:   "{axis}_{type}_{idx}"  e.g. "x_left_hand_0"
# ---------------------------------------------------------------------------
_FS_COLS: list[str] = []
for _col in ALL_COLUMNS:
    _lm_type, _lm_idx, _axis = _col.rsplit("_", 2)
    _FS_COLS.append(f"{_axis}_{_lm_type}_{_lm_idx}")

_AXES = ["x", "y", "z"]
_WRITE_BATCH = 500


def _normalize_wide(data: np.ndarray, col_index: dict[str, int]) -> np.ndarray:
    """Body-relative normalization (nose→shoulder→hip→0) on a wide numpy array."""
    data = data.copy()
    for ax in _AXES:
        def _col(name):
            idx = col_index.get(f"{ax}_{name}")
            return data[:, idx] if idx is not None else np.full(len(data), np.nan)

        nose     = _col("pose_0")
        shoulder = (_col("pose_11") + _col("pose_12")) / 2
        hip      = (_col("pose_23") + _col("pose_24")) / 2

        origin = np.where(~np.isnan(nose),     nose,
                 np.where(~np.isnan(shoulder),  shoulder,
                 np.where(~np.isnan(hip),        hip, 0.0)))

        ax_indices = [i for name, i in col_index.items() if name.startswith(f"{ax}_")]
        data[:, ax_indices] -= origin[:, None]
    return data


def _seq_to_tensor(
    seq_df: pd.DataFrame,
    col_index: dict[str, int],
) -> torch.Tensor:
    """Convert one sequence DataFrame (wide format) to (T, COORD_FEAT) tensor."""
    seq_df = seq_df.sort_values("frame").reset_index(drop=True)
    raw = seq_df.drop(columns=["frame"], errors="ignore").to_numpy(dtype=np.float32)

    raw = _normalize_wide(raw, col_index)

    # Select only the columns that correspond to ALL_COLUMNS
    out = np.zeros((len(raw), COORD_FEAT), dtype=np.float32)
    for feat_i, fs_col in enumerate(_FS_COLS):
        ci = col_index.get(fs_col)
        if ci is not None:
            col_vals = raw[:, ci]
            col_vals = np.where(np.isnan(col_vals), 0.0, col_vals)
            out[:, feat_i] = col_vals

    return torch.from_numpy(out)  # (T, COORD_FEAT)


def build(data_dir: Path, lmdb_path: Path, out_csv: Path, map_size_gb: int) -> None:
    train_csv = pd.read_csv(data_dir / "train.csv")
    print(f"Fingerspelling: {len(train_csv):,} sequences, "
          f"{train_csv['path'].nunique()} parquet files, "
          f"{train_csv['participant_id'].nunique()} signers")

    lmdb_path.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(lmdb_path), map_size=map_size_gb * 1024**3)

    # Pre-scan: skip already-done sequences
    done: set[int] = set()
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, _ in cursor.iternext_dup() if False else []:
            pass
        # Just check membership on demand below (fast enough for <100k entries)

    records: list[dict] = []
    written = skipped = errors = 0
    write_buf: list[tuple[bytes, bytes]] = []

    def _flush():
        with env.begin(write=True) as txn:
            for k, v in write_buf:
                txn.put(k, v)
        write_buf.clear()

    # Process one parquet file at a time to bound peak memory
    for parquet_rel, group in tqdm(
        train_csv.groupby("path", sort=False),
        desc="Parquet files",
        unit="file",
    ):
        parquet_path = data_dir / parquet_rel
        try:
            wide = pd.read_parquet(parquet_path)
        except Exception as e:
            tqdm.write(f"  ERROR reading {parquet_rel}: {e}", file=sys.stderr)
            errors += len(group)
            continue

        # Build column index once per parquet (columns may differ)
        data_cols = [c for c in wide.columns if c != "frame"]
        col_index = {c: i for i, c in enumerate(data_cols)}

        for row in group.itertuples(index=False):
            sid = int(row.sequence_id)
            lmdb_key = f"fs:{sid}".encode()

            # Check if already done
            with env.begin() as txn:
                if txn.get(lmdb_key) is not None:
                    skipped += 1
                    records.append({
                        "sequence_id": sid,
                        "participant_id": row.participant_id,
                        "phrase": row.phrase,
                    })
                    continue

            try:
                seq_df = wide.loc[sid]
                if isinstance(seq_df, pd.Series):
                    seq_df = seq_df.to_frame().T
                # Restore 'frame' column if it's not the index
                if "frame" not in seq_df.columns:
                    seq_df = seq_df.reset_index()

                tensor = _seq_to_tensor(seq_df, col_index)
                buf = io.BytesIO()
                torch.save(tensor, buf)
                write_buf.append((lmdb_key, buf.getvalue()))
                records.append({
                    "sequence_id": sid,
                    "participant_id": row.participant_id,
                    "phrase": row.phrase,
                })
                written += 1

                if len(write_buf) >= _WRITE_BATCH:
                    _flush()

            except Exception as e:
                tqdm.write(f"  ERROR seq {sid}: {e}", file=sys.stderr)
                errors += 1

    if write_buf:
        _flush()
    env.close()

    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"\nDone. written={written}  skipped={skipped}  errors={errors}")
    print(f"LMDB  → {lmdb_path}")
    print(f"CSV   → {out_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True,
                   help="data/ASL_Fingerspelling_Recognition")
    p.add_argument("--lmdb-path", required=True,
                   help="data/cache/fingerspelling/fs.lmdb")
    p.add_argument("--out-csv", required=True,
                   help="data/cache/fingerspelling/train.csv")
    p.add_argument("--map-size-gb", type=int, default=50)
    args = p.parse_args()
    build(Path(args.data_dir), Path(args.lmdb_path), Path(args.out_csv), args.map_size_gb)


if __name__ == "__main__":
    main()
