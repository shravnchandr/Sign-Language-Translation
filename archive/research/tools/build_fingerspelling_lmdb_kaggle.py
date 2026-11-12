"""
Standalone Kaggle script: build LMDB cache from ASL Fingerspelling Recognition dataset.

Paste this file into a Kaggle notebook code cell or upload as a script.
Attach the "asl-fingerspelling" competition dataset before running.

Resume safety: existing keys are skipped, so an interrupted run restarts cleanly.
If the 20 GB /kaggle/working/ cap is hit mid-run, save the version as a Kaggle
dataset, attach it in a new session, copy the partial LMDB into /kaggle/working/,
and re-run — already-written sequences are skipped automatically.

Keys
----
  Data:   b"fs:{sequence_id}"
  Length: b"fs:len:{sequence_id}"

Run:
  python build_fingerspelling_lmdb_kaggle.py
  python build_fingerspelling_lmdb_kaggle.py --map-size-gb 30 --num-workers 4
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

# ---------------------------------------------------------------------------
# Config (inlined from cnn_transformer/config.py — keep in sync)
# ---------------------------------------------------------------------------
INCLUDE_FACE = True
INCLUDE_DEPTH = True

FACE_LANDMARK_INDICES = {
    "left_eyebrow":  [70, 63, 105, 66, 107, 55, 65, 52],
    "right_eyebrow": [300, 293, 334, 296, 336, 285, 295, 282],
    "mouth_outer": [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
    ],
    "mouth_inner": [
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    ],
}

SELECTED_FACE_INDICES: list[int] = []
for _face_idxs in FACE_LANDMARK_INDICES.values():
    SELECTED_FACE_INDICES.extend(_face_idxs)

COORDS_PER_LM = 3 if INCLUDE_DEPTH else 2

_ALL_COLUMNS: list[str] = []
for _lm_type, _count in [("left_hand", 21), ("pose", 33), ("right_hand", 21)]:
    for _i in range(_count):
        for _ax in (["x", "y", "z"] if INCLUDE_DEPTH else ["x", "y"]):
            _ALL_COLUMNS.append(f"{_lm_type}_{_i}_{_ax}")
if INCLUDE_FACE:
    for _fi in SELECTED_FACE_INDICES:
        for _ax in (["x", "y", "z"] if INCLUDE_DEPTH else ["x", "y"]):
            _ALL_COLUMNS.append(f"face_{_fi}_{_ax}")

COORD_FEAT: int = len(_ALL_COLUMNS)  # 393

# ---------------------------------------------------------------------------
# Column mapping
# ALL_COLUMNS fmt:    "{type}_{idx}_{axis}"  e.g. "left_hand_0_x"
# Fingerspelling fmt: "{axis}_{type}_{idx}"  e.g. "x_left_hand_0"
# ---------------------------------------------------------------------------
_FS_COLS: list[str] = []
for _col in _ALL_COLUMNS:
    _lm_type2, _lm_idx2, _axis2 = _col.rsplit("_", 2)
    _FS_COLS.append(f"{_axis2}_{_lm_type2}_{_lm_idx2}")

_WRITE_BATCH = 500


def _lmdb_key(sequence_id: int) -> bytes:
    return f"fs:{sequence_id}".encode()


def _lmdb_length_key(sequence_id: int) -> bytes:
    return f"fs:len:{sequence_id}".encode()


# ---------------------------------------------------------------------------
# Worker (self-contained — fs_cols and coord_feat passed in args tuple)
# ---------------------------------------------------------------------------

def _process_parquet(args: tuple) -> list[tuple]:
    """Load one fingerspelling parquet and process all pending sequences.

    Returns list of (sequence_id, tensor_bytes, length, error).
    """
    import io
    import numpy as np
    import pandas as pd
    import torch

    parquet_path, pending, fs_cols, coord_feat = args
    axes = ["x", "y", "z"]

    def _normalize(data: np.ndarray, col_index: dict) -> np.ndarray:
        data = data.copy()
        for ax in axes:
            def _get(name, ax=ax):
                i = col_index.get(f"{ax}_{name}")
                return data[:, i] if i is not None else np.full(len(data), np.nan)

            nose     = _get("pose_0")
            shoulder = (_get("pose_11") + _get("pose_12")) / 2
            hip      = (_get("pose_23") + _get("pose_24")) / 2
            origin   = np.where(~np.isnan(nose),      nose,
                       np.where(~np.isnan(shoulder),   shoulder,
                       np.where(~np.isnan(hip),         hip, 0.0)))
            ax_idxs = [i for c, i in col_index.items() if c.startswith(f"{ax}_")]
            data[:, ax_idxs] -= origin[:, None]
        return data

    try:
        wide = pd.read_parquet(parquet_path)
    except Exception as e:
        return [(sid, None, None, str(e)) for sid in pending]

    data_cols = [c for c in wide.columns if c != "frame"]
    col_index = {c: i for i, c in enumerate(data_cols)}

    results = []
    for sid in pending:
        try:
            seq_df = wide.loc[sid]
            if isinstance(seq_df, pd.Series):
                seq_df = seq_df.to_frame().T
            if "frame" not in seq_df.columns:
                seq_df = seq_df.reset_index()
            seq_df = seq_df.sort_values("frame").reset_index(drop=True)

            raw = seq_df.drop(columns=["frame"], errors="ignore").to_numpy(dtype=np.float32)
            raw = _normalize(raw, col_index)

            out = np.zeros((len(raw), coord_feat), dtype=np.float32)
            for fi, fs_col in enumerate(fs_cols):
                ci = col_index.get(fs_col)
                if ci is not None:
                    vals = raw[:, ci]
                    out[:, fi] = np.where(np.isnan(vals), 0.0, vals)

            tensor = torch.from_numpy(out)
            buf    = io.BytesIO()
            torch.save(tensor, buf)
            results.append((sid, buf.getvalue(), len(tensor), None))
        except Exception as e:
            results.append((sid, None, None, str(e)))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build(
    data_dir: Path,
    lmdb_path: Path,
    out_csv: Path,
    map_size_gb: int = 25,
    allow_errors: bool = False,
    num_workers: int | None = None,
) -> None:
    if num_workers is None:
        num_workers = os.cpu_count() or 4

    train_csv = pd.read_csv(data_dir / "train.csv")
    total_seqs = len(train_csv)
    print(
        f"Fingerspelling LMDB: {total_seqs:,} sequences, "
        f"{train_csv['path'].nunique()} parquet files → {lmdb_path}  "
        f"(workers={num_workers}, COORD_FEAT={COORD_FEAT})"
    )

    lmdb_path.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(lmdb_path), map_size=map_size_gb * 1024**3)

    written = skipped = errors = 0

    # Pre-scan for already-done sequence_ids
    all_sids = train_csv["sequence_id"].tolist()
    pending_sids: set[int] = set()
    backfill_sids: list[int] = []

    with env.begin() as txn:
        for sid in all_sids:
            has_data = txn.get(_lmdb_key(sid)) is not None
            has_len  = txn.get(_lmdb_length_key(sid)) is not None
            if has_data:
                if not has_len:
                    backfill_sids.append(sid)
                skipped += 1
            else:
                pending_sids.add(sid)

    if backfill_sids:
        print(f"Backfilling length keys for {len(backfill_sids)} existing entries...")
        with env.begin(write=True) as txn:
            for sid in backfill_sids:
                raw    = txn.get(_lmdb_key(sid))
                coords = torch.load(io.BytesIO(bytes(raw)), weights_only=True)
                txn.put(_lmdb_length_key(sid), str(len(coords)).encode())

    # Group pending sids by parquet file
    parquet_tasks: list[tuple] = []
    for parquet_rel, group in train_csv.groupby("path", sort=False):
        sids_in_file = [
            int(sid) for sid in group["sequence_id"].tolist()
            if int(sid) in pending_sids
        ]
        if sids_in_file:
            parquet_tasks.append((
                str(data_dir / parquet_rel),
                sids_in_file,
                _FS_COLS,
                COORD_FEAT,
            ))

    n_pending = sum(len(t[1]) for t in parquet_tasks)
    if n_pending == 0:
        print("All sequences already in LMDB. Nothing to do.")
        env.close()
        _write_csv(train_csv, out_csv)
        return

    print(f"Processing {n_pending:,} sequences across {len(parquet_tasks)} parquet files...")

    write_buf: list[tuple[int, bytes, int]] = []

    def _flush(buf: list) -> None:
        with env.begin(write=True) as txn:
            for sid, data, length in buf:
                txn.put(_lmdb_key(sid),        data)
                txn.put(_lmdb_length_key(sid), str(length).encode())
        buf.clear()

    with tqdm(total=total_seqs, desc="Building FS LMDB", unit="seq") as pbar:
        pbar.update(skipped)
        pbar.set_postfix(written=written, skipped=skipped, errors=errors)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for batch_results in executor.map(
                _process_parquet, parquet_tasks, chunksize=1
            ):
                for sid, data, length, error in batch_results:
                    if error:
                        errors += 1
                        tqdm.write(f"  ERROR seq {sid}: {error}", file=sys.stderr)
                    else:
                        write_buf.append((sid, data, length))
                        written += 1
                        if len(write_buf) >= _WRITE_BATCH:
                            _flush(write_buf)

                    pbar.update(1)
                    pbar.set_postfix(written=written, skipped=skipped, errors=errors)

        if write_buf:
            _flush(write_buf)

    env.close()
    _write_csv(train_csv, out_csv)

    print(
        f"\nDone.  LMDB → {lmdb_path}\n"
        f"  written={written}  skipped={skipped}  errors={errors}"
    )
    print(f"CSV   → {out_csv}")

    if errors > 0 and not allow_errors:
        print(
            f"\nFATAL: {errors} sequence(s) failed. Re-run to retry, or pass "
            "--allow-errors to ignore.",
            file=sys.stderr,
        )
        sys.exit(1)


def _write_csv(train_csv: pd.DataFrame, out_csv: Path) -> None:
    out = train_csv[["sequence_id", "participant_id", "phrase"]].copy()
    out.to_csv(out_csv, index=False)


def main():
    p = argparse.ArgumentParser(
        description="Build LMDB cache for ASL Fingerspelling (Kaggle standalone)"
    )
    p.add_argument(
        "--data-dir",
        default="/kaggle/input/asl-fingerspelling",
        help="Fingerspelling dataset root (default: /kaggle/input/asl-fingerspelling)",
    )
    p.add_argument(
        "--lmdb-path",
        default="/kaggle/working/fs.lmdb",
        help="Output LMDB directory (default: /kaggle/working/fs.lmdb)",
    )
    p.add_argument(
        "--out-csv",
        default="/kaggle/working/train.csv",
        help="Output metadata CSV (default: /kaggle/working/train.csv)",
    )
    p.add_argument(
        "--map-size-gb",
        type=int,
        default=25,
        help="LMDB virtual address space in GB (default 25 — fits within 20 GB cap)",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel workers — one per parquet file (default: os.cpu_count())",
    )
    p.add_argument(
        "--allow-errors",
        action="store_true",
        help="Exit 0 even if some sequences fail",
    )
    args = p.parse_args()
    build(
        Path(args.data_dir),
        Path(args.lmdb_path),
        Path(args.out_csv),
        args.map_size_gb,
        args.allow_errors,
        args.num_workers,
    )


if __name__ == "__main__":
    main()
