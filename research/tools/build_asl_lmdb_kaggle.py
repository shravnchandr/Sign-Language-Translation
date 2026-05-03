"""
Standalone Kaggle script: build LMDB cache from Google ASL Signs dataset.

Paste this file into a Kaggle notebook code cell or upload as a script.
Attach the "asl-signs" competition dataset before running.

Estimated output size: ~15 GB (94k samples × ~100 frames × 393 features × float32).
Fits within the 20 GB /kaggle/working/ cap in a single session.

Resume safety: existing keys are skipped — if the session dies, re-run and it
picks up from where it left off. Save the partial output as a Kaggle dataset
between sessions if needed (see README section on the 20 GB cap).

Keys are versioned by CACHE_VERSION (MD5 of ALL_COLUMNS + norm version), so
changing the landmark selection or normalization logic produces new keys and
never silently reuses stale tensors.

Run:
  python build_asl_lmdb_kaggle.py
  python build_asl_lmdb_kaggle.py --map-size-gb 20 --num-workers 4
"""
import argparse
import hashlib
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lmdb
import numpy as np
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

FACE_LANDMARK_SET = frozenset(SELECTED_FACE_INDICES)
COORDS_PER_LM = 3 if INCLUDE_DEPTH else 2

ALL_COLUMNS: list[str] = []
for _lm_type, _count in [("left_hand", 21), ("pose", 33), ("right_hand", 21)]:
    for _i in range(_count):
        for _ax in (["x", "y", "z"] if INCLUDE_DEPTH else ["x", "y"]):
            ALL_COLUMNS.append(f"{_lm_type}_{_i}_{_ax}")
if INCLUDE_FACE:
    for _fi in SELECTED_FACE_INDICES:
        for _ax in (["x", "y", "z"] if INCLUDE_DEPTH else ["x", "y"]):
            ALL_COLUMNS.append(f"face_{_fi}_{_ax}")

COORD_FEAT: int = len(ALL_COLUMNS)  # 393

# ---------------------------------------------------------------------------
# Cache key versioning (inlined from data/_cache_keys.py)
# Bump _NORM_VERSION whenever normalize_values logic changes.
# ---------------------------------------------------------------------------
_NORM_VERSION = "v2_fallback"
CACHE_VERSION = hashlib.md5(
    ("|".join(ALL_COLUMNS) + "|" + _NORM_VERSION).encode()
).hexdigest()[:8]


def lmdb_key(path: str) -> bytes:
    return f"{CACHE_VERSION}:{path}".encode()


def lmdb_length_key(path: str) -> bytes:
    return f"{CACHE_VERSION}:len:{path}".encode()


# ---------------------------------------------------------------------------
# Worker setup — config passed once per process via initializer
# ---------------------------------------------------------------------------
_w_all_columns: list[str] = []
_w_face_lm_set: frozenset = frozenset()
_w_include_face: bool = True
_w_include_depth: bool = True


def _worker_init(
    all_columns: list[str],
    face_lm_set: frozenset,
    include_face: bool,
    include_depth: bool,
) -> None:
    global _w_all_columns, _w_face_lm_set, _w_include_face, _w_include_depth
    _w_all_columns = all_columns
    _w_face_lm_set = face_lm_set
    _w_include_face = include_face
    _w_include_depth = include_depth


def _process_sample(args: tuple) -> tuple:
    """Worker: parse one parquet, normalize, and return serialised tensor bytes.

    Config globals set once by _worker_init — avoids re-pickling 393-element
    ALL_COLUMNS list for every one of the 94k samples.
    """
    import io
    import numpy as np
    import pandas as pd
    import torch

    path, full_path = args
    all_columns = _w_all_columns
    face_lm_set  = _w_face_lm_set
    include_face  = _w_include_face
    include_depth = _w_include_depth
    axes = ["x", "y", "z"] if include_depth else ["x", "y"]

    def _normalize(dataframe: pd.DataFrame) -> pd.DataFrame:
        all_frames = sorted(dataframe["frame"].unique())

        def pose_lm(idx: int) -> pd.DataFrame:
            mask = (dataframe["type"] == "pose") & (dataframe["landmark_index"] == idx)
            return dataframe[mask].set_index("frame")[axes].reindex(all_frames)

        nose     = pose_lm(0)
        shoulder = (pose_lm(11) + pose_lm(12)) / 2.0
        hip      = (pose_lm(23) + pose_lm(24)) / 2.0
        origin   = nose.combine_first(shoulder).combine_first(hip).fillna(0.0)
        origin.columns = [f"{ax}_origin" for ax in axes]

        dataframe = dataframe.merge(origin, left_on="frame", right_index=True, how="left")
        for ax in axes:
            dataframe[ax] = dataframe[ax] - dataframe[f"{ax}_origin"].fillna(0)
        return dataframe.drop(columns=[f"{ax}_origin" for ax in axes])

    try:
        df = pd.read_parquet(full_path)

        if include_face:
            df = df[~((df["type"] == "face") & ~df["landmark_index"].isin(face_lm_set))]
        else:
            df = df[df["type"] != "face"]

        df = _normalize(df)
        df = df.copy()
        df["uid"] = df["type"].astype(str) + "_" + df["landmark_index"].astype(str)

        try:
            wide = df.pivot(index="frame", columns="uid", values=axes)
        except ValueError:
            wide = df.pivot_table(
                index="frame", columns="uid", values=axes, aggfunc="first"
            )

        wide.columns = [f"{col[1]}_{col[0]}" for col in wide.columns]
        wide = wide.reindex(columns=all_columns)
        arr = wide.ffill().bfill().fillna(0).to_numpy(dtype=np.float32)

        coords = torch.tensor(arr, dtype=torch.float32)
        buf = io.BytesIO()
        torch.save(coords, buf)
        return path, buf.getvalue(), len(coords), None
    except Exception as e:
        return path, None, None, str(e)


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------
_WRITE_BATCH = 2_000


def build(
    data_dir: Path,
    lmdb_path: Path,
    map_size_gb: int = 20,
    allow_errors: bool = False,
    num_workers: int | None = None,
) -> None:
    if num_workers is None:
        num_workers = os.cpu_count() or 4

    df = pd.read_csv(data_dir / "train.csv")
    n = len(df)
    paths = df["path"].tolist()

    print(
        f"ASL LMDB: {n:,} samples → {lmdb_path}  "
        f"(workers={num_workers}, COORD_FEAT={COORD_FEAT}, CACHE_VERSION={CACHE_VERSION})"
    )

    lmdb_path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(lmdb_path), map_size=map_size_gb * 1024**3)

    written = skipped = errors = 0

    # Pre-scan: find pending paths and backfill any missing length keys
    pending_args: list[tuple[str, str]] = []
    backfill: list[str] = []
    with env.begin() as txn:
        for path in paths:
            if txn.get(lmdb_key(path)) is not None:
                if txn.get(lmdb_length_key(path)) is None:
                    backfill.append(path)
                skipped += 1
            else:
                pending_args.append((path, str(data_dir / path)))

    if backfill:
        print(f"Backfilling length keys for {len(backfill)} existing entries...")
        with env.begin(write=True) as txn:
            for path in backfill:
                raw    = txn.get(lmdb_key(path))
                coords = torch.load(io.BytesIO(bytes(raw)), weights_only=True)
                txn.put(lmdb_length_key(path), str(len(coords)).encode())

    if not pending_args:
        print("All samples already in LMDB. Nothing to do.")
        env.close()
        return

    print(f"Processing {len(pending_args):,} samples ({skipped:,} already done)...")

    write_buf: list[tuple[str, bytes, int]] = []

    def _flush(buf: list) -> None:
        with env.begin(write=True) as txn:
            for p, data, length in buf:
                txn.put(lmdb_key(p),        data)
                txn.put(lmdb_length_key(p), str(length).encode())
        buf.clear()

    with tqdm(total=n, desc="Building ASL LMDB", unit="sample") as pbar:
        pbar.update(skipped)
        pbar.set_postfix(written=written, skipped=skipped, errors=errors)

        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_worker_init,
            initargs=(ALL_COLUMNS, FACE_LANDMARK_SET, INCLUDE_FACE, INCLUDE_DEPTH),
        ) as executor:
            for path, data, length, error in executor.map(
                _process_sample, pending_args, chunksize=8
            ):
                if error:
                    errors += 1
                    tqdm.write(f"  ERROR {path}: {error}", file=sys.stderr)
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
        f"\nDone. LMDB → {lmdb_path}\n"
        f"  written={written}  skipped={skipped}  errors={errors}"
    )

    if errors > 0 and not allow_errors:
        print(
            f"\nFATAL: {errors} sample(s) failed. Re-run to retry, or pass "
            "--allow-errors to ignore.",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(
        description="Build LMDB cache for Google ASL Signs (Kaggle standalone)"
    )
    p.add_argument(
        "--data-dir",
        default="/kaggle/input/asl-signs",
        help="ASL Signs dataset root (default: /kaggle/input/asl-signs)",
    )
    p.add_argument(
        "--lmdb-path",
        default="/kaggle/working/asl.lmdb",
        help="Output LMDB directory (default: /kaggle/working/asl.lmdb)",
    )
    p.add_argument(
        "--map-size-gb",
        type=int,
        default=20,
        help="LMDB virtual address space in GB (default 20)",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel workers (default: os.cpu_count())",
    )
    p.add_argument(
        "--allow-errors",
        action="store_true",
        help="Exit 0 even if some samples fail",
    )
    args = p.parse_args()
    build(
        Path(args.data_dir),
        Path(args.lmdb_path),
        args.map_size_gb,
        args.allow_errors,
        args.num_workers,
    )


if __name__ == "__main__":
    main()
