"""
Build an LMDB cache from the ASL Fingerspelling Recognition dataset.

The Fingerspelling parquets are in wide format: one large file per file_id
(~1 GB each), with multiple sequences indexed by sequence_id.  Parallelism
is at the parquet-file level — each worker loads one file and processes all
its pending sequences, so no parquet is read more than once per worker.

Resume safety: existing keys are skipped, so an interrupted run restarts
cleanly.

Keys
----
  Data:   b"fs:{sequence_id}"
  Length: b"fs:len:{sequence_id}"

Run (from project root):
  PYTHONPATH=research/models uv run python -m cnn_transformer.data.build_fingerspelling_lmdb \\
      --data-dir  data/ASL_Fingerspelling_Recognition \\
      --lmdb-path data/cache/fingerspelling/fs.lmdb \\
      --out-csv   data/cache/fingerspelling/train.csv
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

from ..config import ALL_COLUMNS, COORD_FEAT

_WRITE_BATCH = 500  # smaller than ASL build — each entry is larger
_1TB = 1 << 40  # LMDB uses sparse files; 1 TiB costs nothing extra on disk

# ---------------------------------------------------------------------------
# Column mapping  (module-level so workers can import it)
# ALL_COLUMNS fmt:  "{type}_{idx}_{axis}"  e.g. "left_hand_0_x"
# Fingerspelling:   "{axis}_{type}_{idx}"  e.g. "x_left_hand_0"
# ---------------------------------------------------------------------------
_FS_COLS: list[str] = []
for _col in ALL_COLUMNS:
    _lm_type, _lm_idx, _axis = _col.rsplit("_", 2)
    _FS_COLS.append(f"{_axis}_{_lm_type}_{_lm_idx}")


def _lmdb_key(sequence_id: int) -> bytes:
    return f"fs:{sequence_id}".encode()


def _lmdb_length_key(sequence_id: int) -> bytes:
    return f"fs:len:{sequence_id}".encode()


# ---------------------------------------------------------------------------
# Worker function  (self-contained imports for spawn/forkserver safety)
# ---------------------------------------------------------------------------


def _process_parquet(args: tuple) -> list[tuple]:
    """Worker: load one fingerspelling parquet and process all pending sequences.

    Self-contained imports make this safe under spawn/forkserver (Python 3.14
    default on macOS/Linux).  PYTHONPATH is inherited from the parent process.

    Returns list of (sequence_id, lmdb_key, tensor_bytes, length, error).
    """
    import io
    import numpy as np
    import pandas as pd
    import torch
    from cnn_transformer.data.build_fingerspelling_lmdb import (
        _FS_COLS,
        COORD_FEAT,
    )

    parquet_path, pending = args
    # pending: list of sequence_id ints to process from this parquet

    axes = ["x", "y", "z"]

    def _normalize(data: np.ndarray, col_index: dict) -> np.ndarray:
        data = data.copy()
        for ax in axes:

            def _get(name):
                i = col_index.get(f"{ax}_{name}")
                return data[:, i] if i is not None else np.full(len(data), np.nan)

            nose = _get("pose_0")
            shoulder = (_get("pose_11") + _get("pose_12")) / 2
            hip = (_get("pose_23") + _get("pose_24")) / 2
            origin = np.where(
                ~np.isnan(nose),
                nose,
                np.where(
                    ~np.isnan(shoulder), shoulder, np.where(~np.isnan(hip), hip, 0.0)
                ),
            )
            ax_idxs = [i for c, i in col_index.items() if c.startswith(f"{ax}_")]
            data[:, ax_idxs] -= origin[:, None]
        return data

    try:
        wide = pd.read_parquet(parquet_path)
    except Exception as e:
        return [(sid, _FS_COLS, None, None, str(e)) for sid in pending]

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

            raw = seq_df.drop(columns=["frame"], errors="ignore").to_numpy(
                dtype=np.float32
            )
            raw = _normalize(raw, col_index)

            out = np.zeros((len(raw), COORD_FEAT), dtype=np.float32)
            for fi, fs_col in enumerate(_FS_COLS):
                ci = col_index.get(fs_col)
                if ci is not None:
                    vals = raw[:, ci]
                    out[:, fi] = np.where(np.isnan(vals), 0.0, vals)

            tensor = torch.from_numpy(out)
            buf = io.BytesIO()
            torch.save(tensor, buf)
            results.append((sid, buf.getvalue(), len(tensor), None))
        except Exception as e:
            results.append((sid, None, None, str(e)))

    return results


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------


def build(
    data_dir: Path,
    lmdb_path: Path,
    out_csv: Path,
    map_size: int = _1TB,
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
        f"(workers={num_workers})"
    )

    lmdb_path.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(lmdb_path), map_size=map_size)

    written = skipped = errors = 0

    # ── Pre-scan: determine which sequence_ids still need processing ──────────
    all_sids = train_csv["sequence_id"].tolist()
    pending_sids: set[int] = set()
    backfill_sids: list[int] = []

    with env.begin() as txn:
        for sid in all_sids:
            has_data = txn.get(_lmdb_key(sid)) is not None
            has_len = txn.get(_lmdb_length_key(sid)) is not None
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
                raw = txn.get(_lmdb_key(sid))
                coords = torch.load(io.BytesIO(bytes(raw)), weights_only=True)
                txn.put(_lmdb_length_key(sid), str(len(coords)).encode())

    # ── Build per-parquet task list ───────────────────────────────────────────
    # Group pending sequence_ids by parquet file so each worker loads one file.
    parquet_tasks: list[tuple[str, list[int]]] = []
    for parquet_rel, group in train_csv.groupby("path", sort=False):
        sids_in_file = [
            int(sid)
            for sid in group["sequence_id"].tolist()
            if int(sid) in pending_sids
        ]
        if sids_in_file:
            parquet_tasks.append((str(data_dir / parquet_rel), sids_in_file))

    n_pending = sum(len(sids) for _, sids in parquet_tasks)
    if n_pending == 0:
        print("All sequences already in LMDB. Nothing to do.")
        env.close()
        _write_csv(train_csv, out_csv)
        return

    # ── Parallel parse → single-threaded write ────────────────────────────────
    write_buf: list[tuple[int, bytes, int]] = []

    def _flush(buf: list) -> None:
        with env.begin(write=True) as txn:
            for sid, data, length in buf:
                txn.put(_lmdb_key(sid), data)
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
            "--allow-errors to ignore and proceed.",
            file=sys.stderr,
        )
        sys.exit(1)


def _write_csv(train_csv: pd.DataFrame, out_csv: Path) -> None:
    """Write the metadata CSV used by FingerspellingDataset."""
    out = train_csv[["sequence_id", "participant_id", "phrase"]].copy()
    out.to_csv(out_csv, index=False)


def main():
    p = argparse.ArgumentParser(
        description="Build LMDB cache for ASL Fingerspelling dataset"
    )
    p.add_argument(
        "--data-dir", required=True, help="data/ASL_Fingerspelling_Recognition"
    )
    p.add_argument(
        "--lmdb-path",
        required=True,
        help="Output LMDB directory (data/cache/fingerspelling/fs.lmdb)",
    )
    p.add_argument(
        "--out-csv",
        required=True,
        help="Output metadata CSV (data/cache/fingerspelling/train.csv)",
    )
    p.add_argument(
        "--map-size-gb",
        type=int,
        default=None,
        help="LMDB virtual address space in GB. Default: 1 TiB (sparse file, "
        "costs nothing extra on disk).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel workers — one per parquet file (default: os.cpu_count())",
    )
    p.add_argument(
        "--allow-errors", action="store_true", help="Exit 0 even if some sequences fail"
    )
    args = p.parse_args()
    map_size = (args.map_size_gb * 1024**3) if args.map_size_gb else _1TB
    build(
        Path(args.data_dir),
        Path(args.lmdb_path),
        Path(args.out_csv),
        map_size,
        args.allow_errors,
        args.num_workers,
    )


if __name__ == "__main__":
    main()
