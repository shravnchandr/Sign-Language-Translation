"""
Build a single LMDB archive from ASL parquet files.

On network-attached storage (RunPod) individual .pt files incur a per-open()
syscall for every sample every epoch.  A single LMDB archive eliminates that
overhead: one file open per process, then random-access reads.

Run once before training (from the project root):

    PYTHONPATH=research/models uv run python -m islr.core.data.build_lmdb \\
        --data-dir  data/Isolated_ASL_Recognition \\
        --lmdb-path data/asl-is-lmdb-v3/is.lmdb.mdb

A path ending in .mdb is written as one flat file (the released layout the loaders and
validate.py read); the competition train.csv and sign map are copied next to it, with
build_manifest.json. Then: python -m islr.core.data.validate --data-dir <that folder>
--lmdb-path <that file> --raw-root <data-dir>. On Kaggle (the canonical build; the competition
data never leaves it):

    python3 scripts/kaggle_run.py push build-aslsigns-v3 --bare --competition asl-signs --no-copy -- \\
        islr.core.data.build_lmdb --data-dir {COMP_asl_signs} --lmdb-path {OUT}/is.lmdb.mdb :: \\
        islr.core.data.validate --data-dir {OUT} --lmdb-path {OUT}/is.lmdb.mdb --raw-root {COMP_asl_signs}

Resume safety: existing keys are skipped, so an interrupted run can be
restarted without re-processing completed samples.

Keys are versioned (_CACHE_VERSION prefix) so rebuilding after a config
change (e.g. new face landmark set) produces new keys and never silently
reuses stale tensors.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lmdb
import pandas as pd
from tqdm import tqdm

from ..revision import code_revision
from ._cache_keys import lmdb_key as _lmdb_key, lmdb_length_key as _lmdb_length_key
from .contract import load_clip
from .vocab import load_base

# Commit to LMDB every this many successful writes to cap dirty-page memory.
_WRITE_BATCH = 2_000


def _process_sample(args: tuple) -> tuple:
    """Worker: one competition parquet → stored clip bytes (data contract: pose-less frames dropped,
    ghost hands resolved, explicit presence). Self-contained imports (spawn / forkserver safe)."""
    import numpy as np
    import pandas as pd

    from islr.core import config as C
    from islr.core.data.contract import dump_clip, prepare

    path, full_path = args
    try:
        clip = prepare(pd.read_parquet(full_path))  # ASL Signs: portrait phone video (aspect 3:4)
        c = clip["coords"]
        if c.ndim != 2 or c.shape[1] != C.COORD_FEAT or len(c) == 0:
            raise ValueError(f"coords shape {c.shape}, expected (T > 0, {C.COORD_FEAT})")
        if not np.isfinite(c).all():
            raise ValueError("non-finite values")
        return path, dump_clip(clip), {"n": len(c), **clip["stats"]}, None
    except Exception as e:  # noqa: BLE001 — reported per sample
        return path, None, None, str(e)


_1TB = 1 << 40  # LMDB uses sparse files; 1 TiB costs nothing extra on disk


def build_lmdb(
    data_dir: str,
    lmdb_path: str,
    map_size: int = _1TB,
    allow_errors: bool = False,
    num_workers: int | None = None,
) -> None:
    data_dir = Path(data_dir)
    lmdb_path = Path(lmdb_path)
    load_base(data_dir / "sign_to_prediction_index_map.json")  # the release's labels: the canonical map
    flat = lmdb_path.suffix == ".mdb"  # one file (is.lmdb.mdb, the released layout) or a directory
    (lmdb_path.parent if flat else lmdb_path).mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = os.cpu_count() or 4

    df = pd.read_csv(data_dir / "train.csv")
    n = len(df)
    paths = df["path"].tolist()
    print(f"Building LMDB: {n} samples → {lmdb_path}  (workers={num_workers})")

    env = lmdb.open(str(lmdb_path), map_size=map_size, subdir=not flat)

    written = skipped = errors = 0
    stats: dict = {}

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
                txn.put(_lmdb_length_key(path), str(len(load_clip(raw)[0])).encode())

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
            for path, data, info, error in executor.map(
                _process_sample, pending_args, chunksize=8
            ):
                if error:
                    errors += 1
                    tqdm.write(f"  ERROR: skipping {path}: {error}", file=sys.stderr)
                else:
                    write_buf.append((path, data, info["n"]))
                    stats[path] = info
                    written += 1
                    if len(write_buf) >= _WRITE_BATCH:
                        _flush(write_buf)

                pbar.update(1)
                pbar.set_postfix(written=written, skipped=skipped, errors=errors)

        if write_buf:
            _flush(write_buf)

    env.close()
    import json

    from ._cache_keys import CACHE_VERSION
    from .build_wlasl_lmdb import contract_summary
    from .contract import CONTRACT_VERSION

    from .contract import content_digest

    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, subdir=not flat)
    with env.begin() as txn:  # every clip actually stored (written now or by an earlier run)
        stored = [p_ for p_ in paths if txn.get(_lmdb_key(p_)) is not None]
    digest = content_digest(env, stored)
    env.close()
    out = lmdb_path.parent  # the release folder: LMDB + train.csv + sign map + manifest
    if out.resolve() != data_dir.resolve():
        import shutil

        for f in ("train.csv", "sign_to_prediction_index_map.json"):
            shutil.copyfile(data_dir / f, out / f)
    manifest = {"version": "asl-signs-competition", "dataset": "aslsigns", "aspect_correction": False,
                "allow_errors": errors if allow_errors else 0, "content_digest": digest,
                "csv_digest": __import__("hashlib").sha256((data_dir / "train.csv").read_bytes()).hexdigest()[:16],
                "sign_map_digest": __import__("hashlib").sha256(
                    (out / "sign_to_prediction_index_map.json").read_bytes()).hexdigest()[:16],
                "shards": None, "clips_listed": n, "clips_written": written + skipped,
                "errors": errors, "lmdb": lmdb_path.name, "contract": CONTRACT_VERSION, "cache_version": CACHE_VERSION,
                "contract_counts": contract_summary(stats) if stats else None, "resumed_clips": skipped,
                # the competition's train.csv is the selection; every listed clip is attempted
                "coverage": {"selected": n, "listed": n, "missing": 0, "filtered": 0, "build_errors": errors,
                             "excluded_duplicates": 0, "written": written + skipped},
                "prep_aspect": 0.75, "frame_step": 1,  # competition parquets: every frame kept  # contract.prepare default: portrait 3:4 phone video (frame size not recorded)
                "code": code_revision()}
    (lmdb_path.parent / "build_manifest.json").write_text(json.dumps(manifest, indent=1))
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
    parser.add_argument(
        "--data-dir", required=True, help="Directory containing train.csv"
    )
    parser.add_argument("--lmdb-path", required=True, help="output LMDB: a directory, or one flat file when it ends in .mdb (released layout)")
    parser.add_argument(
        "--map-size-gb",
        type=int,
        default=None,
        help="LMDB virtual address space in GB. Default: 1 TiB (sparse file, "
        "costs nothing extra on disk). Override only if your filesystem does "
        "not support sparse files.",
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
    map_size = (args.map_size_gb * 1024**3) if args.map_size_gb else _1TB
    build_lmdb(
        args.data_dir,
        args.lmdb_path,
        map_size,
        args.allow_errors,
        args.num_workers,
    )


if __name__ == "__main__":
    main()
