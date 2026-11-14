"""Build an ASL-Signs-compatible LMDB from PopSign ASL v1.0 landmark shards (CC BY 4.0).

PopSign ASL v1.0 (Starner et al., NeurIPS 2023 Datasets & Benchmarks; Georgia Tech) is licensed
CC BY 4.0 at the source — credit it wherever a model trained on it ships.

Input: the outputs of the `extract-popsign-s<k>` notebooks (`scripts/extract_popsign.py`): each
shard folder holds `shard_<k>.csv` (path, participant, split, gloss, video, n_frames, width,
height, lr_dx) and `train_landmark_files/*.parquet`; --input-root is searched recursively for
`shard_*.csv`. PopSign folder names are matched to the ASL Signs labels ignoring case.

Same worker as `build_wlasl_lmdb` (`frame_stacked_data`) but **no aspect correction**: PopSign is
portrait 1944×2592 phone video from the same game as ASL Signs, which is stored uncorrected too
(the raw shoulder ratio is still reported). Clips filmed mirrored (`lr_dx` ≤ 0) are counted.

PopSign comes from the same game as the Google ASL Signs data, so some signers likely overlap:
ASL Signs held-out scores are not clean for a model trained on it (use ASL Citizen / WLASL).

Writes to --out-dir: `popsign.lmdb.mdb` (same keys as `is.lmdb.mdb`), `train.csv` (path,
participant_id = 700000 + the number after the last '.' of the PopSign user id, sequence_id,
sign, popsign_gloss, popsign_user, synonym (False), fps, source, split, aspect_k (1), ratio_raw,
both_hands, n_frames, width, height) and a copy of the ASL Signs sign map.

Usage (inside `scripts/kaggle_run.py push ... --kernel <user>/extract-popsign-s0 ...`):
  PYTHONPATH=research/models python -m islr.core.data.build_popsign_lmdb \
    --input-root {INPUT} --sign-map {IS_LMDB_DIR}/sign_to_prediction_index_map.json \
    --out-dir /tmp/popsign
"""

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lmdb
import pandas as pd
from tqdm import tqdm

from ._cache_keys import lmdb_key, lmdb_length_key
from .build_wlasl_lmdb import _process

_SIGNER_OFFSET = 700_000  # apart from ASL Signs ids, ASL Citizen (800000 + n) and WLASL (900000 + n)


def participant_number(user: str) -> int:
    """'4a.6001' / 'gtsignstudy4a.8031' → 6001 / 8031 (digits after the last '.')."""
    m = re.search(r"(\d+)$", str(user).rsplit(".", 1)[-1])
    return int(m.group(1)) if m else sum(map(ord, str(user))) % 100_000


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input-root", required=True, type=Path,
                   help="searched recursively for shard_*.csv (each next to train_landmark_files/)")
    p.add_argument("--sign-map", required=True, type=Path,
                   help="ASL Signs sign_to_prediction_index_map.json")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--lmdb-name", default="popsign.lmdb.mdb")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--limit", type=int, default=0, help="first N clips (smoke test)")
    args = p.parse_args()

    sign2idx = json.loads(args.sign_map.read_text())
    by_lower = {s.lower(): s for s in sign2idx}

    shards = sorted(args.input_root.rglob("shard_*.csv"))
    shards = [c for c in shards if (c.parent / "train_landmark_files").is_dir()
              and "popsign" in str(c).lower()] or shards
    if not shards:
        raise SystemExit(f"no shard_*.csv under {args.input_root}")
    parts = []
    for csv in shards:
        d = pd.read_csv(csv)
        d["full_path"] = [str(csv.parent / p_) for p_ in d["path"]]
        parts.append(d)
        print(f"{csv}: {len(d)} clips")
    df = pd.concat(parts, ignore_index=True).drop_duplicates("path")

    df["popsign_gloss"] = df["gloss"]
    df["sign"] = df["gloss"].map(lambda g: by_lower.get(str(g).lower()))
    unmatched = sorted(set(df.loc[df["sign"].isna(), "gloss"]))
    if unmatched:
        print(f"{len(unmatched)} PopSign signs without an ASL Signs label (dropped): {unmatched}")
    df = df[df["sign"].notna()].copy()
    df["synonym"] = False
    df["popsign_user"] = df["participant"]
    df["participant_id"] = [_SIGNER_OFFSET + participant_number(u) for u in df["participant"]]
    clash = df.groupby("participant_id")["popsign_user"].nunique()
    if (clash > 1).any():
        raise SystemExit(f"PopSign user ids collide after numbering: {clash[clash > 1].index.tolist()}")
    df["sequence_id"] = df["video"].map(lambda v: os.path.splitext(str(v))[0])
    df["fps"], df["source"] = float("nan"), "popsign"
    if args.limit:
        df = df.head(args.limit)
    mirrored = int((df["lr_dx"] <= 0).sum()) if "lr_dx" in df else 0
    print(f"{len(df)} clips, {df['sign'].nunique()} signs, {df['participant_id'].nunique()} signers; "
          f"splits {df['split'].value_counts().to_dict()}; frame sizes "
          f"{df.groupby(['width', 'height']).size().to_dict() if 'width' in df else '?'}; "
          f"mirrored clips (pose 11 x ≤ pose 12 x): {mirrored}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lmdb_path = args.out_dir / args.lmdb_name
    env = lmdb.open(str(lmdb_path), subdir=False, map_size=1 << 37)
    stats, errors = {}, []
    jobs = [(p_, f, False) for p_, f in zip(df["path"], df["full_path"])]
    with ProcessPoolExecutor(args.workers) as pool, env.begin(write=True) as txn:
        for path, data, n, k, raw, both, err in tqdm(pool.map(_process, jobs, chunksize=8),
                                                    total=len(jobs)):
            if err:
                errors.append((path, err))
                continue
            txn.put(lmdb_key(path), data)
            txn.put(lmdb_length_key(path), str(n).encode())
            stats[path] = (k, raw, both, n)
    env.close()

    df = df[df["path"].isin(stats)].copy()
    df["aspect_k"], df["ratio_raw"], df["both_hands"], df["n_frames"] = zip(
        *(stats[p_] for p_ in df["path"]))
    cols = ["path", "participant_id", "sequence_id", "sign", "popsign_gloss", "popsign_user", "synonym",
            "fps", "source", "split", "aspect_k", "ratio_raw", "both_hands", "n_frames"]
    cols += [c for c in ("width", "height") if c in df]  # frame size (some clips are 9:16, most 3:4)
    df[cols].to_csv(args.out_dir / "train.csv", index=False)
    (args.out_dir / "sign_to_prediction_index_map.json").write_text(json.dumps(sign2idx))

    print(f"wrote {len(df)} clips → {lmdb_path} ({lmdb_path.stat().st_size / 2**20:.0f} MiB); "
          f"{len(errors)} errors" + (f", e.g. {errors[0]}" if errors else ""))
    print(f"raw shoulder ratio median {df['ratio_raw'].median():.2f} (ASL Signs 0.40; not corrected)")
    per_sign = df.groupby("sign")["both_hands"].apply(lambda s: (s > 0).mean())
    print(f"clips with both hands in some frame: {(df['both_hands'] > 0).mean():.0%}; per sign "
          f"median {per_sign.median():.0%} (ASL Signs 2 %); frames median {df['n_frames'].median():.0f}")


if __name__ == "__main__":
    main()
