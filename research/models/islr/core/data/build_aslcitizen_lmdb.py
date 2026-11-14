"""Build an ASL-Signs-compatible LMDB from ASL Citizen landmark shards (research use only).

RESEARCH USE ONLY: ASL Citizen is under the Microsoft Research License Terms (non-commercial
research, no redistribution of the data or modifications). Build it inside a private Kaggle
notebook or locally; never publish the LMDB, and never export a model trained on it for the app.

Input: the outputs of the `extract-aslcitizen-s<k>` notebooks (`scripts/extract_aslcitizen.py`):
each shard folder holds `shard_<k>.csv` (path, participant, split, gloss, video, n_frames) and
`train_landmark_files/*.parquet` in the competition format. --input-root is searched
recursively for `shard_*.csv`, so mounting all shards under /kaggle/input is enough.

Each clip goes through the same worker as `build_wlasl_lmdb` — the data contract (`contract.prepare`), then the
per-clip aspect correction (ASL Citizen is landscape webcam video; x and z scaled so shoulder
drop / width = 0.40). Glosses map onto ASL Signs labels with `aslcitizen_gloss_map.json`; its
`kind` (identical / variant / synonym) is kept so results can be scored per kind.

Writes to --out-dir: `aslcitizen.lmdb.mdb` (same keys as `is.lmdb.mdb`), `train.csv` (path,
participant_id = 800000 + ASL Citizen participant number, sequence_id, sign, aslc_gloss, kind,
synonym (= kind != identical), fps, source, split (official train / val / test), aspect_k,
ratio_raw, both_hands, n_frames — the columns `islr.evaluation.cross_dataset` reads) and a
copy of the ASL Signs sign map.

Usage (inside `scripts/kaggle_run.py push ... --kernel <user>/extract-aslcitizen-s0 ...`):
  PYTHONPATH=research/models python -m islr.core.data.build_aslcitizen_lmdb \
    --input-root {INPUT} --sign-map {IS_LMDB_DIR}/sign_to_prediction_index_map.json \
    --out-dir /tmp/aslcitizen-lmdb
"""

import argparse
import json
import os
import re
from pathlib import Path

import lmdb
import pandas as pd

from .build_wlasl_lmdb import TARGET_RATIO, finish_build, load_shards, write_clips

_GLOSS_MAP = Path(__file__).with_name("aslcitizen_gloss_map.json")
_SIGNER_OFFSET = 800_000  # apart from ASL Signs participant ids and WLASL's 900000 + signer


def participant_number(p: str) -> int:
    """'P15' / '15' → 15; ids without digits get a stable hash below 100000."""
    m = re.search(r"\d+", str(p))
    return int(m.group()) if m else sum(map(ord, str(p))) % 100_000


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input-root", required=True, type=Path,
                   help="searched recursively for shard_*.csv (each next to train_landmark_files/)")
    p.add_argument("--sign-map", required=True, type=Path,
                   help="ASL Signs sign_to_prediction_index_map.json")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--lmdb-name", default="aslcitizen.lmdb.mdb")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--limit", type=int, default=0, help="first N clips (smoke test)")
    p.add_argument("--expect-shards", type=int, required=True, help="shards 0..N-1 of one extraction version must all be there")
    p.add_argument("--allow-errors", type=int, default=0, help="clips that may fail before the build exits")
    args = p.parse_args()

    sign2idx = json.loads(args.sign_map.read_text())
    gmap = {k: v for k, v in json.loads(_GLOSS_MAP.read_text()).items() if not k.startswith("_")}

    df, manifest = load_shards(args.input_root, "aslcitizen", args.expect_shards)

    df["aslc_gloss"] = df["gloss"]
    df["sign"] = [gmap[g]["sign"] if g in gmap else None for g in df["gloss"]]
    df["kind"] = [gmap[g]["kind"] if g in gmap else None for g in df["gloss"]]
    df = df[df["sign"].isin(sign2idx)].copy()
    df["synonym"] = df["kind"] != "identical"
    df["participant_id"] = [_SIGNER_OFFSET + participant_number(x) for x in df["participant"]]
    df["sequence_id"] = df["video"].map(lambda v: os.path.splitext(str(v))[0])
    df["fps"], df["source"] = float("nan"), "aslcitizen"
    if args.limit:
        df = df.head(args.limit)
    print(f"{len(df)} clips, {df['sign'].nunique()} signs, {df['participant_id'].nunique()} signers; "
          f"kinds {df['kind'].value_counts().to_dict()}; splits {df['split'].value_counts().to_dict()}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lmdb_path = args.out_dir / args.lmdb_name
    env = lmdb.open(str(lmdb_path), subdir=False, map_size=1 << 36)
    aspect = (df["width"] / df["height"]) if {"width", "height"} <= set(df) else pd.Series(float("nan"), df.index)
    stats, errors = write_clips(env, list(zip(df["path"], df["full_path"], [True] * len(df), aspect)), args.workers)
    env.close()

    df = df[df["path"].isin(stats)].copy()
    df["aspect_k"], df["ratio_raw"], df["both_hands"], df["n_frames"] = zip(
        *((stats[p_]["k"], stats[p_]["raw"], stats[p_]["both"], stats[p_]["n"]) for p_ in df["path"]))
    cols = ["path", "participant_id", "sequence_id", "sign", "aslc_gloss", "kind", "synonym", "fps",
            "source", "split", "aspect_k", "ratio_raw", "both_hands", "n_frames"]
    df[cols].to_csv(args.out_dir / "train.csv", index=False)
    (args.out_dir / "sign_to_prediction_index_map.json").write_text(json.dumps(sign2idx))
    finish_build(lmdb_path, df["path"].tolist(), errors, args.allow_errors, manifest, stats)

    print(f"wrote {len(df)} clips → {lmdb_path} ({lmdb_path.stat().st_size / 2**20:.0f} MiB); "
          f"{len(errors)} errors" + (f", e.g. {errors[0]}" if errors else ""))
    print(f"raw shoulder ratio median {df['ratio_raw'].median():.2f} → corrected {TARGET_RATIO}; "
          f"aspect k median {df['aspect_k'].median():.2f} (p10 {df['aspect_k'].quantile(.1):.2f}, "
          f"p90 {df['aspect_k'].quantile(.9):.2f}); k = 1 fallback {int((df['aspect_k'] == 1).sum())}")
    per_sign = df.groupby("sign")["both_hands"].apply(lambda s: (s > 0).mean())
    print(f"clips with both hands in some frame: {(df['both_hands'] > 0).mean():.0%}; per sign "
          f"median {per_sign.median():.0%}; frames median {df['n_frames'].median():.0f}")


if __name__ == "__main__":
    main()
