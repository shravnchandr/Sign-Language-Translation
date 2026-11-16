"""Build an ASL-Signs-compatible LMDB from PopSign ASL v1.0 landmark shards (CC BY 4.0).

PopSign ASL v1.0 (Starner et al., NeurIPS 2023 Datasets & Benchmarks; Georgia Tech) is licensed
CC BY 4.0 at the source — credit it wherever a model trained on it ships.

Input: the outputs of the `extract-popsign-s<k>` notebooks (`scripts/extract_popsign.py`): each
shard folder holds `shard_<k>.csv` (path, participant, split, gloss, video, n_frames, width,
height, lr_dx) and `train_landmark_files/*.parquet`; --input-root is searched recursively for
`shard_*.csv`. PopSign folder names are matched to the ASL Signs labels ignoring case.

Same worker as `build_wlasl_lmdb` (data contract, `contract.prepare` — ghost hands, which PopSign
has many of, are resolved there) but **no aspect correction**: PopSign is
portrait 1944×2592 phone video from the same game as ASL Signs, which is stored uncorrected too
(the raw shoulder ratio is still reported). Clips filmed mirrored (`lr_dx` ≤ 0) are counted.

PopSign comes from the same game as the Google ASL Signs data, so some signers likely overlap:
ASL Signs held-out scores are not clean for a model trained on it (use ASL Citizen / WLASL).
Clips whose pose track matches an ASL Signs clip (`audit-popsign`: same sign and frame count,
<= 0.095 frame-wise — the same recording, or the same person in the same posture) are left out
(`popsign_asl_signs_matches.csv`, --exclude-csv; listed in excluded_listed.csv, counted as filtered).
Frame rate: ~19 % of the videos are 120 fps; --fps-csv (every video probed) gives each clip's fps
and frame_step = round(fps / 30) raw frames per kept frame (contract step 0), both in train.csv.

Writes to --out-dir: `popsign.lmdb.mdb` (same keys as `is.lmdb.mdb`), `train.csv` (path,
participant_id = 700000 + the number after the last '.' of the PopSign user id, sequence_id,
sign, popsign_gloss, popsign_user, synonym (False), fps, source, split, aspect_k (1), ratio_raw,
both_hands, n_frames (after the frame step), prep_aspect, frame_step, width, height) and a copy of the
ASL Signs sign map.

Usage (inside `scripts/kaggle_run.py push ... --kernel <user>/extract-popsign-s0 ...`):
  PYTHONPATH=research/models python -m islr.core.data.build_popsign_lmdb \
    --input-root {INPUT} --sign-map {IS_LMDB_DIR}/sign_to_prediction_index_map.json \
    --out-dir /tmp/popsign --expect-shards 12 --fps-csv {INPUT}/fps-popsign/video_probe.csv
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import lmdb
import pandas as pd

from .contract import frame_step_for
from .vocab import load_base
from .build_wlasl_lmdb import exclude_duplicates, finish_build, load_shards, write_clips

_MATCHES = Path(__file__).with_name("popsign_asl_signs_matches.csv")  # audit-popsign: same recording as an ASL Signs clip
_SIGNER_OFFSET = 700_000  # apart from ASL Signs ids, ASL Citizen (800000 + n) and WLASL (900000 + n)


def participant_number(user) -> int:
    """'4a.6001' / 'gtsignstudy4a.8031' → 6001 / 8031: the whole id must be <letters/digits>.<digits>.
    Anything else raises ('-1', '4a.-1', a missing id): a number pulled out of a malformed id would
    look like a real, separate signer."""
    m = re.fullmatch(r"[A-Za-z0-9]+\.(\d+)", user.strip()) if isinstance(user, str) else None
    if m is None:
        raise ValueError(f"PopSign user id {user!r} is not <letters/digits>.<digits>")
    return int(m.group(1))


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
    p.add_argument("--expect-shards", nargs="+", required=True,
                   help="N: shards 0..N-1 of one extraction version must all be there; or VERSION=N ... for several "
                        "versions built together (a nested extraction added to an accepted one)")
    p.add_argument("--allow-errors", type=int, default=0, help="clips that may fail before the build exits")
    p.add_argument("--inventory", type=Path, help="selection inventory (scripts/extraction_inventory.py) for shards "
                   "without their own selected_<k>.csv")
    p.add_argument("--allow-missing", type=int, default=0, help="selected videos that may be absent from the shard lists")
    p.add_argument("--fps-csv", type=Path, nargs="+", required=True,
                   help="video_probe.csv of `extraction_inventory.py push popsign --probe-all` (path, fps for every "
                        "selected video): ~19 %% of PopSign is 120 fps, kept every round(fps / 30)-th frame")
    p.add_argument("--exclude-csv", type=Path, default=_MATCHES,
                   help="clips to leave out (column path): default the PopSign clips matching an ASL Signs recording")
    args = p.parse_args()

    sign2idx = load_base(args.sign_map)
    by_lower = {s.lower(): s for s in sign2idx}

    if len(args.expect_shards) == 1 and "=" not in args.expect_shards[0]:
        expect = int(args.expect_shards[0])
    else:
        expect = {v: int(n) for v, _, n in (e.partition("=") for e in args.expect_shards)}
    df, manifest = load_shards(args.input_root, "popsign", expect, args.inventory, args.allow_missing)

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
    df["source"] = "popsign"
    probe = pd.concat([pd.read_csv(f) for f in args.fps_csv]).drop_duplicates("path").set_index("path")["fps"]
    df["fps"] = df["path"].map(probe)  # the video's own frame rate (probed)
    if df["fps"].isna().any() or (df["fps"] <= 0).any():
        bad = df.loc[df["fps"].isna() | (df["fps"] <= 0), "path"].tolist()
        raise SystemExit(f"{len(bad)} clips without a probed frame rate in {[str(f) for f in args.fps_csv]}, e.g. {bad[:3]}")
    df["frame_step"] = [frame_step_for(f) for f in df["fps"]]
    print(f"frame steps (raw frames per kept frame): {df['frame_step'].value_counts().sort_index().to_dict()}")
    excl = pd.read_csv(args.exclude_csv)
    hit = df["path"].isin(set(excl["path"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    excl[excl["path"].isin(set(df["path"]))].to_csv(args.out_dir / "excluded_listed.csv", index=False)
    print(f"excluded before preparation ({args.exclude_csv.name}): {int(hit.sum())} of {len(excl)} listed clips")
    manifest["excluded_listed"] = {"file": args.exclude_csv.name, "sha256": hashlib.sha256(args.exclude_csv.read_bytes()).hexdigest()[:16],
                                   "listed": len(excl), "excluded": int(hit.sum())}
    manifest["fps_probe"] = {"file": str(args.fps_csv[0]) if len(args.fps_csv) == 1 else [str(f) for f in args.fps_csv], "frame_steps": {int(k): int(v) for k, v in df["frame_step"].value_counts().items()}}
    df = df[~hit].copy()
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
    aspect = (df["width"] / df["height"]) if {"width", "height"} <= set(df) else pd.Series(float("nan"), df.index)
    if "frame_step" not in df:
        df["frame_step"] = 1
    jobs = list(zip(df["path"], df["full_path"], [False] * len(df), aspect, df["frame_step"]))
    stats, errors = write_clips(env, jobs, args.workers)
    env.close()

    df = df[df["path"].isin(stats)].copy()
    df = exclude_duplicates(df, stats, args.out_dir)
    df["aspect_k"], df["ratio_raw"], df["both_hands"], df["n_frames"], df["prep_aspect"] = zip(
        *((stats[p_]["k"], stats[p_]["raw"], stats[p_]["both"], stats[p_]["n"], stats[p_]["prep_aspect"]) for p_ in df["path"]))
    cols = ["path", "participant_id", "sequence_id", "sign", "popsign_gloss", "popsign_user", "synonym",
            "fps", "source", "split", "aspect_k", "ratio_raw", "both_hands", "n_frames", "prep_aspect", "frame_step"]
    cols += [c for c in ("width", "height") if c in df]  # frame size (some clips are 9:16, most 3:4)
    df[cols].to_csv(args.out_dir / "train.csv", index=False)
    (args.out_dir / "sign_to_prediction_index_map.json").write_text(json.dumps(sign2idx))
    finish_build(lmdb_path, df["path"].tolist(), errors, args.allow_errors, manifest, stats, "popsign", False, attempted=len(jobs))

    print(f"wrote {len(df)} clips → {lmdb_path} ({lmdb_path.stat().st_size / 2**20:.0f} MiB); "
          f"{len(errors)} errors" + (f", e.g. {errors[0]}" if errors else ""))
    print(f"raw shoulder ratio median {df['ratio_raw'].median():.2f} (ASL Signs 0.40; not corrected)")
    per_sign = df.groupby("sign")["both_hands"].apply(lambda s: (s > 0).mean())
    print(f"clips with both hands in some frame: {(df['both_hands'] > 0).mean():.0%}; per sign "
          f"median {per_sign.median():.0%} (ASL Signs 2 %); frames median {df['n_frames'].median():.0f}")


if __name__ == "__main__":
    main()
