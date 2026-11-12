"""Build an ASL-Signs-compatible LMDB from WLASL landmark parquets (shared signs only).

Input: the output of the `shravnchandr/preprocess-wlasl-mediapipe` Kaggle notebook
(competition-format parquets from legacy MediaPipe Holistic, `train.csv` with
path / sign / video_id / n_frames) and optionally `WLASL_v0.3.json` for signer, fps,
source and split. WLASL glosses are mapped onto ASL Signs labels with
`wlasl_gloss_map.json` (233 glosses → 206 signs; 32 are synonyms such as
father → dad — `--strict` keeps identical names only).

Each clip goes through the same `frame_stacked_data` as the ASL Signs LMDB, then a
per-clip **aspect correction**: WLASL is mostly landscape video, ASL Signs portrait
≈ 3:4, and normalised x is a fraction of the frame width, y of its height. x and z
(z is in x's units) are scaled so that (shoulder drop below the nose) / (shoulder
width) equals ASL Signs' median 0.40. This also removes natural body variation in
that ratio (ASL Signs p10–p90 0.34–0.45) — the video dimensions are not recorded.

--all-glosses keeps every WLASL gloss (pre-training data): unshared glosses become extra
labels numbered after the 250 ASL Signs signs in the written sign map (the first 250
entries are unchanged), and `wlasl_half` records each signer's half of `signer_split`,
computed on the shared-sign clips only, so it equals the split of the 206-sign LMDB
(signers without shared-sign clips → train).

Writes to --out-dir: `wlasl.lmdb.mdb` (or --lmdb-name; same key scheme as `is.lmdb.mdb`, so
`ASLDataset` reads it), `train.csv` (path, participant_id = 900000 + WLASL signer,
sequence_id, sign, wlasl_gloss, synonym, fps, source, split, aspect_k, ratio_raw,
both_hands, n_frames) and a copy of the ASL Signs sign map.

Usage:
  PYTHONPATH=research/models python -m cnn_transformer.data.build_wlasl_lmdb \
    --wlasl-dir /kaggle/input/preprocess-wlasl-mediapipe/wlasl_landmarks \
    --wlasl-json /kaggle/input/wlasl-processed/WLASL_v0.3.json \
    --sign-map /kaggle/input/asl-signs/sign_to_prediction_index_map.json \
    --out-dir /kaggle/working/wlasl-lmdb
"""

import argparse
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ._cache_keys import lmdb_key, lmdb_length_key

_GLOSS_MAP = Path(__file__).with_name("wlasl_gloss_map.json")
TARGET_RATIO = 0.40  # ASL Signs median shoulder drop / shoulder width
_K_RANGE = (0.5, 4.0)
_SIGNER_OFFSET = 900_000  # keeps WLASL signer ids apart from ASL Signs participant ids


def aspect_factor(coords: np.ndarray) -> tuple[float, float]:
    """(k, raw ratio): x and z scale that brings the shoulder ratio to TARGET_RATIO.

    coords: (T, COORD_FEAT) nose-relative. Solves sy / sqrt((k·dx)² + dy²) = TARGET
    on per-clip medians; k = 1 when the shoulders are missing or k is implausible.
    """
    from ..config import POSE_START

    l, r = POSE_START + 11 * 3, POSE_START + 12 * 3
    valid = (np.abs(coords[:, l : l + 2]).sum(1) > 0) & (np.abs(coords[:, r : r + 2]).sum(1) > 0)
    if not valid.any():
        return 1.0, float("nan")
    c = coords[valid]
    sy = float(np.median((c[:, l + 1] + c[:, r + 1]) / 2))
    dx = float(np.median(np.abs(c[:, l] - c[:, r])))
    dy = float(np.median(c[:, l + 1] - c[:, r + 1]))
    raw = sy / max(float(np.hypot(dx, dy)), 1e-6)
    want = (sy / TARGET_RATIO) ** 2 - dy**2
    if dx <= 1e-6 or want <= 0:
        return 1.0, raw
    k = float(np.sqrt(want) / dx)
    return (k if _K_RANGE[0] <= k <= _K_RANGE[1] else 1.0), raw


def signer_split(df: pd.DataFrame) -> dict:
    """Fixed WLASL signer split {"train": [...], "test": [...]} with balanced clip counts.

    Signers sorted by (clips desc, id) are dealt snake-wise (train, test, test, train, ...),
    so the two halves get about the same number of clips and of signers. "train" signers
    may be mixed into training (train_phrases --extra-data-dir); "test" signers stay a
    cross-dataset test (cross_dataset_eval --signers test).
    """
    counts = df.groupby("participant_id").size()
    order = sorted(counts.index, key=lambda s: (-counts[s], s))
    split = {"train": [], "test": []}
    for i, s in enumerate(order):
        split["train" if i % 4 in (0, 3) else "test"].append(int(s))
    return split


def _process(args: tuple) -> tuple:
    """Worker: parquet → corrected coords bytes + stats (self-contained imports)."""
    import numpy as np
    import torch

    from cnn_transformer.data.build_wlasl_lmdb import aspect_factor
    from cnn_transformer.data.preprocessing import frame_stacked_data, hand_presence

    path, full_path = args
    try:
        coords = frame_stacked_data(full_path)
        k, raw = aspect_factor(coords)
        coords[:, 0::3] *= k  # x
        coords[:, 2::3] *= k  # z
        t = torch.from_numpy(coords)
        _, pres = hand_presence(t)
        both = float(((pres[:, 0] > 0) & (pres[:, 1] > 0)).float().mean())
        buf = io.BytesIO()
        torch.save(t.clone(), buf)
        return path, buf.getvalue(), len(coords), k, raw, both, None
    except Exception as e:  # noqa: BLE001 — reported per clip
        return path, None, None, None, None, None, str(e)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--wlasl-dir", required=True, type=Path,
                   help="folder with train.csv and train_landmark_files/")
    p.add_argument("--wlasl-json", type=Path, help="WLASL_v0.3.json (signer, fps, source, split)")
    p.add_argument("--sign-map", required=True, type=Path,
                   help="ASL Signs sign_to_prediction_index_map.json")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--strict", action="store_true", help="identical gloss names only")
    p.add_argument("--all-glosses", action="store_true",
                   help="keep unshared glosses as extra labels (after the 250) + wlasl_half column")
    p.add_argument("--lmdb-name", default="wlasl.lmdb.mdb")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--limit", type=int, default=0, help="first N clips (smoke test)")
    args = p.parse_args()

    sign2idx = json.loads(args.sign_map.read_text())
    idx2sign = {v: k for k, v in sign2idx.items()}
    gmap = json.loads(_GLOSS_MAP.read_text())

    df = pd.read_csv(args.wlasl_dir / "train.csv", dtype={"video_id": str})
    df["wlasl_gloss"] = df["sign"]
    df["sign"] = df["wlasl_gloss"].map(lambda g: idx2sign.get(gmap[g]) if g in gmap
                                       else g if g in sign2idx else None)
    df["synonym"] = df["sign"].notna() & (df["wlasl_gloss"] != df["sign"].str.lower())
    if args.strict:
        df = df[~df["synonym"]]
    vocab = dict(sign2idx)
    if args.all_glosses:
        extra = sorted(set(df.loc[df["sign"].isna(), "wlasl_gloss"]))
        vocab.update({g: len(sign2idx) + i for i, g in enumerate(extra)})
        df["sign"] = df["sign"].fillna(df["wlasl_gloss"])
    df = df[df["sign"].notna()].copy()
    df = df[[(args.wlasl_dir / p_).exists() for p_ in df["path"]]]
    if args.limit:
        df = df.head(args.limit)

    meta = {}
    if args.wlasl_json and args.wlasl_json.exists():
        for entry in json.loads(args.wlasl_json.read_text()):
            for inst in entry["instances"]:
                meta[str(inst["video_id"])] = inst
    df["participant_id"] = [_SIGNER_OFFSET + int(meta[v]["signer_id"]) if v in meta else -1
                            for v in df["video_id"]]
    for col in ("fps", "source", "split"):
        df[col] = [meta.get(v, {}).get(col) for v in df["video_id"]]
    df["sequence_id"] = df["video_id"]
    print(f"{len(df)} clips, {df['sign'].nunique()} signs ({int(df['synonym'].sum())} via synonym "
          f"glosses), {df['participant_id'].nunique()} signers")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lmdb_path = args.out_dir / args.lmdb_name
    env = lmdb.open(str(lmdb_path), subdir=False, map_size=1 << 36)
    stats, errors = {}, []
    jobs = [(p_, str(args.wlasl_dir / p_)) for p_ in df["path"]]
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
    cols = ["path", "participant_id", "sequence_id", "sign", "wlasl_gloss", "synonym", "fps",
            "source", "split", "aspect_k", "ratio_raw", "both_hands", "n_frames"]
    if args.all_glosses:
        test = set(signer_split(df[df["sign"].isin(sign2idx)])["test"])
        df["wlasl_half"] = np.where(df["participant_id"].isin(test), "test", "train")
        cols.append("wlasl_half")
        print(f"{len(vocab)} labels ({len(vocab) - len(sign2idx)} extra glosses); halves "
              f"{df.groupby('wlasl_half')['participant_id'].nunique().to_dict()} signers, "
              f"{df['wlasl_half'].value_counts().to_dict()} clips")
    df[cols].to_csv(args.out_dir / "train.csv", index=False)
    (args.out_dir / "sign_to_prediction_index_map.json").write_text(json.dumps(vocab))

    print(f"wrote {len(df)} clips → {lmdb_path} ({lmdb_path.stat().st_size / 2**20:.0f} MiB); "
          f"{len(errors)} errors" + (f", e.g. {errors[0]}" if errors else ""))
    print(f"raw shoulder ratio median {df['ratio_raw'].median():.2f} → corrected {TARGET_RATIO}; "
          f"aspect k median {df['aspect_k'].median():.2f} (p10 {df['aspect_k'].quantile(.1):.2f}, "
          f"p90 {df['aspect_k'].quantile(.9):.2f}); k = 1 fallback {int((df['aspect_k'] == 1).sum())}")
    df = df[df["sign"].isin(sign2idx)]  # stats below: shared signs only
    per_sign = df.groupby("sign")["both_hands"].apply(lambda s: (s > 0).mean())
    print(f"clips with both hands in some frame: {(df['both_hands'] > 0).mean():.0%}; per sign "
          f"median {per_sign.median():.0%}, {(per_sign > 0.5).sum()} signs > 50 % "
          f"(ASL Signs: per-sign median 2 %); fps: {df['fps'].value_counts().head(5).to_dict()}")


if __name__ == "__main__":
    main()
