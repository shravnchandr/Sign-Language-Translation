"""Golden test fixtures for a browser port of the phrase model's preprocessing.

Each fixture pairs raw per-frame landmarks — the shape a MediaPipe pipeline
produces (null for an undetected part) — with what the Python pipeline makes of
them: the segment, the (T, IN_FEAT) model input and the ONNX logits / greedy
decode. A browser implementation passes when, fed `frames`, it reproduces
`segment`, `subsample`, `features` and `decoded`.

Contract: `islr/core/data/contract.py` (also written to preprocess.json) — the same functions
the LMDB builders, the training datasets and the evaluators use, so a fixture is exactly what
training and evaluation see:
  1. drop frames without a pose (`kept_frames` = indices into `frames` that remain)
  2. ghost hands: one physical hand fitted into both slots is resolved (`hand_status` per kept
     frame and slot: 0 not detected, 1 kept, 2 ghost — rejected, 3 ambiguous — kept)
  3. per frame, subtract pose landmark 0 (nose) from x, y, z of every landmark
  4. presence = hand_status 1 or 3 (never re-inferred)
  5. a missing landmark holds its nose-relative value from the previous frame where it was
     present (the next one before the first); never present → 0
  6. segment = first..last kept frame with a hand present (`segment`, indices into kept frames)
  7. more than --max-frames frames → keep positions (i * (T - 1)) // (M - 1), i < M (`subsample`)
  8. Δ1 = frame difference of the positions after 7; 0 for the first frame

Input: competition-format parquets (frame, type, landmark_index, x, y, z; NaN when
a part is not detected). --concat also writes all inputs back to back as one
multi-sign phrase.

Usage:
  PYTHONPATH=research/models uv run --with onnxruntime python -m islr.export.browser_fixtures \
    --parquet data/Isolated_ASL_Recognition/train_landmark_files/16069/100015657.parquet \
              data/Isolated_ASL_Recognition/train_landmark_files/18796/1001373962.parquet \
    --concat --out-dir exports/phrases04/fixtures
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..core import config as C
from ..core.data import contract

_PARTS = {"pose": 33, "left_hand": 21, "right_hand": 21, "face": 468}
_MIN_FRAMES = 8


def raw_frames(df: pd.DataFrame) -> list:
    """Per frame: {part: [[x, y, z], ...] or None if the part was not detected}."""
    out = []
    for _, f in df.sort_values(["frame", "type", "landmark_index"]).groupby("frame", sort=True):
        frame = {}
        for part, n in _PARTS.items():
            rows = f[f["type"] == part].set_index("landmark_index").reindex(range(n))
            xyz = rows[["x", "y", "z"]].to_numpy(dtype=np.float64)
            frame[part] = None if np.isnan(xyz).all() else np.round(xyz, 7).tolist()
        out.append(frame)
    return out


def fixture(name: str, df: pd.DataFrame, label: str, sess, labels: list,
            max_frames: int) -> dict | None:
    frames = raw_frames(df)
    clip = contract.prepare(df)
    all_frames = list(np.sort(df["frame"].unique()))
    kept = [all_frames.index(f) for f in clip["frames"]]
    coords, presence = torch.from_numpy(clip["coords"]), torch.from_numpy(clip["presence"]).float()
    seg = contract.segment(presence)
    if not len(kept) or not bool(presence.any()):
        print(f"skip {name}: no frame with both a pose and a hand")
        return None
    T = seg.stop - seg.start
    keep = contract.subsample_index(T, max_frames)
    feats = contract.clip_input(coords, presence, max_frames, trim=True).numpy()
    fx = {
        "name": name,
        "label": label,
        "contract_version": contract.CONTRACT_VERSION,
        "frames": frames,
        "kept_frames": kept,  # indices into `frames` (rule 1)
        "hand_status": clip["status"].tolist(),  # per kept frame: [left_hand, right_hand] (rule 2)
        "segment": list(range(seg.start, seg.stop)),  # indices into kept frames (rule 6)
        "subsample": keep.tolist() if T > max_frames else None,  # positions into `segment` (rule 7)
        "features": np.round(feats.astype(np.float64), 7).tolist(),
        "logits": None,
        "decoded": None,
    }
    if feats.shape[0] >= _MIN_FRAMES:
        logits = sess.run(None, {"features": feats[None].astype(np.float32)})[0][0]
        best, prev, decoded = logits.argmax(-1).tolist(), None, []
        for c in best:
            if c != prev and c != len(labels):
                decoded.append(labels[c])
            prev = c
        fx["logits"] = np.round(logits.astype(np.float64), 5).tolist()
        fx["decoded"] = decoded
    print(f"{name}: {len(frames)} frames, {len(kept)} kept, segment {seg.start}..{seg.stop - 1}, T {feats.shape[0]}, "
          f"ghost frames {clip['stats']['ghost_frames']}, label {label!r}, decoded {fx['decoded']}")
    return fx


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--parquet", nargs="+", required=True, type=Path)
    p.add_argument("--concat", action="store_true", help="also write all inputs as one phrase")
    p.add_argument("--train-csv", default="data/asl-is-lmdb/train.csv")
    p.add_argument("--export-dir", type=Path, default=Path("exports/phrases04"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-frames", type=int, default=384)
    args = p.parse_args()

    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.export_dir / "sign_ctc.onnx"),
                                providers=["CPUExecutionProvider"])
    labels = json.loads((args.export_dir / "labels.json").read_text())
    signs = {}
    if Path(args.train_csv).exists():
        tc = pd.read_csv(args.train_csv)
        signs = dict(zip(tc["sequence_id"].astype(str), tc["sign"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dfs, names = [], []
    for path in args.parquet:
        df = pd.read_parquet(path)
        label = signs.get(path.stem, "")
        fx = fixture(path.stem, df, label, sess, labels, args.max_frames)
        if fx:
            (args.out_dir / f"{path.stem}.json").write_text(json.dumps(fx))
        dfs.append(df)
        names.append(label or path.stem)
    if args.concat and len(dfs) > 1:
        parts, offset = [], 0
        for df in dfs:  # renumber frames so the clips play back to back
            df = df.copy()
            frames = sorted(df["frame"].unique())
            df["frame"] = df["frame"].map({f: offset + i for i, f in enumerate(frames)})
            offset += len(frames)
            parts.append(df)
        fx = fixture("concat", pd.concat(parts), " ".join(names), sess, labels, args.max_frames)
        if fx:
            (args.out_dir / "concat.json").write_text(json.dumps(fx))


if __name__ == "__main__":
    main()
