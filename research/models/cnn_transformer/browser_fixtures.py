"""Golden test fixtures for a browser port of the phrase model's preprocessing.

Each fixture pairs raw per-frame landmarks — the shape a MediaPipe pipeline
produces (null for an undetected part) — with what the Python pipeline makes of
them: the trimmed segment, the (T, IN_FEAT) model input and the ONNX logits /
greedy decode. A browser implementation passes when, fed `frames`, it reproduces
`trim`, `features` and `decoded` (rules: `preprocess.json` from export_onnx.py).

Input: competition-format parquets (frame, type, landmark_index, x, y, z; NaN when
a part is not detected). --concat also writes all inputs back to back as one
multi-sign phrase.

Usage:
  PYTHONPATH=research/models uv run --with onnxruntime python -m cnn_transformer.browser_fixtures \
    --parquet data/Isolated_ASL_Recognition/train_landmark_files/16069/100015657.parquet \
              data/Isolated_ASL_Recognition/train_landmark_files/18796/1001373962.parquet \
    --concat --out-dir exports/phrases01/fixtures
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data.preprocessing import frame_stacked_data
from .export_onnx import clip_features

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


def hand_span(frames: list) -> tuple[int, int] | None:
    """First..last frame (inclusive) with either hand detected."""
    hands = [i for i, f in enumerate(frames) if f["left_hand"] or f["right_hand"]]
    return (hands[0], hands[-1]) if hands else None


def fixture(name: str, df: pd.DataFrame, label: str, sess, labels: list) -> dict | None:
    frames = raw_frames(df)
    span = hand_span(frames)
    if span is None:
        print(f"skip {name}: no hand detected")
        return None
    keep = sorted(df["frame"].unique())[span[0] : span[1] + 1]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "seg.parquet"
        df[df["frame"].isin(keep)].to_parquet(path)
        coords = torch.tensor(frame_stacked_data(str(path)), dtype=torch.float32)
    feats = clip_features(coords)
    fx = {
        "name": name,
        "label": label,
        "frames": frames,
        "trim": list(span),  # inclusive frame indices into `frames`
        "features": np.round(feats.numpy().astype(np.float64), 7).tolist(),
        "logits": None,
        "decoded": None,
    }
    if feats.shape[0] >= _MIN_FRAMES:
        logits = sess.run(None, {"features": feats[None].numpy()})[0][0]
        best, prev, decoded = logits.argmax(-1).tolist(), None, []
        for c in best:
            if c != prev and c != len(labels):
                decoded.append(labels[c])
            prev = c
        fx["logits"] = np.round(logits.astype(np.float64), 5).tolist()
        fx["decoded"] = decoded
    print(f"{name}: {len(frames)} frames, trim {span}, T {feats.shape[0]}, label {label!r}, "
          f"decoded {fx['decoded']}")
    return fx


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--parquet", nargs="+", required=True, type=Path)
    p.add_argument("--concat", action="store_true", help="also write all inputs as one phrase")
    p.add_argument("--train-csv", default="data/Isolated_ASL_Recognition/train.csv")
    p.add_argument("--export-dir", type=Path, default=Path("exports/phrases01"))
    p.add_argument("--out-dir", type=Path, required=True)
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
        fx = fixture(path.stem, df, label, sess, labels)
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
        fx = fixture("concat", pd.concat(parts), " ".join(names), sess, labels)
        if fx:
            (args.out_dir / "concat.json").write_text(json.dumps(fx))


if __name__ == "__main__":
    main()
