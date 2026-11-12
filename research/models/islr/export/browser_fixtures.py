"""Golden test fixtures for a browser port of the phrase model's preprocessing.

Each fixture pairs raw per-frame landmarks — the shape a MediaPipe pipeline
produces (null for an undetected part) — with what the Python pipeline makes of
them: the segment, the (T, IN_FEAT) model input and the ONNX logits / greedy
decode. A browser implementation passes when, fed `frames`, it reproduces
`segment`, `subsample`, `features` and `decoded`.

Contract (built from the raw frames by these rules, so a fixture can never
contradict its own `frames`):
  1. drop frames without a pose; segment = first..last remaining frame with a hand
  2. per frame, subtract pose landmark 0 (nose) from x, y, z of every landmark
  3. presence = the hand was detected in that frame (never re-inferred)
  4. a missing landmark holds its nose-relative value from the previous frame where
     it was detected (the next one before the first detection); never detected → 0
  5. more than --max-frames frames → keep positions (i * (T - 1)) // (M - 1), i < M
  6. Δ1 = frame difference of the positions after 5; 0 for the first frame
This equals the training pipeline (frame_stacked_data + hand_presence) except
where hand_presence cannot tell which frame a lone detection came from; the
script reports such presence differences as a cross-check, not an error.

Input: competition-format parquets (frame, type, landmark_index, x, y, z; NaN when
a part is not detected). --concat also writes all inputs back to back as one
multi-sign phrase.

Usage:
  PYTHONPATH=research/models uv run --with onnxruntime python -m islr.export.browser_fixtures \
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

from ..core import config as C
from ..core.data.preprocessing import frame_stacked_data
from .export_onnx import clip_features

_PARTS = {"pose": 33, "left_hand": 21, "right_hand": 21, "face": 468}
_MIN_FRAMES = 8
_LANDMARKS = (
    [("left_hand", i) for i in range(C.N_LH)]
    + [("pose", i) for i in range(C.N_POSE)]
    + [("right_hand", i) for i in range(C.N_RH)]
    + [("face", i) for i in C.SELECTED_FACE_INDICES]
)


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


def segment(frames: list) -> list:
    """Rule 1: indices of frames with a pose, first..last such frame with a hand."""
    posed = [i for i, f in enumerate(frames) if f["pose"] is not None]
    hands = [i for i in posed if frames[i]["left_hand"] or frames[i]["right_hand"]]
    return [i for i in posed if hands and hands[0] <= i <= hands[-1]]


def contract_features(frames: list, max_frames: int) -> tuple[np.ndarray, list | None]:
    """Rules 2–6 on the segment's frames → ((T, IN_FEAT) float32, subsample positions)."""
    T, n = len(frames), len(_LANDMARKS)
    pos = np.full((T, n, 3), np.nan)
    pres = np.zeros((T, 2), dtype=np.float32)
    for t, f in enumerate(frames):
        origin = np.asarray(f["pose"][0])
        for k, (part, i) in enumerate(_LANDMARKS):
            if f[part] is not None:
                pos[t, k] = np.asarray(f[part][i]) - origin
        pres[t] = [f["left_hand"] is not None, f["right_hand"] is not None]
    for k in range(n):
        seen = ~np.isnan(pos[:, k, 0])
        if not seen.any():
            pos[:, k] = 0.0
            continue
        src = np.maximum.accumulate(np.where(seen, np.arange(T), -1))
        src[src < 0] = int(np.argmax(seen))
        pos[:, k] = pos[src, k]
    pos = pos.reshape(T, -1).astype(np.float32)
    keep = None
    if T > max_frames:
        keep = [(i * (T - 1)) // (max_frames - 1) for i in range(max_frames)]
        pos, pres = pos[keep], pres[keep]
    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    return np.concatenate([pos, vel, pres], axis=1), keep


def pipeline_check(df: pd.DataFrame, seg_frames: list, feats: np.ndarray) -> str:
    """Training pipeline on the same frames (only when no frame was dropped or subsampled)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "seg.parquet"
        df[df["frame"].isin(seg_frames)].to_parquet(path)
        ref = clip_features(torch.tensor(frame_stacked_data(str(path)), dtype=torch.float32)).numpy()
    coord = float(np.abs(ref[:, : C.PRESENCE_START] - feats[:, : C.PRESENCE_START]).max())
    flags = int((ref[:, C.PRESENCE_START :] != feats[:, C.PRESENCE_START :]).sum())
    return f"training pipeline: max |Δcoord| {coord:.1e}, presence differences {flags}"


def fixture(name: str, df: pd.DataFrame, label: str, sess, labels: list,
            max_frames: int) -> dict | None:
    frames = raw_frames(df)
    seg = segment(frames)
    if not seg:
        print(f"skip {name}: no frame with both a pose and a hand")
        return None
    feats, keep = contract_features([frames[i] for i in seg], max_frames)
    all_frames = sorted(df["frame"].unique())
    check = (pipeline_check(df, [all_frames[i] for i in seg], feats)
             if keep is None and seg == list(range(seg[0], seg[-1] + 1)) else "training pipeline: n/a")
    fx = {
        "name": name,
        "label": label,
        "frames": frames,
        "segment": seg,  # indices into `frames` (rule 1)
        "subsample": keep,  # positions into `segment` (rule 5), null if not subsampled
        "features": np.round(feats.astype(np.float64), 7).tolist(),
        "logits": None,
        "decoded": None,
    }
    if feats.shape[0] >= _MIN_FRAMES:
        logits = sess.run(None, {"features": feats[None]})[0][0]
        best, prev, decoded = logits.argmax(-1).tolist(), None, []
        for c in best:
            if c != prev and c != len(labels):
                decoded.append(labels[c])
            prev = c
        fx["logits"] = np.round(logits.astype(np.float64), 5).tolist()
        fx["decoded"] = decoded
    print(f"{name}: {len(frames)} frames, segment {seg[0]}..{seg[-1]} ({len(seg)}), "
          f"T {feats.shape[0]}, label {label!r}, decoded {fx['decoded']} | {check}")
    return fx


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--parquet", nargs="+", required=True, type=Path)
    p.add_argument("--concat", action="store_true", help="also write all inputs as one phrase")
    p.add_argument("--train-csv", default="data/asl-is-lmdb/train.csv")
    p.add_argument("--export-dir", type=Path, default=Path("exports/phrases01"))
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
