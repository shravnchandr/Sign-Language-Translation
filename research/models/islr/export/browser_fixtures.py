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


def _exact(a: np.ndarray) -> list:
    """float32 values as JSON numbers that parse back to the same float32 (no rounding: a rounded
    coordinate can flip a ghost-hand decision near the IoU threshold); NaN → null."""
    a = np.asarray(a, dtype=np.float32).astype(np.float64)
    return np.where(np.isnan(a), None, a).tolist()


def raw_frames(df: pd.DataFrame) -> list:
    """Per frame: {part: [[x, y, z], ...] or None if the part was not detected}, values exactly the
    float32 the contract reads."""
    out = []
    for _, f in df.sort_values(["frame", "type", "landmark_index"]).groupby("frame", sort=True):
        frame = {}
        for part, n in _PARTS.items():
            rows = f[f["type"] == part].set_index("landmark_index").reindex(range(n))
            xyz = rows[["x", "y", "z"]].to_numpy(dtype=np.float32)
            frame[part] = None if np.isnan(xyz).all() else _exact(xyz)
        out.append(frame)
    return out


LIVE_FPS = contract.TARGET_FPS
NEAREST_MIN = 0.95  # median capture rate >= 95 % of LIVE_FPS: pick frames; slower: interpolate
SNAP_MS = 0.5  # a tick this close to a capture timestamp takes that frame exactly (timestamps are rounded)


def resample_plan(ts_ms: list, fps: float = LIVE_FPS) -> list:
    """Live input, before anything else (the browser's counterpart of contract step 0): a 30 fps grid
    t_k = ts[0] + k * 1000 / fps (t_k <= ts[-1]) over the capture timestamps. Per tick
    (i0, i1, w): the output frame is (1 - w) · frame i0 + w · frame i1. Mode chosen once per stream from
    the median capture interval: at >= NEAREST_MIN · fps, the nearest frame (w = 0, i0 = i1; ties → the
    earlier) — what training's frame step does; slower, linear interpolation between the frames around
    the tick, except that a tick within SNAP_MS of a frame takes that frame (repeating frames would put zero-velocity steps into the features, which training never
    sees). Timestamps must be strictly increasing."""
    t = np.asarray(ts_ms, dtype=np.float64)
    if len(t) == 0:
        return []
    if len(t) > 1 and not (np.diff(t) > 0).all():
        raise ValueError("capture timestamps must be strictly increasing")
    step = 1000.0 / fps
    n = int(np.floor((t[-1] - t[0] + SNAP_MS) / step)) + 1
    ticks = t[0] + step * np.arange(n)
    nearest = len(t) < 2 or float(np.median(np.diff(t))) <= step / NEAREST_MIN
    plan = []
    for tk in ticks:
        j = min(int(np.searchsorted(t, tk, side="right")) - 1, len(t) - 1)
        if j == len(t) - 1 or tk - t[j] <= SNAP_MS:
            plan.append((j, j, 0.0))
        elif t[j + 1] - tk <= SNAP_MS:
            plan.append((j + 1, j + 1, 0.0))
        elif nearest:
            k = j if tk - t[j] <= t[j + 1] - tk else j + 1
            plan.append((k, k, 0.0))
        else:
            plan.append((j, j + 1, float((tk - t[j]) / (t[j + 1] - t[j]))))
    return plan


def apply_plan(frames: list, plan: list) -> list:
    """Resampled raw frames (fixture format: part → (n, 3) list or None). A part detected in both
    frames is interpolated; otherwise the nearer frame's part (or its absence) is taken as it is."""
    out = []
    for i0, i1, w in plan:
        f = {}
        for part in _PARTS:
            a, b = frames[i0][part], frames[i1][part]
            if w == 0.0 or a is None or b is None:
                f[part] = a if w < 0.5 else b
            else:
                f[part] = ((1 - w) * np.asarray(a, np.float64) + w * np.asarray(b, np.float64)).astype(np.float32).tolist()
        out.append(f)
    return out


def timing_vectors() -> list:
    """Reference cases for a port of resample_plan: capture at 15 / 24 / 30 / 60 / 120 fps over 2 s,
    and 30 fps with ±3 ms jitter."""
    rng = np.random.default_rng(0)
    cases = []
    for name, fps_in, jitter in (("15fps", 15, 0), ("24fps", 24, 0), ("30fps", 30, 0), ("60fps", 60, 0),
                                 ("120fps", 120, 0), ("30fps_jitter3ms", 30, 3.0)):
        n = int(2 * fps_in)
        ts = 1000.0 * np.arange(n) / fps_in + 50.0
        if jitter:
            ts[1:-1] += rng.uniform(-jitter, jitter, n - 2)
        ts = np.round(ts, 3)
        cases.append({"name": name, "timestamps_ms": ts.tolist(),
                      "plan": [{"i0": i0, "i1": i1, "w": w} for i0, i1, w in resample_plan(ts.tolist())]})
    return cases


def frames_to_rows(frames: list) -> pd.DataFrame:
    """A fixture's serialized `frames` back to competition-format rows (frame = position)."""
    rows = []
    for t, frame in enumerate(frames):
        for part, n in _PARTS.items():
            xyz = frame[part]
            for i in range(n):
                v = xyz[i] if xyz is not None else (None, None, None)
                rows.append((t, part, i, *(np.nan if c is None else c for c in v)))
    df = pd.DataFrame(rows, columns=["frame", "type", "landmark_index", "x", "y", "z"])
    df[["x", "y", "z"]] = df[["x", "y", "z"]].astype(np.float32)
    return df


def verify(fx: dict, max_frames: int) -> None:
    """The fixture must be reproducible from its own serialized frames: kept frames, hand status,
    assignment_uncertain, segment, subsample and features bit-exact. Raises otherwise."""
    clip = contract.prepare(frames_to_rows(fx["frames"]))
    coords, presence = torch.from_numpy(clip["coords"]), torch.from_numpy(clip["presence"]).float()
    seg = contract.segment(presence)
    T = seg.stop - seg.start
    feats = contract.clip_input(coords, presence, max_frames, trim=True).numpy()
    got = {"kept_frames": [int(f) for f in clip["frames"]], "hand_status": clip["status"].tolist(),
           "assignment_uncertain": clip["assignment_uncertain"].tolist(),
           "segment": list(range(seg.start, seg.stop)),
           "subsample": contract.subsample_index(T, max_frames).tolist() if T > max_frames else None,
           "features": _exact(feats)}
    bad = [k for k, v in got.items() if v != fx[k]]
    if bad:
        raise SystemExit(f"fixture {fx['name']}: {bad} not reproducible from its serialized frames")


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
        "assignment_uncertain": clip["assignment_uncertain"].tolist(),  # per kept frame (rule 2)
        "segment": list(range(seg.start, seg.stop)),  # indices into kept frames (rule 6)
        "subsample": keep.tolist() if T > max_frames else None,  # positions into `segment` (rule 7)
        "features": _exact(feats),  # float32, exact
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
        fx["logits"] = np.round(logits.astype(np.float64), 5).tolist()  # compare with a tolerance (kernels differ)
        fx["decoded"] = decoded
    verify(fx, max_frames)
    print(f"{name}: {len(frames)} frames, {len(kept)} kept, segment {seg.start}..{seg.stop - 1}, T {feats.shape[0]}, "
          f"copy frames {clip['stats']['copy_frames']}, label {label!r}, decoded {fx['decoded']}")
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
