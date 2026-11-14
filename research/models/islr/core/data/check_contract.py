"""Contract checks on synthetic clips: expected presence / ghost status per case, and bit-exact
feature agreement between every consumer (isolated training, phrase training, evaluation, browser
fixtures). Run before any LMDB rebuild or training:

  PYTHONPATH=research/models uv run python -m islr.core.data.check_contract

Cases: leading / trailing hand gaps, a lone detection, frames without a pose, crossed hands, two
real hands touching, a ghost (one hand in both slots), an ambiguous overlap, a clip longer than
max_frames. Exits non-zero on the first failure.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import torch

from ..config import SELECTED_FACE_INDICES
from . import contract as K
from .build_wlasl_lmdb import write_clips
from .dataset import ASLDataset

MAX_FRAMES = 384
RNG = np.random.default_rng(0)


def _hand(cx: float, cy: float, size: float = 0.08) -> np.ndarray:
    pts = RNG.uniform(-size / 2, size / 2, (21, 3))
    pts[:, 0] += cx
    pts[:, 1] += cy
    pts[0, :2] = cx, cy + size / 2  # wrist at the bottom
    return pts


def clip(T: int, lh=None, rh=None, no_pose=(), ghost=(), ambiguous=(), seed=0) -> pd.DataFrame:
    """Competition-format rows. lh / rh: frames where that hand is detected. ghost: frames where
    rh copies lh (+ tiny jitter); ambiguous: frames where rh partly overlaps lh (IoU ≈ 0.3)."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(T):
        drift = 0.002 * t
        pose = np.column_stack([rng.uniform(0.3, 0.7, 33), rng.uniform(0.2, 0.9, 33), rng.uniform(-1, 0, 33)])
        pose[0, :2] = 0.5 + drift, 0.3                 # nose
        pose[15, :2], pose[16, :2] = (0.65, 0.7), (0.35, 0.7)  # left / right wrist (unmirrored)
        has_pose = t not in no_pose
        parts = {"pose": pose if has_pose else None,
                 "face": np.column_stack([rng.uniform(0.4, 0.6, 468), rng.uniform(0.2, 0.4, 468), rng.uniform(-0.1, 0.1, 468)]) if has_pose else None,
                 "left_hand": _hand(0.62 + drift, 0.6) if lh is not None and t in lh else None,
                 "right_hand": _hand(0.38 - drift, 0.6) if rh is not None and t in rh else None}
        if t in ghost and parts["left_hand"] is not None:
            parts["right_hand"] = parts["left_hand"] + rng.normal(0, 0.002, (21, 3))
        if t in ambiguous and parts["left_hand"] is not None:
            parts["right_hand"] = parts["left_hand"] + np.array([0.04, 0.0, 0.0])  # half a hand's width over
        for part, n in (("face", 468), ("left_hand", 21), ("pose", 33), ("right_hand", 21)):
            xyz = parts[part] if parts[part] is not None else np.full((n, 3), np.nan)
            for i in range(n):
                rows.append((t, part, i, *xyz[i]))
    return pd.DataFrame(rows, columns=["frame", "type", "landmark_index", "x", "y", "z"])


CASES = {
    # name: (frames, clip kwargs, expected (presence lh, rh) by kept frame or None, expected status checks)
    "leading_gap": (40, dict(lh=range(10, 40))),
    "trailing_gap": (40, dict(lh=range(0, 25))),
    "lone_detection": (30, dict(rh=[12])),
    "no_pose_frames": (30, dict(lh=range(30), no_pose=(0, 13, 29))),
    "crossed_hands": (30, dict(lh=range(30), rh=range(30))),  # rh drawn on the lh side below
    "two_hands_touching": (30, dict(lh=range(30), rh=range(30))),
    "ghost": (40, dict(lh=range(40), rh=range(10, 16), ghost=range(10, 16))),
    "ambiguous": (40, dict(lh=range(40), rh=[20], ambiguous=[20])),
    "long_clip": (500, dict(lh=range(30, 480), rh=range(200, 260))),
}


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sign-map", default="data/asl-is-lmdb/sign_to_prediction_index_map.json",
                    help="the 250 ASL Signs labels (the phrase grammar needs the real names)")
    idx2sign = {v: k for k, v in json.loads(Path(ap.parse_args().sign_map).read_text()).items()}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "train_landmark_files").mkdir()
        rows, raw = [], {}
        for k, (name, (T, kw)) in enumerate(CASES.items()):
            df = clip(T, seed=k, **kw)
            if name == "crossed_hands":  # right hand drawn at the signer's left wrist and vice versa
                for part, dx in (("left_hand", -0.24), ("right_hand", 0.24)):
                    m = df["type"] == part
                    df.loc[m, "x"] += dx
            if name == "two_hands_touching":  # boxes touch (IoU ≈ 0.05), distinct hands
                m = df["type"] == "right_hand"
                df.loc[m, "x"] += 0.16
            path = f"train_landmark_files/{name}.parquet"
            df.to_parquet(tmp / path)
            raw[name] = df
            rows.append({"path": path, "participant_id": 1000 + k, "sequence_id": k, "sign": k})
        csv = pd.DataFrame(rows)
        env = lmdb.open(str(tmp / "c.lmdb.mdb"), subdir=False, map_size=1 << 30)
        stats, errors = write_clips(env, [(p, str(tmp / p), False, 0.75) for p in csv["path"]], 2)
        env.close()
        if errors:
            fail(f"builder errors {errors}")

        ds = ASLDataset(csv.assign(sign=csv["sign"]), str(tmp), lmdb_path=str(tmp / "c.lmdb.mdb"),
                        max_frames=MAX_FRAMES, augment=False)
        from ...export import browser_fixtures as BF
        from ...phrases.dataset import SyntheticPhraseDataset

        class _Sess:  # browser_fixtures needs an ONNX session; logits are not under test here
            def run(self, _, feeds):
                return [np.zeros((1, feeds["features"].shape[1], len(idx2sign) + 1), np.float32)]

        phr = SyntheticPhraseDataset(ds, idx2sign, n_phrases=1, augment=False,
                                     max_frames=MAX_FRAMES, min_detected=1)
        for j, name in enumerate(CASES):
            coords, presence = ds._load_clip(j)
            st = stats[f"train_landmark_files/{name}.parquet"]
            status = K.prepare(raw[name])["status"]
            # 1. expected stored presence / provenance per case
            if name == "no_pose_frames" and (st["frames_no_pose"] != 3 or len(coords) != 27):
                fail(f"{name}: {st['frames_no_pose']} pose-less frames dropped, {len(coords)} kept (want 3, 27)")
            if name == "lone_detection" and int(presence[:, 1].sum()) != 1:
                fail(f"{name}: right-hand presence {int(presence[:, 1].sum())} frames, want 1")
            if name == "leading_gap" and (presence[:10, 0].any() or not presence[10:, 0].all()):
                fail(f"{name}: presence must start at frame 10")
            if name == "trailing_gap" and (presence[25:, 0].any() or not presence[:25, 0].all()):
                fail(f"{name}: presence must end at frame 24")
            if name in ("crossed_hands", "two_hands_touching") and (not presence.all() or (status != K.KEPT).any()):
                fail(f"{name}: both hands must be kept in every frame (status {np.unique(status)})")
            if name == "ghost":
                if not ((status[10:16, 1] == K.GHOST).all() and (status[10:16, 0] == K.KEPT).all()
                        and presence[10:16, 1].sum() == 0):
                    fail(f"{name}: right-hand ghost not rejected in frames 10–15: {status[8:18].tolist()}")
            if name == "ambiguous" and not (status[20] == K.AMBIGUOUS).all():
                fail(f"{name}: frame 20 must be AMBIGUOUS (kept), got {status[20].tolist()}")
            # held hands: a gap frame repeats the nearest detection exactly
            if name == "leading_gap" and not torch.equal(coords[0, :63], coords[10, :63]):
                fail(f"{name}: leading frames must hold the first detection")
            # 2. feature agreement: every consumer builds the same input
            ref_iso = K.clip_input(coords, presence, MAX_FRAMES, trim=False)
            got_iso = ds[j][0]
            if not torch.equal(ref_iso, got_iso):
                fail(f"{name}: isolated training features ≠ evaluation features")
            ref = K.clip_input(coords, presence, MAX_FRAMES, trim=True)
            c, p_ = phr._clip(np.random.default_rng(0), str(1000 + j), j, j)
            k = K.subsample_index(c.shape[0], MAX_FRAMES)
            if not torch.equal(K.model_input(c[k], p_[k]), ref):
                fail(f"{name}: phrase training features ≠ evaluation features")
            fx = BF.fixture(name, raw[name], name, _Sess(), [idx2sign[i] for i in range(len(idx2sign))], MAX_FRAMES)
            if fx is None or not np.array_equal(np.asarray(fx["features"], np.float32), np.round(ref.numpy().astype(np.float64), 7).astype(np.float32)):
                fail(f"{name}: browser fixture features ≠ evaluation features")
            if float(ref[0, coords.shape[1]: 2 * coords.shape[1]].abs().max()) != 0.0:
                fail(f"{name}: Δ1 of the first frame must be 0")
            if name == "long_clip" and ref.shape[0] != MAX_FRAMES:
                fail(f"{name}: {ref.shape[0]} frames after subsample, want {MAX_FRAMES}")
            print(f"ok  {name:20s} T {len(coords):3d} → {ref.shape[0]:3d} | presence frames lh {int(presence[:, 0].sum())} "
                  f"rh {int(presence[:, 1].sum())} | ghost {st['ghost_frames']} ambiguous {st['ambiguous_frames']} "
                  f"no-pose {st['frames_no_pose']}")
    print("contract checks passed")


if __name__ == "__main__":
    main()
