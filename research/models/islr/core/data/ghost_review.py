"""Visual review of the ghost-hand rule: raw hand slots and pose wrists drawn on the video frames.

The rule (contract.resolve_ghosts) rejects one hand slot when the two hands' 2D boxes overlap.
Statistics cannot say whether a rejected hand was a duplicate fit of one physical hand (a ghost)
or a real second hand touching the first (PRAY, HELP). This draws, for the chosen clips, frames
where the rule fires: left_hand slot red, right_hand slot blue (a rejected slot dashed), pose
wrists 15 / 16 as filled circles in the slot colours, with IoU and statuses. It also logs per
frame the evidence the rule does not use: how far each pose wrist is from the nearer of the two
hand wrists, in shoulder widths — a real second hand has its own pose wrist next to it.

Writes to --out: `<video id>.png` contact sheets and `ghost_review.csv` (one row per frame with
both hands detected).

  python -m islr.core.data.ghost_review --input-root {INPUT} --video-root {INPUT} \
    --keyword wlasl --ids 44132,01627 --out {OUT}
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import contract as K
from .audit_raw import shard_clips

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
RED, BLUE = (40, 40, 230), (230, 120, 30)  # BGR: left_hand slot, right_hand slot
NAMES = {K.ABSENT: "-", K.KEPT: "kept", K.GHOST: "GHOST", K.AMBIGUOUS: "ambig"}


def frame_evidence(raw: dict, status: np.ndarray, aspect: float) -> pd.DataFrame:
    """Per frame with both hands detected: IoU, statuses, each slot's wrist distance to both pose
    wrists, each pose wrist's distance to the nearer hand wrist, the two hand wrists' and the two pose
    wrists' distance (shoulder widths, aspect-scaled) and the slots' mean landmark distance / hand size."""
    lh, rh, pose = raw["left_hand"], raw["right_hand"], raw["pose"]
    scale = np.array([aspect, 1.0])
    sw = np.nanmedian(np.linalg.norm((pose[:, 11, :2] - pose[:, 12, :2]) * scale, axis=-1))
    rows = []
    for t in np.nonzero(~np.isnan(lh[:, 0, 0]) & ~np.isnan(rh[:, 0, 0]))[0]:
        hw = [lh[t, 0, :2], rh[t, 0, :2]]
        near = [min(np.linalg.norm((pose[t, w, :2] - h) * scale) for h in hw) / sw for w in (15, 16)]
        d = {f"{s}_to_pose{w}": float(np.linalg.norm((h[t, 0, :2] - pose[t, w, :2]) * scale) / sw)
             for s, h in (("lh", lh), ("rh", rh)) for w in (15, 16)}
        size = np.mean([np.linalg.norm((h[t, 9, :2] - h[t, 0, :2]) * scale) for h in (lh, rh)])
        rows.append({"frame": int(t), "iou": K._iou(K._box(lh[t]), K._box(rh[t])), **d,
                     # mean landmark distance between the two slots / hand size: ~0 = one hand fitted twice
                     "slot_landmark_dist": float(np.linalg.norm((lh[t, :, :2] - rh[t, :, :2]) * scale, axis=-1).mean()
                                                 / max(size, 1e-6)),
                     "left_status": NAMES[int(status[t, 0])], "right_status": NAMES[int(status[t, 1])],
                     "pose15_to_hand": near[0], "pose16_to_hand": near[1],
                     "hand_wrists_apart": float(np.linalg.norm((hw[0] - hw[1]) * scale) / sw),
                     "pose_wrists_apart": float(np.linalg.norm((pose[t, 15, :2] - pose[t, 16, :2]) * scale) / sw)})
    return pd.DataFrame(rows)


def _draw_hand(img, h, color, dashed):
    import cv2

    H, W = img.shape[:2]
    pts = [(int(x * W), int(y * H)) for x, y in h[:, :2]]
    for a, b in HAND_EDGES:
        if dashed:
            for s in np.linspace(0, 1, 6)[::2]:
                p = tuple(int(pts[a][i] + (pts[b][i] - pts[a][i]) * s) for i in (0, 1))
                q = tuple(int(pts[a][i] + (pts[b][i] - pts[a][i]) * min(1, s + 0.2)) for i in (0, 1))
                cv2.line(img, p, q, color, 1, cv2.LINE_AA)
        else:
            cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)


def contact_sheet(video: str, raw: dict, status: np.ndarray, ev: pd.DataFrame, title: str, n: int = 6):
    """Up to n frames where the rule fired (or, failing that, any two-hand frames), evenly spaced."""
    import cv2

    fired = ev[(ev["left_status"] != "kept") | (ev["right_status"] != "kept")]
    pick = (fired if len(fired) else ev)["frame"].to_numpy()
    pick = pick[np.linspace(0, len(pick) - 1, min(n, len(pick))).round().astype(int)] if len(pick) else []
    cap = cv2.VideoCapture(video)
    decoded, t_ = {}, 0  # read in decode order, as the extraction numbered frames (seeking can be inexact)
    while len(decoded) < len(pick):
        ok, img = cap.read()
        if not ok:
            break
        if t_ in set(int(x) for x in pick):
            decoded[t_] = img
        t_ += 1
    cap.release()
    tiles = []
    for t in pick:
        img = decoded.get(int(t))
        if img is None:
            continue
        f = 360 / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1] * f), 360), interpolation=cv2.INTER_CUBIC)
        H, W = img.shape[:2]
        for slot, part, color in ((0, "left_hand", RED), (1, "right_hand", BLUE)):
            _draw_hand(img, raw[part][t], color, dashed=status[t, slot] == K.GHOST)
        for w, color in ((15, RED), (16, BLUE)):
            x, y = raw["pose"][t, w, :2]
            if x == x:
                cv2.circle(img, (int(x * W), int(y * H)), 7, color, -1, cv2.LINE_AA)
                cv2.circle(img, (int(x * W), int(y * H)), 7, (255, 255, 255), 1, cv2.LINE_AA)
        r = ev[ev["frame"] == t].iloc[0]
        for i, s in enumerate([f"t{t} IoU {r.iou:.2f} L {r.left_status} R {r.right_status}",
                               f"wrists {r.pose15_to_hand:.2f} / {r.pose16_to_hand:.2f} sw"]):
            cv2.putText(img, s, (4, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, s, (4, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    if not tiles:
        return None
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles]
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 3)
    sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    head = np.full((30, sheet.shape[1], 3), 255, np.uint8)
    cv2.putText(head, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([head, sheet])


def _clip_evidence(job: tuple):
    """(stem, gloss, full path, aspect, draw) → (evidence frame rows, and for a clip to draw: pose-kept
    raw arrays, status, original frame ids)."""
    stem, gloss, full, aspect, draw = job
    raw = K.raw_arrays(pd.read_parquet(full))
    orig = np.nonzero(~np.isnan(raw["pose"][:, 0, 0]))[0]  # contract step 1: pose-less frames dropped
    raw = {k: v[orig] for k, v in raw.items()}
    status, _ = K.resolve_ghosts(raw, aspect)
    ev = frame_evidence(raw, status, aspect)
    if not ev.empty:
        ev.insert(0, "video", stem)
        ev.insert(1, "gloss", gloss)
    return (ev, raw, status, orig) if draw else (ev, None, None, None)


def main() -> None:
    import cv2
    from concurrent.futures import ProcessPoolExecutor

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input-root", required=True, type=Path, help="extraction shard folders")
    p.add_argument("--video-root", required=True, type=Path, help="searched recursively for the videos")
    p.add_argument("--keyword", required=True, help="shard csv path keyword (wlasl / popsign / aslcitizen)")
    p.add_argument("--ids", required=True, help="comma-separated video ids (file stems) to draw")
    p.add_argument("--evidence-all", action="store_true",
                   help="frame evidence for every clip in the shards (ghost_review.csv), not only --ids")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ids = [s.strip() for s in a.ids.split(",") if s.strip()]
    clips = shard_clips(a.input_root, a.keyword)
    clips["stem"] = [Path(str(v)).stem for v in clips["video"]]
    clips["aspect"] = (clips["width"] / clips["height"]).fillna(0.75) if "width" in clips else 0.75
    if not a.evidence_all:
        clips = clips[clips["stem"].isin(ids)]
    videos = {Path(v).stem: v for ext in ("mp4", "mov", "MOV", "webm")
              for v in glob.glob(f"{a.video_root}/**/*.{ext}", recursive=True) if Path(v).stem in ids}
    print(f"{len(clips)} clips; {int(clips['stem'].isin(ids).sum())} of {len(ids)} to draw, {len(videos)} videos found")
    jobs = list(zip(clips["stem"], clips["gloss"], clips["full_path"], clips["aspect"], clips["stem"].isin(videos)))
    rows = []
    with ProcessPoolExecutor(os.cpu_count() or 4) as pool:
        for r, (ev, raw, status, orig) in zip(clips.itertuples(), pool.map(_clip_evidence, jobs, chunksize=16)):
            if ev.empty:
                continue
            rows.append(ev)
            if raw is None:
                continue
            # the extraction numbers frames from 0 in decode order; map back past dropped pose-less frames
            n = int(orig.max()) + 1
            raw_v = {k: np.full((n,) + v.shape[1:], np.nan) for k, v in raw.items()}
            st_v = np.zeros((n, 2), np.int8)
            for k in raw:
                raw_v[k][orig] = raw[k]
            st_v[orig] = status
            sheet = contact_sheet(videos[r.stem], raw_v, st_v, ev.assign(frame=orig[ev["frame"].to_numpy()]),
                                  f"{r.stem} {r.gloss} signer {r.participant} {int(r.width)}x{int(r.height)}: "
                                  f"ghost {int((status == K.GHOST).any(1).sum())}/{len(status)} frames")
            if sheet is not None:
                cv2.imwrite(str(a.out / f"{r.stem}.png"), sheet)
    if not rows:
        raise SystemExit("no clip with a frame where both hands are detected")
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(a.out / "ghost_review.csv", index=False)
    fired = (out["left_status"] == "GHOST") | (out["right_status"] == "GHOST")
    both = fired & (out["pose15_to_hand"] < 0.3) & (out["pose16_to_hand"] < 0.3)
    print(f"{int(fired.sum())} ghost frames in {out.loc[fired, 'video'].nunique()} clips; both pose wrists "
          f"within 0.3 shoulder widths of a hand in {int(both.sum())} of them "
          f"({out.loc[both, 'video'].nunique()} clips)")
    print(f"{len(list(a.out.glob('*.png')))} contact sheets → {a.out}")


if __name__ == "__main__":
    main()
