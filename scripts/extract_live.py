# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["mediapipe==0.10.14", "opencv-python<4.12", "numpy<2", "pandas", "pyarrow<18"]
# ///
"""Landmarks from your own recordings (live camera-acceptance videos), on this machine.

Same MediaPipe Holistic settings as the Kaggle extraction cell (scripts/extract_aslcitizen.py
CELL_RUN: 0.10.14, model_complexity 1, confidence 0.5 / 0.5, a fresh tracker per video, full-size
frames), so the landmarks match the training data's. Every video under --videos (any depth; the first
folder below --videos is its condition) → <out>/<condition>/<name>.parquet in the competition format,
plus <name>.meta.json: decoded frame size (after rotation metadata), capture timestamps (ms), fps, and
the mirroring check (pose 11 x − pose 12 x, median; > 0 = unmirrored like training). A mirrored video
(front camera with mirroring on) is re-extracted with every frame flipped back and flagged.

  uv run --script scripts/extract_live.py --videos data/live-recordings/videos --out data/live-recordings/landmarks
  uv run --script scripts/extract_live.py --sheet data/live-recordings/landmarks/<condition>/segments.json   # contact sheet

The videos and landmarks stay in data/ (gitignored): recordings of a person are never committed.
"""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

PARTS = (("pose", 33, "pose_landmarks"), ("left_hand", 21, "left_hand_landmarks"),
         ("right_hand", 21, "right_hand_landmarks"), ("face", 468, "face_landmarks"))
N_LM = sum(n for _, n, _ in PARTS)
TYPES = np.concatenate([[part] * n for part, n, _ in PARTS])
INDEX = np.concatenate([np.arange(n) for _, n, _ in PARTS])
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}
VERSION = "live-mp0.10.14-v1"  # bump whenever what this writes changes


def track(video: str, flip: bool) -> tuple[list, list, tuple]:
    """(per-frame (543, 3) landmarks, capture timestamps ms, (width, height)) of one video."""
    holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1,
                                              min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video)
    frames, ts, size = [], [], (None, None)
    while True:
        ok, img = cap.read()
        if not ok:
            break
        ts.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)))
        if flip:
            img = cv2.flip(img, 1)
        size = img.shape[1], img.shape[0]
        res = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        xyz = np.full((N_LM, 3), np.nan, dtype=np.float32)
        i = 0
        for _, n, attr in PARTS:
            lms = getattr(res, attr)
            if lms:
                xyz[i:i + n] = [[l.x, l.y, l.z] for l in lms.landmark[:n]]
            i += n
        frames.append(xyz)
    cap.release()
    holistic.close()
    return frames, ts, size


def lr_dx(frames: list) -> float:
    a = np.stack(frames)
    d = a[:, 11, 0] - a[:, 12, 0]
    return float(np.nanmedian(d)) if np.isfinite(d).any() else float("nan")


def extract(job: tuple) -> dict:
    video, out = job
    meta_path = out.with_suffix(".meta.json")
    if out.exists() and meta_path.exists() and json.loads(meta_path.read_text()).get("version") == VERSION:
        return json.loads(meta_path.read_text())
    frames, ts, size = track(video, flip=False)
    mirrored = bool(frames) and lr_dx(frames) < 0
    if mirrored:  # selfie-mirrored recording: training frames are unmirrored
        frames, ts, size = track(video, flip=True)
    meta = {"version": VERSION, "video": video, "n_frames": len(frames), "width": size[0], "height": size[1],
            "fps_header": None, "timestamps_ms": ts, "mirrored_in_file": mirrored,
            "lr_dx": lr_dx(frames) if frames else None}
    cap = cv2.VideoCapture(video)
    meta["fps_header"] = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if len(ts) > 1:
        meta["fps_measured"] = float((len(ts) - 1) / ((ts[-1] - ts[0]) / 1000)) if ts[-1] > ts[0] else None
    if frames:
        T = len(frames)
        xyz = np.concatenate(frames)
        pd.DataFrame({"frame": np.repeat(np.arange(T), N_LM), "type": np.tile(TYPES, T),
                      "landmark_index": np.tile(INDEX, T), "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
                     ).to_parquet(out, index=False)
        pose = np.isfinite(np.stack(frames)[:, 0, 0])
        hands = np.isfinite(np.stack(frames)[:, 33, 0]) | np.isfinite(np.stack(frames)[:, 54, 0])
        meta.update(pose_frac=float(pose.mean()), hand_frac=float(hands.mean()))
    meta_path.write_text(json.dumps(meta))
    return meta


def sheet(seg_path: Path) -> None:
    """One thumbnail (middle frame) per detected segment, labelled, in a grid → segments.png."""
    segs = json.loads(seg_path.read_text())
    tiles = []
    for s in segs["segments"]:
        cap = cv2.VideoCapture(segs["videos"][s["video"]])
        cap.set(cv2.CAP_PROP_POS_FRAMES, s["mid_raw_frame"])
        ok, img = cap.read()
        cap.release()
        if not ok:
            img = np.zeros((480, 360, 3), np.uint8)
        if segs.get("mirrored", {}).get(s["video"]):
            img = cv2.flip(img, 1)
        img = cv2.resize(img, (240, round(240 * img.shape[0] / img.shape[1])))
        cv2.putText(img, f"{s['index']}: {s['label']}", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(img, f"{s['index']}: {s['label']}", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(img)
    if not tiles:
        return
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    cols = 8
    rows = [np.hstack(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols])))
            for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(seg_path.with_suffix(".png")), np.vstack(rows))
    print("wrote", seg_path.with_suffix(".png"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--videos", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--sheet", type=Path, help="segments.json written by islr.evaluation.live_videos")
    a = p.parse_args()
    if a.sheet:
        return sheet(a.sheet)
    vids = sorted(v for v in a.videos.rglob("*") if v.suffix.lower() in VIDEO_EXT and not v.name.startswith("."))
    jobs = []
    for v in vids:
        rel = v.relative_to(a.videos)
        out = a.out / rel.with_suffix(".parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((str(v), out))
    print(f"{len(jobs)} videos")
    with Pool(a.workers) as pool:
        for m in pool.imap(extract, jobs):
            print(f"{Path(m['video']).name}: {m['n_frames']} frames {m['width']}x{m['height']} "
                  f"fps {m.get('fps_measured') or 0:.1f} pose {m.get('pose_frac', 0):.0%} hands {m.get('hand_frac', 0):.0%}"
                  + (" (mirrored in file: flipped back)" if m["mirrored_in_file"] else ""), flush=True)


if __name__ == "__main__":
    main()
