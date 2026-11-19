# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["mediapipe==0.10.14", "opencv-python<4.12", "numpy<2", "pandas", "pyarrow<18"]
# ///
"""Landmarks from your own recordings (live camera-acceptance videos), on this machine.

Same MediaPipe Holistic settings as the Kaggle extraction cell (scripts/extract_aslcitizen.py
CELL_RUN: 0.10.14, model_complexity 1, confidence 0.5 / 0.5, a fresh tracker per video), so the
landmarks are training-like. Each frame is first centre-cropped to 3:4 exactly as the app's
`portraitCrop` (ASL-App signPhrase.ts) does before detection. Every video under --videos (the folder
below --videos is its condition) → <out>/<condition>/<name>.parquet in the competition format, plus
<name>.meta.json: crop and source frame size (after rotation metadata), capture timestamps (ms), fps,
the source file's sha256 (a cached extraction is reused only for the same bytes and VERSION) and the
mirroring check (pose 11 x − pose 12 x, median; > 0 = unmirrored like training). A mirrored video
(front camera with mirroring on) is re-extracted with every frame flipped back and flagged. A video
that cannot be opened, or decodes fewer frames than its header lists, fails (nothing written, exit 1).

  uv run --script scripts/extract_live.py --videos data/live-recordings/videos --out data/live-recordings/landmarks
  uv run --script scripts/extract_live.py --sheet <live_videos out>/segments/<condition>__<video>.json   # contact sheet

The videos and landmarks stay in data/ (gitignored): recordings of a person are never committed.
"""

import argparse
import hashlib
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
VERSION = "live-mp0.10.14-v2"  # v2: 3:4 portraitCrop before detection  # bump whenever what this writes changes


def portrait_crop(width: int, height: int) -> tuple[int, int, int, int]:
    """ASL-App portraitCrop: (x, y, w, h) centre crop to the training aspect 3:4 (JS Math.round)."""
    rnd = lambda v: int(np.floor(v + 0.5))  # noqa: E731 — Math.round, not banker's rounding
    if width / height > 0.75:
        w = rnd(height * 0.75)
        return rnd((width - w) / 2), 0, w, height
    h = min(height, rnd(width / 0.75))
    return 0, rnd((height - h) / 2), width, h


def track(video: str, flip: bool) -> tuple[list, list, tuple, tuple]:
    """(per-frame (543, 3) landmarks, capture timestamps ms, crop (w, h), source (w, h)) of one video;
    raises when the video does not decode completely."""
    holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1,
                                              min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError("cannot open")
    listed = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, ts, size, src = [], [], (None, None), (None, None)
    while True:
        ok, img = cap.read()
        if not ok:
            break
        ts.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)))
        if flip:
            img = cv2.flip(img, 1)
        src = img.shape[1], img.shape[0]
        x, y, w, h = portrait_crop(*src)
        img = img[y:y + h, x:x + w]
        size = w, h
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
    if not frames:
        raise RuntimeError("no frame decoded")
    if listed > 0 and len(frames) < listed - 1:  # the header count can be one off; fewer is a broken decode
        raise RuntimeError(f"decoded {len(frames)} of the {listed} frames the header lists")
    if not all(b > a for a, b in zip(ts, ts[1:])):
        raise RuntimeError("capture timestamps not strictly increasing")
    return frames, ts, size, src


def lr_dx(frames: list) -> float:
    a = np.stack(frames)
    d = a[:, 11, 0] - a[:, 12, 0]
    return float(np.nanmedian(d)) if np.isfinite(d).any() else float("nan")


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(job: tuple) -> dict:
    video, out = job
    meta_path = out.with_suffix(".meta.json")
    digest = file_sha256(video)
    if out.exists() and meta_path.exists():
        m = json.loads(meta_path.read_text())
        if m.get("version") == VERSION and m.get("source_sha256") == digest:
            return m
    for stale in (out, meta_path):  # never leave landmarks of an older file under this name
        stale.unlink(missing_ok=True)
    try:
        frames, ts, size, src = track(video, flip=False)
        mirrored = lr_dx(frames) < 0
        if mirrored:  # selfie-mirrored recording: training frames are unmirrored
            frames, ts, size, src = track(video, flip=True)
    except Exception as e:  # noqa: BLE001 — reported, nothing written
        return {"video": video, "error": str(e)}
    cap = cv2.VideoCapture(video)
    fps_header = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    T = len(frames)
    stack = np.stack(frames)
    meta = {"version": VERSION, "video": video, "source_sha256": digest, "n_frames": T,
            "width": size[0], "height": size[1], "source_width": src[0], "source_height": src[1],
            "fps_header": fps_header, "timestamps_ms": ts, "mirrored_in_file": mirrored, "lr_dx": lr_dx(frames),
            "fps_measured": float((T - 1) / ((ts[-1] - ts[0]) / 1000)) if T > 1 else None,
            "pose_frac": float(np.isfinite(stack[:, 0, 0]).mean()),
            "hand_frac": float((np.isfinite(stack[:, 33, 0]) | np.isfinite(stack[:, 54, 0])).mean())}
    xyz = np.concatenate(frames)
    pd.DataFrame({"frame": np.repeat(np.arange(T), N_LM), "type": np.tile(TYPES, T),
                  "landmark_index": np.tile(INDEX, T), "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
                 ).to_parquet(out, index=False)
    meta_path.write_text(json.dumps(meta))  # last: a meta file means a complete extraction
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
    failed = []
    with Pool(a.workers) as pool:
        for m in pool.imap(extract, jobs):
            if "error" in m:
                failed.append(m)
                print(f"FAILED {m['video']}: {m['error']}", flush=True)
                continue
            print(f"{Path(m['video']).name}: {m['n_frames']} frames, crop {m['width']}x{m['height']} of "
                  f"{m['source_width']}x{m['source_height']}, fps {m['fps_measured'] or 0:.1f}, pose {m['pose_frac']:.0%} "
                  f"hands {m['hand_frac']:.0%}" + (" (mirrored in file: flipped back)" if m["mirrored_in_file"] else ""), flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} of {len(jobs)} videos failed")

if __name__ == "__main__":
    main()
