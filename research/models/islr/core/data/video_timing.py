"""Decoded timing of sampled videos per header frame rate: is a "120 fps" video really 120 frames per second?

The fps probe (`extraction_inventory.py --probe-all`) reads container metadata only. This decodes
--n videos of each header-fps class and records, per video: the decoded frame count, each frame's
timestamp (OpenCV CAP_PROP_POS_MSEC, i.e. the stream's presentation times), the median and spread of
the frame intervals, the duration they span, and the share of consecutive frames that are near
duplicates (mean absolute grey-level difference < 0.5 on a 64×64 thumbnail — a 30 fps stream padded to
120 would show ~75 %). With ffprobe available, packet times are read too (a second source).
A genuine 120 fps video: median interval ≈ 8.3 ms, duration ≈ frames / 120, few duplicates.

  PYTHONPATH=research/models python -m islr.core.data.video_timing --probe <fps-popsign>/video_probe.csv \
      --video-root /kaggle/input --n 25 --out {OUT}
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def decode(src: str) -> dict:
    import cv2

    cap = cv2.VideoCapture(src)
    ts, dup, prev = [], 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev is not None and float(np.abs(g - prev).mean()) < 0.5:
            dup += 1
        prev = g
    out = {"header_fps": cap.get(cv2.CAP_PROP_FPS), "header_frames": cap.get(cv2.CAP_PROP_FRAME_COUNT)}
    cap.release()
    iv = np.diff(ts) if len(ts) > 1 else np.array([np.nan])
    out.update({"decoded_frames": len(ts), "median_interval_ms": float(np.median(iv)),
                "p5_interval_ms": float(np.percentile(iv, 5)), "p95_interval_ms": float(np.percentile(iv, 95)),
                "span_s": float((ts[-1] - ts[0]) / 1000) if len(ts) > 1 else np.nan,
                "duplicate_share": dup / max(len(ts) - 1, 1)})
    if shutil.which("ffprobe"):
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                            "-of", "csv=p=0", src], capture_output=True, text=True)
        pts = np.sort([float(x) for x in r.stdout.split() if x.replace(".", "", 1).isdigit()])
        if len(pts) > 1:
            out.update({"ffprobe_packets": len(pts), "ffprobe_median_interval_ms": float(np.median(np.diff(pts)) * 1000),
                        "ffprobe_span_s": float(pts[-1] - pts[0])})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--probe", required=True, type=Path, help="video_probe.csv (path, fps, frames, width, height)")
    p.add_argument("--video-root", required=True, type=Path)
    p.add_argument("--n", type=int, default=25, help="videos per header-fps class")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()

    probe = pd.read_csv(a.probe)
    probe["fps_class"] = probe["fps"].round(0).astype(int)
    pick = probe.groupby("fps_class", group_keys=False).apply(lambda g: g.sample(min(a.n, len(g)), random_state=a.seed))
    want = {Path(x).stem: x for x in pick["path"]}
    found = {}
    for dirpath, _, files in os.walk(a.video_root):
        for f in files:
            if f.endswith(".mp4") and Path(f).stem in want:
                found[want[Path(f).stem]] = os.path.join(dirpath, f)
    rows = []
    for r in pick.itertuples():
        if r.path in found:
            rows.append({"path": r.path, "fps_class": r.fps_class, "probe_frames": r.frames, **decode(found[r.path])})
    res = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out / "video_timing.csv", index=False)
    cols = [c for c in ("decoded_frames", "median_interval_ms", "p5_interval_ms", "p95_interval_ms", "span_s",
                        "duplicate_share", "ffprobe_median_interval_ms", "ffprobe_span_s") if c in res]
    summary = {"sampled": len(pick), "decoded": len(res), "classes": probe["fps_class"].value_counts().to_dict(),
               "per_class_median": res.groupby("fps_class")[cols].median().round(3).to_dict("index")}
    (a.out / "video_timing.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    if len(res) < len(pick):
        raise SystemExit(f"{len(pick) - len(res)} sampled videos not found under {a.video_root}")


if __name__ == "__main__":
    main()
