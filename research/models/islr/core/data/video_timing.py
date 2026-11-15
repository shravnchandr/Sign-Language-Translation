"""Decoded timing of sampled videos per header frame rate: is a "120 fps" video really 120 frames per second?

The fps probe (`extraction_inventory.py --probe-all`) reads container metadata only. This decodes
--n videos of each header-fps class and records, per video: the decoded frame count, each frame's
timestamp (OpenCV CAP_PROP_POS_MSEC, i.e. the stream's presentation times), the median and spread of
the frame intervals, the duration they span, and the share of consecutive frames that are near
duplicates (mean absolute grey-level difference < 0.5 on a 64×64 thumbnail — a 30 fps stream padded to
120 would show ~75 %). With ffprobe available, packet times are read too (a second source).
A genuine 120 fps video: median interval ≈ 8.3 ms, duration ≈ frames / 120, few duplicates.
Every frame's timestamp and duplicate flag is kept (video_timing_frames.csv). A video counts as
decoded only if it gives >= 2 frames, at least 95 % of the header's frame count, and finite, strictly
increasing timestamps; anything else is listed as a failure and the run exits non-zero.

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


MIN_DECODED = 0.95  # decoded frames / header frame count below this = a truncated decode


def decode(src: str) -> tuple[dict, list, str | None]:
    """→ (summary, per-frame rows (frame, ts_ms, duplicate), problem or None)."""
    import cv2

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return {}, [], "could not open"
    ts, dups, prev = [], [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
        dups.append(prev is not None and float(np.abs(g - prev).mean()) < 0.5)
        prev = g
    out = {"header_fps": cap.get(cv2.CAP_PROP_FPS), "header_frames": cap.get(cv2.CAP_PROP_FRAME_COUNT)}
    cap.release()
    frames = [{"frame": i, "ts_ms": t, "duplicate": d} for i, (t, d) in enumerate(zip(ts, dups))]
    t = np.asarray(ts, dtype=float)
    if len(t) < 2:
        return out, frames, f"decoded {len(t)} frames"
    if out["header_frames"] > 0 and len(t) < MIN_DECODED * out["header_frames"]:
        return out, frames, f"truncated: decoded {len(t)} of {int(out['header_frames'])} header frames"
    if not np.isfinite(t).all() or not (np.diff(t) > 0).all():
        return out, frames, "timestamps not finite and strictly increasing"
    iv = np.diff(t)
    dup = int(sum(dups))
    out.update({"decoded_frames": len(ts), "median_interval_ms": float(np.median(iv)),
                "p5_interval_ms": float(np.percentile(iv, 5)), "p95_interval_ms": float(np.percentile(iv, 95)),
                "span_s": float((ts[-1] - ts[0]) / 1000) if len(ts) > 1 else np.nan,
                "duplicate_share": dup / (len(ts) - 1)})
    if shutil.which("ffprobe"):
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                            "-of", "csv=p=0", src], capture_output=True, text=True)
        pts = np.sort([float(x) for x in r.stdout.split() if x.replace(".", "", 1).isdigit()])
        if len(pts) > 1:
            out.update({"ffprobe_packets": len(pts), "ffprobe_median_interval_ms": float(np.median(np.diff(pts)) * 1000),
                        "ffprobe_span_s": float(pts[-1] - pts[0])})
    return out, frames, None


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
    rows, per_frame, failures = [], [], []
    for r in pick.itertuples():
        if r.path not in found:
            failures.append({"path": r.path, "fps_class": r.fps_class, "problem": "video not found"})
            continue
        summary, frames, problem = decode(found[r.path])
        per_frame += [{"path": r.path, "fps_class": r.fps_class, **f} for f in frames]
        if problem:
            failures.append({"path": r.path, "fps_class": r.fps_class, "problem": problem})
        else:
            rows.append({"path": r.path, "fps_class": r.fps_class, "probe_frames": r.frames, **summary})
    res = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out / "video_timing.csv", index=False)
    pd.DataFrame(per_frame).to_csv(a.out / "video_timing_frames.csv", index=False)
    pd.DataFrame(failures, columns=["path", "fps_class", "problem"]).to_csv(a.out / "video_timing_failures.csv", index=False)
    cols = [c for c in ("decoded_frames", "median_interval_ms", "p5_interval_ms", "p95_interval_ms", "span_s",
                        "duplicate_share", "ffprobe_median_interval_ms", "ffprobe_span_s") if c in res]
    summary = {"sampled": len(pick), "decoded": len(res), "failures": failures[:20],
               "classes": probe["fps_class"].value_counts().to_dict(),
               "per_class_median": res.groupby("fps_class")[cols].median().round(3).to_dict("index") if len(res) else {}}
    (a.out / "video_timing.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    if failures:
        raise SystemExit(f"{len(failures)} of {len(pick)} sampled videos not decoded usably (video_timing_failures.csv)")


if __name__ == "__main__":
    main()
