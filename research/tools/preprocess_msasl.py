"""
Preprocess MSASL videos → MediaPipe Holistic landmarks → parquet files.

Produces the same parquet format as Google ASL Signs so the AnatomicalConformer
pipeline can evaluate cross-dataset generalization on MSASL-200/500/1000.

Required extras (not in main deps):
    pip install mediapipe opencv-python-headless

MSASL annotation JSONs (MSASL_train.json, MSASL_val.json, MSASL_test.json) can
be downloaded from https://github.com/bhavanamahajan/MSASL — place all three in
the same directory and pass that directory as --ann-dir.

Videos: each annotation entry has a `file` field (YouTube video ID). The script
searches --video-dir recursively for `{file}.mp4` (and other extensions). Most
MSASL downloaders pre-trim clips; if a clip is already trimmed, the start/end
frame IDs in the annotation are ignored automatically.

Usage:
    # Report sign overlap with Google ASL Signs (no video processing)
    python research/tools/preprocess_msasl.py --compare-labels \\
        --ann-dir  data/MSASL/annotations \\
        --sign-map data/Isolated_ASL_Recognition/sign_to_prediction_index_map.json \\
        --split 200

    # Extract landmarks from all MSASL-200 videos
    python research/tools/preprocess_msasl.py \\
        --video-dir data/MSASL/videos \\
        --ann-dir   data/MSASL/annotations \\
        --out-dir   data/MSASL_Landmarks \\
        --split 200 --num-workers 4
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import cv2
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

# ── Constants ────────────────────────────────────────────────────────────────

_ANN_FILES = ["MSASL_train.json", "MSASL_val.json", "MSASL_test.json"]
_VIDEO_EXTS = [".mp4", ".mov", ".avi", ".webm", ".mkv"]

# Match the WLASL preprocessor (no iris refinement; all FACE_LANDMARK_SET
# indices are within the standard 0-467 range).
_MP_COMPLEXITY = 1
_MP_DETECTION_CONF = 0.5
_MP_TRACKING_CONF = 0.5
_N_POSE = 33
_N_HAND = 21
_N_FACE = 468


# ── Annotation loading ────────────────────────────────────────────────────────

def load_msasl_annotations(ann_dir: str, split: int) -> pd.DataFrame:
    """Load MSASL_train/val/test.json and filter to top-`split` classes."""
    ann_dir = Path(ann_dir)
    records = []
    for fname in _ANN_FILES:
        fpath = ann_dir / fname
        if not fpath.exists():
            print(f"  Warning: {fpath} not found, skipping")
            continue
        with open(fpath) as f:
            entries = json.load(f)
        for e in entries:
            if e.get("label", split) < split:
                records.append({
                    "sign":        e["clean_text"].lower().strip(),
                    "label":       int(e["label"]),
                    "file":        e["file"],
                    "signer_id":   int(e.get("signer_id", 0)),
                    "start_frame": int(e.get("start_frame_id", 0)),
                    "end_frame":   int(e.get("end_frame_id", -1)),
                    "ann_split":   fname.replace("MSASL_", "").replace(".json", ""),
                })
    if not records:
        sys.exit(f"No annotations found in {ann_dir}. Expected MSASL_train/val/test.json")
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} samples across {df['sign'].nunique()} signs (top-{split})")
    return df


# ── Label alignment ───────────────────────────────────────────────────────────

def compare_labels(ann_dir: str, sign_map_path: str, split: int) -> None:
    """Print sign overlap between MSASL-{split} and Google ASL Signs."""
    df = load_msasl_annotations(ann_dir, split)
    msasl_signs = set(df["sign"].unique())

    with open(sign_map_path) as f:
        google_signs = {k.lower().strip() for k in json.load(f)}

    overlap   = msasl_signs & google_signs
    msasl_only  = msasl_signs - google_signs
    google_only = google_signs - msasl_signs

    print(f"\n── Label overlap: MSASL-{split} vs Google ASL Signs ──────────────")
    print(f"  MSASL-{split} signs:   {len(msasl_signs)}")
    print(f"  Google ASL signs:  {len(google_signs)}")
    print(f"  Overlap:           {len(overlap)}  "
          f"({100 * len(overlap) / len(msasl_signs):.0f}% of MSASL-{split}, "
          f"{100 * len(overlap) / len(google_signs):.0f}% of Google ASL Signs)")
    print(f"\nOverlapping signs ({len(overlap)}):")
    for s in sorted(overlap):
        print(f"    {s}")
    print(f"\nMSASL-only signs ({len(msasl_only)}):")
    for s in sorted(msasl_only):
        print(f"    {s}")
    print(f"\nGoogle-only signs ({len(google_only)}):")
    for s in sorted(google_only):
        print(f"    {s}")

    # Write overlap map: msasl_sign → google_index (useful for cross-dataset eval)
    overlap_map_path = Path(sign_map_path).parent / "msasl_to_google_index.json"
    with open(sign_map_path) as f:
        google_map = {k.lower().strip(): v for k, v in json.load(f).items()}
    overlap_map = {s: google_map[s] for s in overlap}
    with open(overlap_map_path, "w") as f:
        json.dump(overlap_map, f, indent=2, sort_keys=True)
    print(f"\nOverlap map saved → {overlap_map_path}")
    print("Use this to evaluate the Google-trained model on matched MSASL signs.")


# ── Video / landmark extraction ───────────────────────────────────────────────

def _find_video(video_dir: str, file_id: str) -> Optional[str]:
    """Find a video file by base name; direct check first, then recursive."""
    for ext in _VIDEO_EXTS:
        p = Path(video_dir) / f"{file_id}{ext}"
        if p.exists():
            return str(p)
    for ext in _VIDEO_EXTS:
        matches = list(Path(video_dir).rglob(f"{file_id}{ext}"))
        if matches:
            return str(matches[0])
    return None


# Per-worker MediaPipe instance (initialised once per worker process)
_holistic = None


def _init_worker():
    global _holistic
    _holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=_MP_COMPLEXITY,
        min_detection_confidence=_MP_DETECTION_CONF,
        min_tracking_confidence=_MP_TRACKING_CONF,
    )


def _lm_rows(frame_idx: int, lm_list, lm_type: str, n: int) -> List[dict]:
    if lm_list:
        return [
            {"frame": frame_idx, "type": lm_type, "landmark_index": i,
             "x": lm.x, "y": lm.y, "z": lm.z}
            for i, lm in enumerate(lm_list.landmark)
        ]
    return [
        {"frame": frame_idx, "type": lm_type, "landmark_index": i,
         "x": np.nan, "y": np.nan, "z": np.nan}
        for i in range(n)
    ]


def _extract_landmarks(video_path: str, start_frame: int, end_frame: int) -> Optional[pd.DataFrame]:
    """
    Extract MediaPipe landmarks from a video file.

    If start_frame > total frames, the video is assumed to be a pre-trimmed
    clip and is read from the beginning without frame-range restriction.
    """
    global _holistic
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pre_trimmed = start_frame >= total_frames
    if not pre_trimmed and start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    rows: List[dict] = []
    frame_idx = 0
    while True:
        if not pre_trimmed and end_frame > 0:
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if pos > end_frame:
                break
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = _holistic.process(rgb)
        rows.extend(_lm_rows(frame_idx, res.pose_landmarks,       "pose",       _N_POSE))
        rows.extend(_lm_rows(frame_idx, res.left_hand_landmarks,  "left_hand",  _N_HAND))
        rows.extend(_lm_rows(frame_idx, res.right_hand_landmarks, "right_hand", _N_HAND))
        rows.extend(_lm_rows(frame_idx, res.face_landmarks,       "face",       _N_FACE))
        frame_idx += 1

    cap.release()
    return pd.DataFrame(rows) if rows else None


def _process_one(args: Tuple) -> Optional[Dict]:
    video_dir, file_id, sign, signer_id, start_frame, end_frame, out_landmarks_dir = args

    # Unique filename: signer + clip so repeated clips across signers don't collide
    video_name = f"{signer_id}_{file_id}"
    parquet_path = Path(out_landmarks_dir) / f"{video_name}.parquet"
    rel_path = f"train_landmark_files/{video_name}.parquet"

    if parquet_path.exists():
        return {"path": rel_path, "sign": sign, "participant_id": signer_id}

    video_path = _find_video(video_dir, file_id)
    if video_path is None:
        return None

    try:
        df = _extract_landmarks(video_path, start_frame, end_frame)
        if df is None or df.empty:
            return None
        df.to_parquet(parquet_path, index=False)
        return {"path": rel_path, "sign": sign, "participant_id": signer_id}
    except Exception as e:
        print(f"  ERROR {file_id}: {e}", file=sys.stderr)
        return None


# ── Main pipeline ─────────────────────────────────────────────────────────────

def preprocess(
    video_dir: str,
    ann_dir: str,
    out_dir: str,
    split: int,
    num_workers: int,
) -> None:
    if not _MP_AVAILABLE:
        sys.exit(
            "mediapipe and opencv-python are required.\n"
            "  pip install mediapipe opencv-python-headless"
        )

    ann_df = load_msasl_annotations(ann_dir, split)

    out_dir = Path(out_dir)
    landmarks_dir = out_dir / "train_landmark_files"
    landmarks_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (video_dir, row.file, row.sign, row.signer_id,
         row.start_frame, row.end_frame, str(landmarks_dir))
        for row in ann_df.itertuples(index=False)
    ]

    already_done = sum(
        1 for _, fid, _, sid, *_ , ld in tasks
        if (Path(ld) / f"{sid}_{fid}.parquet").exists()
    )
    print(f"Processing {len(tasks)} clips ({already_done} already done) "
          f"with {num_workers} workers...")

    t0 = time.time()
    with Pool(processes=num_workers, initializer=_init_worker) as pool:
        results = list(pool.imap(_process_one, tasks, chunksize=8))
    elapsed = time.time() - t0

    records = [r for r in results if r is not None]
    failed = len(tasks) - len(records)
    print(f"\nDone in {elapsed / 60:.1f} min — "
          f"{len(records)}/{len(tasks)} succeeded, {failed} failed/missing")

    if not records:
        sys.exit("No clips processed. Check --video-dir and that videos are present.")

    # train.csv — participant_id enables signer-independent split in get_data_loaders
    train_df = pd.DataFrame(records)
    train_df.to_csv(out_dir / "train.csv", index=False)

    # sign_to_prediction_index_map.json using original MSASL label indices
    sign_map = (
        ann_df[["sign", "label"]]
        .drop_duplicates()
        .set_index("sign")["label"]
        .to_dict()
    )
    with open(out_dir / "sign_to_prediction_index_map.json", "w") as f:
        json.dump(sign_map, f, indent=2, sort_keys=True)

    print(f"train.csv          → {out_dir}/train.csv  ({len(train_df)} rows)")
    print(f"sign_to_index map  → {out_dir}/sign_to_prediction_index_map.json  ({len(sign_map)} signs)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Preprocess MSASL videos → MediaPipe landmarks (parquet)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video-dir",  help="Root directory containing MSASL video files")
    p.add_argument("--ann-dir",    required=True,
                   help="Directory with MSASL_train/val/test.json")
    p.add_argument("--out-dir",    help="Output directory for parquets + train.csv")
    p.add_argument("--sign-map",
                   help="Google ASL sign_to_prediction_index_map.json (--compare-labels only)")
    p.add_argument("--split",      type=int, default=200, choices=[200, 500, 1000],
                   help="Use top-N MSASL classes")
    p.add_argument("--num-workers", type=int, default=max(1, cpu_count() - 1),
                   help="Parallel worker processes")
    p.add_argument("--compare-labels", action="store_true",
                   help="Print sign overlap with Google ASL Signs and exit (no video processing)")
    args = p.parse_args()

    if args.compare_labels:
        if not args.sign_map:
            p.error("--sign-map is required with --compare-labels")
        compare_labels(args.ann_dir, args.sign_map, args.split)
        return

    if not args.video_dir or not args.out_dir:
        p.error("--video-dir and --out-dir are required for landmark extraction")

    preprocess(args.video_dir, args.ann_dir, args.out_dir, args.split, args.num_workers)


if __name__ == "__main__":
    main()
