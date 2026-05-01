"""
Extract MediaPipe Holistic landmarks from ASL video datasets → parquet files.

Supports WLASL and MSASL. Output format matches Google ASL Signs so the
AnatomicalConformer pipeline can be trained or evaluated on either dataset.

Required extras (not in main deps):
    pip install mediapipe opencv-python-headless

── MSASL ────────────────────────────────────────────────────────────────────
    # Report sign overlap with Google ASL Signs (no video processing)
    python research/tools/preprocess_landmarks.py msasl --compare-labels \\
        --ann-dir  data/MSASL/annotations \\
        --sign-map data/Isolated_ASL_Recognition/sign_to_prediction_index_map.json

    # Extract landmarks
    python research/tools/preprocess_landmarks.py msasl \\
        --video-dir data/MSASL/videos \\
        --ann-dir   data/MSASL/annotations \\
        --out-dir   data/MSASL_Landmarks --split 200

── WLASL ────────────────────────────────────────────────────────────────────
    # Extract landmarks (full dataset or a gloss range for multi-session runs)
    python research/tools/preprocess_landmarks.py wlasl \\
        --video-dir data/WLASL/videos \\
        --ann-json  data/WLASL/WLASL_v0.3.json \\
        --out-dir   data/WLASL_Landmarks [--start-gloss 0 --end-gloss 500]

    # Merge batch CSVs after all sessions complete
    python research/tools/preprocess_landmarks.py wlasl --merge-batches \\
        --out-dir data/WLASL_Landmarks
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

# ── Shared constants ──────────────────────────────────────────────────────────

_VIDEO_EXTS = [".mp4", ".mov", ".avi", ".webm", ".mkv"]
_MP_COMPLEXITY      = 1
_MP_DETECTION_CONF  = 0.5
_MP_TRACKING_CONF   = 0.5
_N_POSE, _N_HAND, _N_FACE = 33, 21, 468

# ── Shared MediaPipe extraction ───────────────────────────────────────────────

_holistic = None  # per-worker instance


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
        return [{"frame": frame_idx, "type": lm_type, "landmark_index": i,
                 "x": lm.x, "y": lm.y, "z": lm.z}
                for i, lm in enumerate(lm_list.landmark)]
    return [{"frame": frame_idx, "type": lm_type, "landmark_index": i,
             "x": np.nan, "y": np.nan, "z": np.nan}
            for i in range(n)]


def _find_video(video_dir: str, file_id: str) -> Optional[str]:
    for ext in _VIDEO_EXTS:
        p = Path(video_dir) / f"{file_id}{ext}"
        if p.exists():
            return str(p)
    for ext in _VIDEO_EXTS:
        hits = list(Path(video_dir).rglob(f"{file_id}{ext}"))
        if hits:
            return str(hits[0])
    return None


def _extract_landmarks(video_path: str, start_frame: int, end_frame: int) -> Optional[pd.DataFrame]:
    """
    Read a video and return a landmark DataFrame.

    start_frame / end_frame are the original annotation indices.  If
    start_frame >= total_frames the clip is assumed to be pre-trimmed and is
    read from the beginning without a frame-range restriction.
    """
    global _holistic
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pre_trimmed = (start_frame >= total)
    if not pre_trimmed and start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    rows: List[dict] = []
    out_idx = 0
    while True:
        if not pre_trimmed and end_frame > 0 and int(cap.get(cv2.CAP_PROP_POS_FRAMES)) > end_frame:
            break
        ret, frame = cap.read()
        if not ret:
            break
        res = _holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        rows.extend(_lm_rows(out_idx, res.pose_landmarks,       "pose",       _N_POSE))
        rows.extend(_lm_rows(out_idx, res.left_hand_landmarks,  "left_hand",  _N_HAND))
        rows.extend(_lm_rows(out_idx, res.right_hand_landmarks, "right_hand", _N_HAND))
        rows.extend(_lm_rows(out_idx, res.face_landmarks,       "face",       _N_FACE))
        out_idx += 1

    cap.release()
    return pd.DataFrame(rows) if rows else None


def _process_one(args: Tuple) -> Optional[Dict]:
    """Worker function — shared by both MSASL and WLASL."""
    video_dir, file_id, participant_id, sign, start_frame, end_frame, out_landmarks_dir = args

    parquet_name = f"{participant_id}_{file_id}.parquet"
    parquet_path = Path(out_landmarks_dir) / parquet_name
    rel_path     = f"train_landmark_files/{parquet_name}"

    if parquet_path.exists():
        return {"path": rel_path, "sign": sign, "participant_id": participant_id}

    video_path = _find_video(video_dir, file_id)
    if video_path is None:
        return None

    try:
        df = _extract_landmarks(video_path, start_frame, end_frame)
        if df is None or df.empty:
            return None
        df.to_parquet(parquet_path, index=False)
        return {"path": rel_path, "sign": sign, "participant_id": participant_id}
    except Exception as e:
        print(f"  ERROR {file_id}: {e}", file=sys.stderr)
        return None


def _run_extraction(tasks: List[Tuple], num_workers: int) -> List[Dict]:
    already = sum(1 for t in tasks if (Path(t[-1]) / f"{t[2]}_{t[1]}.parquet").exists())
    print(f"Processing {len(tasks)} clips ({already} already done) with {num_workers} workers...")
    t0 = time.time()
    with Pool(processes=num_workers, initializer=_init_worker) as pool:
        results = list(pool.imap(_process_one, tasks, chunksize=8))
    elapsed = time.time() - t0
    records = [r for r in results if r is not None]
    print(f"Done in {elapsed / 60:.1f} min — {len(records)}/{len(tasks)} succeeded, "
          f"{len(tasks) - len(records)} failed/missing")
    return records


# ── MSASL ─────────────────────────────────────────────────────────────────────

_MSASL_ANN_FILES = ["MSASL_train.json", "MSASL_val.json", "MSASL_test.json"]


def _load_msasl(ann_dir: str, split: int) -> pd.DataFrame:
    ann_dir = Path(ann_dir)
    records = []
    for fname in _MSASL_ANN_FILES:
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
                    "participant_id": int(e.get("signer_id", 0)),
                    "start_frame": int(e.get("start_frame_id", 0)),
                    "end_frame":   int(e.get("end_frame_id", -1)),
                })
    if not records:
        sys.exit(f"No MSASL annotations found in {ann_dir}")
    df = pd.DataFrame(records)
    print(f"MSASL-{split}: {len(df)} samples, {df['sign'].nunique()} signs")
    return df


def cmd_msasl(args):
    if args.compare_labels:
        if not args.sign_map:
            sys.exit("--sign-map is required with --compare-labels")
        _msasl_compare_labels(args.ann_dir, args.sign_map, args.split)
        return

    if not _MP_AVAILABLE:
        sys.exit("mediapipe and opencv-python are required.\n  pip install mediapipe opencv-python-headless")
    if not args.video_dir or not args.out_dir:
        sys.exit("--video-dir and --out-dir are required for landmark extraction")

    ann_df = _load_msasl(args.ann_dir, args.split)
    out_dir = Path(args.out_dir)
    landmarks_dir = out_dir / "train_landmark_files"
    landmarks_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (args.video_dir, row.file, row.participant_id, row.sign,
         row.start_frame, row.end_frame, str(landmarks_dir))
        for row in ann_df.itertuples(index=False)
    ]
    records = _run_extraction(tasks, args.num_workers)
    if not records:
        sys.exit("No clips processed. Check --video-dir.")

    pd.DataFrame(records).to_csv(out_dir / "train.csv", index=False)
    sign_map = ann_df[["sign", "label"]].drop_duplicates().set_index("sign")["label"].to_dict()
    with open(out_dir / "sign_to_prediction_index_map.json", "w") as f:
        json.dump(sign_map, f, indent=2, sort_keys=True)
    print(f"train.csv + sign map → {out_dir}  ({len(sign_map)} signs)")


def _msasl_compare_labels(ann_dir: str, sign_map_path: str, split: int) -> None:
    df = _load_msasl(ann_dir, split)
    msasl_signs = set(df["sign"].unique())

    with open(sign_map_path) as f:
        raw = json.load(f)
    google_signs = {k.lower().strip() for k in raw}
    google_map   = {k.lower().strip(): v for k, v in raw.items()}

    overlap     = msasl_signs & google_signs
    msasl_only  = msasl_signs - google_signs
    google_only = google_signs - msasl_signs

    print(f"\n── MSASL-{split} vs Google ASL Signs ──────────────────────────────")
    print(f"  MSASL-{split} signs:  {len(msasl_signs)}")
    print(f"  Google ASL signs: {len(google_signs)}")
    print(f"  Overlap:          {len(overlap)}  "
          f"({100 * len(overlap) / len(msasl_signs):.0f}% of MSASL-{split}, "
          f"{100 * len(overlap) / len(google_signs):.0f}% of Google ASL)")
    print(f"\nOverlapping signs ({len(overlap)}): {', '.join(sorted(overlap))}")
    print(f"\nMSASL-only ({len(msasl_only)}): {', '.join(sorted(msasl_only))}")
    print(f"\nGoogle-only ({len(google_only)}): {', '.join(sorted(google_only))}")

    overlap_map_path = Path(sign_map_path).parent / "msasl_to_google_index.json"
    with open(overlap_map_path, "w") as f:
        json.dump({s: google_map[s] for s in overlap}, f, indent=2, sort_keys=True)
    print(f"\nOverlap map (for cross-dataset eval) → {overlap_map_path}")


# ── WLASL ─────────────────────────────────────────────────────────────────────

def _load_wlasl(ann_json: str, start_gloss: int, end_gloss: Optional[int]) -> pd.DataFrame:
    with open(ann_json) as f:
        data = json.load(f)
    if end_gloss is None:
        end_gloss = len(data)
    records = []
    for entry in data[start_gloss:end_gloss]:
        gloss = entry["gloss"].lower().strip()
        for inst in entry.get("instances", []):
            records.append({
                "sign":        gloss,
                "file":        inst.get("video_id", ""),
                "participant_id": int(inst.get("signer_id", 0)),
                "start_frame": 0,
                "end_frame":   -1,  # WLASL clips are pre-trimmed
            })
    df = pd.DataFrame(records)
    print(f"WLASL glosses {start_gloss}–{end_gloss}: {len(df)} clips, {df['sign'].nunique()} signs")
    return df


def cmd_wlasl(args):
    if args.merge_batches:
        _wlasl_merge_batches(args.out_dir)
        return

    if not _MP_AVAILABLE:
        sys.exit("mediapipe and opencv-python are required.\n  pip install mediapipe opencv-python-headless")
    if not args.video_dir or not args.ann_json or not args.out_dir:
        sys.exit("--video-dir, --ann-json, and --out-dir are required")

    ann_df = _load_wlasl(args.ann_json, args.start_gloss, args.end_gloss)
    out_dir = Path(args.out_dir)
    landmarks_dir = out_dir / "train_landmark_files"
    landmarks_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (args.video_dir, row.file, row.participant_id, row.sign,
         row.start_frame, row.end_frame, str(landmarks_dir))
        for row in ann_df.itertuples(index=False)
        if row.file  # skip entries without a video_id
    ]
    records = _run_extraction(tasks, args.num_workers)
    if not records:
        sys.exit("No clips processed. Check --video-dir.")

    train_df = pd.DataFrame(records)

    # Batch suffix when processing a gloss range; otherwise final train.csv
    with open(args.ann_json) as f:
        total_glosses = len(json.load(f))
    end = args.end_gloss if args.end_gloss is not None else total_glosses
    is_partial = args.start_gloss > 0 or end < total_glosses
    csv_name = f"train_batch_{args.start_gloss}_{end}.csv" if is_partial else "train.csv"
    train_df.to_csv(out_dir / csv_name, index=False)

    if not is_partial:
        _wlasl_write_sign_map(train_df, out_dir)

    print(f"{csv_name} → {out_dir}  ({len(train_df)} rows)")
    if is_partial:
        print("Run --merge-batches after all sessions complete to produce train.csv + sign map.")


def _wlasl_write_sign_map(train_df: pd.DataFrame, out_dir: Path) -> None:
    unique_signs = sorted(train_df["sign"].unique())
    sign_map = {s: i for i, s in enumerate(unique_signs)}
    with open(out_dir / "sign_to_prediction_index_map.json", "w") as f:
        json.dump(sign_map, f, indent=2, sort_keys=True)
    print(f"sign_to_prediction_index_map.json → {out_dir}  ({len(sign_map)} signs)")


def _wlasl_merge_batches(out_dir: str) -> None:
    out_dir = Path(out_dir)
    batch_files = sorted(out_dir.glob("train_batch_*.csv"))
    if not batch_files:
        sys.exit(f"No train_batch_*.csv files found in {out_dir}")
    print(f"Merging {len(batch_files)} batch files...")
    merged = pd.concat([pd.read_csv(f) for f in batch_files], ignore_index=True)
    merged = merged.drop_duplicates(subset=["path"])
    merged.to_csv(out_dir / "train.csv", index=False)
    _wlasl_write_sign_map(merged, out_dir)
    print(f"Merged {len(merged)} clips, {merged['sign'].nunique()} signs → {out_dir}/train.csv")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Extract MediaPipe landmarks from ASL video datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="dataset", required=True)

    # ── msasl subcommand ──
    msasl = sub.add_parser("msasl", help="Preprocess MSASL dataset")
    msasl.add_argument("--video-dir")
    msasl.add_argument("--ann-dir",  required=True,
                       help="Directory with MSASL_train/val/test.json")
    msasl.add_argument("--out-dir")
    msasl.add_argument("--sign-map", help="Google ASL sign_to_prediction_index_map.json")
    msasl.add_argument("--split", type=int, default=200, choices=[200, 500, 1000])
    msasl.add_argument("--num-workers", type=int, default=max(1, cpu_count() - 1))
    msasl.add_argument("--compare-labels", action="store_true",
                       help="Print sign overlap with Google ASL Signs and exit")

    # ── wlasl subcommand ──
    wlasl = sub.add_parser("wlasl", help="Preprocess WLASL dataset")
    wlasl.add_argument("--video-dir")
    wlasl.add_argument("--ann-json",  help="Path to WLASL_v0.3.json")
    wlasl.add_argument("--out-dir")
    wlasl.add_argument("--start-gloss", type=int, default=0)
    wlasl.add_argument("--end-gloss",   type=int, default=None,
                       help="Exclusive end index (default: all)")
    wlasl.add_argument("--num-workers", type=int, default=max(1, cpu_count() - 1))
    wlasl.add_argument("--merge-batches", action="store_true",
                       help="Merge train_batch_*.csv files into train.csv and exit")

    args = p.parse_args()
    {"msasl": cmd_msasl, "wlasl": cmd_wlasl}[args.dataset](args)


if __name__ == "__main__":
    main()
