# # Preprocess WLASL Videos with MediaPipe (Parallelized)
#
# This notebook extracts MediaPipe landmarks from WLASL videos to match the Google ASL competition format.
#
# **Optimizations**:
# - Uses multiprocessing (4 workers) for ~3-4x speedup
# - Supports batch processing with start/end indices for multi-session runs
# - Checkpointing to resume interrupted runs
#
# **Expected Time**: ~10-12 hours for full dataset (vs 40 hours sequential)
#
# **Key Requirements**:
# 1. Use MediaPipe Holistic (same as competition)
# 2. Extract: pose (33), left_hand (21), right_hand (21), face (468) landmarks
# 3. Save as parquet with same column format: frame, type, landmark_index, x, y, z


# Install MediaPipe if needed
# !pip install mediapipe opencv-python-headless pyarrow --quiet

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
import time

print(f"MediaPipe version: {mp.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"CPU cores available: {cpu_count()}")

# ============== CONFIGURATION ==============

# WLASL dataset paths (adjust based on your Kaggle dataset structure)
WLASL_VIDEO_DIR = "/kaggle/input/wlasl-complete/videos"  # or wherever videos are
WLASL_JSON = "/kaggle/input/wlasl-complete/WLASL_v0.3.json"  # metadata

# Output directory for processed landmarks
OUTPUT_DIR = "/kaggle/working/wlasl_landmarks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parallelization settings
NUM_WORKERS = 4  # Kaggle typically has 4 CPU cores

# Batch processing settings (for multi-session runs)
# Set these to process a subset of glosses
# Session 1: START_GLOSS=0, END_GLOSS=500
# Session 2: START_GLOSS=500, END_GLOSS=1000
# Session 3: START_GLOSS=1000, END_GLOSS=1500
# Session 4: START_GLOSS=1500, END_GLOSS=2000
START_GLOSS = 0  # Starting gloss index (inclusive)
END_GLOSS = None  # Ending gloss index (exclusive), None = process all

# MediaPipe settings (match Google competition)
MP_MODEL_COMPLEXITY = 1  # 0, 1, or 2
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE = 0.5


def init_worker():
    """
    Initialize MediaPipe Holistic for each worker process.
    Called once per worker when the pool is created.
    """
    global holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=MP_MODEL_COMPLEXITY,
        min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
    )


def extract_landmarks_from_video(video_path: str) -> Optional[pd.DataFrame]:
    """
    Extract MediaPipe landmarks from a video file.
    Uses the global holistic model initialized by init_worker.
    """
    global holistic

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    all_rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        # Extract pose landmarks (33)
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                all_rows.append(
                    {
                        "frame": frame_idx,
                        "type": "pose",
                        "landmark_index": i,
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                    }
                )
        else:
            for i in range(33):
                all_rows.append(
                    {
                        "frame": frame_idx,
                        "type": "pose",
                        "landmark_index": i,
                        "x": np.nan,
                        "y": np.nan,
                        "z": np.nan,
                    }
                )

        # Extract left hand landmarks (21)
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                all_rows.append(
                    {
                        "frame": frame_idx,
                        "type": "left_hand",
                        "landmark_index": i,
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                    }
                )
        else:
            for i in range(21):
                all_rows.append(
                    {
                        "frame": frame_idx,
                        "type": "left_hand",
                        "landmark_index": i,
                        "x": np.nan,
                        "y": np.nan,
                        "z": np.nan,
                    }
                )

        # Extract right hand landmarks (21)
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                all_rows.append(
                    {
                        "frame": frame_idx,
                        "type": "right_hand",
                        "landmark_index": i,
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                    }
                )
        else:
            for i in range(21):
                all_rows.append(
                    {
                        "frame": frame_idx,
                        "type": "right_hand",
                        "landmark_index": i,
                        "x": np.nan,
                        "y": np.nan,
                        "z": np.nan,
                    }
                )

        # Extract face landmarks (468)
        if results.face_landmarks:
            for i, lm in enumerate(results.face_landmarks.landmark):
                all_rows.append(
                    {
                        "frame": frame_idx,
                        "type": "face",
                        "landmark_index": i,
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                    }
                )
        else:
            for i in range(468):
                all_rows.append(
                    {
                        "frame": frame_idx,
                        "type": "face",
                        "landmark_index": i,
                        "x": np.nan,
                        "y": np.nan,
                        "z": np.nan,
                    }
                )

        frame_idx += 1

    cap.release()

    if len(all_rows) == 0:
        return None

    df = pd.DataFrame(all_rows)
    df["row_id"] = (
        df["frame"].astype(str)
        + "-"
        + df["type"]
        + "-"
        + df["landmark_index"].astype(str)
    )
    return df


def process_single_video(args: Tuple[str, str, str, str]) -> Optional[Dict]:
    """
    Process a single video file. Called by multiprocessing pool.

    Args:
        args: (video_path, video_id, gloss, output_dir)

    Returns:
        Record dict or None if failed
    """
    video_path, video_id, gloss, landmark_dir = args

    try:
        # Check if already processed (for resume capability)
        parquet_filename = f"{video_id}.parquet"
        parquet_path = os.path.join(landmark_dir, parquet_filename)

        if os.path.exists(parquet_path):
            # Already processed, just return the record
            existing_df = pd.read_parquet(parquet_path)
            return {
                "path": f"train_landmark_files/{parquet_filename}",
                "sign": gloss,
                "video_id": video_id,
                "n_frames": existing_df["frame"].nunique(),
            }

        # Extract landmarks
        df = extract_landmarks_from_video(video_path)

        if df is None or len(df) == 0:
            return None

        # Save as parquet
        df.to_parquet(parquet_path, index=False)

        return {
            "path": f"train_landmark_files/{parquet_filename}",
            "sign": gloss,
            "video_id": video_id,
            "n_frames": df["frame"].nunique(),
        }

    except Exception as e:
        return None


def process_wlasl_parallel(
    video_dir: str,
    metadata_json: str,
    output_dir: str,
    start_gloss: int = 0,
    end_gloss: Optional[int] = None,
    num_workers: int = 4,
) -> pd.DataFrame:
    """
    Process WLASL videos in parallel using multiprocessing.
    """
    # Load metadata
    with open(metadata_json, "r") as f:
        wlasl_data = json.load(f)

    total_glosses = len(wlasl_data)
    if end_gloss is None:
        end_gloss = total_glosses

    print(f"WLASL metadata loaded: {total_glosses} total glosses")
    print(
        f"Processing glosses {start_gloss} to {end_gloss} ({end_gloss - start_gloss} glosses)"
    )

    # Create output directory
    landmark_dir = os.path.join(output_dir, "train_landmark_files")
    os.makedirs(landmark_dir, exist_ok=True)

    # Build list of all videos to process
    video_tasks = []

    for gloss_entry in wlasl_data[start_gloss:end_gloss]:
        gloss = gloss_entry["gloss"]
        instances = gloss_entry.get("instances", [])

        for instance in instances:
            video_id = instance.get("video_id", "")

            # Find video file
            video_path = None
            for ext in [".mp4", ".mov", ".avi", ".webm"]:
                potential_path = os.path.join(video_dir, f"{video_id}{ext}")
                if os.path.exists(potential_path):
                    video_path = potential_path
                    break

            if video_path:
                video_tasks.append((video_path, video_id, gloss, landmark_dir))

    print(f"Found {len(video_tasks)} videos to process")

    # Process in parallel
    start_time = time.time()
    processed_records = []

    with Pool(processes=num_workers, initializer=init_worker) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_video, video_tasks),
                total=len(video_tasks),
                desc="Processing videos",
            )
        )

    # Filter successful results
    processed_records = [r for r in results if r is not None]

    elapsed = time.time() - start_time
    print(f"\nProcessing completed in {elapsed/3600:.2f} hours")
    print(
        f"Successfully processed: {len(processed_records)} / {len(video_tasks)} videos"
    )

    # Save results
    train_df = pd.DataFrame(processed_records)

    # Save with batch suffix if doing partial processing
    if start_gloss > 0 or end_gloss < total_glosses:
        csv_filename = f"train_batch_{start_gloss}_{end_gloss}.csv"
    else:
        csv_filename = "train.csv"

    train_csv_path = os.path.join(output_dir, csv_filename)
    train_df.to_csv(train_csv_path, index=False)

    print(f"\nSaved to: {train_csv_path}")
    print(f"Unique signs: {train_df['sign'].nunique()}")

    return train_df


# Check if WLASL dataset is available and process
if os.path.exists(WLASL_VIDEO_DIR) and os.path.exists(WLASL_JSON):
    print("WLASL dataset found!")
    print(f"Using {NUM_WORKERS} parallel workers")
    print()

    train_df = process_wlasl_parallel(
        video_dir=WLASL_VIDEO_DIR,
        metadata_json=WLASL_JSON,
        output_dir=OUTPUT_DIR,
        start_gloss=START_GLOSS,
        end_gloss=END_GLOSS,
        num_workers=NUM_WORKERS,
    )
else:
    print("WLASL dataset not found. Please check paths:")
    print(f"  Video dir: {WLASL_VIDEO_DIR}")
    print(f"  Metadata: {WLASL_JSON}")


# If you ran multiple batches, merge them here
def merge_batch_csvs(output_dir: str) -> pd.DataFrame:
    """
    Merge all batch CSV files into a single train.csv
    """
    batch_files = sorted(Path(output_dir).glob("train_batch_*.csv"))

    if not batch_files:
        print("No batch files found")
        return None

    print(f"Found {len(batch_files)} batch files:")
    for f in batch_files:
        print(f"  - {f.name}")

    # Merge all batches
    dfs = [pd.read_csv(f) for f in batch_files]
    merged_df = pd.concat(dfs, ignore_index=True)

    # Remove duplicates (in case of overlapping batches)
    merged_df = merged_df.drop_duplicates(subset=["video_id"])

    # Save merged file
    merged_path = os.path.join(output_dir, "train.csv")
    merged_df.to_csv(merged_path, index=False)

    print(f"\nMerged {len(merged_df)} videos")
    print(f"Unique signs: {merged_df['sign'].nunique()}")
    print(f"Saved to: {merged_path}")

    return merged_df


# Uncomment to merge batches:
# merged_df = merge_batch_csvs(OUTPUT_DIR)

# Verify output format
train_csv_path = os.path.join(OUTPUT_DIR, "train.csv")
if not os.path.exists(train_csv_path):
    # Try batch file
    batch_files = list(Path(OUTPUT_DIR).glob("train_batch_*.csv"))
    if batch_files:
        train_csv_path = str(batch_files[0])

if os.path.exists(train_csv_path):
    train_df = pd.read_csv(train_csv_path)
    print("Train CSV columns:", train_df.columns.tolist())
    print(f"Total videos: {len(train_df)}")
    print(f"Unique signs: {train_df['sign'].nunique()}")
    print(f"\nSample rows:")
    print(train_df.head())

    # Check a parquet file
    sample_path = train_df["path"].iloc[0]
    sample_df = pd.read_parquet(os.path.join(OUTPUT_DIR, sample_path))
    print(f"\nParquet columns: {sample_df.columns.tolist()}")
    print(f"Landmark types: {sample_df['type'].unique()}")
    print(f"Frames: {sample_df['frame'].nunique()}")

# Create sign-to-index mapping
train_csv_path = os.path.join(OUTPUT_DIR, "train.csv")
if os.path.exists(train_csv_path):
    train_df = pd.read_csv(train_csv_path)

    unique_signs = sorted(train_df["sign"].unique())
    sign_to_index = {sign: idx for idx, sign in enumerate(unique_signs)}

    with open(os.path.join(OUTPUT_DIR, "sign_to_prediction_index_map.json"), "w") as f:
        json.dump(sign_to_index, f, indent=2)

    print(f"Created sign mapping with {len(sign_to_index)} signs")
    print(f"Sample signs: {list(sign_to_index.keys())[:10]}")

# ## Multi-Session Processing Guide
#
# If you need to split processing across multiple Kaggle sessions:
#
# **Session 1** (glosses 0-500):
# ```python
# START_GLOSS = 0
# END_GLOSS = 500
# ```
#
# **Session 2** (glosses 500-1000):
# ```python
# START_GLOSS = 500
# END_GLOSS = 1000
# ```
#
# **Session 3** (glosses 1000-1500):
# ```python
# START_GLOSS = 1000
# END_GLOSS = 1500
# ```
#
# **Session 4** (glosses 1500-2000):
# ```python
# START_GLOSS = 1500
# END_GLOSS = 2000
# ```
#
# After all sessions complete, download all `train_landmark_files/` and batch CSVs, then run `merge_batch_csvs()` to combine them.


print("Done! WLASL landmarks extracted and saved.")
