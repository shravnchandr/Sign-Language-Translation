# Convert ASL Fingerspelling from Wide Format to Long Format
#
# This script converts the fingerspelling dataset (wide format) to the same
# long format used by Isolated ASL and WLASL datasets.
#
# Wide format: columns are x_face_0, y_face_0, z_face_0, x_left_hand_0, ...
# Long format: columns are frame, type, landmark_index, x, y, z
#
# This allows us to use a single data loader for all datasets.

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
import gc

# ============== CONFIGURATION ==============

IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:
    INPUT_DIR = "/kaggle/input/asl-fingerspelling"
    INPUT_CSV = "/kaggle/input/asl-fingerspelling/train.csv"
    OUTPUT_DIR = "/kaggle/working/fingerspelling_landmarks"
else:
    INPUT_DIR = "/Users/shravnchandr/Projects/Isolated_Sign_Language_Recognition/data/ASL_Fingerspelling_Recognition"
    INPUT_CSV = f"{INPUT_DIR}/train.csv"
    OUTPUT_DIR = "/Users/shravnchandr/Projects/Isolated_Sign_Language_Recognition/data/Fingerspelling_Long_Format"

os.makedirs(OUTPUT_DIR, exist_ok=True)
LANDMARK_DIR = os.path.join(OUTPUT_DIR, "train_landmark_files")
os.makedirs(LANDMARK_DIR, exist_ok=True)

# Parallelization
NUM_WORKERS = 4 if IS_KAGGLE else 2

# Batch processing (for multi-session runs on Kaggle)
START_IDX = 0  # Starting row index (inclusive)
END_IDX = None  # Ending row index (exclusive), None = process all

print(f"Environment: {'Kaggle' if IS_KAGGLE else 'Local'}")
print(f"Input: {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Workers: {NUM_WORKERS}")


# ============== WIDE TO LONG CONVERSION ==============


def parse_wide_column(col_name):
    """
    Parse wide format column name to extract type, landmark_index, and coordinate.

    Examples:
        'x_face_0' -> ('face', 0, 'x')
        'y_left_hand_15' -> ('left_hand', 15, 'y')
        'z_pose_33' -> ('pose', 33, 'z')
    """
    # Pattern: {coord}_{type}_{index} where type can have underscores (left_hand, right_hand)
    match = re.match(r"^([xyz])_(.+)_(\d+)$", col_name)
    if match:
        coord = match.group(1)
        type_and_idx = match.group(2) + "_" + match.group(3)

        # Now extract type and index
        # Types: face, pose, left_hand, right_hand
        for lm_type in ["left_hand", "right_hand", "face", "pose"]:
            if type_and_idx.startswith(lm_type + "_"):
                idx_str = type_and_idx[len(lm_type) + 1 :]
                return (lm_type, int(idx_str), coord)

    return None


def convert_sequence_to_long(seq_df, sequence_id):
    """
    Convert a single sequence from wide to long format.

    Args:
        seq_df: DataFrame with wide format columns (one row per frame)
        sequence_id: Unique identifier for this sequence

    Returns:
        DataFrame in long format with columns: frame, type, landmark_index, x, y, z
    """
    rows = []

    # Get all landmark columns (x_*, y_*, z_*)
    coord_cols = [c for c in seq_df.columns if c.startswith(("x_", "y_", "z_"))]

    # Group columns by (type, landmark_index)
    landmarks = {}
    for col in coord_cols:
        parsed = parse_wide_column(col)
        if parsed:
            lm_type, lm_idx, coord = parsed
            key = (lm_type, lm_idx)
            if key not in landmarks:
                landmarks[key] = {}
            landmarks[key][coord] = col

    # Convert each frame
    for frame_idx, (_, frame_row) in enumerate(seq_df.iterrows()):
        for (lm_type, lm_idx), coord_cols_dict in landmarks.items():
            x = (
                frame_row.get(coord_cols_dict.get("x"), np.nan)
                if "x" in coord_cols_dict
                else np.nan
            )
            y = (
                frame_row.get(coord_cols_dict.get("y"), np.nan)
                if "y" in coord_cols_dict
                else np.nan
            )
            z = (
                frame_row.get(coord_cols_dict.get("z"), np.nan)
                if "z" in coord_cols_dict
                else np.nan
            )

            rows.append(
                {
                    "frame": frame_idx,
                    "type": lm_type,
                    "landmark_index": lm_idx,
                    "x": x if pd.notna(x) else np.nan,
                    "y": y if pd.notna(y) else np.nan,
                    "z": z if pd.notna(z) else np.nan,
                }
            )

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["row_id"] = (
        df["frame"].astype(str)
        + "-"
        + df["type"]
        + "-"
        + df["landmark_index"].astype(str)
    )
    return df


def process_parquet_file(args):
    """
    Process a single parquet file containing multiple sequences.
    Convert all sequences to long format and save individually.

    Args:
        args: (parquet_path, base_path, output_dir, sequences_info)
              sequences_info is list of (sequence_id, phrase)

    Returns:
        List of record dicts for successfully processed sequences
    """
    parquet_path, base_path, output_dir, sequences_info = args
    records = []

    try:
        full_path = os.path.join(base_path, parquet_path)
        pq_df = pd.read_parquet(full_path)

        for sequence_id, phrase in sequences_info:
            try:
                # Extract sequence
                if pq_df.index.name == "sequence_id":
                    seq_df = pq_df.loc[sequence_id]
                    if isinstance(seq_df, pd.Series):
                        seq_df = seq_df.to_frame().T
                elif "sequence_id" in pq_df.columns:
                    seq_df = pq_df[pq_df["sequence_id"] == sequence_id]
                else:
                    continue

                if len(seq_df) == 0:
                    continue

                # Convert to long format
                long_df = convert_sequence_to_long(seq_df, sequence_id)

                if long_df is None or len(long_df) == 0:
                    continue

                # Save as parquet
                out_filename = f"{sequence_id}.parquet"
                out_path = os.path.join(output_dir, out_filename)
                long_df.to_parquet(out_path, index=False)

                records.append(
                    {
                        "path": f"train_landmark_files/{out_filename}",
                        "sign": phrase,  # Using 'sign' to match other datasets
                        "sequence_id": sequence_id,
                        "n_frames": long_df["frame"].nunique(),
                    }
                )

            except Exception as e:
                continue

        return records

    except Exception as e:
        return []


def convert_fingerspelling_parallel(
    input_csv: str,
    input_dir: str,
    output_dir: str,
    start_idx: int = 0,
    end_idx: int = None,
    num_workers: int = 4,
) -> pd.DataFrame:
    """
    Convert fingerspelling dataset from wide to long format using parallel processing.
    """
    # Load metadata
    train_df = pd.read_csv(input_csv)
    total_rows = len(train_df)

    if end_idx is None:
        end_idx = total_rows

    train_df = train_df.iloc[start_idx:end_idx]

    print(f"Total sequences in CSV: {total_rows:,}")
    print(f"Processing rows {start_idx} to {end_idx} ({len(train_df):,} sequences)")

    # Group sequences by parquet file for efficient processing
    # Structure: {parquet_path: [(sequence_id, phrase), ...]}
    file_to_sequences = {}

    for _, row in train_df.iterrows():
        pq_path = row["path"]
        seq_id = row["sequence_id"]
        phrase = row["phrase"]

        if pq_path not in file_to_sequences:
            file_to_sequences[pq_path] = []
        file_to_sequences[pq_path].append((seq_id, phrase))

    print(f"Grouped into {len(file_to_sequences):,} parquet files")

    # Create task list
    landmark_dir = os.path.join(output_dir, "train_landmark_files")
    tasks = [
        (pq_path, input_dir, landmark_dir, seq_list)
        for pq_path, seq_list in file_to_sequences.items()
    ]

    # Process in parallel
    all_records = []

    print(f"\nConverting with {num_workers} workers...")
    with Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_parquet_file, tasks, chunksize=10),
                total=len(tasks),
                desc="Converting files",
            )
        )

    for records in results:
        all_records.extend(records)

    print(f"\nSuccessfully converted: {len(all_records):,} sequences")

    # Save train.csv
    result_df = pd.DataFrame(all_records)

    # Save with batch suffix if doing partial processing
    if start_idx > 0 or end_idx < total_rows:
        csv_filename = f"train_batch_{start_idx}_{end_idx}.csv"
    else:
        csv_filename = "train.csv"

    csv_path = os.path.join(output_dir, csv_filename)
    result_df.to_csv(csv_path, index=False)
    print(f"Saved to: {csv_path}")

    return result_df


# ============== MAIN ==============

if __name__ == "__main__":
    if os.path.exists(INPUT_CSV):
        print("Fingerspelling dataset found!")
        print()

        result_df = convert_fingerspelling_parallel(
            input_csv=INPUT_CSV,
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            start_idx=START_IDX,
            end_idx=END_IDX,
            num_workers=NUM_WORKERS,
        )

        # Verify output
        if len(result_df) > 0:
            print(f"\n{'='*60}")
            print("Verification:")
            print(f"  Total sequences: {len(result_df):,}")
            print(f"  Unique phrases: {result_df['sign'].nunique():,}")

            # Check a sample parquet
            sample_path = result_df["path"].iloc[0]
            sample_full_path = os.path.join(OUTPUT_DIR, sample_path)
            if os.path.exists(sample_full_path):
                sample_df = pd.read_parquet(sample_full_path)
                print(f"\nSample parquet ({sample_path}):")
                print(f"  Columns: {sample_df.columns.tolist()}")
                print(f"  Landmark types: {sample_df['type'].unique().tolist()}")
                print(f"  Frames: {sample_df['frame'].nunique()}")
                print(f"  Rows: {len(sample_df):,}")
    else:
        print(f"Input CSV not found: {INPUT_CSV}")
        print("Please check the path and try again.")


# ============== MERGE BATCHES ==============


def merge_batch_csvs(output_dir: str) -> pd.DataFrame:
    """Merge all batch CSV files into a single train.csv"""
    batch_files = sorted(Path(output_dir).glob("train_batch_*.csv"))

    if not batch_files:
        print("No batch files found")
        return pd.DataFrame()

    print(f"Found {len(batch_files)} batch files")

    dfs = [pd.read_csv(f) for f in batch_files]
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=["sequence_id"])

    merged_path = os.path.join(output_dir, "train.csv")
    merged_df.to_csv(merged_path, index=False)

    print(f"Merged {len(merged_df):,} sequences")
    print(f"Saved to: {merged_path}")

    return merged_df


# Uncomment to merge batches after multi-session processing:
# merge_batch_csvs(OUTPUT_DIR)
