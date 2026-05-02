import numpy as np
import pandas as pd
from ..config import (
    INCLUDE_FACE,
    INCLUDE_DEPTH,
    ALL_COLUMNS,
    FACE_LANDMARK_SET,
)


def normalize_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Subtract a per-frame body origin: nose → shoulder center → hip center → 0.

    The fallback chain matches RobustNormalization's intent but is applied once
    at LMDB build time rather than per-forward-pass in the model. Storing
    normalized coordinates keeps the velocity computed in ASLDataset (Δ1 =
    frame difference of stored positions) body-relative rather than absolute.
    """
    axes = ["x", "y", "z"] if INCLUDE_DEPTH else ["x", "y"]
    all_frames = sorted(dataframe["frame"].unique())

    def pose_lm(idx: int) -> pd.DataFrame:
        mask = (dataframe["type"] == "pose") & (dataframe["landmark_index"] == idx)
        return dataframe[mask].set_index("frame")[axes].reindex(all_frames)

    nose = pose_lm(0)
    shoulder = (pose_lm(11) + pose_lm(12)) / 2.0   # mid-point of left/right shoulder
    hip = (pose_lm(23) + pose_lm(24)) / 2.0         # mid-point of left/right hip

    # combine_first: prefer nose; where nose is NaN, use shoulder; then hip; then 0.
    origin = nose.combine_first(shoulder).combine_first(hip).fillna(0.0)
    origin.columns = [f"{ax}_origin" for ax in axes]

    dataframe = dataframe.merge(origin, left_on="frame", right_index=True, how="left")
    for axis in axes:
        dataframe[axis] = dataframe[axis] - dataframe[f"{axis}_origin"].fillna(0)
    return dataframe.drop(columns=[f"{ax}_origin" for ax in axes])


def frame_stacked_data(file_path: str) -> np.ndarray:
    df = pd.read_parquet(file_path)
    axes = ["x", "y", "z"] if INCLUDE_DEPTH else ["x", "y"]

    # Filter landmarks — frozenset gives O(1) per-row membership test
    if INCLUDE_FACE:
        df = df[
            ~((df["type"] == "face") & ~df["landmark_index"].isin(FACE_LANDMARK_SET))
        ]
    else:
        df = df[df["type"] != "face"]

    df = normalize_values(df)
    df = df.copy()
    df["uid"] = df["type"].astype(str) + "_" + df["landmark_index"].astype(str)

    # pivot is faster than pivot_table for unique (frame, uid) pairs
    try:
        wide = df.pivot(index="frame", columns="uid", values=axes)
    except ValueError:
        # Rare duplicate (frame, uid) rows — fall back to aggregation
        wide = df.pivot_table(
            index="frame", columns="uid", values=axes, aggfunc="first"
        )

    wide.columns = [f"{col[1]}_{col[0]}" for col in wide.columns]
    wide = wide.reindex(columns=ALL_COLUMNS)
    return wide.ffill().bfill().fillna(0).to_numpy(dtype=np.float32)
