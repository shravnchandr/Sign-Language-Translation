import numpy as np
import pandas as pd
from ..config import (
    INCLUDE_FACE,
    INCLUDE_DEPTH,
    ALL_COLUMNS,
    FACE_LANDMARK_SET,
)


def normalize_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    axes = ["x", "y", "z"] if INCLUDE_DEPTH else ["x", "y"]
    origins = (
        dataframe[(dataframe["type"] == "pose") & (dataframe["landmark_index"] == 0)]
        .set_index("frame")[axes]
        .rename(columns={ax: f"{ax}_origin" for ax in axes})
    )
    dataframe = dataframe.merge(origins, left_on="frame", right_index=True, how="left")
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
