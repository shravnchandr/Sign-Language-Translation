import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_PATH = r"/kaggle/input/asl-signs"
TRAIN_FILE = r"/kaggle/input/asl-signs/train.csv"
SIGN_INDEX_FILE = r"/kaggle/input/asl-signs/sign_to_prediction_index_map.json"

with open(SIGN_INDEX_FILE, "r") as json_file:
    SIGN2INDEX_JSON = json.load(json_file)

INCLUDE_FACE = True
INCLUDE_DEPTH = False

FACE_LANDMARK_INDICES = {
    "nose": [1, 2, 4, 5, 6, 19, 94, 168, 197, 195],
    "left_eye": [
        33,
        133,
        160,
        159,
        158,
        157,
        173,
        144,
        145,
        153,
        154,
        155,
        156,
        246,
        7,
        163,
    ],
    "right_eye": [
        263,
        362,
        387,
        386,
        385,
        384,
        398,
        373,
        374,
        380,
        381,
        382,
        466,
        388,
        390,
        249,
    ],
    "left_eyebrow": [70, 63, 105, 66, 107, 55, 65, 52],
    "right_eyebrow": [300, 293, 334, 296, 336, 285, 295, 282],
    "mouth_outer": [
        61,
        146,
        91,
        181,
        84,
        17,
        314,
        405,
        321,
        375,
        291,
        409,
        270,
        269,
        267,
        0,
        37,
        39,
        40,
        185,
    ],
    "mouth_inner": [
        78,
        191,
        80,
        81,
        82,
        13,
        312,
        311,
        310,
        415,
        308,
        324,
        318,
        402,
        317,
        14,
        87,
        178,
        88,
        95,
    ],
    "face_oval": [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
    ],
}

SELECTED_FACE_INDICES = []
for feature_indices in FACE_LANDMARK_INDICES.values():
    SELECTED_FACE_INDICES.extend(feature_indices)


def generate_full_column_list() -> List[str]:
    """
    Generates the standardized list of 1629 column names (x/y/z for 543 landmarks).
    """
    landmark_specs = {
        "left_hand": 21,  # Indices 0 to 20
        "pose": 33,  # Indices 0 to 32
        "right_hand": 21,  # Indices 0 to 20
    }

    axes = ["x", "y", "z"] if INCLUDE_DEPTH else ["x", "y"]

    full_columns = []

    for landmark_type, count in landmark_specs.items():
        for i in range(count):
            for axis in axes:
                full_columns.append(f"{landmark_type}_{i}_{axis}")

    if INCLUDE_FACE:
        for face_idx in SELECTED_FACE_INDICES:
            for axis in axes:
                full_columns.append(f"face_{face_idx}_{axis}")

    return full_columns


ALL_COLUMNS = generate_full_column_list()


def normalize_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize coordinates using the nose coordinates

    Args:
        df (pd.DataFrame): Unnormalaized dataframe

    Returns:
        pd.Dataframe: Normalized dataframe / series
    """
    axes = ["x", "y", "z"] if INCLUDE_DEPTH else ["x", "y"]

    origins = df[(df["type"] == "pose") & (df["landmark_index"] == 0)].set_index(
        "frame"
    )[axes]

    def normalize_frame(frame_df: pd.DataFrame) -> pd.DataFrame:
        frame = frame_df.name
        if frame not in origins.index:
            return frame_df  # or raise an error
        frame_df[axes] = frame_df[axes] - origins.loc[frame]
        return frame_df

    normalized_df = df.groupby("frame", group_keys=False).apply(normalize_frame)
    return normalized_df


def frame_stacked_data(file_path: str) -> np.ndarray:
    """
    Read landmark data from parquet files, normalize and stack them for each frame

    Args:
        file_path (str): File path for the parquet file

    Returns:
        np.ndarray: The normlaized stacked coordinates
    """
    df = pd.read_parquet(os.path.join(BASE_PATH, file_path))
    if INCLUDE_FACE:
        face_df = df[df["type"] == "face"]
        face_df = face_df[face_df["landmark_index"].isin(SELECTED_FACE_INDICES)]
        other_df = df[df["type"] != "face"]
        df = pd.concat([face_df, other_df], ignore_index=True)
    else:
        df = df[df["type"] != "face"]
    axes = ["x", "y", "z"] if INCLUDE_DEPTH else ["x", "y"]

    df["UID"] = df["type"].astype("str") + "_" + df["landmark_index"].astype("str")
    df = df.sort_values(["frame", "UID"])

    df = normalize_values(df)

    pivot_df = df.pivot_table(index="frame", columns="UID", values=axes)
    pivot_df.columns = [f"{col[1]}_{col[0]}" for col in pivot_df.columns]
    pivot_df = pivot_df.reindex(columns=ALL_COLUMNS)

    final_data = pd.DataFrame(pivot_df).ffill().bfill().fillna(0).to_numpy()
    return final_data


def augment_sample(
    video_coordinates: np.ndarray, noise_std: float = 3e-3, spatial_shift: float = 2e-2
) -> np.ndarray:
    video_coordinates = video_coordinates.copy()

    if np.random.random() > 0.5:
        noise = np.random.normal(0, noise_std, video_coordinates.shape)
        video_coordinates = video_coordinates + noise

    if np.random.random() > 0.5:
        shift = np.random.uniform(
            -spatial_shift, spatial_shift, (1, video_coordinates.shape[1])
        )
        video_coordinates = video_coordinates + shift

    if np.random.random() > 0.5 and video_coordinates.shape[0] > 20:
        start_idx = np.random.randint(0, max(1, video_coordinates.shape[0] // 10))
        end_idx = video_coordinates.shape[0] - np.random.randint(
            0, max(1, video_coordinates.shape[0] // 10)
        )
        video_coordinates = video_coordinates[start_idx:end_idx]

    return video_coordinates


class ASLDataset(Dataset):
    def __init__(self, video_coordinates, video_labels, max_frames=128, augment=False):
        self.video_coordinates = video_coordinates
        self.video_labels = video_labels
        self.max_frames = max_frames
        self.augment = augment

    def __len__(self):
        return len(self.video_coordinates)

    def __getitem__(self, idx):
        x = self.video_coordinates[idx]
        y = self.video_labels[idx]

        if self.augment:
            x = augment_sample(x)

        if x.shape[0] > self.max_frames:
            idxs = np.linspace(0, x.shape[0] - 1, self.max_frames).astype(int)
            x = x[idxs]

        vel = x[1:] - x[:-1]
        vel = np.vstack([np.zeros_like(x[:1]), vel])
        x = np.concatenate([x, vel], axis=1)

        return torch.tensor(x, dtype=torch.float32), y


def collate_fn(batch):
    sequences, labels = zip(*batch)

    lengths = torch.tensor([seq.shape[0] for seq in sequences])
    max_len = int(lengths.max())

    D = sequences[0].shape[1]
    B = len(sequences)

    padded = torch.zeros(B, max_len, D)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, seq in enumerate(sequences):
        T = seq.shape[0]
        padded[i, :T] = seq
        mask[i, :T] = 1

    return padded, mask, torch.tensor(labels)


class BucketBatchSampler(Sampler):
    """
    Sampler that groups sequences by length into buckets and shuffles buckets each epoch
    """

    def __init__(self, lengths, batch_size, drop_last=False):
        self.lengths = lengths
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        sorted_idxs = np.argsort(self.lengths)

        buckets = []
        for i in range(0, len(sorted_idxs), self.batch_size):
            bucket = sorted_idxs[i : i + self.batch_size]
            if len(bucket) == self.batch_size or not self.drop_last:
                buckets.append(bucket)

        np.random.shuffle(buckets)

        for bucket in buckets:
            yield list(bucket)

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def bucket_dataloader(dataset, batch_size: int = 128) -> DataLoader:
    lengths = [min(x.shape[0], dataset.max_frames) for x in dataset.video_coordinates]

    sampler = BucketBatchSampler(lengths, batch_size=batch_size, drop_last=False)

    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)


def get_data_loaders() -> Tuple[DataLoader, DataLoader]:
    """
    Reads dataloaders from the normalized coordinates

    Returns:
        Tuple[DataLoader, DataLoader]: train and test dataloaders
    """
    train_df = pd.read_csv(TRAIN_FILE)

    train_df["sign"] = train_df["sign"].map(SIGN2INDEX_JSON)
    train_split, test_split = train_test_split(
        train_df, test_size=0.1, stratify=train_df["sign"], random_state=42
    )

    all_train_videos = []
    for path in train_split["path"].to_list():
        all_train_videos.append(frame_stacked_data(path))

    all_test_videos = []
    for path in test_split["path"].to_list():
        all_test_videos.append(frame_stacked_data(path))

    train_dataset = ASLDataset(
        all_train_videos, train_split["sign"].to_numpy(), augment=True
    )
    test_dataset = ASLDataset(all_test_videos, test_split["sign"].to_numpy())

    train_loader = bucket_dataloader(train_dataset)
    test_loader = bucket_dataloader(test_dataset)

    return train_loader, test_loader
