import hashlib
import json
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Sampler
from typing import List, Optional, Tuple
from sklearn.model_selection import train_test_split
from .augmentation import augment_sample
from .preprocessing import frame_stacked_data
from ..config import INCLUDE_FACE, INCLUDE_DEPTH, ALL_COLUMNS

# Hash of the exact column list serialized into each cached tensor.
# ALL_COLUMNS encodes INCLUDE_FACE, INCLUDE_DEPTH, and the full face landmark
# selection (including ordering), so any change that shifts column semantics
# produces a new hash and forces a clean cache rebuild.
_CACHE_VERSION = hashlib.md5("|".join(ALL_COLUMNS).encode()).hexdigest()[:8]


class ASLDataset(Dataset):
    """
    ASL landmark dataset with per-sample .pt caching.

    First access processes each parquet and saves a .pt tensor under cache_dir,
    mirroring the parquet's relative path. Subsequent accesses skip parquet parsing.
    Velocity (body-relative frame differences) is computed at runtime so augmented
    coordinates produce the correct velocity.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        base_path: str,
        cache_dir: Optional[str] = None,
        max_frames: int = 128,
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.base_path = Path(base_path)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_frames = max_frames
        self.augment = augment

        # Compute (and optionally cache) sequence lengths so BucketBatchSampler
        # can group similar-length sequences without reloading every sample.
        self.lengths = self._load_or_compute_lengths()

    def __len__(self) -> int:
        return len(self.df)

    def _cache_path(self, idx: int) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        rel = self.df.iloc[idx]["path"]
        return self.cache_dir / Path(rel).with_suffix(f".{_CACHE_VERSION}.pt")

    def _load_coords(self, idx: int) -> torch.Tensor:
        """Return raw position coordinates (T, D_pos) as a float32 tensor."""
        cp = self._cache_path(idx)
        if cp is not None and cp.exists():
            return torch.load(cp, weights_only=True)
        # Parse from parquet and cache
        full_path = str(self.base_path / self.df.iloc[idx]["path"])
        coords = torch.tensor(frame_stacked_data(full_path), dtype=torch.float32)
        if cp is not None:
            cp.parent.mkdir(parents=True, exist_ok=True)
            torch.save(coords, cp)
        return coords

    def _load_or_compute_lengths(self) -> List[int]:
        """Load sequence lengths from a sidecar JSON, or compute and save them."""
        lengths_file = (
            (self.cache_dir / f"_lengths_{_CACHE_VERSION}.json") if self.cache_dir else None
        )
        if lengths_file is not None and lengths_file.exists():
            with open(lengths_file) as f:
                lengths = json.load(f)
            # Stored length list is for the full dataset, re-index to our subset
            if len(lengths) == len(self.df):
                return lengths

        # Load (and cache) every sample to get its length; this only happens once
        lengths = []
        for i in range(len(self.df)):
            coords = self._load_coords(i)
            lengths.append(min(len(coords), self.max_frames))

        if lengths_file is not None:
            lengths_file.parent.mkdir(parents=True, exist_ok=True)
            with open(lengths_file, "w") as f:
                json.dump(lengths, f)
        return lengths

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        coords = self._load_coords(idx)  # (T, D_pos)
        label = int(self.df.iloc[idx]["sign"])

        # Optional augmentation on raw positions before velocity computation
        if self.augment:
            coords = torch.tensor(augment_sample(coords.numpy()), dtype=torch.float32)

        # Truncate long sequences
        if coords.shape[0] > self.max_frames:
            idxs = torch.linspace(0, coords.shape[0] - 1, self.max_frames).long()
            coords = coords[idxs]

        # Velocity: body-relative because coords are already origin-subtracted
        vel = torch.zeros_like(coords)
        vel[1:] = coords[1:] - coords[:-1]

        return torch.cat([coords, vel], dim=-1), label  # (T, 2*D_pos)


def collate_batch(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([seq.shape[0] for seq in sequences])
    max_len = int(lengths.max())
    B, D = len(sequences), sequences[0].shape[1]
    padded = torch.zeros(B, max_len, D)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, seq in enumerate(sequences):
        T = seq.shape[0]
        padded[i, :T] = seq
        mask[i, :T] = True
    return padded, mask, torch.tensor(labels)


class BucketBatchSampler(Sampler):
    """Groups sequences by length to minimise padding waste within each batch."""

    def __init__(self, lengths: List[int], batch_size: int, drop_last: bool = False):
        self.lengths = lengths
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        sorted_idxs = np.argsort(self.lengths)
        buckets = [
            sorted_idxs[i : i + self.batch_size]
            for i in range(0, len(sorted_idxs), self.batch_size)
        ]
        if self.drop_last and len(buckets[-1]) < self.batch_size:
            buckets = buckets[:-1]
        np.random.shuffle(buckets)
        for b in buckets:
            yield list(b)

    def __len__(self) -> int:
        n = len(self.lengths)
        return (
            n // self.batch_size
            if self.drop_last
            else (n + self.batch_size - 1) // self.batch_size
        )


def get_data_loaders(
    data_dir: str,
    cache_dir: Optional[str] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    max_frames: int = 128,
) -> Tuple[DataLoader, DataLoader]:
    """
    Args:
        data_dir:    Directory containing train.csv and sign_to_prediction_index_map.json.
        cache_dir:   Directory for per-sample .pt cache files. Built automatically on
                     first run; subsequent runs skip parquet parsing.
        batch_size:  Samples per batch.
        num_workers: DataLoader worker processes.
        max_frames:  Truncate sequences longer than this.
    """
    sign_map_file = Path(data_dir) / "sign_to_prediction_index_map.json"
    train_csv = Path(data_dir) / "train.csv"

    with open(sign_map_file) as f:
        sign2idx = json.load(f)

    df = pd.read_csv(train_csv)
    df["sign"] = df["sign"].map(sign2idx)

    train_df, test_df = train_test_split(
        df, test_size=0.1, stratify=df["sign"], random_state=42
    )

    # Separate cache subdirs so the lengths sidecar files don't collide
    train_cache = str(Path(cache_dir) / "train") if cache_dir else None
    test_cache = str(Path(cache_dir) / "test") if cache_dir else None

    train_dataset = ASLDataset(
        train_df, data_dir, cache_dir=train_cache, max_frames=max_frames, augment=True
    )
    test_dataset = ASLDataset(
        test_df, data_dir, cache_dir=test_cache, max_frames=max_frames, augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=BucketBatchSampler(train_dataset.lengths, batch_size),
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=BucketBatchSampler(test_dataset.lengths, batch_size),
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
