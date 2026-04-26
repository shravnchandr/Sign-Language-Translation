"""PyTorch datasets for VQ-VAE and translation training."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path

from .preprocessing import LandmarkProcessor, LandmarkConfig


class VQVAEDataset(Dataset):
    """
    Dataset for VQ-VAE pre-training on landmark sequences.

    Returns raw landmark data for reconstruction learning.
    No labels needed - unsupervised pre-training.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        base_path: str,
        config: Optional[LandmarkConfig] = None,
        max_frames: int = 256,
        augment: bool = False,
        augment_fn: Optional[Callable] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            df: DataFrame with 'path' column (and optionally 'full_path')
            base_path: Base directory for landmark files
            config: Landmark processing configuration
            max_frames: Maximum sequence length (longer sequences are subsampled)
            augment: Whether to apply augmentation
            augment_fn: Custom augmentation function
            cache_dir: Directory to cache processed tensors. On first access each
                sample is processed from parquet and saved as a .pt file; subsequent
                accesses load the cached tensor directly, skipping parquet parsing.
        """
        self.df = df.reset_index(drop=True)
        self.base_path = Path(base_path)
        self.processor = LandmarkProcessor(config)
        self.max_frames = max_frames
        self.augment = augment
        self.augment_fn = augment_fn
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def __len__(self) -> int:
        return len(self.df)

    def _get_path(self, idx: int) -> str:
        """Get full path for a sample."""
        row = self.df.iloc[idx]
        if "full_path" in self.df.columns:
            return row["full_path"]
        return str(self.base_path / row["path"])

    def _get_cache_path(self, idx: int) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        row = self.df.iloc[idx]
        rel = row["path"] if "path" in self.df.columns else str(idx)
        return self.cache_dir / Path(rel).with_suffix(".pt")

    def _process_parquet(self, path: str) -> Dict[str, torch.Tensor]:
        """Process a parquet file into a tensor dict (no augmentation)."""
        data = self.processor.process(path)

        T = data["left_hand"].shape[0]
        if T > self.max_frames:
            indices = np.linspace(0, T - 1, self.max_frames).astype(int)
            for key in data:
                data[key] = data[key][indices]
            T = self.max_frames

        landmarks = np.concatenate(
            [data["left_hand"], data["right_hand"], data["pose"], data["face"]],
            axis=1,
        )
        return {
            "landmarks": torch.tensor(landmarks, dtype=torch.float32),
            "left_hand": torch.tensor(data["left_hand"], dtype=torch.float32),
            "right_hand": torch.tensor(data["right_hand"], dtype=torch.float32),
            "pose": torch.tensor(data["pose"], dtype=torch.float32),
            "face": torch.tensor(data["face"], dtype=torch.float32),
            "length": torch.tensor(T, dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cache_path = self._get_cache_path(idx)

        if cache_path is not None and cache_path.exists():
            result = torch.load(cache_path, weights_only=True)
        else:
            result = self._process_parquet(self._get_path(idx))
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_path.with_suffix(".tmp")
                torch.save(result, tmp)
                tmp.rename(cache_path)  # atomic write

        if self.augment and self.augment_fn is not None:
            result = self.augment_fn(result)

        return result


class TranslationDataset(Dataset):
    """
    Dataset for seq2seq translation training.

    Returns tokenized landmark sequences with gloss labels.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        base_path: str,
        sign_to_idx: Dict[str, int],
        config: Optional[LandmarkConfig] = None,
        max_frames: int = 256,
        augment: bool = False,
        augment_fn: Optional[Callable] = None,
        cache_dir: Optional[str] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.base_path = Path(base_path)
        self.sign_to_idx = sign_to_idx
        self.processor = LandmarkProcessor(config)
        self.max_frames = max_frames
        self.augment = augment
        self.augment_fn = augment_fn
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def __len__(self) -> int:
        return len(self.df)

    def _get_path(self, idx: int) -> str:
        row = self.df.iloc[idx]
        if "full_path" in self.df.columns:
            return row["full_path"]
        return str(self.base_path / row["path"])

    def _get_cache_path(self, idx: int) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        row = self.df.iloc[idx]
        rel = row["path"] if "path" in self.df.columns else str(idx)
        return self.cache_dir / Path(rel).with_suffix(".pt")

    def _process_parquet(self, path: str) -> Dict[str, torch.Tensor]:
        data = self.processor.process(path)

        T = data["left_hand"].shape[0]
        if T > self.max_frames:
            indices = np.linspace(0, T - 1, self.max_frames).astype(int)
            for key in data:
                data[key] = data[key][indices]
            T = self.max_frames

        landmarks = np.concatenate(
            [data["left_hand"], data["right_hand"], data["pose"], data["face"]],
            axis=1,
        )
        return {
            "landmarks": torch.tensor(landmarks, dtype=torch.float32),
            "left_hand": torch.tensor(data["left_hand"], dtype=torch.float32),
            "right_hand": torch.tensor(data["right_hand"], dtype=torch.float32),
            "pose": torch.tensor(data["pose"], dtype=torch.float32),
            "face": torch.tensor(data["face"], dtype=torch.float32),
            "length": torch.tensor(T, dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        cache_path = self._get_cache_path(idx)

        if cache_path is not None and cache_path.exists():
            result = torch.load(cache_path, weights_only=True)
        else:
            result = self._process_parquet(self._get_path(idx))
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_path.with_suffix(".tmp")
                torch.save(result, tmp)
                tmp.rename(cache_path)

        sign = row["sign"]
        label = self.sign_to_idx.get(sign, -1)
        if label == -1:
            raise ValueError(f"Unknown sign: {sign}")

        result = dict(result)
        result["label"] = torch.tensor(label, dtype=torch.long)

        if self.augment and self.augment_fn is not None:
            result = self.augment_fn(result)

        return result


def collate_vqvae(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for VQ-VAE dataset.

    Pads sequences to the same length and creates attention masks.
    """
    # Get max length in batch
    lengths = torch.stack([item["length"] for item in batch])
    max_len = lengths.max().item()

    batch_size = len(batch)

    # Get dimensions from first item
    first = batch[0]
    n_landmarks = first["landmarks"].shape[1]
    n_coords = first["landmarks"].shape[2]

    # Initialize padded tensors
    landmarks = torch.zeros(batch_size, max_len, n_landmarks, n_coords)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    # Separate landmark types
    left_hand = torch.zeros(batch_size, max_len, 21, n_coords)
    right_hand = torch.zeros(batch_size, max_len, 21, n_coords)
    pose = torch.zeros(batch_size, max_len, 33, n_coords)

    # Face dimension may vary
    face_dim = first["face"].shape[1]
    face = torch.zeros(batch_size, max_len, face_dim, n_coords)

    for i, item in enumerate(batch):
        T = item["length"].item()
        landmarks[i, :T] = item["landmarks"]
        mask[i, :T] = True
        left_hand[i, :T] = item["left_hand"]
        right_hand[i, :T] = item["right_hand"]
        pose[i, :T] = item["pose"]
        face[i, :T] = item["face"]

    return {
        "landmarks": landmarks,
        "mask": mask,
        "lengths": lengths,
        "left_hand": left_hand,
        "right_hand": right_hand,
        "pose": pose,
        "face": face,
    }


def collate_translation(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Collate function for translation dataset.

    Same as VQ-VAE collate but includes labels.
    """
    result = collate_vqvae(batch)
    result["labels"] = torch.stack([item["label"] for item in batch])
    return result


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    collate_fn: Callable = collate_vqvae,
) -> DataLoader:
    """Create a DataLoader with appropriate settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=False,
    )
