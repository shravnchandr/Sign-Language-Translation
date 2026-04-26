"""
Standalone training script for Improved VQ-VAE.

This file consolidates all dependencies from vqvae_seq2seq/ into a single file.
Run with: uv run python train_vqvae_standalone.py --data-dir data/Isolated_ASL_Recognition
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from tqdm import tqdm

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ImprovedVQVAEConfig:
    """
    Configuration for the improved VQ-VAE tokenizer.

    Addresses blindspots:
    - #1: Z-coordinate included (3D input)
    - #2: Dedicated face encoder
    - #3: Multi-scale temporal encoding
    - #10: Increased codebook sizes
    """

    # Input dimensions (3D coordinates)
    hand_dim: int = 63  # 21 landmarks * 3 coords
    pose_dim: int = 99  # 33 landmarks * 3 coords
    face_dim: int = 402  # 134 landmarks * 3 coords (compact subset)
    n_coords: int = 3  # x, y, z

    # Landmark counts
    hand_landmarks: int = 21
    pose_landmarks: int = 33
    face_landmarks: int = 134  # Compact subset

    # Multi-scale temporal encoding (blindspot #3)
    temporal_scales: Tuple[int, ...] = (4, 8, 16)
    base_chunk_size: int = 8  # Reference chunk size

    # Encoder architecture
    encoder_hidden_dim: int = 256
    encoder_n_layers: int = 3
    encoder_dropout: float = 0.1

    # Codebook sizes (blindspot #10: increased sizes)
    pose_codebook_size: int = 1024  # Was 512
    motion_codebook_size: int = 512  # Was 256
    dynamics_codebook_size: int = 256  # Was 128
    face_codebook_size: int = 256  # New dedicated face codebook

    # Embedding dimensions
    embed_dim: int = 128  # Codebook embedding dimension
    latent_dim: int = 256  # Encoder output dimension

    # Vector quantizer settings
    commitment_weight: float = 0.25
    ema_decay: float = 0.99
    codebook_reset_threshold: float = 0.01  # Reset codes used < 1% of time
    codebook_reset_patience: int = 100  # Steps before resetting

    # Cross-factor attention (blindspot #8)
    use_cross_attention: bool = True
    cross_attention_heads: int = 4
    cross_attention_layers: int = 2

    # Face NMM encoder (blindspot #2)
    face_region_dims: dict = field(
        default_factory=lambda: {
            "eyebrows": 16 * 3,  # 16 landmarks * 3 coords
            "eyes": 32 * 3,
            "nose": 10 * 3,
            "mouth": 40 * 3,
            "face_oval": 36 * 3,
        }
    )

    # Decoder architecture
    decoder_hidden_dim: int = 256
    decoder_n_layers: int = 3

    # Training settings
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 100
    max_patience: int = 10

    # Loss weights
    reconstruction_weight: float = 1.0
    velocity_reconstruction_weight: float = 0.5
    codebook_diversity_weight: float = 0.1

    # Augmentation (blindspot #9)
    augment_speed_range: Tuple[float, float] = (0.8, 1.2)
    augment_frame_dropout_prob: float = 0.1
    augment_temporal_jitter_std: float = 0.02

    # Device
    device: str = "cuda"

    def get_total_landmarks(self) -> int:
        """Get total number of landmarks across all body parts."""
        return (
            self.hand_landmarks * 2  # Both hands
            + self.pose_landmarks
            + self.face_landmarks
        )

    def get_total_input_dim(self) -> int:
        """Get total input dimension (flattened)."""
        return self.get_total_landmarks() * self.n_coords


# =============================================================================
# Data Preprocessing
# =============================================================================

# MediaPipe landmark indices
POSE_INDICES = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
}

HAND_LANDMARKS = 21
POSE_LANDMARKS = 33
FACE_LANDMARKS = 478  # Full MediaPipe face mesh


@dataclass
class LandmarkConfig:
    """Configuration for landmark processing."""

    include_z: bool = True
    face_subset: Optional[List[int]] = None  # None = use all 478
    interpolate_missing: bool = True
    max_missing_ratio: float = 0.3  # Max ratio of missing frames before rejection


# Commonly used face landmark subsets
FACE_LANDMARK_SUBSETS = {
    "lips": list(range(0, 40)),  # Inner and outer lips
    "eyes": list(range(33, 133)) + list(range(263, 363)),  # Both eyes
    "eyebrows": [70, 63, 105, 66, 107, 55, 65, 52]
    + [300, 293, 334, 296, 336, 285, 295, 282],
    "nose": [1, 2, 4, 5, 6, 19, 94, 168, 197, 195],
    # Compact subset for efficient processing (134 landmarks)
    "compact": (
        [1, 2, 4, 5, 6, 19, 94, 168, 197, 195]  # nose (10)
        + [
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
        ]  # left eye (16)
        + [
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
        ]  # right eye (16)
        + [70, 63, 105, 66, 107, 55, 65, 52]  # left eyebrow (8)
        + [300, 293, 334, 296, 336, 285, 295, 282]  # right eyebrow (8)
        + [
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
        ]  # outer mouth (20)
        + [
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
        ]  # inner mouth (20)
        + [
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
        ]  # face oval (36)
    ),
}


class FastLandmarkProcessor:
    """
    Fast landmark processor using pure NumPy operations.

    Replaces the slow pandas-based implementation with vectorized operations
    for ~10-50x speedup in data loading.
    """

    # Type string to integer mapping for fast comparison
    TYPE_MAP = {"face": 0, "left_hand": 1, "pose": 2, "right_hand": 3}

    def __init__(self, config: Optional[LandmarkConfig] = None):
        self.config = config or LandmarkConfig()
        self.n_coords = 3 if self.config.include_z else 2

        # Pre-compute face subset lookup if needed
        if self.config.face_subset is not None:
            self.face_subset_set = set(self.config.face_subset)
            self.face_subset_map = {
                idx: i for i, idx in enumerate(self.config.face_subset)
            }
            self.n_face = len(self.config.face_subset)
        else:
            self.face_subset_set = None
            self.face_subset_map = None
            self.n_face = FACE_LANDMARKS

    def load_parquet_numpy(
        self, path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load parquet file directly into NumPy arrays using PyArrow.

        Returns:
            frames: (N,) int array of frame indices
            types: (N,) int array of landmark types (0=face, 1=left_hand, 2=pose, 3=right_hand)
            landmark_indices: (N,) int array of landmark indices within type
            coords: (N, 3) float array of x, y, z coordinates
        """
        table = pq.read_table(path)

        frames = table.column("frame").to_numpy()
        types_str = (
            table.column("type").to_pandas().values
        )  # Need pandas for string handling
        landmark_indices = table.column("landmark_index").to_numpy()

        x = table.column("x").to_numpy()
        y = table.column("y").to_numpy()

        if self.config.include_z and "z" in table.column_names:
            z = table.column("z").to_numpy()
        else:
            z = np.zeros_like(x)

        coords = np.stack([x, y, z], axis=1).astype(np.float32)

        # Convert type strings to integers for fast comparison
        types = np.zeros(len(types_str), dtype=np.int32)
        for type_str, type_int in self.TYPE_MAP.items():
            types[types_str == type_str] = type_int

        return frames.astype(np.int32), types, landmark_indices.astype(np.int32), coords

    def _extract_landmarks_fast(
        self,
        frames: np.ndarray,
        types: np.ndarray,
        landmark_indices: np.ndarray,
        coords: np.ndarray,
        type_int: int,
        n_landmarks: int,
        n_frames: int,
        frame_to_idx: np.ndarray,
        subset_map: Optional[Dict[int, int]] = None,
    ) -> np.ndarray:
        """
        Extract landmarks of a specific type using vectorized operations.

        Args:
            frames: Frame indices array
            types: Type integers array
            landmark_indices: Landmark indices array
            coords: Coordinates array (N, 3)
            type_int: Integer type to extract
            n_landmarks: Number of landmarks for this type
            n_frames: Total number of frames
            frame_to_idx: Mapping from frame number to index
            subset_map: Optional mapping for face subset

        Returns:
            Array of shape (T, N, C)
        """
        # Filter by type
        type_mask = types == type_int
        type_frames = frames[type_mask]
        type_landmarks = landmark_indices[type_mask]
        type_coords = coords[type_mask]

        # Initialize output with NaN
        output = np.full(
            (n_frames, n_landmarks, self.n_coords), np.nan, dtype=np.float32
        )

        if len(type_frames) == 0:
            return output

        # Map frames to indices
        frame_indices = frame_to_idx[type_frames]
        valid_frame_mask = frame_indices >= 0

        if subset_map is not None:
            # Map landmark indices for face subset
            mapped_landmarks = np.array([subset_map.get(l, -1) for l in type_landmarks])
            valid_landmark_mask = (mapped_landmarks >= 0) & (
                mapped_landmarks < n_landmarks
            )
            valid_mask = valid_frame_mask & valid_landmark_mask

            valid_frame_idx = frame_indices[valid_mask]
            valid_landmark_idx = mapped_landmarks[valid_mask]
        else:
            valid_landmark_mask = (type_landmarks >= 0) & (type_landmarks < n_landmarks)
            valid_mask = valid_frame_mask & valid_landmark_mask

            valid_frame_idx = frame_indices[valid_mask]
            valid_landmark_idx = type_landmarks[valid_mask]

        valid_coords = type_coords[valid_mask, : self.n_coords]

        # Vectorized assignment
        output[valid_frame_idx, valid_landmark_idx] = valid_coords

        return output

    def _normalize_vectorized(
        self,
        pose: np.ndarray,
    ) -> np.ndarray:
        """
        Compute normalization origin using vectorized operations.

        Uses fallback chain: nose -> shoulder center -> hip center

        Args:
            pose: (T, 33, 3) pose landmarks

        Returns:
            origin: (T, 3) normalization origins per frame
        """
        T = pose.shape[0]
        origin = np.zeros((T, self.n_coords), dtype=np.float32)

        # Get candidate landmarks
        nose = pose[:, POSE_INDICES["nose"], : self.n_coords]  # (T, C)
        left_shoulder = pose[:, POSE_INDICES["left_shoulder"], : self.n_coords]
        right_shoulder = pose[:, POSE_INDICES["right_shoulder"], : self.n_coords]
        left_hip = pose[:, POSE_INDICES["left_hip"], : self.n_coords]
        right_hip = pose[:, POSE_INDICES["right_hip"], : self.n_coords]

        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2

        # Check validity (not NaN and not all zeros)
        def is_valid(arr):
            return ~np.isnan(arr).any(axis=1) & (np.abs(arr).sum(axis=1) > 1e-6)

        nose_valid = is_valid(nose)
        shoulder_valid = is_valid(left_shoulder) & is_valid(right_shoulder)
        hip_valid = is_valid(left_hip) & is_valid(right_hip)

        # Apply fallback chain (lowest priority first, highest last)
        origin[hip_valid] = hip_center[hip_valid]
        origin[shoulder_valid] = shoulder_center[shoulder_valid]
        origin[nose_valid] = nose[nose_valid]

        return origin

    def _interpolate_missing_vectorized(self, arr: np.ndarray) -> np.ndarray:
        """
        Interpolate missing values using vectorized operations where possible.

        Args:
            arr: (T, N, C) array with NaN for missing values

        Returns:
            Interpolated array with no NaN values
        """
        if not self.config.interpolate_missing:
            return np.nan_to_num(arr, nan=0.0)

        T, N, C = arr.shape
        result = arr.copy()

        # Reshape to (T, N*C) for efficient processing
        flat = result.reshape(T, -1)

        for col_idx in range(flat.shape[1]):
            col = flat[:, col_idx]
            nan_mask = np.isnan(col)

            if nan_mask.all():
                # All missing - fill with zeros
                flat[:, col_idx] = 0.0
            elif nan_mask.any():
                # Interpolate
                valid_idx = np.where(~nan_mask)[0]
                valid_vals = col[~nan_mask]
                all_idx = np.arange(T)
                flat[:, col_idx] = np.interp(all_idx, valid_idx, valid_vals)

        return flat.reshape(T, N, C)

    def process(self, path: str) -> Dict[str, np.ndarray]:
        """
        Process a parquet file into model-ready tensors.

        Args:
            path: Path to parquet file

        Returns:
            Dictionary with keys:
            - 'left_hand': (T, 21, 3)
            - 'right_hand': (T, 21, 3)
            - 'pose': (T, 33, 3)
            - 'face': (T, N_face, 3)
        """
        # Load data as numpy arrays
        frames, types, landmark_indices, coords = self.load_parquet_numpy(path)

        # Get unique frames and create mapping
        unique_frames = np.unique(frames)
        n_frames = len(unique_frames)

        # Create frame to index mapping (use max frame + 1 for direct indexing)
        max_frame = frames.max() + 1
        frame_to_idx = np.full(max_frame, -1, dtype=np.int32)
        frame_to_idx[unique_frames] = np.arange(n_frames)

        # Extract each landmark type
        left_hand = self._extract_landmarks_fast(
            frames,
            types,
            landmark_indices,
            coords,
            self.TYPE_MAP["left_hand"],
            HAND_LANDMARKS,
            n_frames,
            frame_to_idx,
        )
        right_hand = self._extract_landmarks_fast(
            frames,
            types,
            landmark_indices,
            coords,
            self.TYPE_MAP["right_hand"],
            HAND_LANDMARKS,
            n_frames,
            frame_to_idx,
        )
        pose = self._extract_landmarks_fast(
            frames,
            types,
            landmark_indices,
            coords,
            self.TYPE_MAP["pose"],
            POSE_LANDMARKS,
            n_frames,
            frame_to_idx,
        )
        face = self._extract_landmarks_fast(
            frames,
            types,
            landmark_indices,
            coords,
            self.TYPE_MAP["face"],
            self.n_face,
            n_frames,
            frame_to_idx,
            subset_map=self.face_subset_map,
        )

        # Compute normalization origin from pose landmarks
        origin = self._normalize_vectorized(pose)

        # Apply normalization to all landmarks
        origin_expanded = origin[:, np.newaxis, :]  # (T, 1, C)
        left_hand = left_hand - origin_expanded
        right_hand = right_hand - origin_expanded
        pose = pose - origin_expanded
        face = face - origin_expanded

        # Interpolate missing values
        result = {
            "left_hand": self._interpolate_missing_vectorized(left_hand),
            "right_hand": self._interpolate_missing_vectorized(right_hand),
            "pose": self._interpolate_missing_vectorized(pose),
            "face": self._interpolate_missing_vectorized(face),
        }

        return result


# Keep alias for backward compatibility
LandmarkProcessor = FastLandmarkProcessor


# =============================================================================
# Signer Split
# =============================================================================


@dataclass
class SplitConfig:
    """Configuration for signer-independent splits."""

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    min_samples_per_signer: int = 10


class SignerIndependentSplitter:
    """
    Creates signer-independent train/val/test splits.

    Ensures no signer appears in multiple splits, providing
    unbiased evaluation of model generalization to new signers.
    """

    def __init__(self, config: Optional[SplitConfig] = None):
        self.config = config or SplitConfig()
        self._validate_ratios()

    def _validate_ratios(self):
        total = self.config.train_ratio + self.config.val_ratio + self.config.test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    def split(
        self, df: pd.DataFrame, signer_col: str = "participant_id"
    ) -> Dict[str, pd.DataFrame]:
        """
        Split dataframe by signer ID.

        Args:
            df: DataFrame with samples and signer information
            signer_col: Column name containing signer IDs

        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        if signer_col not in df.columns:
            raise ValueError(f"Column '{signer_col}' not found in dataframe")

        # Get unique signers and their sample counts
        signer_counts = df[signer_col].value_counts()

        # Filter signers with minimum samples
        valid_signers = signer_counts[
            signer_counts >= self.config.min_samples_per_signer
        ].index.tolist()

        if len(valid_signers) < 3:
            raise ValueError(
                f"Need at least 3 signers with >= {self.config.min_samples_per_signer} "
                f"samples, got {len(valid_signers)}"
            )

        # Shuffle signers
        np.random.seed(self.config.random_seed)
        shuffled_signers = np.random.permutation(valid_signers)

        # Calculate split points
        n_signers = len(shuffled_signers)
        n_train = int(n_signers * self.config.train_ratio)
        n_val = int(n_signers * self.config.val_ratio)

        # Assign signers to splits
        train_signers = set(shuffled_signers[:n_train])
        val_signers = set(shuffled_signers[n_train : n_train + n_val])
        test_signers = set(shuffled_signers[n_train + n_val :])

        # Filter valid signers only
        df_valid = df[df[signer_col].isin(valid_signers)]

        # Create splits
        splits = {
            "train": df_valid[df_valid[signer_col].isin(train_signers)].copy(),
            "val": df_valid[df_valid[signer_col].isin(val_signers)].copy(),
            "test": df_valid[df_valid[signer_col].isin(test_signers)].copy(),
        }

        # Store metadata
        self.split_info = {
            "n_signers": {
                k: len(s)
                for k, s in [
                    ("train", train_signers),
                    ("val", val_signers),
                    ("test", test_signers),
                ]
            },
            "n_samples": {k: len(v) for k, v in splits.items()},
            "signers": {
                "train": sorted(train_signers),
                "val": sorted(val_signers),
                "test": sorted(test_signers),
            },
        }

        return splits

    def get_split_info(self) -> Dict:
        """Return information about the last split performed."""
        if not hasattr(self, "split_info"):
            raise RuntimeError("No split has been performed yet")
        return self.split_info

    def print_split_summary(self):
        """Print a summary of the split."""
        info = self.get_split_info()
        print("Signer-Independent Split Summary")
        print("=" * 40)
        for split_name in ["train", "val", "test"]:
            print(
                f"{split_name.capitalize():>8}: {info['n_signers'][split_name]:>3} signers, "
                f"{info['n_samples'][split_name]:>5} samples"
            )


def create_signer_splits(
    csv_path: str, base_path: Optional[str] = None, config: Optional[SplitConfig] = None
) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    """
    Convenience function to create signer-independent splits from a CSV file.

    Args:
        csv_path: Path to the train.csv file
        base_path: Base path for landmark files (prepended to 'path' column)
        config: Split configuration

    Returns:
        Tuple of (splits dict, split info dict)
    """
    df = pd.read_csv(csv_path)

    # Add full path if base_path provided
    if base_path:
        df["full_path"] = df["path"].apply(lambda p: str(Path(base_path) / p))

    splitter = SignerIndependentSplitter(config)
    splits = splitter.split(df)

    splitter.print_split_summary()

    return splits, splitter.get_split_info()


# =============================================================================
# Dataset
# =============================================================================


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
    ):
        """
        Args:
            df: DataFrame with 'path' column (and optionally 'full_path')
            base_path: Base directory for landmark files
            config: Landmark processing configuration
            max_frames: Maximum sequence length (longer sequences are subsampled)
            augment: Whether to apply augmentation
            augment_fn: Custom augmentation function
        """
        self.df = df.reset_index(drop=True)
        self.base_path = Path(base_path)
        self.processor = LandmarkProcessor(config)
        self.max_frames = max_frames
        self.augment = augment
        self.augment_fn = augment_fn

    def __len__(self) -> int:
        return len(self.df)

    def _get_path(self, idx: int) -> str:
        """Get full path for a sample."""
        row = self.df.iloc[idx]
        if "full_path" in self.df.columns:
            return row["full_path"]
        return str(self.base_path / row["path"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
            - 'landmarks': (T, N, 3) tensor of all landmarks
            - 'left_hand': (T, 21, 3)
            - 'right_hand': (T, 21, 3)
            - 'pose': (T, 33, 3)
            - 'face': (T, N_face, 3)
            - 'length': original sequence length
        """
        path = self._get_path(idx)
        data = self.processor.process(path)

        # Apply augmentation if enabled
        if self.augment and self.augment_fn is not None:
            data = self.augment_fn(data)

        # Subsample if needed
        T = data["left_hand"].shape[0]
        if T > self.max_frames:
            indices = np.linspace(0, T - 1, self.max_frames).astype(int)
            for key in data:
                data[key] = data[key][indices]
            T = self.max_frames

        # Concatenate all landmarks
        landmarks = np.concatenate(
            [
                data["left_hand"],
                data["right_hand"],
                data["pose"],
                data["face"],
            ],
            axis=1,
        )

        # Convert to tensors
        result = {
            "landmarks": torch.tensor(landmarks, dtype=torch.float32),
            "left_hand": torch.tensor(data["left_hand"], dtype=torch.float32),
            "right_hand": torch.tensor(data["right_hand"], dtype=torch.float32),
            "pose": torch.tensor(data["pose"], dtype=torch.float32),
            "face": torch.tensor(data["face"], dtype=torch.float32),
            "length": torch.tensor(T, dtype=torch.long),
        }

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


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn: Callable = collate_vqvae,
) -> DataLoader:
    """Create a DataLoader with appropriate settings.

    Note: num_workers=0 is recommended for Kaggle environments.
    For local training with fast SSDs, use num_workers=2-4.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if num_workers > 0 else False,
        drop_last=False,
    )


# =============================================================================
# Normalization Modules
# =============================================================================


class RobustNormalization(nn.Module):
    """
    Robust normalization with fallback chain: nose -> shoulder center -> hip center.

    Addresses blindspot #7: Fragile nose normalization.
    """

    # Pose landmark indices (MediaPipe)
    NOSE_IDX = 0
    LEFT_SHOULDER_IDX = 11
    RIGHT_SHOULDER_IDX = 12
    LEFT_HIP_IDX = 23
    RIGHT_HIP_IDX = 24

    def __init__(
        self,
        pose_start_idx: int = 42,  # After left_hand (21) + right_hand (21)
        n_coords: int = 3,
        missing_threshold: float = 0.0,
    ):
        super().__init__()
        self.pose_start_idx = pose_start_idx
        self.n_coords = n_coords
        self.missing_threshold = missing_threshold

    def _get_pose_landmark(
        self, landmarks: torch.Tensor, landmark_idx: int
    ) -> torch.Tensor:
        """Get a specific pose landmark."""
        full_idx = self.pose_start_idx + landmark_idx
        return landmarks[:, :, full_idx, :]

    def _is_valid_landmark(self, landmark: torch.Tensor) -> torch.Tensor:
        """Check if a landmark is valid (not missing/zero)."""
        abs_vals = landmark.abs()
        return (abs_vals > self.missing_threshold).any(dim=-1)

    def _get_center(
        self, landmarks: torch.Tensor, idx1: int, idx2: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get center point between two landmarks."""
        lm1 = self._get_pose_landmark(landmarks, idx1)
        lm2 = self._get_pose_landmark(landmarks, idx2)

        valid1 = self._is_valid_landmark(lm1)
        valid2 = self._is_valid_landmark(lm2)

        center = (lm1 + lm2) / 2
        valid = valid1 & valid2

        return center, valid

    def forward(
        self, landmarks: torch.Tensor, return_origins: bool = False
    ) -> torch.Tensor:
        """Apply robust normalization."""
        B, T, N, C = landmarks.shape

        # Get candidate origins
        nose = self._get_pose_landmark(landmarks, self.NOSE_IDX)
        nose_valid = self._is_valid_landmark(nose)

        shoulder_center, shoulder_valid = self._get_center(
            landmarks, self.LEFT_SHOULDER_IDX, self.RIGHT_SHOULDER_IDX
        )

        hip_center, hip_valid = self._get_center(
            landmarks, self.LEFT_HIP_IDX, self.RIGHT_HIP_IDX
        )

        # Initialize origin with zeros (fallback)
        origin = torch.zeros(B, T, C, device=landmarks.device, dtype=landmarks.dtype)
        origin_type = torch.zeros(B, T, device=landmarks.device, dtype=torch.long)

        # Apply fallback chain (reverse priority so nose gets priority)
        # 3: hip center
        origin = torch.where(hip_valid.unsqueeze(-1), hip_center, origin)
        origin_type = torch.where(
            hip_valid, torch.full_like(origin_type, 3), origin_type
        )

        # 2: shoulder center
        origin = torch.where(shoulder_valid.unsqueeze(-1), shoulder_center, origin)
        origin_type = torch.where(
            shoulder_valid, torch.full_like(origin_type, 2), origin_type
        )

        # 1: nose (highest priority)
        origin = torch.where(nose_valid.unsqueeze(-1), nose, origin)
        origin_type = torch.where(
            nose_valid, torch.full_like(origin_type, 1), origin_type
        )

        # Subtract origin from all landmarks
        normalized = landmarks - origin.unsqueeze(2)

        if return_origins:
            return normalized, origin, origin_type

        return normalized


class ScaleNormalization(nn.Module):
    """
    Scale normalization based on body proportions (e.g., shoulder width).
    """

    def __init__(
        self,
        pose_start_idx: int = 42,
        target_shoulder_width: float = 0.4,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.pose_start_idx = pose_start_idx
        self.target_shoulder_width = target_shoulder_width
        self.eps = eps

        self.LEFT_SHOULDER_IDX = 11
        self.RIGHT_SHOULDER_IDX = 12

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Scale landmarks based on shoulder width."""
        # Get shoulder landmarks
        left_shoulder = landmarks[:, :, self.pose_start_idx + self.LEFT_SHOULDER_IDX, :]
        right_shoulder = landmarks[
            :, :, self.pose_start_idx + self.RIGHT_SHOULDER_IDX, :
        ]

        # Compute shoulder width (use only x,y for 2D distance)
        shoulder_diff = left_shoulder[:, :, :2] - right_shoulder[:, :, :2]
        shoulder_width = torch.norm(shoulder_diff, dim=-1, keepdim=True)  # (B, T, 1)

        # Compute per-sequence average shoulder width
        avg_width = shoulder_width.mean(dim=1, keepdim=True)  # (B, 1, 1)

        # Compute scale factor
        scale = self.target_shoulder_width / (avg_width + self.eps)

        # Apply scale to all landmarks
        scaled = landmarks * scale.unsqueeze(-1)

        return scaled


# =============================================================================
# Hand Dominance Module
# =============================================================================


class HandDominanceModule(nn.Module):
    """
    Detects hand dominance and reorders hands consistently.

    Addresses blindspot #5: No hand dominance normalization.
    """

    def __init__(
        self,
        hand_landmarks: int = 21,
        n_coords: int = 3,
        motion_smoothing: int = 3,
    ):
        super().__init__()
        self.hand_landmarks = hand_landmarks
        self.n_coords = n_coords
        self.motion_smoothing = motion_smoothing

    def compute_motion_energy(self, hand: torch.Tensor) -> torch.Tensor:
        """Compute motion energy for a hand sequence."""
        velocity = hand[:, 1:] - hand[:, :-1]
        motion_magnitude = torch.norm(velocity, dim=-1)
        total_motion = motion_magnitude.sum(dim=(1, 2))
        return total_motion

    def detect_dominant_hand(
        self, left_hand: torch.Tensor, right_hand: torch.Tensor
    ) -> torch.Tensor:
        """Detect which hand is dominant based on motion."""
        left_energy = self.compute_motion_energy(left_hand)
        right_energy = self.compute_motion_energy(right_hand)
        dominant = (right_energy > left_energy).long()
        return dominant

    def forward(
        self,
        left_hand: torch.Tensor,
        right_hand: torch.Tensor,
        return_swap_mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reorder hands so dominant hand is always first."""
        dominant_mask = self.detect_dominant_hand(left_hand, right_hand)

        B, T, N, C = left_hand.shape
        mask_expanded = dominant_mask.view(B, 1, 1, 1).expand_as(left_hand)

        dominant_hand = torch.where(mask_expanded == 0, left_hand, right_hand)
        non_dominant_hand = torch.where(mask_expanded == 0, right_hand, left_hand)

        if return_swap_mask:
            swap_mask = dominant_mask == 0
            return dominant_hand, non_dominant_hand, swap_mask

        return dominant_hand, non_dominant_hand


class TwoHandFusion(nn.Module):
    """
    Fuses information from both hands for signs that use both hands together.
    """

    def __init__(
        self,
        hand_dim: int = 63,
        hidden_dim: int = 128,
        output_dim: int = 64,
    ):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(hand_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.relative_encoder = nn.Sequential(
            nn.Linear(hand_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(
        self, dominant_hand: torch.Tensor, non_dominant_hand: torch.Tensor
    ) -> torch.Tensor:
        """Compute fused two-hand representation."""
        B, T, N, C = dominant_hand.shape

        dom_flat = dominant_hand.reshape(B, T, -1)
        nondom_flat = non_dominant_hand.reshape(B, T, -1)

        concat = torch.cat([dom_flat, nondom_flat], dim=-1)
        fused = self.fusion(concat)

        relative = dom_flat - nondom_flat
        relative_features = self.relative_encoder(relative)

        return fused + relative_features


# =============================================================================
# Temporal Augmentation
# =============================================================================


class TemporalAugmentation(nn.Module):
    """
    Temporal augmentation module for landmark sequences.

    Addresses blindspot #9: No temporal augmentation.
    """

    def __init__(
        self,
        speed_range: Tuple[float, float] = (0.8, 1.2),
        frame_dropout_prob: float = 0.1,
        temporal_jitter_std: float = 0.02,
        noise_std: float = 0.01,
        spatial_noise_std: float = 0.005,
    ):
        super().__init__()
        self.speed_range = speed_range
        self.frame_dropout_prob = frame_dropout_prob
        self.temporal_jitter_std = temporal_jitter_std
        self.noise_std = noise_std
        self.spatial_noise_std = spatial_noise_std

    def speed_augment(
        self, x: torch.Tensor, speed: Optional[float] = None
    ) -> torch.Tensor:
        """Apply speed variation by resampling the temporal dimension."""
        if speed is None:
            speed = np.random.uniform(*self.speed_range)

        B, T, N, C = x.shape
        new_T = int(T / speed)

        if new_T < 2:
            return x

        x_reshape = x.permute(0, 2, 3, 1).reshape(B, N * C, T)

        x_resampled = F.interpolate(
            x_reshape.float(), size=new_T, mode="linear", align_corners=True
        )

        x_out = x_resampled.reshape(B, N, C, new_T).permute(0, 3, 1, 2)

        return x_out

    def frame_dropout(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Randomly drop frames from the sequence."""
        B, T, N, C = x.shape
        device = x.device

        if T < 4:
            return x, mask

        keep_mask = torch.rand(B, T, device=device) > self.frame_dropout_prob

        keep_mask[:, 0] = True
        keep_mask[:, -1] = True

        min_frames = max(T // 2, 2)
        for b in range(B):
            if keep_mask[b].sum() < min_frames:
                drop_indices = (~keep_mask[b]).nonzero().squeeze(-1)
                n_to_add = min_frames - keep_mask[b].sum().item()
                if len(drop_indices) > 0:
                    add_indices = drop_indices[
                        torch.randperm(len(drop_indices))[: int(n_to_add)]
                    ]
                    keep_mask[b, add_indices] = True

        x_masked = x * keep_mask.unsqueeze(-1).unsqueeze(-1)

        x_interp = self._interpolate_dropped(x_masked, keep_mask)

        if mask is not None:
            mask = mask & keep_mask

        return x_interp, mask

    def _interpolate_dropped(
        self, x: torch.Tensor, keep_mask: torch.Tensor
    ) -> torch.Tensor:
        """Interpolate values for dropped frames."""
        B, T, N, C = x.shape
        result = x.clone()

        for b in range(B):
            mask = keep_mask[b]
            if mask.all():
                continue

            kept_indices = mask.nonzero().squeeze(-1).cpu().numpy()
            dropped_indices = (~mask).nonzero().squeeze(-1).cpu().numpy()

            if len(kept_indices) < 2:
                continue

            for n in range(N):
                for c in range(C):
                    kept_values = x[b, kept_indices, n, c].cpu().numpy()
                    interp_values = np.interp(
                        dropped_indices, kept_indices, kept_values
                    )
                    result[b, dropped_indices, n, c] = torch.tensor(
                        interp_values, device=x.device, dtype=x.dtype
                    )

        return result

    def temporal_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Add temporal jitter by slightly shifting coordinate values over time."""
        B, T, N, C = x.shape

        jitter_freq = max(1, T // 8)

        base_jitter = (
            torch.randn(B, jitter_freq, device=x.device) * self.temporal_jitter_std
        )

        base_jitter = base_jitter.unsqueeze(1)
        jitter = F.interpolate(base_jitter, size=T, mode="linear", align_corners=True)
        jitter = jitter.squeeze(1)

        velocity = x[:, 1:] - x[:, :-1]
        jitter_scale = jitter[:, 1:].unsqueeze(-1).unsqueeze(-1)

        x_jittered = x.clone()
        x_jittered[:, 1:] = x[:, 1:] + velocity * jitter_scale

        return x_jittered

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to coordinates."""
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

    def add_spatial_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add per-landmark spatial noise that's consistent across time."""
        B, T, N, C = x.shape

        spatial_noise = (
            torch.randn(B, 1, N, C, device=x.device) * self.spatial_noise_std
        )

        return x + spatial_noise

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply all augmentations."""
        if not training:
            return x, mask

        if torch.rand(1).item() < 0.5:
            x = self.speed_augment(x)
            if mask is not None and x.shape[1] != mask.shape[1]:
                new_T = x.shape[1]
                mask = (
                    F.interpolate(mask.float().unsqueeze(1), size=new_T, mode="nearest")
                    .squeeze(1)
                    .bool()
                )

        if torch.rand(1).item() < 0.5:
            x, mask = self.frame_dropout(x, mask)

        if torch.rand(1).item() < 0.5:
            x = self.temporal_jitter(x)

        if torch.rand(1).item() < 0.7:
            x = self.add_noise(x)

        if torch.rand(1).item() < 0.3:
            x = self.add_spatial_noise(x)

        return x, mask


# =============================================================================
# Multi-Scale Encoder
# =============================================================================


class TemporalConvBlock(nn.Module):
    """Single temporal convolution block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = (kernel_size - 1) // 2

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding),
            nn.BatchNorm1d(out_channels),
        )

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, 1)
        )

        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        if self.conv[0].stride[0] > 1:
            residual = F.avg_pool1d(residual, self.conv[0].stride[0])

        out = self.conv(x)

        if out.shape[2] != residual.shape[2]:
            residual = F.interpolate(
                residual, size=out.shape[2], mode="linear", align_corners=True
            )

        return self.activation(out + residual)


class ScaleEncoder(nn.Module):
    """Encoder for a single temporal scale."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        chunk_size: int,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.chunk_size = chunk_size

        layers = []
        in_dim = input_dim
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else output_dim
            layers.append(
                TemporalConvBlock(in_dim, out_dim, kernel_size=3, dropout=dropout)
            )
            in_dim = out_dim

        self.encoder = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, D = x.shape

        pad_len = (self.chunk_size - T % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            if mask is not None:
                mask = F.pad(mask.float(), (0, pad_len)).bool()

        T_padded = x.shape[1]
        n_chunks = T_padded // self.chunk_size

        x_chunks = x.reshape(B, n_chunks, self.chunk_size, D)

        x_conv = x_chunks.reshape(B * n_chunks, self.chunk_size, D).permute(0, 2, 1)

        encoded = self.encoder(x_conv)

        pooled = self.pool(encoded).squeeze(-1)

        output = pooled.reshape(B, n_chunks, -1)

        return output


class MultiScaleEncoder(nn.Module):
    """
    Multi-scale temporal encoder that processes at multiple chunk sizes.

    Addresses blindspot #3: Fixed chunk size.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        scales: Tuple[int, ...] = (4, 8, 16),
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.scales = scales
        self.output_dim = output_dim

        self.scale_encoders = nn.ModuleList(
            [
                ScaleEncoder(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    chunk_size=scale,
                    n_layers=n_layers,
                    dropout=dropout,
                )
                for scale in scales
            ]
        )

        self.fusion = nn.Sequential(
            nn.Linear(output_dim * len(scales), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self.cross_scale_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

    def _align_scales(
        self, scale_outputs: List[torch.Tensor], target_len: int
    ) -> torch.Tensor:
        """Align outputs from different scales to the same temporal length."""
        aligned = []
        for output in scale_outputs:
            if output.shape[1] != target_len:
                output = output.permute(0, 2, 1)
                output = F.interpolate(
                    output, size=target_len, mode="linear", align_corners=True
                )
                output = output.permute(0, 2, 1)
            aligned.append(output)

        return torch.cat(aligned, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_all_scales: bool = False,
    ) -> torch.Tensor:
        B, T, D = x.shape

        scale_outputs = []
        for encoder in self.scale_encoders:
            encoded = encoder(x, mask)
            scale_outputs.append(encoded)

        ref_idx = len(self.scales) // 2
        target_len = scale_outputs[ref_idx].shape[1]

        aligned = self._align_scales(scale_outputs, target_len)

        fused = self.fusion(aligned)

        fused_attn, _ = self.cross_scale_attn(fused, fused, fused)
        fused = fused + fused_attn

        if return_all_scales:
            return fused, scale_outputs

        return fused


class MultiScaleMotionEncoder(nn.Module):
    """
    Combines multi-scale encoding with motion/dynamics extraction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        scales: Tuple[int, ...] = (4, 8, 16),
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim

        self.pose_encoder = MultiScaleEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            scales=scales,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.motion_encoder = MultiScaleEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            scales=scales,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.dynamics_encoder = MultiScaleEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            scales=scales,
            n_layers=n_layers,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        velocity = x[:, 1:] - x[:, :-1]
        acceleration = velocity[:, 1:] - velocity[:, :-1]

        velocity = F.pad(velocity, (0, 0, 0, 1))
        acceleration = F.pad(acceleration, (0, 0, 0, 2))

        vel_mask = mask
        acc_mask = mask

        pose_latent = self.pose_encoder(x, mask)
        motion_latent = self.motion_encoder(velocity, vel_mask)
        dynamics_latent = self.dynamics_encoder(acceleration, acc_mask)

        return pose_latent, motion_latent, dynamics_latent


# =============================================================================
# Face Encoder
# =============================================================================

# Face region landmark ranges (for compact 134-landmark subset)
FACE_REGIONS = {
    "nose": (0, 10),
    "left_eye": (10, 26),
    "right_eye": (26, 42),
    "left_eyebrow": (42, 50),
    "right_eyebrow": (50, 58),
    "outer_mouth": (58, 78),
    "inner_mouth": (78, 98),
    "face_oval": (98, 134),
}

NMM_GROUPS = {
    "eyebrows": ["left_eyebrow", "right_eyebrow"],
    "eyes": ["left_eye", "right_eye"],
    "mouth": ["outer_mouth", "inner_mouth"],
    "face_shape": ["nose", "face_oval"],
}


class RegionEncoder(nn.Module):
    """Encoder for a single face region."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class FaceNMMEncoder(nn.Module):
    """
    Face encoder with region-specific processing for Non-Manual Markers.

    Addresses blindspot #2: Face underweighted.
    """

    def __init__(
        self,
        n_face_landmarks: int = 134,
        n_coords: int = 3,
        hidden_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.1,
        temporal_kernel: int = 5,
    ):
        super().__init__()
        self.n_face_landmarks = n_face_landmarks
        self.n_coords = n_coords
        self.output_dim = output_dim

        self.region_encoders = nn.ModuleDict()
        region_output_dim = hidden_dim // len(FACE_REGIONS)

        for region_name, (start, end) in FACE_REGIONS.items():
            n_landmarks = end - start
            input_dim = n_landmarks * n_coords
            self.region_encoders[region_name] = RegionEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=region_output_dim,
                dropout=dropout,
            )

        self.temporal_conv = nn.ModuleDict()
        for region_name in FACE_REGIONS:
            self.temporal_conv[region_name] = nn.Sequential(
                nn.Conv1d(
                    region_output_dim,
                    region_output_dim,
                    temporal_kernel,
                    padding=temporal_kernel // 2,
                ),
                nn.BatchNorm1d(region_output_dim),
                nn.GELU(),
            )

        self.nmm_fusion = nn.ModuleDict()
        group_input_dim = region_output_dim * 2
        for group_name in NMM_GROUPS:
            self.nmm_fusion[group_name] = nn.Sequential(
                nn.Linear(group_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim // len(NMM_GROUPS)),
            )

        self.final_fusion = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self.region_attention = nn.MultiheadAttention(
            embed_dim=region_output_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

    def _extract_region(
        self, face_landmarks: torch.Tensor, region_name: str
    ) -> torch.Tensor:
        """Extract landmarks for a specific region."""
        start, end = FACE_REGIONS[region_name]
        return face_landmarks[:, :, start:end, :]

    def forward(
        self, face_landmarks: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, N, C = face_landmarks.shape

        region_features = {}
        for region_name in FACE_REGIONS:
            region = self._extract_region(face_landmarks, region_name)
            region_flat = region.reshape(B, T, -1)

            encoded = self.region_encoders[region_name](region_flat)

            encoded_t = encoded.permute(0, 2, 1)
            encoded_t = self.temporal_conv[region_name](encoded_t)
            encoded = encoded_t.permute(0, 2, 1)

            region_features[region_name] = encoded

        all_regions = torch.stack(list(region_features.values()), dim=2)
        all_regions_flat = all_regions.reshape(B * T, len(FACE_REGIONS), -1)

        attended, _ = self.region_attention(
            all_regions_flat, all_regions_flat, all_regions_flat
        )
        attended = attended.reshape(B, T, len(FACE_REGIONS), -1)

        for i, region_name in enumerate(FACE_REGIONS):
            region_features[region_name] = (
                region_features[region_name] + attended[:, :, i]
            )

        group_features = []
        for group_name, region_names in NMM_GROUPS.items():
            group_concat = torch.cat([region_features[r] for r in region_names], dim=-1)
            group_feat = self.nmm_fusion[group_name](group_concat)
            group_features.append(group_feat)

        combined = torch.cat(group_features, dim=-1)

        output = self.final_fusion(combined)

        return output


class FaceTemporalEncoder(nn.Module):
    """
    Temporal encoder specifically for face NMM patterns.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.conv_scales = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(input_dim, hidden_dim, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                )
                for k in [3, 5, 7]
            ]
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, D = x.shape

        x_t = x.permute(0, 2, 1)
        conv_outputs = [conv(x_t) for conv in self.conv_scales]

        concat = torch.cat(conv_outputs, dim=1)
        concat = concat.permute(0, 2, 1)
        fused = self.fusion(concat)

        attn_mask = None
        if mask is not None:
            attn_mask = ~mask

        transformed = self.transformer(fused, src_key_padding_mask=attn_mask)

        output = self.output_proj(transformed)

        return output


class FaceChunkEncoder(nn.Module):
    """
    Chunk-based encoder for face that outputs per-chunk representations.
    """

    def __init__(
        self,
        n_face_landmarks: int = 134,
        n_coords: int = 3,
        hidden_dim: int = 256,
        output_dim: int = 128,
        chunk_size: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.output_dim = output_dim

        self.face_encoder = FaceNMMEncoder(
            n_face_landmarks=n_face_landmarks,
            n_coords=n_coords,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )

        self.temporal_encoder = FaceTemporalEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=2,
            dropout=dropout,
        )

        self.chunk_pool = nn.Sequential(
            nn.Linear(output_dim * chunk_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self, face_landmarks: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, N, C = face_landmarks.shape

        face_features = self.face_encoder(face_landmarks, mask)

        temporal_features = self.temporal_encoder(face_features, mask)

        pad_len = (self.chunk_size - T % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            temporal_features = F.pad(temporal_features, (0, 0, 0, pad_len))

        T_padded = temporal_features.shape[1]
        n_chunks = T_padded // self.chunk_size

        chunked = temporal_features.reshape(
            B, n_chunks, self.chunk_size, self.output_dim
        )
        chunked_flat = chunked.reshape(B, n_chunks, -1)

        output = self.chunk_pool(chunked_flat)

        return output


# =============================================================================
# Cross-Factor Attention
# =============================================================================


class CrossFactorAttention(nn.Module):
    """
    Cross-attention between different body part factors.

    Addresses blindspot #8: No cross-factor interaction.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        self.cross_attn_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "hand_to_pose": nn.MultiheadAttention(
                            embed_dim, num_heads, dropout, batch_first=True
                        ),
                        "pose_to_hand": nn.MultiheadAttention(
                            embed_dim, num_heads, dropout, batch_first=True
                        ),
                        "hand_to_face": nn.MultiheadAttention(
                            embed_dim, num_heads, dropout, batch_first=True
                        ),
                        "face_to_hand": nn.MultiheadAttention(
                            embed_dim, num_heads, dropout, batch_first=True
                        ),
                        "pose_to_face": nn.MultiheadAttention(
                            embed_dim, num_heads, dropout, batch_first=True
                        ),
                        "face_to_pose": nn.MultiheadAttention(
                            embed_dim, num_heads, dropout, batch_first=True
                        ),
                    }
                )
                for _ in range(n_layers)
            ]
        )

        self.layer_norms = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "dominant_hand": nn.LayerNorm(embed_dim),
                        "non_dominant_hand": nn.LayerNorm(embed_dim),
                        "pose": nn.LayerNorm(embed_dim),
                        "face": nn.LayerNorm(embed_dim),
                    }
                )
                for _ in range(n_layers)
            ]
        )

        self.ffns = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        factor: nn.Sequential(
                            nn.Linear(embed_dim, embed_dim * 4),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(embed_dim * 4, embed_dim),
                            nn.Dropout(dropout),
                        )
                        for factor in [
                            "dominant_hand",
                            "non_dominant_hand",
                            "pose",
                            "face",
                        ]
                    }
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norms = nn.ModuleDict(
            {
                "dominant_hand": nn.LayerNorm(embed_dim),
                "non_dominant_hand": nn.LayerNorm(embed_dim),
                "pose": nn.LayerNorm(embed_dim),
                "face": nn.LayerNorm(embed_dim),
            }
        )

    def forward(
        self, factors: Dict[str, torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        dom_hand = factors["dominant_hand"]
        nondom_hand = factors["non_dominant_hand"]
        pose = factors["pose"]
        face = factors["face"]

        attn_mask = None
        if mask is not None:
            attn_mask = ~mask

        for layer_idx in range(self.n_layers):
            cross_attn = self.cross_attn_layers[layer_idx]
            layer_norm = self.layer_norms[layer_idx]
            ffn = self.ffns[layer_idx]

            # Hand-Pose Cross-Attention
            dom_to_pose, _ = cross_attn["hand_to_pose"](
                dom_hand, pose, pose, key_padding_mask=attn_mask
            )
            dom_hand = layer_norm["dominant_hand"](dom_hand + dom_to_pose)

            nondom_to_pose, _ = cross_attn["hand_to_pose"](
                nondom_hand, pose, pose, key_padding_mask=attn_mask
            )
            nondom_hand = layer_norm["non_dominant_hand"](nondom_hand + nondom_to_pose)

            hands_combined = (dom_hand + nondom_hand) / 2
            pose_to_hand, _ = cross_attn["pose_to_hand"](
                pose, hands_combined, hands_combined, key_padding_mask=attn_mask
            )
            pose = layer_norm["pose"](pose + pose_to_hand)

            # Hand-Face Cross-Attention
            dom_to_face, _ = cross_attn["hand_to_face"](
                dom_hand, face, face, key_padding_mask=attn_mask
            )
            dom_hand = dom_hand + dom_to_face

            nondom_to_face, _ = cross_attn["hand_to_face"](
                nondom_hand, face, face, key_padding_mask=attn_mask
            )
            nondom_hand = nondom_hand + nondom_to_face

            face_to_hand, _ = cross_attn["face_to_hand"](
                face, hands_combined, hands_combined, key_padding_mask=attn_mask
            )
            face = layer_norm["face"](face + face_to_hand)

            # Pose-Face Cross-Attention
            pose_to_face, _ = cross_attn["pose_to_face"](
                pose, face, face, key_padding_mask=attn_mask
            )
            pose = pose + pose_to_face

            face_to_pose, _ = cross_attn["face_to_pose"](
                face, pose, pose, key_padding_mask=attn_mask
            )
            face = face + face_to_pose

            # FFN
            dom_hand = dom_hand + ffn["dominant_hand"](dom_hand)
            nondom_hand = nondom_hand + ffn["non_dominant_hand"](nondom_hand)
            pose = pose + ffn["pose"](pose)
            face = face + ffn["face"](face)

        return {
            "dominant_hand": self.final_norms["dominant_hand"](dom_hand),
            "non_dominant_hand": self.final_norms["non_dominant_hand"](nondom_hand),
            "pose": self.final_norms["pose"](pose),
            "face": self.final_norms["face"](face),
        }


class FactorFusion(nn.Module):
    """
    Fuses multiple factor representations into a single representation.
    """

    def __init__(
        self,
        factor_dim: int = 128,
        output_dim: int = 256,
        n_factors: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.factor_weights = nn.Parameter(torch.ones(n_factors))

        self.fusion = nn.Sequential(
            nn.Linear(factor_dim * n_factors, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
        )

        self.attn_pool = nn.Sequential(
            nn.Linear(factor_dim, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, factors: Dict[str, torch.Tensor]) -> torch.Tensor:
        factor_list = [
            factors["dominant_hand"],
            factors["non_dominant_hand"],
            factors["pose"],
            factors["face"],
        ]
        stacked = torch.stack(factor_list, dim=2)

        weights = F.softmax(self.factor_weights, dim=0)
        weighted = stacked * weights.view(1, 1, -1, 1)

        B, T, N, D = weighted.shape
        concat = weighted.reshape(B, T, N * D)

        fused = self.fusion(concat)

        return fused


# =============================================================================
# Vector Quantizer
# =============================================================================


class EMAVectorQuantizer(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average updates.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
        reset_threshold: float = 0.01,
        reset_patience: int = 100,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        self.ema_decay = ema_decay
        self.reset_threshold = reset_threshold
        self.reset_patience = reset_patience
        self.epsilon = epsilon

        self.register_buffer("embeddings", torch.randn(num_embeddings, embedding_dim))
        nn.init.xavier_uniform_(self.embeddings)

        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_embed_sum", self.embeddings.clone())

        self.register_buffer("usage_count", torch.zeros(num_embeddings))
        self.register_buffer("steps_since_reset", torch.tensor(0))

    def forward(
        self, z: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        input_shape = z.shape
        D = input_shape[-1]

        z_flat = z.reshape(-1, D)

        z_sq = (z_flat**2).sum(dim=1, keepdim=True)
        e_sq = (self.embeddings**2).sum(dim=1)
        ze = torch.matmul(z_flat, self.embeddings.t())
        distances = z_sq + e_sq - 2 * ze

        indices = distances.argmin(dim=1)
        z_q_flat = self.embeddings[indices]

        if training and self.training:
            self._ema_update(z_flat, indices)

        z_q = z_q_flat.reshape(input_shape)
        indices = indices.reshape(input_shape[:-1])

        commitment_loss = F.mse_loss(z_q.detach(), z)

        z_q = z + (z_q - z).detach()

        losses = {
            "commitment_loss": commitment_loss * self.commitment_weight,
            "vq_loss": commitment_loss * self.commitment_weight,
        }

        return z_q, indices, losses

    def _ema_update(self, z_flat: torch.Tensor, indices: torch.Tensor):
        """Update codebook using EMA."""
        # Cast z_flat to buffer dtype for AMP compatibility
        z_flat_fp32 = z_flat.to(self.embeddings.dtype)

        one_hot = F.one_hot(indices, self.num_embeddings).to(self.embeddings.dtype)

        cluster_size = one_hot.sum(dim=0)
        self.ema_cluster_size.mul_(self.ema_decay).add_(
            cluster_size, alpha=1 - self.ema_decay
        )

        embed_sum = torch.matmul(one_hot.t(), z_flat_fp32)
        self.ema_embed_sum.mul_(self.ema_decay).add_(
            embed_sum, alpha=1 - self.ema_decay
        )

        n = self.ema_cluster_size.unsqueeze(1)
        self.embeddings.copy_(self.ema_embed_sum / (n + self.epsilon))

        self.usage_count.add_(cluster_size)
        self.steps_since_reset.add_(1)

        if self.steps_since_reset >= self.reset_patience:
            self._maybe_reset_codes(z_flat_fp32)

    def _maybe_reset_codes(self, z_flat: torch.Tensor):
        """Reset underused codes."""
        total_usage = self.usage_count.sum()
        if total_usage == 0:
            return

        usage_rate = self.usage_count / total_usage

        underused = usage_rate < self.reset_threshold

        if underused.any():
            num_reset = underused.sum().item()
            random_indices = torch.randint(
                0, z_flat.shape[0], (num_reset,), device=z_flat.device
            )
            new_embeddings = z_flat[random_indices]

            noise = torch.randn_like(new_embeddings) * 0.01
            new_embeddings = new_embeddings + noise

            # Cast to match buffer dtype (handles AMP float16 -> float32)
            new_embeddings = new_embeddings.to(self.embeddings.dtype)

            self.embeddings[underused] = new_embeddings
            self.ema_embed_sum[underused] = new_embeddings
            self.ema_cluster_size[underused] = 1.0

        self.usage_count.zero_()
        self.steps_since_reset.zero_()

    def get_codebook_usage(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute histogram of codebook usage."""
        return torch.bincount(indices.flatten(), minlength=self.num_embeddings).float()


class DiversityLoss(nn.Module):
    """
    Loss to encourage diverse codebook usage.
    """

    def __init__(self, target_perplexity_ratio: float = 0.8):
        super().__init__()
        self.target_perplexity_ratio = target_perplexity_ratio

    def forward(self, indices: torch.Tensor, num_embeddings: int) -> torch.Tensor:
        usage = torch.bincount(indices.flatten(), minlength=num_embeddings).float()
        probs = usage / (usage.sum() + 1e-10)

        target_probs = torch.ones_like(probs) / num_embeddings

        kl_div = F.kl_div((probs + 1e-10).log(), target_probs, reduction="sum")

        return kl_div


class FactorizedVectorQuantizer(nn.Module):
    """
    Factorized quantizer with multiple codebooks.
    """

    def __init__(
        self,
        codebook_configs: Dict[str, Tuple[int, int]],
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()

        self.codebook_names = list(codebook_configs.keys())
        self.quantizers = nn.ModuleDict()

        for name, (num_codes, dim) in codebook_configs.items():
            self.quantizers[name] = EMAVectorQuantizer(
                num_embeddings=num_codes,
                embedding_dim=dim,
                commitment_weight=commitment_weight,
                ema_decay=ema_decay,
            )

    def forward(
        self, latents: Dict[str, torch.Tensor], training: bool = True
    ) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]
    ]:
        quantized = {}
        indices = {}
        losses = {}

        for name in self.codebook_names:
            if name in latents:
                z_q, idx, loss = self.quantizers[name](latents[name], training)
                quantized[name] = z_q
                indices[name] = idx
                losses[name] = loss

        total_loss = sum(l["vq_loss"] for l in losses.values())
        losses["total"] = {"vq_loss": total_loss}

        return quantized, indices, losses


# =============================================================================
# Decoder
# =============================================================================


class Decoder(nn.Module):
    """Decoder to reconstruct landmarks from quantized representations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        chunk_size: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.output_dim = output_dim

        self.upsample = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * chunk_size),
            nn.GELU(),
        )

        self.temporal_refine = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor, target_len: int) -> torch.Tensor:
        B, n_chunks, D = z.shape

        upsampled = self.upsample(z)
        upsampled = upsampled.reshape(B, n_chunks * self.chunk_size, -1)

        upsampled_t = upsampled.permute(0, 2, 1)
        refined = self.temporal_refine(upsampled_t)
        refined = refined.permute(0, 2, 1)

        if refined.shape[1] != target_len:
            refined_t = refined.permute(0, 2, 1)
            refined_t = F.interpolate(
                refined_t, size=target_len, mode="linear", align_corners=True
            )
            refined = refined_t.permute(0, 2, 1)

        output = self.output_proj(refined)

        return output


# =============================================================================
# Improved VQ-VAE Model
# =============================================================================


class ImprovedVQVAE(nn.Module):
    """
    Improved VQ-VAE for sign language tokenization.

    Addresses all 10 blindspots:
    1. Z-coordinate: Full 3D landmarks
    2. Face underweighted: Dedicated FaceNMMEncoder
    3. Fixed chunk size: Multi-scale encoders
    4. Signer-dependent: Handled in data pipeline
    5. Hand dominance: HandDominanceModule
    6. Domain shift: Trained on combined datasets
    7. Fragile normalization: RobustNormalization
    8. No cross-factor: CrossFactorAttention
    9. No temporal augmentation: TemporalAugmentation
    10. Suboptimal codebooks: Increased sizes
    """

    def __init__(self, config: Optional[ImprovedVQVAEConfig] = None):
        super().__init__()
        self.config = config or ImprovedVQVAEConfig()

        # Normalization modules
        self.robust_norm = RobustNormalization(
            pose_start_idx=42,
            n_coords=self.config.n_coords,
        )
        self.scale_norm = ScaleNormalization(pose_start_idx=42)

        # Hand dominance
        self.hand_dominance = HandDominanceModule(
            hand_landmarks=self.config.hand_landmarks,
            n_coords=self.config.n_coords,
        )

        # Two-hand fusion
        self.two_hand_fusion = TwoHandFusion(
            hand_dim=self.config.hand_dim,
            hidden_dim=self.config.encoder_hidden_dim,
            output_dim=self.config.embed_dim,
        )

        # Augmentation
        self.augmentation = TemporalAugmentation(
            speed_range=self.config.augment_speed_range,
            frame_dropout_prob=self.config.augment_frame_dropout_prob,
            temporal_jitter_std=self.config.augment_temporal_jitter_std,
        )

        # Multi-scale encoders for hands and pose
        self.dominant_hand_encoder = MultiScaleMotionEncoder(
            input_dim=self.config.hand_dim,
            hidden_dim=self.config.encoder_hidden_dim,
            output_dim=self.config.embed_dim,
            scales=self.config.temporal_scales,
            n_layers=self.config.encoder_n_layers,
            dropout=self.config.encoder_dropout,
        )

        self.non_dominant_hand_encoder = MultiScaleMotionEncoder(
            input_dim=self.config.hand_dim,
            hidden_dim=self.config.encoder_hidden_dim,
            output_dim=self.config.embed_dim,
            scales=self.config.temporal_scales,
            n_layers=self.config.encoder_n_layers,
            dropout=self.config.encoder_dropout,
        )

        self.pose_encoder = MultiScaleMotionEncoder(
            input_dim=self.config.pose_dim,
            hidden_dim=self.config.encoder_hidden_dim,
            output_dim=self.config.embed_dim,
            scales=self.config.temporal_scales,
            n_layers=self.config.encoder_n_layers,
            dropout=self.config.encoder_dropout,
        )

        # Face encoder
        self.face_encoder = FaceChunkEncoder(
            n_face_landmarks=self.config.face_landmarks,
            n_coords=self.config.n_coords,
            hidden_dim=self.config.encoder_hidden_dim,
            output_dim=self.config.embed_dim,
            chunk_size=self.config.base_chunk_size,
            dropout=self.config.encoder_dropout,
        )

        # Cross-factor attention
        if self.config.use_cross_attention:
            self.cross_attention = CrossFactorAttention(
                embed_dim=self.config.embed_dim,
                num_heads=self.config.cross_attention_heads,
                n_layers=self.config.cross_attention_layers,
                dropout=self.config.encoder_dropout,
            )

        # Factor fusion
        self.factor_fusion = FactorFusion(
            factor_dim=self.config.embed_dim,
            output_dim=self.config.latent_dim,
            n_factors=4,
            dropout=self.config.encoder_dropout,
        )

        # Vector quantizers
        self.quantizers = FactorizedVectorQuantizer(
            codebook_configs={
                "pose": (self.config.pose_codebook_size, self.config.embed_dim),
                "motion": (self.config.motion_codebook_size, self.config.embed_dim),
                "dynamics": (self.config.dynamics_codebook_size, self.config.embed_dim),
                "face": (self.config.face_codebook_size, self.config.embed_dim),
            },
            commitment_weight=self.config.commitment_weight,
            ema_decay=self.config.ema_decay,
        )

        # Diversity loss
        self.diversity_loss = DiversityLoss()

        # Decoder
        self.decoder = Decoder(
            input_dim=self.config.latent_dim,
            hidden_dim=self.config.decoder_hidden_dim,
            output_dim=self.config.get_total_input_dim(),
            chunk_size=self.config.base_chunk_size,
            n_layers=self.config.decoder_n_layers,
            dropout=self.config.encoder_dropout,
        )

    def _extract_body_parts(self, landmarks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract different body parts from combined landmarks tensor."""
        return {
            "left_hand": landmarks[:, :, :21],
            "right_hand": landmarks[:, :, 21:42],
            "pose": landmarks[:, :, 42:75],
            "face": landmarks[:, :, 75:],
        }

    def encode(
        self,
        landmarks: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Encode landmarks to quantized representations."""
        B, T, N, C = landmarks.shape

        # Apply normalization
        landmarks = self.robust_norm(landmarks)
        landmarks = self.scale_norm(landmarks)

        # Extract body parts
        parts = self._extract_body_parts(landmarks)

        # Apply hand dominance reordering
        dom_hand, nondom_hand = self.hand_dominance(
            parts["left_hand"], parts["right_hand"]
        )

        # Flatten landmark dimensions for encoding
        dom_hand_flat = dom_hand.reshape(B, T, -1)
        nondom_hand_flat = nondom_hand.reshape(B, T, -1)
        pose_flat = parts["pose"].reshape(B, T, -1)

        # Encode each factor
        dom_pose, dom_motion, dom_dynamics = self.dominant_hand_encoder(
            dom_hand_flat, mask
        )
        nondom_pose, nondom_motion, nondom_dynamics = self.non_dominant_hand_encoder(
            nondom_hand_flat, mask
        )
        body_pose, body_motion, body_dynamics = self.pose_encoder(pose_flat, mask)
        face_features = self.face_encoder(parts["face"], mask)

        # Aggregate pose features
        pose_combined = (dom_pose + nondom_pose + body_pose) / 3
        motion_combined = (dom_motion + nondom_motion + body_motion) / 3
        dynamics_combined = (dom_dynamics + nondom_dynamics + body_dynamics) / 3

        # Apply cross-factor attention if enabled
        if self.config.use_cross_attention:
            n_chunks = pose_combined.shape[1]

            # Align face features to same chunk count
            if face_features.shape[1] != n_chunks:
                face_t = face_features.permute(0, 2, 1)
                face_t = F.interpolate(
                    face_t, size=n_chunks, mode="linear", align_corners=True
                )
                face_features = face_t.permute(0, 2, 1)

            factors = {
                "dominant_hand": dom_pose,
                "non_dominant_hand": nondom_pose,
                "pose": body_pose,
                "face": face_features,
            }

            attended_factors = self.cross_attention(factors)

            pose_combined = (
                attended_factors["dominant_hand"]
                + attended_factors["non_dominant_hand"]
                + attended_factors["pose"]
            ) / 3

        # Prepare latents for quantization
        latents = {
            "pose": pose_combined,
            "motion": motion_combined,
            "dynamics": dynamics_combined,
            "face": face_features,
        }

        # Quantize
        quantized, indices, vq_losses = self.quantizers(latents, training=self.training)

        return quantized, indices, vq_losses

    def decode(
        self, quantized: Dict[str, torch.Tensor], target_len: int
    ) -> torch.Tensor:
        """Decode quantized representations back to landmarks."""
        factors = {
            "dominant_hand": quantized["pose"],
            "non_dominant_hand": quantized["pose"],
            "pose": quantized["motion"],
            "face": quantized.get("face", quantized["dynamics"]),
        }
        fused = self.factor_fusion(factors)

        reconstructed = self.decoder(fused, target_len)

        return reconstructed

    def forward(
        self,
        landmarks: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: encode, quantize, decode."""
        B, T, N, C = landmarks.shape

        # Apply augmentation during training
        if self.training:
            landmarks, mask = self.augmentation(landmarks, mask, training=True)
            T = landmarks.shape[1]

        # Encode and quantize
        quantized, indices, vq_losses = self.encode(landmarks, mask)

        # Decode
        reconstructed = self.decode(quantized, T)

        # Compute reconstruction loss
        landmarks_flat = landmarks.reshape(B, T, -1)
        recon_loss = F.mse_loss(reconstructed, landmarks_flat)

        # Compute velocity reconstruction loss
        target_velocity = landmarks_flat[:, 1:] - landmarks_flat[:, :-1]
        pred_velocity = reconstructed[:, 1:] - reconstructed[:, :-1]
        velocity_loss = F.mse_loss(pred_velocity, target_velocity)

        # Compute diversity losses
        diversity_losses = {}
        for name, idx in indices.items():
            codebook_size = getattr(self.config, f"{name}_codebook_size", 256)
            diversity_losses[name] = self.diversity_loss(idx, codebook_size)

        total_diversity = sum(diversity_losses.values()) / len(diversity_losses)

        # Aggregate losses
        losses = {
            "reconstruction": recon_loss * self.config.reconstruction_weight,
            "velocity_reconstruction": velocity_loss
            * self.config.velocity_reconstruction_weight,
            "vq": vq_losses["total"]["vq_loss"],
            "diversity": total_diversity * self.config.codebook_diversity_weight,
        }
        losses["total"] = sum(losses.values())

        return {
            "reconstructed": reconstructed,
            "indices": indices,
            "quantized": quantized,
            "losses": losses,
        }

    def tokenize(
        self, landmarks: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Tokenize landmarks into discrete codes."""
        self.eval()
        with torch.no_grad():
            _, indices, _ = self.encode(landmarks, mask)
        return indices

    def get_codebook_embeddings(self) -> Dict[str, torch.Tensor]:
        """Get all codebook embeddings for downstream use."""
        return {
            name: quantizer.embeddings.data
            for name, quantizer in self.quantizers.quantizers.items()
        }


# =============================================================================
# Training Functions
# =============================================================================


def train_epoch(
    model: ImprovedVQVAE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Train for one epoch with optional mixed precision."""
    model.train()
    total_losses = {
        "total": 0,
        "reconstruction": 0,
        "velocity_reconstruction": 0,
        "vq": 0,
        "diversity": 0,
    }
    n_batches = 0

    # Determine if we should use AMP
    use_amp = use_amp and scaler is not None and device.type == "cuda"

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move to device
        landmarks = batch["landmarks"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()

        # Forward pass with optional mixed precision
        if use_amp:
            with autocast(device_type="cuda"):
                outputs = model(landmarks, mask)
                loss = outputs["losses"]["total"]

            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(landmarks, mask)
            loss = outputs["losses"]["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Accumulate losses
        for key in total_losses:
            if key in outputs["losses"]:
                total_losses[key] += outputs["losses"][key].item()
        n_batches += 1

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "recon": f"{outputs['losses']['reconstruction'].item():.4f}",
            }
        )

        # Periodic cache clearing to prevent OOM
        if n_batches % 200 == 0:
            torch.cuda.empty_cache()

    # Average losses
    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def validate(
    model: ImprovedVQVAE,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Validate model with optional mixed precision."""
    model.eval()
    total_losses = {
        "total": 0,
        "reconstruction": 0,
        "velocity_reconstruction": 0,
        "vq": 0,
        "diversity": 0,
    }
    codebook_usage = {
        name: torch.zeros(model.config.pose_codebook_size) for name in ["pose"]
    }
    n_batches = 0

    # Determine if we should use AMP
    use_amp = use_amp and device.type == "cuda"

    for batch in tqdm(dataloader, desc="Validation"):
        landmarks = batch["landmarks"].to(device)
        mask = batch["mask"].to(device)

        if use_amp:
            with autocast(device_type="cuda"):
                outputs = model(landmarks, mask)
        else:
            outputs = model(landmarks, mask)

        for key in total_losses:
            if key in outputs["losses"]:
                total_losses[key] += outputs["losses"][key].item()

        # Track codebook usage
        for name, indices in outputs["indices"].items():
            usage = torch.bincount(
                indices.flatten().cpu(),
                minlength=model.quantizers.quantizers[name].num_embeddings,
            ).float()
            if name not in codebook_usage:
                codebook_usage[name] = usage
            else:
                codebook_usage[name] += usage

        n_batches += 1

    # Average losses
    avg_losses = {k: v / n_batches for k, v in total_losses.items()}

    # Compute codebook utilization
    for name in codebook_usage:
        total = codebook_usage[name].sum()
        if total > 0:
            probs = codebook_usage[name] / total
            used = (probs > 0.001).sum().item()
            total_codes = len(probs)
            avg_losses[f"{name}_utilization"] = used / total_codes

    return avg_losses


def save_checkpoint(
    model: ImprovedVQVAE,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    losses: Dict[str, float],
    save_path: str,
):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "losses": losses,
        "config": asdict(model.config),
    }
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: ImprovedVQVAE,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> int:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return checkpoint["epoch"]


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train Improved VQ-VAE")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/Isolated_ASL_Recognition",
        help="Path to data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/vqvae",
        help="Output directory for checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=10, help="Patience count")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--num-workers", type=int, default=0, help="DataLoader workers (0 for Kaggle)"
    )
    parser.add_argument(
        "--use-amp", action="store_true", default=True, help="Use mixed precision (AMP)"
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    # Handle AMP flag
    use_amp = args.use_amp and not args.no_amp

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Create config
    config = ImprovedVQVAEConfig(
        device=str(device),
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        max_patience=args.patience,
    )

    # Save config
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Create data loaders
    print("Loading data...")
    csv_path = os.path.join(args.data_dir, "train.csv")
    splits, split_info = create_signer_splits(csv_path, args.data_dir)

    # Landmark config with compact face subset
    landmark_config = LandmarkConfig(
        include_z=True,
        face_subset=FACE_LANDMARK_SUBSETS["compact"],
    )

    train_dataset = VQVAEDataset(
        df=splits["train"],
        base_path=args.data_dir,
        config=landmark_config,
        augment=True,
    )

    val_dataset = VQVAEDataset(
        df=splits["val"],
        base_path=args.data_dir,
        config=landmark_config,
        augment=False,
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_vqvae,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_vqvae,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create model
    model = ImprovedVQVAE(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create GradScaler for mixed precision training
    scaler = GradScaler("cuda") if use_amp and device.type == "cuda" else None
    if scaler:
        print("Mixed precision training (AMP) enabled")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.max_epochs,
        eta_min=config.learning_rate / 100,
    )

    # Resume if specified
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(start_epoch, config.max_epochs):
        # Train
        train_losses = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            scaler=scaler,
            use_amp=use_amp,
        )

        # Validate
        val_losses = validate(model, val_loader, device, use_amp=use_amp)

        # Step scheduler
        scheduler.step()

        # Log
        print(f"\nEpoch {epoch}:")
        print(
            f"  Train - Loss: {train_losses['total']:.4f}, "
            f"Recon: {train_losses['reconstruction']:.4f}, "
            f"VQ: {train_losses['vq']:.4f}"
        )
        print(
            f"  Val   - Loss: {val_losses['total']:.4f}, "
            f"Recon: {val_losses['reconstruction']:.4f}"
        )

        # Print codebook utilization
        for name in ["pose", "motion", "dynamics", "face"]:
            util_key = f"{name}_utilization"
            if util_key in val_losses:
                print(
                    f"  {name.capitalize()} codebook utilization: {val_losses[util_key]:.1%}"
                )

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_losses,
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt"),
            )

        # Save best model
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_losses,
                os.path.join(args.output_dir, "best_model.pt"),
            )
            print(f"  New best model saved!")

            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > config.max_patience:
            break

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
