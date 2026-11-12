# # Sign Language VQ-VAE with Temporal Dynamics
#
# ## Improved Version: Factorized Codebooks for Pose + Motion
#
# This notebook addresses a key limitation: sign language requires understanding both:
# 1. **WHAT** - The pose/handshape (static)
# 2. **HOW** - The motion dynamics (speed, acceleration, rhythm)
#
# We use **factorized codebooks**:
# - **Pose Codebook**: Captures hand configurations and body positions
# - **Motion Codebook**: Captures velocity patterns (direction + speed)
# - **Dynamics Codebook**: Captures acceleration (sharp vs smooth)
#
# This is analogous to how speech has phonemes (sounds) AND prosody (rhythm, stress, intonation).
#
# ## Datasets Supported
# - **Isolated ASL Recognition** (Google ASL Signs) - Long format
# - **WLASL Landmarks** - Long format (same as above)
# - **ASL Fingerspelling** - Wide format (different structure)


import json
import os
import math
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
print(f"Device: {device}")


@dataclass
class TemporalVQVAEConfig:
    # Chunk settings
    chunk_size: int = 8
    chunk_stride: int = 4

    # Feature dimensions (will be computed)
    pose_dim: int = 418  # Position features
    motion_dim: int = 418  # Velocity features
    dynamics_dim: int = 418  # Acceleration features

    # Codebook settings - FACTORIZED
    pose_codes: int = 384  # Codebook for poses/handshapes
    motion_codes: int = 192  # Codebook for velocity patterns
    dynamics_codes: int = 96  # Codebook for acceleration patterns

    embed_dim: int = 128  # Embedding dimension per codebook
    hidden_dim: int = 512

    # Training
    batch_size: int = 256
    learning_rate: float = 3e-4
    epochs: int = 100
    commitment_cost: float = 0.01  # Lower to prevent codebook collapse (was 0.25)

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.99


config = TemporalVQVAEConfig()
print(
    f"Factorized codebooks: Pose={config.pose_codes}, Motion={config.motion_codes}, Dynamics={config.dynamics_codes}"
)
print(
    f"Total vocabulary: {config.pose_codes * config.motion_codes * config.dynamics_codes:,} possible combinations"
)
print(
    f"Commitment cost: {config.commitment_cost} (lower helps prevent codebook collapse)"
)

# ============== DATASET CONFIGURATION ==============
# Auto-detect environment (Kaggle vs Local)
# ALL datasets use LONG format (frame, type, landmark_index, x, y, z)
# Fingerspelling must be converted first using convert-fingerspelling-to-long.py

IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:
    # Kaggle paths - use input datasets directly or outputs from other notebooks
    DATASETS = {
        "isolated_asl": {
            "base_path": "/kaggle/input/asl-signs",
            "train_csv": "/kaggle/input/asl-signs/train.csv",
            "enabled": True,
        },
        "wlasl": {
            "base_path": "/kaggle/input/preprocess-wlasl-mediapipe/wlasl_landmarks",
            "train_csv": "/kaggle/input/preprocess-wlasl-mediapipe/wlasl_landmarks/train.csv",
            "enabled": True,
        },
        "fingerspelling": {
            # Use converted long-format version
            "base_path": "/kaggle/input/convert-fingerspelling-to-long/fingerspelling_landmarks",
            "train_csv": "/kaggle/input/convert-fingerspelling-to-long/fingerspelling_landmarks/train.csv",
            "enabled": True,
        },
    }
else:
    # Local paths
    DATA_ROOT = "/Users/shravnchandr/Projects/Isolated_Sign_Language_Recognition/data"
    DATASETS = {
        "isolated_asl": {
            "base_path": f"{DATA_ROOT}/Isolated_ASL_Recognition",
            "train_csv": f"{DATA_ROOT}/Isolated_ASL_Recognition/train.csv",
            "enabled": True,
        },
        "wlasl": {
            "base_path": f"{DATA_ROOT}/WLASL_Landmarks",
            "train_csv": f"{DATA_ROOT}/WLASL_Landmarks/train.csv",
            "enabled": True,
        },
        "fingerspelling": {
            # Use converted long-format version
            "base_path": f"{DATA_ROOT}/Fingerspelling_Long_Format",
            "train_csv": f"{DATA_ROOT}/Fingerspelling_Long_Format/train.csv",
            "enabled": True,
        },
    }

print(f"Environment: {'Kaggle' if IS_KAGGLE else 'Local'}")
for name, cfg in DATASETS.items():
    exists = os.path.exists(cfg["base_path"])
    print(f"  {name}: {'✓' if exists else '✗'} (long format)")

# ============== LANDMARK CONFIGURATION ==============
# Selected face landmarks (key regions for sign language)

FACE_LANDMARKS = {
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
SELECTED_FACE = sorted(set([i for v in FACE_LANDMARKS.values() for i in v]))


def generate_columns():
    """Generate standardized column names for our feature vector"""
    specs = {"left_hand": 21, "pose": 33, "right_hand": 21}
    columns = []
    for lm_type, count in specs.items():
        for i in range(count):
            for ax in ["x", "y"]:
                columns.append(f"{lm_type}_{i}_{ax}")
    for face_idx in SELECTED_FACE:
        for ax in ["x", "y"]:
            columns.append(f"face_{face_idx}_{ax}")
    return columns


ALL_COLUMNS = generate_columns()
POSE_DIM = len(ALL_COLUMNS)

# Update config
config.pose_dim = POSE_DIM
config.motion_dim = POSE_DIM
config.dynamics_dim = POSE_DIM

print(f"Feature dimension: {POSE_DIM}")
print(f"  - Hands: {21*2*2} (21 landmarks × 2 hands × 2 coords)")
print(f"  - Pose: {33*2} (33 landmarks × 2 coords)")
print(f"  - Face: {len(SELECTED_FACE)*2} ({len(SELECTED_FACE)} landmarks × 2 coords)")

# ============== DATA LOADING FUNCTIONS ==============


def load_long_format(file_path: str, base_path: str) -> Optional[np.ndarray]:
    """
    Load parquet in LONG format (Isolated ASL, WLASL).
    Columns: frame, type, landmark_index, x, y, z
    Returns: (T, D) array where D = POSE_DIM
    """
    try:
        full_path = os.path.join(base_path, file_path)
        if not os.path.exists(full_path):
            return None

        df = pd.read_parquet(full_path)

        # Filter face landmarks to selected subset
        face_df = df[df["type"] == "face"]
        face_df = face_df[face_df["landmark_index"].isin(SELECTED_FACE)]
        other_df = df[df["type"] != "face"]
        df = pd.concat([face_df, other_df], ignore_index=True)

        # Create unique ID for each landmark
        df["UID"] = df["type"].astype(str) + "_" + df["landmark_index"].astype(str)
        df = df.sort_values(["frame", "UID"])

        # Get nose positions for normalization (vectorized approach)
        nose = df[(df["type"] == "pose") & (df["landmark_index"] == 0)][
            ["frame", "x", "y"]
        ]
        nose = nose.rename(columns={"x": "nose_x", "y": "nose_y"})

        # Merge nose positions and normalize
        df = df.merge(nose, on="frame", how="left")
        df["x"] = df["x"] - df["nose_x"].fillna(0)
        df["y"] = df["y"] - df["nose_y"].fillna(0)
        df = df.drop(columns=["nose_x", "nose_y"])

        # Pivot to wide format
        pivot = df.pivot_table(index="frame", columns="UID", values=["x", "y"])
        pivot.columns = [f"{col[1]}_{col[0]}" for col in pivot.columns]
        pivot = pivot.reindex(columns=ALL_COLUMNS)

        return pivot.ffill().bfill().fillna(0).values.astype(np.float32)
    except Exception as e:
        return None


def extract_temporal_chunks(
    video: np.ndarray, chunk_size: int, stride: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract chunks with SEPARATE pose, velocity, and acceleration features.

    Returns tuple of 3 arrays (N_chunks, chunk_dim) in float16 for memory efficiency:
    - poses: Position features
    - motions: Velocity features
    - dynamics: Acceleration features 
    """
    T = video.shape[0]

    # Compute velocity and acceleration for entire video
    velocity = np.zeros_like(video)
    velocity[1:] = video[1:] - video[:-1]

    acceleration = np.zeros_like(video)
    acceleration[1:] = velocity[1:] - velocity[:-1]

    poses = []
    motions = []
    dynamics_list = []

    for start in range(0, T - chunk_size + 1, stride):
        end = start + chunk_size

        # Extract chunk for each feature type (flatten each chunk)
        poses.append(video[start:end].flatten())
        motions.append(velocity[start:end].flatten())
        dynamics_list.append(acceleration[start:end].flatten())

    if not poses:
        return None, None, None

    # Stack and convert to float16 for memory efficiency
    poses = np.stack(poses).astype(np.float16)
    motions = np.stack(motions).astype(np.float16)
    dynamics_arr = np.stack(dynamics_list).astype(np.float16)

    return poses, motions, dynamics_arr


class TemporalChunkDataset(Dataset):
    """
    Dataset with separate pose, motion, and dynamics features.
    Uses flattened NumPy arrays (float16) for memory efficiency.
    Converts to float32 tensors on-the-fly in __getitem__.
    """

    def __init__(
        self,
        poses: np.ndarray,
        motions: np.ndarray,
        dynamics: np.ndarray,
        augment: bool = True,
    ):
        """
        Args:
            poses: (N, chunk_dim) array in float16
            motions: (N, chunk_dim) array in float16
            dynamics: (N, chunk_dim) array in float16
            augment: Whether to apply data augmentation
        """
        self.poses = poses
        self.motions = motions
        self.dynamics = dynamics
        self.augment = augment

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        # Convert from float16 to float32 on-the-fly
        pose = self.poses[idx].astype(np.float32)
        motion = self.motions[idx].astype(np.float32)
        dynamics = self.dynamics[idx].astype(np.float32)

        if self.augment:
            # Add noise (less noise to motion/dynamics to preserve temporal info)
            if random.random() > 0.5:
                pose += np.random.normal(0, 0.003, pose.shape).astype(np.float32)
                motion += np.random.normal(0, 0.001, motion.shape).astype(np.float32)
                dynamics += np.random.normal(0, 0.0005, dynamics.shape).astype(
                    np.float32
                )

            # Random scale (affects pose, not motion/dynamics as much)
            if random.random() > 0.5:
                scale = np.random.uniform(0.9, 1.1)
                pose *= scale

        return (
            torch.tensor(pose, dtype=torch.float32),
            torch.tensor(motion, dtype=torch.float32),
            torch.tensor(dynamics, dtype=torch.float32),
        )


# ============== STREAMING DATA LOADER WITH PARALLELIZATION ==============
# Simplified: ALL datasets use LONG format only
from multiprocessing import Pool, cpu_count
import gc


def process_single_video(args):
    """
    Process a single video file (for parallel processing).
    Returns tuple of (poses, motions, dynamics) arrays or (None, None, None) on failure.
    """
    path, base_path, chunk_size, chunk_stride = args
    try:
        video = load_long_format(path, base_path)
        if video is None or video.shape[0] < chunk_size:
            return None, None, None
        return extract_temporal_chunks(video, chunk_size, chunk_stride)
    except Exception as e:
        return None, None, None


class StreamingChunkLoader:
    """
    Memory-efficient streaming loader that processes videos in batches.

    All datasets must be in LONG format (frame, type, landmark_index, x, y, z).
    Use convert-fingerspelling-to-long.py to convert fingerspelling first.
    """

    def __init__(
        self,
        datasets_config: Dict,
        config,
        videos_per_batch: int = 30000,
        num_workers: int = 4,
    ):
        self.datasets_config = datasets_config
        self.config = config
        self.videos_per_batch = videos_per_batch
        self.num_workers = min(num_workers, cpu_count())

        # Collect all video paths (just paths, not data)
        self.video_tasks = []
        self._collect_video_tasks()

        print(f"StreamingChunkLoader initialized:")
        print(f"  Total videos: {len(self.video_tasks):,}")
        print(f"  Videos per batch: {videos_per_batch:,}")
        print(f"  Num batches: {self.num_batches}")
        print(f"  Parallel workers: {self.num_workers}")

    def _collect_video_tasks(self):
        """Collect all video paths without loading data"""
        for name, ds_config in self.datasets_config.items():
            if not ds_config.get("enabled", False):
                print(f"  {name}: disabled")
                continue
            if not os.path.exists(ds_config["base_path"]):
                print(f"  {name}: path not found")
                continue
            if not os.path.exists(ds_config["train_csv"]):
                print(f"  {name}: train.csv not found")
                continue

            df = pd.read_csv(ds_config["train_csv"])
            paths = df["path"].unique().tolist()

            for path in paths:
                self.video_tasks.append((path, ds_config["base_path"]))

            print(f"  {name}: {len(paths):,} videos")

        # Shuffle to mix datasets
        random.shuffle(self.video_tasks)

    def load_batch(
        self, batch_idx: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load a specific batch of videos and return flattened arrays.

        Returns:
            Tuple of (poses, motions, dynamics) arrays in float16, or (None, None, None) if empty.
        """
        start_idx = batch_idx * self.videos_per_batch
        end_idx = min(start_idx + self.videos_per_batch, len(self.video_tasks))

        if start_idx >= len(self.video_tasks):
            return None, None, None

        batch_tasks = [
            (t[0], t[1], self.config.chunk_size, self.config.chunk_stride)
            for t in self.video_tasks[start_idx:end_idx]
        ]

        print(f"\nLoading batch {batch_idx} ({start_idx:,} to {end_idx:,})...")
        print(f"  Processing {len(batch_tasks):,} videos...")

        with Pool(self.num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(process_single_video, batch_tasks, chunksize=100),
                    total=len(batch_tasks),
                    desc="  Videos",
                )
            )

        # Collect valid results (filter out failures)
        pose_list = []
        motion_list = []
        dynamics_list = []

        for poses, motions, dynamics in results:
            if poses is not None:
                pose_list.append(poses)
                motion_list.append(motions)
                dynamics_list.append(dynamics)

        del results
        gc.collect()

        if not pose_list:
            return None, None, None

        # Concatenate all arrays (already float16)
        all_poses = np.concatenate(pose_list, axis=0)
        all_motions = np.concatenate(motion_list, axis=0)
        all_dynamics = np.concatenate(dynamics_list, axis=0)

        # Free intermediate lists
        del pose_list, motion_list, dynamics_list
        gc.collect()

        print(f"  Loaded {len(all_poses):,} chunks")
        print(
            f"  Memory: {(all_poses.nbytes + all_motions.nbytes + all_dynamics.nbytes) / 1024**2:.1f} MB"
        )

        return all_poses, all_motions, all_dynamics

    @property
    def num_batches(self) -> int:
        return max(
            1,
            (len(self.video_tasks) + self.videos_per_batch - 1)
            // self.videos_per_batch,
        )


def create_streaming_dataloaders(
    streaming_loader: StreamingChunkLoader,
    batch_idx: int,
    batch_size: int,
    val_split: float = 0.05,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders from a specific video batch"""

    # Load flattened arrays for this batch
    poses, motions, dynamics = streaming_loader.load_batch(batch_idx)

    if poses is None:
        return None, None

    n_chunks = len(poses)

    # Shuffle indices and split
    indices = np.random.permutation(n_chunks)
    split_idx = int((1 - val_split) * n_chunks)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    # Split arrays using indices
    train_poses = poses[train_idx]
    train_motions = motions[train_idx]
    train_dynamics = dynamics[train_idx]

    val_poses = poses[val_idx]
    val_motions = motions[val_idx]
    val_dynamics = dynamics[val_idx]

    # Free original arrays
    del poses, motions, dynamics
    gc.collect()

    # Create datasets
    train_dataset = TemporalChunkDataset(
        train_poses, train_motions, train_dynamics, augment=True
    )
    val_dataset = TemporalChunkDataset(
        val_poses, val_motions, val_dynamics, augment=False
    )

    # Create dataloaders (no multiprocessing here to avoid issues)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"  Train: {len(train_idx):,} chunks, {len(train_loader)} batches")
    print(f"  Val: {len(val_idx):,} chunks, {len(val_loader)} batches")

    return train_loader, val_loader


# Initialize streaming loader (only collects paths, doesn't load data yet)
print("Initializing streaming data loader...")
print("=" * 60)

# Configure based on available memory
# Kaggle has ~13GB RAM, so we use smaller batches
VIDEOS_PER_BATCH = 5000  # Reduced to prevent OOM
NUM_WORKERS = 4 if IS_KAGGLE else 2

streaming_loader = StreamingChunkLoader(
    DATASETS, config, videos_per_batch=VIDEOS_PER_BATCH, num_workers=NUM_WORKERS
)

print(f"\nWill process {streaming_loader.num_batches} data batches")
print(f"Each batch trains for multiple epochs before loading next batch")

# ============== FACTORIZED VQ-VAE MODEL ==============


class VectorQuantizerEMA(nn.Module):
    """VQ layer with EMA updates, CODEBOOK RESET, and DIVERSITY LOSS to prevent collapse"""

    def __init__(
        self,
        num_codes,
        embed_dim,
        commitment_cost=0.25,
        decay=0.99,
        reset_threshold=1.0,
        diversity_weight=0.1,  # Weight for entropy-based diversity loss
    ):
        super().__init__()
        self.num_codes = num_codes
        self.embed_dim = embed_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.reset_threshold = reset_threshold
        self.diversity_weight = diversity_weight

        self.codebook = nn.Embedding(num_codes, embed_dim)
        self.codebook.weight.data.uniform_(-1 / num_codes, 1 / num_codes)

        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_embed_sum", self.codebook.weight.data.clone())
        self.register_buffer("usage_count", torch.zeros(num_codes))

    def reset_unused_codes(self, z_e, aggressive=False):
        """Reset unused codebook entries to random encoder outputs"""
        with torch.no_grad():
            if aggressive:
                # Aggressive reset: reset bottom 50% of codes by usage
                # This forces exploration even when codes have some usage
                sorted_usage, sorted_idx = self.usage_count.sort()
                num_to_reset = self.num_codes // 2  # Reset bottom 50%
                unused_mask = torch.zeros(
                    self.num_codes, dtype=torch.bool, device=z_e.device
                )
                unused_mask[sorted_idx[:num_to_reset]] = True
            else:
                # Normal reset: codes with very low usage
                avg_usage = self.usage_count.sum() / self.num_codes
                threshold = max(self.reset_threshold, avg_usage * 0.01)
                unused_mask = self.usage_count < threshold

            num_unused = unused_mask.sum().item()

            if num_unused > 0 and z_e.shape[0] >= num_unused:
                # Sample random encoder outputs
                rand_idx = torch.randperm(z_e.shape[0])[:num_unused]
                new_codes = z_e[rand_idx].detach()

                # Add noise to spread them out
                new_codes = new_codes + torch.randn_like(new_codes) * 0.1

                # Reset the unused codes
                self.codebook.weight.data[unused_mask] = new_codes
                self.ema_embed_sum[unused_mask] = new_codes
                self.ema_cluster_size[unused_mask] = 1.0
                self.usage_count[unused_mask] = 0

                return num_unused
        return 0

    def forward(self, z_e, training_noise=0.0):
        # Add noise during training to force codebook exploration
        if self.training and training_noise > 0:
            z_e = z_e + torch.randn_like(z_e) * training_noise * z_e.std()

        # Use standard Euclidean distance (more stable than cosine)
        distances = torch.cdist(z_e, self.codebook.weight)
        indices = distances.argmin(dim=-1)
        z_q = self.codebook(indices)

        if self.training:
            encodings = F.one_hot(indices, self.num_codes).float()

            with torch.no_grad():
                # Update usage tracking
                self.usage_count = 0.99 * self.usage_count + encodings.sum(0)

                # EMA updates
                self.ema_cluster_size = self.decay * self.ema_cluster_size + (
                    1 - self.decay
                ) * encodings.sum(0)
                embed_sum = encodings.T @ z_e
                self.ema_embed_sum = (
                    self.decay * self.ema_embed_sum + (1 - self.decay) * embed_sum
                )

                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + 1e-5) / (n + self.num_codes * 1e-5) * n
                )
                self.codebook.weight.data = self.ema_embed_sum / cluster_size.unsqueeze(
                    1
                )

            # Commitment loss
            vq_loss = self.commitment_cost * F.mse_loss(z_e, z_q.detach())

            # Diversity loss: encourage uniform codebook usage via entropy
            # Higher entropy = more uniform usage = better
            avg_probs = encodings.mean(dim=0)  # Average probability of each code
            avg_probs = avg_probs + 1e-10  # Avoid log(0)
            entropy = -(avg_probs * torch.log(avg_probs)).sum()
            max_entropy = math.log(self.num_codes)  # Maximum possible entropy
            diversity_loss = self.diversity_weight * (max_entropy - entropy)

            vq_loss = vq_loss + diversity_loss
        else:
            vq_loss = torch.tensor(0.0, device=z_e.device)

        z_q = z_e + (z_q - z_e).detach()
        return z_q, indices, vq_loss

    def get_utilization(self):
        """Return fraction of codebook being used"""
        with torch.no_grad():
            avg_usage = self.usage_count.sum() / self.num_codes
            used = (self.usage_count > avg_usage * 0.01).sum()
            return used.item() / self.num_codes


class ResBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class FactorizedTemporalVQVAE(nn.Module):
    """
    VQ-VAE with FACTORIZED codebooks for pose, motion, and dynamics.

    Each factor captures different aspects of sign language:
    - Pose: What handshape/position (static)
    - Motion: How fast and which direction (velocity)
    - Dynamics: Sharp or smooth movement (acceleration)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        chunk_dim = config.chunk_size * config.pose_dim

        # Separate encoders for each factor
        self.pose_encoder = nn.Sequential(
            nn.Linear(chunk_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            ResBlock(config.hidden_dim, config.hidden_dim * 2),
            ResBlock(config.hidden_dim, config.hidden_dim * 2),
            nn.Linear(config.hidden_dim, config.embed_dim),
        )

        self.motion_encoder = nn.Sequential(
            nn.Linear(chunk_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            ResBlock(config.hidden_dim, config.hidden_dim * 2),
            nn.Linear(config.hidden_dim, config.embed_dim),
        )

        self.dynamics_encoder = nn.Sequential(
            nn.Linear(chunk_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            ResBlock(config.hidden_dim // 2, config.hidden_dim),
            nn.Linear(config.hidden_dim // 2, config.embed_dim),
        )

        # Separate codebooks with diversity loss to prevent collapse
        # Pose and Dynamics need HIGHER diversity weight since they tend to collapse
        self.pose_vq = VectorQuantizerEMA(
            config.pose_codes,
            config.embed_dim,
            config.commitment_cost,
            diversity_weight=1.0,  # Very high for pose
        )
        self.motion_vq = VectorQuantizerEMA(
            config.motion_codes,
            config.embed_dim,
            config.commitment_cost,
            diversity_weight=0.1,  # Motion is fine
        )
        self.dynamics_vq = VectorQuantizerEMA(
            config.dynamics_codes,
            config.embed_dim,
            config.commitment_cost,
            diversity_weight=0.5,  # Higher for dynamics (also collapsing)
        )

        # Decoder takes concatenated quantized embeddings
        combined_embed_dim = config.embed_dim * 3

        self.decoder = nn.Sequential(
            nn.Linear(combined_embed_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            ResBlock(config.hidden_dim, config.hidden_dim * 2),
            ResBlock(config.hidden_dim, config.hidden_dim * 2),
            nn.Linear(
                config.hidden_dim, chunk_dim * 3
            ),  # Reconstruct pose + motion + dynamics
        )

    def encode(self, pose, motion, dynamics):
        """Encode to continuous embeddings"""
        z_pose = self.pose_encoder(pose)
        z_motion = self.motion_encoder(motion)
        z_dynamics = self.dynamics_encoder(dynamics)
        return z_pose, z_motion, z_dynamics

    def quantize(self, z_pose, z_motion, z_dynamics):
        """Quantize each factor separately"""
        # Add MORE noise to pose and dynamics during training to force codebook exploration
        # Both pose and dynamics tend to collapse
        zq_pose, idx_pose, loss_pose = self.pose_vq(z_pose, training_noise=1.0)
        zq_motion, idx_motion, loss_motion = self.motion_vq(
            z_motion, training_noise=0.1
        )
        zq_dynamics, idx_dynamics, loss_dynamics = self.dynamics_vq(
            z_dynamics, training_noise=0.5
        )

        vq_loss = loss_pose + loss_motion + loss_dynamics

        # Add variance regularization for pose encoder
        # Penalize if encoder outputs have low variance (collapsing to same point)
        if self.training:
            pose_var = z_pose.var(dim=0).mean()  # Variance across batch
            # We want high variance, so penalize low variance
            # Target variance ~1.0, penalize if much lower
            variance_loss = 0.5 * F.relu(0.1 - pose_var)  # Penalize if var < 0.1
            vq_loss = vq_loss + variance_loss

        indices = (idx_pose, idx_motion, idx_dynamics)

        return zq_pose, zq_motion, zq_dynamics, indices, vq_loss

    def decode(self, zq_pose, zq_motion, zq_dynamics):
        """Decode from quantized embeddings"""
        combined = torch.cat([zq_pose, zq_motion, zq_dynamics], dim=-1)
        recon = self.decoder(combined)

        # Split reconstruction into pose, motion, dynamics
        chunk_dim = self.config.chunk_size * self.config.pose_dim
        recon_pose = recon[:, :chunk_dim]
        recon_motion = recon[:, chunk_dim : 2 * chunk_dim]
        recon_dynamics = recon[:, 2 * chunk_dim :]

        return recon_pose, recon_motion, recon_dynamics

    def forward(self, pose, motion, dynamics):
        # Encode
        z_pose, z_motion, z_dynamics = self.encode(pose, motion, dynamics)

        # Quantize
        zq_pose, zq_motion, zq_dynamics, indices, vq_loss = self.quantize(
            z_pose, z_motion, z_dynamics
        )

        # Decode
        recon_pose, recon_motion, recon_dynamics = self.decode(
            zq_pose, zq_motion, zq_dynamics
        )

        return (recon_pose, recon_motion, recon_dynamics), indices, vq_loss

    def get_tokens(self, pose, motion, dynamics):
        """Get factorized token indices"""
        z_pose, z_motion, z_dynamics = self.encode(pose, motion, dynamics)
        _, _, _, indices, _ = self.quantize(z_pose, z_motion, z_dynamics)
        return indices  # (pose_idx, motion_idx, dynamics_idx)


# Create model
model = FactorizedTemporalVQVAE(config).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {num_params:,}")
print(f"\nCodebooks:")
print(f"  Pose: {config.pose_codes} codes")
print(f"  Motion: {config.motion_codes} codes")
print(f"  Dynamics: {config.dynamics_codes} codes")

# Training setup
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

# Use a more flexible scheduler that doesn't need to know total steps upfront
# ReduceLROnPlateau reduces LR when validation loss plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
)

scaler = GradScaler(enabled=use_amp)


def compute_loss(recon, target, vq_loss):
    """Weighted reconstruction loss"""
    recon_pose, recon_motion, recon_dynamics = recon
    pose, motion, dynamics = target

    # Weight pose reconstruction higher (primary information)
    loss_pose = F.mse_loss(recon_pose, pose)
    loss_motion = F.mse_loss(recon_motion, motion)
    loss_dynamics = F.mse_loss(recon_dynamics, dynamics)

    recon_loss = loss_pose + 0.5 * loss_motion + 0.25 * loss_dynamics

    return recon_loss + vq_loss, loss_pose, loss_motion, loss_dynamics


def train_epoch(loader, reset_codebook_every=100):
    """Training epoch with periodic codebook reset"""
    model.train()
    total_loss = 0
    total_pose_loss = 0
    total_motion_loss = 0
    reset_counts = {"pose": 0, "motion": 0, "dynamics": 0}

    for batch_idx, (pose, motion, dynamics) in enumerate(loader):
        pose, motion, dynamics = pose.to(device), motion.to(device), dynamics.to(device)

        optimizer.zero_grad()

        with autocast(enabled=use_amp, device_type=device.type):
            recon, indices, vq_loss = model(pose, motion, dynamics)
            loss, loss_pose, loss_motion, loss_dynamics = compute_loss(
                recon, (pose, motion, dynamics), vq_loss
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_pose_loss += loss_pose.item()
        total_motion_loss += loss_motion.item()

        # Periodically reset unused codebook entries
        if (batch_idx + 1) % reset_codebook_every == 0:
            with torch.no_grad():
                # Get encoder outputs for reset
                z_pose = model.pose_encoder(pose)
                z_motion = model.motion_encoder(motion)
                z_dynamics = model.dynamics_encoder(dynamics)

                # Use AGGRESSIVE reset only for pose (tends to collapse badly)
                # Motion and dynamics use normal reset (they're stable enough)
                reset_counts["pose"] += model.pose_vq.reset_unused_codes(
                    z_pose, aggressive=True
                )
                reset_counts["motion"] += model.motion_vq.reset_unused_codes(
                    z_motion, aggressive=False
                )
                reset_counts["dynamics"] += model.dynamics_vq.reset_unused_codes(
                    z_dynamics, aggressive=False  # Changed: was capping at 50%
                )

    n = len(loader)
    return total_loss / n, total_pose_loss / n, total_motion_loss / n, reset_counts


@torch.no_grad()
def validate_epoch(loader):
    model.eval()
    total_loss = 0
    all_pose_idx, all_motion_idx, all_dynamics_idx = [], [], []

    for pose, motion, dynamics in loader:
        pose, motion, dynamics = pose.to(device), motion.to(device), dynamics.to(device)

        recon, indices, vq_loss = model(pose, motion, dynamics)
        loss, _, _, _ = compute_loss(recon, (pose, motion, dynamics), vq_loss)

        total_loss += loss.item()

        pose_idx, motion_idx, dynamics_idx = indices
        all_pose_idx.append(pose_idx.cpu())
        all_motion_idx.append(motion_idx.cpu())
        all_dynamics_idx.append(dynamics_idx.cpu())

    # Compute utilization for each codebook
    pose_util = len(torch.unique(torch.cat(all_pose_idx))) / config.pose_codes
    motion_util = len(torch.unique(torch.cat(all_motion_idx))) / config.motion_codes
    dynamics_util = (
        len(torch.unique(torch.cat(all_dynamics_idx))) / config.dynamics_codes
    )

    return total_loss / len(loader), pose_util, motion_util, dynamics_util


# ============== STREAMING TRAINING LOOP ==============
# Train on batches of videos sequentially to manage memory

best_loss = float("inf")
patience = 5  # Early stopping patience (across all data batches)
patience_counter = 0
best_epoch = 0
global_epoch = 0

# How many epochs to train on each video batch before moving to next
EPOCHS_PER_BATCH = 15 if IS_KAGGLE else 5  # More epochs per batch on Kaggle
TOTAL_PASSES = 3  # How many times to cycle through all video batches

print("Training Factorized Temporal VQ-VAE (Streaming Mode)")
print(f"Video batches: {streaming_loader.num_batches}")
print(f"Epochs per batch: {EPOCHS_PER_BATCH}")
print(f"Total passes through data: {TOTAL_PASSES}")
print(f"Early stopping patience: {patience}")
print("=" * 80)

for pass_idx in range(TOTAL_PASSES):
    print(f"\n{'='*80}")
    print(f"DATA PASS {pass_idx + 1}/{TOTAL_PASSES}")
    print(f"{'='*80}")

    # Shuffle video order for each pass
    random.shuffle(streaming_loader.video_tasks)

    for batch_idx in range(streaming_loader.num_batches):
        print(f"\n--- Video Batch {batch_idx + 1}/{streaming_loader.num_batches} ---")

        # Load this batch of videos
        train_loader, val_loader = create_streaming_dataloaders(
            streaming_loader, batch_idx, config.batch_size, val_split=0.05
        )

        if train_loader is None:
            print("  No data in this batch, skipping...")
            continue
        
        patience_counter = 0

        # Train for several epochs on this batch
        for local_epoch in range(EPOCHS_PER_BATCH):
            train_loss, pose_loss, motion_loss, reset_counts = train_epoch(
                train_loader, reset_codebook_every=50
            )
            val_loss, pose_util, motion_util, dynamics_util = validate_epoch(val_loader)

            # Step scheduler with validation loss
            scheduler.step(val_loss)

            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {global_epoch:3d} (local {local_epoch}) | Loss: {train_loss:.5f} | Val: {val_loss:.5f} | "
                f"Util: {pose_util:.0%}/{motion_util:.0%}/{dynamics_util:.0%} | LR: {lr:.2e}"
            )

            # Show reset counts
            total_resets = sum(reset_counts.values())
            if total_resets > 0:
                print(
                    f"    Reset codes: P={reset_counts['pose']}, M={reset_counts['motion']}, D={reset_counts['dynamics']}"
                )

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = global_epoch
                patience_counter = 0
                torch.save(
                    {
                        "epoch": global_epoch,
                        "model_state_dict": model.state_dict(),
                        "config": config.__dict__,
                        "val_loss": val_loss,
                        "utilization": {
                            "pose": pose_util,
                            "motion": motion_util,
                            "dynamics": dynamics_util,
                        },
                    },
                    "temporal_vqvae_best.pth",
                )
                print(f"    -> Saved best model!")
            else:
                patience_counter += 1

            global_epoch += 1

            # Check early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping! No improvement for {patience} epochs.")
                break

        # Clear memory after each video batch
        del train_loader, val_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    #     if patience_counter >= patience:
    #         break

    # if patience_counter >= patience:
    #     break

print(f"\n{'='*80}")
print(f"Training complete!")
print(f"Best validation loss: {best_loss:.5f} at epoch {best_epoch}")
print(f"Total epochs trained: {global_epoch}")

# Save tokenizer for downstream use
# NOTE: Save config as dict (not dataclass) for PyTorch 2.6+ serialization compatibility
torch.save(
    {
        "config": config.__dict__,  # Convert dataclass to dict for safe serialization
        "pose_encoder": model.pose_encoder.state_dict(),
        "motion_encoder": model.motion_encoder.state_dict(),
        "dynamics_encoder": model.dynamics_encoder.state_dict(),
        "pose_codebook": model.pose_vq.codebook.weight.data.cpu(),
        "motion_codebook": model.motion_vq.codebook.weight.data.cpu(),
        "dynamics_codebook": model.dynamics_vq.codebook.weight.data.cpu(),
        "columns": ALL_COLUMNS,
        "selected_face": SELECTED_FACE,
    },
    "temporal_sign_tokenizer.pth",
)

print("Saved temporal_sign_tokenizer.pth!")
print(f"\nTokenizer outputs 3 indices per chunk:")
print(f"  - Pose token: 0-{config.pose_codes-1} (what handshape/position)")
print(f"  - Motion token: 0-{config.motion_codes-1} (how fast, which direction)")
print(f"  - Dynamics token: 0-{config.dynamics_codes-1} (sharp or smooth)")

# ## How This Captures Temporal Dynamics
#
# ### What Each Codebook Learns
#
# | Codebook | Input | What it captures | Sign example |
# |----------|-------|------------------|-------------|
# | **Pose** | Positions | Handshapes, locations | "A" vs "B" handshape |
# | **Motion** | Velocities | Speed, direction | "HURRY" (fast) vs "SLOW" (slow) |
# | **Dynamics** | Accelerations | Sharp vs smooth | "STOP" (sharp) vs "FINISH" (smooth) |
#
# ### Combined Token Representation
#
# Each chunk gets 3 tokens: `(pose_id, motion_id, dynamics_id)`
#
# For example:
# - `(42, 15, 3)` = Handshape #42, moving fast-right (#15), with sharp stop (#3)
# - `(42, 89, 7)` = Same handshape #42, moving slow-up (#89), smooth motion (#7)
#
# This factorization allows the model to learn that:
# - Same handshape can have different motions
# - Same motion pattern can apply to different handshapes
# - Dynamics (sharp/smooth) are independent of both
