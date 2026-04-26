import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, Sampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"

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
    """Generate standardized column names for 543 landmarks (x/y/z coordinates)."""
    landmark_specs = {
        "left_hand": 21,
        "pose": 33,
        "right_hand": 21,
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


def normalize_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize coordinates relative to nose position.

    Args:
        dataframe: Unnormalized landmark dataframe.

    Returns:
        Normalized dataframe.
    """
    axes = ["x", "y", "z"] if INCLUDE_DEPTH else ["x", "y"]

    origins = dataframe[
        (dataframe["type"] == "pose") & (dataframe["landmark_index"] == 0)
    ].set_index("frame")[axes]

    def normalize_frame(frame_df: pd.DataFrame) -> pd.DataFrame:
        frame = frame_df.name
        if frame not in origins.index:
            return frame_df
        frame_df[axes] = frame_df[axes] - origins.loc[frame]
        return frame_df

    return dataframe.groupby("frame", group_keys=False).apply(normalize_frame)


def frame_stacked_data(file_path: str) -> np.ndarray:
    """Read landmark data from parquet files and stack frames.

    Args:
        file_path: Path to parquet file.

    Returns:
        Normalized stacked coordinates as numpy array.
    """
    dataframe = pd.read_parquet(os.path.join(BASE_PATH, file_path))

    if INCLUDE_FACE:
        face_df = dataframe[dataframe["type"] == "face"]
        face_df = face_df[face_df["landmark_index"].isin(SELECTED_FACE_INDICES)]
        other_df = dataframe[dataframe["type"] != "face"]
        dataframe = pd.concat([face_df, other_df], ignore_index=True)
    else:
        dataframe = dataframe[dataframe["type"] != "face"]

    axes = ["x", "y", "z"] if INCLUDE_DEPTH else ["x", "y"]

    dataframe["uid"] = (
        dataframe["type"].astype("str")
        + "_"
        + dataframe["landmark_index"].astype("str")
    )
    dataframe = dataframe.sort_values(["frame", "uid"])
    dataframe = normalize_values(dataframe)

    pivot_df = dataframe.pivot_table(index="frame", columns="uid", values=axes)
    pivot_df.columns = [f"{col[1]}_{col[0]}" for col in pivot_df.columns]
    pivot_df = pivot_df.reindex(columns=ALL_COLUMNS)

    return pd.DataFrame(pivot_df).ffill().bfill().fillna(0).to_numpy()


def augment_sample(
    video_coordinates: np.ndarray, noise_std: float = 3e-3, spatial_shift: float = 2e-2
) -> np.ndarray:
    """Apply data augmentation to video coordinates.

    Args:
        video_coordinates: Input coordinates.
        noise_std: Standard deviation of Gaussian noise.
        spatial_shift: Maximum spatial shift magnitude.

    Returns:
        Augmented coordinates.
    """
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
    """Dataset for ASL sign recognition."""

    def __init__(self, video_coordinates, video_labels, max_frames=128, augment=False):
        """Initialize dataset.

        Args:
            video_coordinates: List of coordinate arrays.
            video_labels: List of sign labels.
            max_frames: Maximum number of frames per video.
            augment: Whether to apply augmentation.
        """
        self.video_coordinates = video_coordinates
        self.video_labels = video_labels
        self.max_frames = max_frames
        self.augment = augment

    def __len__(self):
        return len(self.video_coordinates)

    def __getitem__(self, idx):
        coordinates = self.video_coordinates[idx]
        label = self.video_labels[idx]

        if self.augment:
            coordinates = augment_sample(coordinates)

        if coordinates.shape[0] > self.max_frames:
            idxs = np.linspace(0, coordinates.shape[0] - 1, self.max_frames).astype(int)
            coordinates = coordinates[idxs]

        velocities = coordinates[1:] - coordinates[:-1]
        velocities = np.vstack([np.zeros_like(coordinates[:1]), velocities])
        coordinates = np.concatenate([coordinates, velocities], axis=1)

        return torch.tensor(coordinates, dtype=torch.float32), label


def collate_batch(batch):
    """Collate function for variable-length sequences.

    Args:
        batch: List of (sequence, label) tuples.

    Returns:
        Padded sequences, attention masks, and labels.
    """
    sequences, labels = zip(*batch)

    lengths = torch.tensor([seq.shape[0] for seq in sequences])
    max_len = int(lengths.max())

    feature_dim = sequences[0].shape[1]
    batch_size = len(sequences)

    padded = torch.zeros(batch_size, max_len, feature_dim)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, seq in enumerate(sequences):
        seq_len = seq.shape[0]
        padded[i, :seq_len] = seq
        mask[i, :seq_len] = 1

    return padded, mask, torch.tensor(labels)


class BucketBatchSampler(Sampler):
    """Sampler that groups sequences by length into buckets."""

    def __init__(self, lengths, batch_size, drop_last=False):
        """Initialize sampler.

        Args:
            lengths: List of sequence lengths.
            batch_size: Batch size.
            drop_last: Whether to drop incomplete batches.
        """
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
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def get_bucket_dataloader(dataset, batch_size: int = 128) -> DataLoader:
    """Create dataloader with bucket batch sampler.

    Args:
        dataset: ASLDataset instance.
        batch_size: Batch size.

    Returns:
        DataLoader with bucketing.
    """
    lengths = [min(x.shape[0], dataset.max_frames) for x in dataset.video_coordinates]
    sampler = BucketBatchSampler(lengths, batch_size=batch_size, drop_last=False)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_batch)


def get_data_loaders(use_npz: bool = False) -> Tuple[DataLoader, DataLoader]:
    """Load and prepare train/test dataloaders.

    Args:
        use_npz: Use saved npz instance.

    Returns:
        Tuple of train and test dataloaders.
    """
    if use_npz:
        data = np.load(
            "/kaggle/input/face-data-coordinates/asl_preprocessed.npz",
            allow_pickle=True,
        )

        train_videos = data["train_videos"]
        test_videos = data["test_videos"]

        train_labels = data["train_labels"]
        test_labels = data["test_labels"]

    else:
        train_df = pd.read_csv(TRAIN_FILE)
        train_df["sign"] = train_df["sign"].map(SIGN2INDEX_JSON)

        train_split, test_split = train_test_split(
            train_df, test_size=0.1, stratify=train_df["sign"], random_state=42
        )

        train_videos = [
            frame_stacked_data(path) for path in train_split["path"].to_list()
        ]
        test_videos = [
            frame_stacked_data(path) for path in test_split["path"].to_list()
        ]

        train_labels = train_split["sign"].to_numpy()
        test_labels = test_split["sign"].to_numpy()

    train_dataset = ASLDataset(train_videos, train_labels, augment=True)
    test_dataset = ASLDataset(test_videos, test_labels)

    train_loader = get_bucket_dataloader(train_dataset)
    test_loader = get_bucket_dataloader(test_dataset)

    return train_loader, test_loader


class AdvancedAugmentation:
    """Advanced augmentation strategies for landmarks"""

    @staticmethod
    def temporal_cropping(x, mask, min_ratio=0.7, max_ratio=0.95):
        """Randomly crop temporal sequences"""
        B, T, D = x.shape
        crop_len = np.random.randint(int(T * min_ratio), int(T * max_ratio))

        # Random start position
        start = np.random.randint(0, T - crop_len + 1)
        x_cropped = x[:, start : start + crop_len, :]
        mask_cropped = mask[:, start : start + crop_len]

        # Pad back to original length
        if crop_len < T:
            pad_len = T - crop_len
            x_padded = torch.nn.functional.pad(x_cropped, (0, 0, 0, pad_len), value=0)
            mask_padded = torch.nn.functional.pad(
                mask_cropped, (0, pad_len), value=False
            )
            return x_padded, mask_padded

        return x_cropped, mask_cropped

    @staticmethod
    def random_flip(x, probability=0.3):
        """Randomly flip left-right landmarks (mirror movement)"""
        if np.random.random() < probability:
            # Flip x-coordinates
            x_flipped = x.clone()
            x_flipped[..., 0] = 1.0 - x_flipped[..., 0]  # Flip x-axis
            return x_flipped
        return x

    @staticmethod
    def gaussian_noise(x, std=0.01):
        """Add learnable Gaussian noise"""
        return x + torch.randn_like(x) * std

    @staticmethod
    def temporal_interpolation(x, mask):
        """Interpolate missing frames for smoother motion"""
        # Fill small gaps in sequences
        for i in range(1, x.shape[1] - 1):
            if not mask[:, i].all() and mask[:, i - 1].all() and mask[:, i + 1].all():
                x[:, i] = (x[:, i - 1] + x[:, i + 1]) / 2
                mask[:, i] = True
        return x, mask

    @staticmethod
    def time_stretch(x, mask, min_stretch=0.8, max_stretch=1.3):
        """Speed up or slow down temporal sequences via interpolation.

        Args:
            x: Input tensor (B, T, D).
            mask: Attention mask (B, T).
            min_stretch: Minimum stretch factor (< 1 = speed up).
            max_stretch: Maximum stretch factor (> 1 = slow down).

        Returns:
            Stretched tensor and updated mask.
        """
        B, T, D = x.shape
        stretch_factor = np.random.uniform(min_stretch, max_stretch)

        # Calculate new length
        new_len = int(T * stretch_factor)
        new_len = min(new_len, T)  # Don't exceed original length

        if new_len == T:
            return x, mask
        

        # Use linear interpolation for each batch and dimension
        x_reshaped = x.permute(0, 2, 1).reshape(B * D, 1, T)
        
        # Interpolate to new length
        x_stretched = F.interpolate(
            x_reshaped, size=new_len, mode='linear', align_corners=False
        )
        x_stretched = x_stretched.reshape(B, D, new_len).permute(0, 2, 1)
        
        # Interpolate mask
        mask_float = mask.float().unsqueeze(1)  # (B, 1, T)
        mask_stretched = F.interpolate(
            mask_float, size=new_len, mode='linear', align_corners=False
        )
        mask_stretched = (mask_stretched.squeeze(1) > 0.5).bool()  # (B, new_len)

        # Pad back to original length
        if new_len < T:
            pad_len = T - new_len
            x_stretched = torch.nn.functional.pad(
                x_stretched, (0, 0, 0, pad_len), value=0
            )
            mask_stretched = torch.nn.functional.pad(
                mask_stretched, (0, pad_len), value=False
            )

        return x_stretched, mask_stretched

    @staticmethod
    def finger_dropout(
        x, mask, hand_indices_left=None, hand_indices_right=None, dropout_prob=0.3
    ):
        """Randomly zero out entire fingers or hand regions.

        Args:
            x: Input tensor (B, T, D).
            mask: Attention mask (B, T).
            hand_indices_left: Indices for left hand landmarks.
            hand_indices_right: Indices for right hand landmarks.
            dropout_prob: Probability to dropout a finger.

        Returns:
            Tensor with dropped fingers.
        """
        x = x.clone()

        # Define finger regions (approximate based on MediaPipe hand skeleton)
        # Each finger has ~4 landmarks
        fingers = {
            "thumb": list(range(1, 5)),
            "index": list(range(5, 9)),
            "middle": list(range(9, 13)),
            "ring": list(range(13, 17)),
            "pinky": list(range(17, 21)),
        }

        # Apply dropout to left hand
        if hand_indices_left is not None:
            for finger_landmarks in fingers.values():
                if np.random.random() < dropout_prob:
                    for idx in finger_landmarks:
                        if idx < len(hand_indices_left):
                            x[:, :, hand_indices_left[idx]] = 0

        # Apply dropout to right hand
        if hand_indices_right is not None:
            for finger_landmarks in fingers.values():
                if np.random.random() < dropout_prob:
                    for idx in finger_landmarks:
                        if idx < len(hand_indices_right):
                            x[:, :, hand_indices_right[idx]] = 0

        return x

    @staticmethod
    def spatial_rotation(x, max_angle=15):
        """Apply 3D spatial rotation to landmarks.

        Rotates around z-axis (most relevant for viewing angle changes).

        Args:
            x: Input tensor (B, T, D) with coordinates [..., x, y, z, ...].
            max_angle: Maximum rotation angle in degrees.

        Returns:
            Rotated tensor.
        """
        x = x.clone()

        # Random rotation angle in radians
        angle = np.radians(np.random.uniform(-max_angle, max_angle))
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        B, T, D = x.shape

        # Process every 3 coordinates (assuming x, y, z order in features)
        for i in range(0, D - 2, 3):
            x_coords = x[:, :, i]
            y_coords = x[:, :, i + 1]

            x[:, :, i] = x_coords * cos_a - y_coords * sin_a
            x[:, :, i + 1] = x_coords * sin_a + y_coords * cos_a
            # z-coordinate unchanged

        return x

    @staticmethod
    def random_scale(x, min_scale=0.9, max_scale=1.1):
        """Randomly scale spatial coordinates (zoom in/out).

        Args:
            x: Input tensor (B, T, D).
            min_scale: Minimum scale factor.
            max_scale: Maximum scale factor.

        Returns:
            Scaled tensor.
        """
        scale = np.random.uniform(min_scale, max_scale)
        return x * scale


class AdvancedFeatureExtractor:
    """Extract advanced features from landmark sequences."""

    @staticmethod
    def compute_velocities(landmarks: np.ndarray) -> np.ndarray:
        """Compute frame-to-frame velocities."""
        velocities = np.diff(landmarks, axis=0)
        return np.vstack([np.zeros_like(landmarks[:1]), velocities])

    @staticmethod
    def compute_accelerations(landmarks: np.ndarray) -> np.ndarray:
        """Compute frame-to-frame accelerations."""
        velocities = np.diff(landmarks, axis=0)
        accelerations = np.diff(velocities, axis=0)
        return np.vstack([np.zeros_like(landmarks[:2]), accelerations])

    @staticmethod
    def compute_angles(landmarks: np.ndarray, hand_indices: List[int]) -> np.ndarray:
        """Compute joint angles for hand skeleton.

        Args:
            landmarks: Sequence of landmarks.
            hand_indices: Indices for hand landmarks.

        Returns:
            Array of joint angles.
        """
        angles = []
        connections = [(0, 1, 2), (0, 5, 6), (0, 9, 10), (0, 13, 14), (0, 17, 18)]

        for i, j, k in connections:
            v1 = landmarks[:, hand_indices[j], :] - landmarks[:, hand_indices[i], :]
            v2 = landmarks[:, hand_indices[k], :] - landmarks[:, hand_indices[j], :]

            cos_angle = np.sum(v1 * v2, axis=1) / (
                np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-6
            )
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            angles.append(angle)

        return np.stack(angles, axis=1)

    @staticmethod
    def compute_distances(landmarks: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between key landmarks.

        Args:
            landmarks: Sequence of landmarks.

        Returns:
            Array of pairwise distances.
        """
        distances = []
        pairs = [(0, 1), (1, 2), (5, 6), (11, 12)]

        for i, j in pairs:
            dist = np.linalg.norm(landmarks[:, i, :] - landmarks[:, j, :], axis=1)
            distances.append(dist)

        return np.stack(distances, axis=1)


class DimensionalityReducer(nn.Module):
    """Reduce high-dimensional landmark data to manageable size."""

    def __init__(self, input_dim, output_dim=128, dropout=0.1):
        """Initialize reducer.

        Args:
            input_dim: Input feature dimension.
            output_dim: Output feature dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Linear(input_dim, min(input_dim // 2, 512)),
            nn.LayerNorm(min(input_dim // 2, 512)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(min(input_dim // 2, 512), output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.reduce(x)


class ImprovedTemporalCNN(nn.Module):
    """CNN with residual connections for temporal feature extraction."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        """Initialize CNN.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden feature dimension.
            output_dim: Output feature dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.conv_blocks = nn.ModuleList(
            [
                self._build_conv_block(hidden_dim, hidden_dim, 3, dropout),
                self._build_conv_block(hidden_dim, hidden_dim, 5, dropout),
                self._build_conv_block(hidden_dim, hidden_dim, 7, dropout),
            ]
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    @staticmethod
    def _build_conv_block(in_channels, out_channels, kernel_size, dropout):
        """Build convolutional block."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x.transpose(1, 2)

        residual = x
        for conv_block in self.conv_blocks:
            x = conv_block(x) + residual
            residual = x

        x = x.transpose(1, 2)
        return self.output_proj(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        """Initialize positional encoding.

        Args:
            d_model: Model dimension.
            max_len: Maximum sequence length.
            dropout: Dropout rate.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(1)]
        return self.dropout(x)


class ImprovedTransformerEncoder(nn.Module):
    """Transformer encoder with pre-LayerNorm."""

    def __init__(self, d_model, n_heads, n_layers, dropout=0.1):
        """Initialize encoder.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            n_layers: Number of transformer layers.
            dropout: Dropout rate.
        """
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.encoder(x, src_key_padding_mask=~mask if mask is not None else None)
        return self.norm(x)


class ImprovedCNNTransformer(nn.Module):
    """Hybrid CNN-Transformer architecture for sign recognition."""

    def __init__(
        self,
        input_dim,
        num_classes,
        cnn_hidden=256,
        d_model=256,
        n_heads=8,
        n_layers=6,
        dropout=0.2,
        reduced_dim=128,
    ):
        """Initialize model.

        Args:
            input_dim: Input feature dimension.
            num_classes: Number of sign classes.
            cnn_hidden: CNN hidden dimension.
            d_model: Transformer model dimension.
            n_heads: Number of attention heads.
            n_layers: Number of transformer layers.
            dropout: Dropout rate.
            reduced_dim: Reduced dimension for dimensionality reducer.
        """
        super().__init__()

        self.reducer = DimensionalityReducer(
            input_dim, output_dim=reduced_dim, dropout=dropout
        )

        self.spatial_encoder = ImprovedTemporalCNN(
            input_dim=reduced_dim,
            hidden_dim=cnn_hidden,
            output_dim=d_model,
            dropout=dropout,
        )

        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        self.transformer = ImprovedTransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x, mask):
        batch_size = x.size(0)

        x = self.reducer(x)
        x = self.spatial_encoder(x)
        x = self.pos_enc(x)

        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        cls_mask = torch.ones(batch_size, 1, device=mask.device, dtype=torch.bool)
        mask = torch.cat([cls_mask, mask], dim=1)

        x = self.transformer(x, mask)
        x = x[:, 0]

        return self.head(x)


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, alpha=0.25, gamma=2.0):
        """Initialize loss.

        Args:
            alpha: Weighting factor.
            gamma: Focusing parameter.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return loss.mean()


def mixup_batch(x, y, alpha=0.2):
    """Apply mixup data augmentation.

    Args:
        x: Input batch.
        y: Target batch.
        alpha: Beta distribution parameter.

    Returns:
        Mixed inputs, targets, and mixing coefficient.
    """
    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    lam = np.random.beta(alpha, alpha)
    mixed_x = lam * x + (1 - lam) * x[index]

    return mixed_x, y, y[index], lam


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    accumulation_steps: int = 4,
    use_mixup: bool = True,
) -> Tuple[float, float]:
    """Train for one epoch.

    Args:
        model: Neural network model.
        data_loader: Training dataloader.
        optimizer: Optimizer instance.
        criterion: Loss function.
        scaler: Gradient scaler for AMP.
        use_mixup: Whether to use mixup augmentation.

    Returns:
        Tuple of average loss and accuracy.
    """
    model.train()
    train_loss = 0
    correct, total = 0, 0

    for idx, (x, mask, y) in enumerate(data_loader):
        x, mask, y = x.to(device), mask.to(device), y.to(device)

        # Augmentation
        if np.random.random() > 0.5:
            x = AdvancedAugmentation.random_flip(x)

        if np.random.random() > 0.5:
            x = AdvancedAugmentation.gaussian_noise(x)

        if np.random.random() > 0.5:
            x, mask = AdvancedAugmentation.temporal_interpolation(x, mask)

        if np.random.random() > 0.5:
            x, mask = AdvancedAugmentation.time_stretch(x, mask)

        if np.random.random() > 0.5:
            x = AdvancedAugmentation.spatial_rotation(x, max_angle=15)

        if np.random.random() > 0.5:
            x = AdvancedAugmentation.finger_dropout(x, mask, dropout_prob=0.25)

        if use_mixup:
            x, y_a, y_b, lam = mixup_batch(x, y)

        optimizer.zero_grad()

        with autocast(enabled=use_amp, device_type=device.type):
            logits = model(x, mask)

            if use_mixup:
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            else:
                loss = criterion(logits, y)

        # scaler.scale(loss).backward()
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # scaler.step(optimizer)
        # scaler.update()

        # train_loss += loss.item()

        loss = loss / accumulation_steps  # Scale loss

        scaler.scale(loss).backward()

        # Accumulate gradients
        if (idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss += loss.item() * accumulation_steps

        prediction = logits.argmax(dim=1)
        correct += (prediction == y).sum().item()
        total += y.size(0)

    return train_loss / len(data_loader), correct / total


@torch.no_grad()
def predict_with_tta(model, x, mask, n_augmentations=5):
    """Test-time augmentation for better predictions"""
    model.eval()
    predictions = []

    # Original prediction
    pred = model(x, mask)
    predictions.append(pred)

    # Augmented predictions
    for _ in range(n_augmentations - 1):
        # Slight temporal jittering
        if np.random.random() > 0.5:
            x_aug = x + torch.randn_like(x) * 0.01
        else:
            x_aug = x

        pred = model(x_aug, mask)
        predictions.append(pred)

    # Average predictions
    ensemble_pred = torch.stack(predictions).mean(dim=0)
    return ensemble_pred


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module, data_loader: DataLoader, criterion: nn.Module
) -> Tuple[float, float]:
    """Evaluate model on validation/test set.

    Args:
        model: Neural network model.
        data_loader: Evaluation dataloader.
        criterion: Loss function.

    Returns:
        Tuple of average loss and accuracy.
    """
    model.eval()
    test_loss = 0
    correct, total = 0, 0

    for x, mask, y in data_loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)

        logits = predict_with_tta(model, x, mask, n_augmentations=5)
        loss = criterion(logits, y)
        test_loss += loss.item()

        prediction = logits.argmax(dim=1)
        correct += (prediction == y).sum().item()
        total += y.size(0)

    return test_loss / len(data_loader), correct / total


INPUT_DIM = 836
NUM_CLASSES = 250
MAX_PATIENCE = 20
NUM_EPOCHS = 100

model = ImprovedCNNTransformer(
    input_dim=INPUT_DIM,
    num_classes=NUM_CLASSES,
    cnn_hidden=256,
    d_model=256,
    n_heads=8,
    n_layers=8,
    dropout=0.4,
).to(device)

train_loader, test_loader = get_data_loaders(use_npz=True)

print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = FocalLoss(alpha=0.25, gamma=2.0)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
scaler = GradScaler(enabled=use_amp)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

best_accuracy = 0.0
patience = 0

print("Training improved model...")
print("-" * 80)

for epoch in range(NUM_EPOCHS):
    train_loss, train_accuracy = train_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        scaler,
        accumulation_steps=4,
        use_mixup=True,
    )
    test_loss, test_accuracy = evaluate_epoch(model, test_loader, criterion)

    scheduler.step()

    print(
        f"Epoch {epoch:3d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_accuracy:.4f} | "
    )

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        patience = 0
        torch.save(model.state_dict(), "best_improved_model.pth")
    else:
        patience += 1

    if patience >= MAX_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}")
        break

print("-" * 80)
