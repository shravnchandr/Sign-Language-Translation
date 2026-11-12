# # Sign Language Token Classifier
#
# ## Phase 2: Classification using Pre-trained Factorized Sign Tokenizer
#
# This notebook uses the pre-trained **Factorized VQ-VAE** to:
# 1. Convert videos into sequences of discrete **factorized tokens** (pose, motion, dynamics)
# 2. Train a Transformer to classify token sequences
#
# Each video chunk produces 3 tokens:
# - **Pose token**: What handshape/position (512 codes)
# - **Motion token**: Speed and direction (256 codes)
# - **Dynamics token**: Sharp vs smooth movement (128 codes)
#
# This is analogous to how NLP models work:
# - **Text**: "Hello world" → [15496, 995] → Transformer → Classification
# - **Sign**: [landmarks...] → [(42,15,3), (89,67,5), ...] → Transformer → Classification


import json
import os
import math
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler

from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
print(f"Device: {device}")

# ============== LOAD PRE-TRAINED FACTORIZED TOKENIZER ==============

# Path to the tokenizer saved from VQ-VAE pre-training
IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:
    TOKENIZER_PATH = "/kaggle/input/sign-vqvae-temporal/temporal_sign_tokenizer.pth"
else:
    TOKENIZER_PATH = "/Users/shravnchandr/Projects/Isolated_Sign_Language_Recognition/notebooks/vqvae/temporal_sign_tokenizer.pth"

# Load tokenizer - use weights_only=False for backward compatibility, or handle dict config
tokenizer_data = torch.load(TOKENIZER_PATH, map_location="cpu", weights_only=False)

# Extract config - now saved as dict for PyTorch 2.6+ compatibility
saved_config = tokenizer_data["config"]

# Handle both dict and dataclass formats for backward compatibility
if isinstance(saved_config, dict):
    # New format: config saved as dict
    @dataclass
    class TemporalVQVAEConfig:
        chunk_size: int = 8
        chunk_stride: int = 4
        pose_dim: int = 418
        motion_dim: int = 418
        dynamics_dim: int = 418
        pose_codes: int = 384
        motion_codes: int = 192
        dynamics_codes: int = 96
        embed_dim: int = 128
        hidden_dim: int = 512
        batch_size: int = 256
        learning_rate: float = 3e-4
        epochs: int = 100
        commitment_cost: float = 0.25
        use_ema: bool = True
        ema_decay: float = 0.99

    tok_config = TemporalVQVAEConfig(
        **{
            k: v
            for k, v in saved_config.items()
            if k in TemporalVQVAEConfig.__dataclass_fields__
        }
    )
else:
    # Old format: config saved as dataclass
    tok_config = saved_config

ALL_COLUMNS = tokenizer_data.get("columns", None)
SELECTED_FACE = tokenizer_data.get("selected_face", None)

# Load factorized codebooks
POSE_CODEBOOK = tokenizer_data["pose_codebook"]
MOTION_CODEBOOK = tokenizer_data["motion_codebook"]
DYNAMICS_CODEBOOK = tokenizer_data["dynamics_codebook"]

print(f"Loaded factorized tokenizer:")
print(
    f"  Pose codes: {tok_config.pose_codes}, Motion codes: {tok_config.motion_codes}, Dynamics codes: {tok_config.dynamics_codes}"
)
print(f"  Chunk size: {tok_config.chunk_size}, Stride: {tok_config.chunk_stride}")
print(f"  Embedding dim: {tok_config.embed_dim}")

# ============== REBUILD FACTORIZED ENCODERS FOR TOKENIZATION ==============


class ResBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class FactorizedSignEncoder(nn.Module):
    """Factorized encoder from VQ-VAE (without decoder) for tokenization"""

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

        # Codebooks
        self.pose_codebook = nn.Embedding(config.pose_codes, config.embed_dim)
        self.motion_codebook = nn.Embedding(config.motion_codes, config.embed_dim)
        self.dynamics_codebook = nn.Embedding(config.dynamics_codes, config.embed_dim)

    def forward(self, pose, motion, dynamics):
        """Returns (pose_idx, motion_idx, dynamics_idx) for each chunk"""
        # Encode
        z_pose = self.pose_encoder(pose)
        z_motion = self.motion_encoder(motion)
        z_dynamics = self.dynamics_encoder(dynamics)

        # Quantize (find nearest codebook entry)
        pose_idx = torch.cdist(z_pose, self.pose_codebook.weight).argmin(dim=-1)
        motion_idx = torch.cdist(z_motion, self.motion_codebook.weight).argmin(dim=-1)
        dynamics_idx = torch.cdist(z_dynamics, self.dynamics_codebook.weight).argmin(
            dim=-1
        )

        return pose_idx, motion_idx, dynamics_idx


# Create and load encoder
sign_encoder = FactorizedSignEncoder(tok_config).to(device)
sign_encoder.pose_encoder.load_state_dict(tokenizer_data["pose_encoder"])
sign_encoder.motion_encoder.load_state_dict(tokenizer_data["motion_encoder"])
sign_encoder.dynamics_encoder.load_state_dict(tokenizer_data["dynamics_encoder"])
sign_encoder.pose_codebook.weight.data = POSE_CODEBOOK.to(device)
sign_encoder.motion_codebook.weight.data = MOTION_CODEBOOK.to(device)
sign_encoder.dynamics_codebook.weight.data = DYNAMICS_CODEBOOK.to(device)
sign_encoder.eval()

print("Loaded pre-trained factorized encoder")
print(f"  Pose codebook: {tok_config.pose_codes} × {tok_config.embed_dim}")
print(f"  Motion codebook: {tok_config.motion_codes} × {tok_config.embed_dim}")
print(f"  Dynamics codebook: {tok_config.dynamics_codes} × {tok_config.embed_dim}")

# ============== DATA LOADING ==============

if IS_KAGGLE:
    BASE_PATH = "/kaggle/input/asl-signs"
    TRAIN_FILE = "/kaggle/input/asl-signs/train.csv"
    SIGN_INDEX_FILE = "/kaggle/input/asl-signs/sign_to_prediction_index_map.json"
else:
    BASE_PATH = "/Users/shravnchandr/Projects/Isolated_Sign_Language_Recognition/data/Isolated_ASL_Recognition"
    TRAIN_FILE = f"{BASE_PATH}/train.csv"
    SIGN_INDEX_FILE = f"{BASE_PATH}/sign_to_prediction_index_map.json"

with open(SIGN_INDEX_FILE, "r") as f:
    SIGN2INDEX = json.load(f)

NUM_CLASSES = len(SIGN2INDEX)
print(f"Classes: {NUM_CLASSES}")

# Regenerate column names if not loaded
if ALL_COLUMNS is None:
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
    SELECTED_FACE = [i for v in FACE_LANDMARKS.values() for i in v]

    specs = {"left_hand": 21, "pose": 33, "right_hand": 21}
    ALL_COLUMNS = []
    for lm_type, count in specs.items():
        for i in range(count):
            for ax in ["x", "y"]:
                ALL_COLUMNS.append(f"{lm_type}_{i}_{ax}")
    for face_idx in SELECTED_FACE:
        for ax in ["x", "y"]:
            ALL_COLUMNS.append(f"face_{face_idx}_{ax}")

print(f"Landmark columns: {len(ALL_COLUMNS)}")


def load_parquet(file_path: str) -> np.ndarray:
    """Load and normalize parquet file"""
    df = pd.read_parquet(os.path.join(BASE_PATH, file_path))

    # Filter face
    face_df = df[df["type"] == "face"]
    face_df = face_df[face_df["landmark_index"].isin(SELECTED_FACE)]
    other_df = df[df["type"] != "face"]
    df = pd.concat([face_df, other_df], ignore_index=True)

    df["UID"] = df["type"].astype(str) + "_" + df["landmark_index"].astype(str)
    df = df.sort_values(["frame", "UID"])

    # Get nose positions for normalization (vectorized approach)
    nose = df[(df["type"] == "pose") & (df["landmark_index"] == 0)][["frame", "x", "y"]]
    nose = nose.rename(columns={"x": "nose_x", "y": "nose_y"})

    # Merge nose positions and normalize
    df = df.merge(nose, on="frame", how="left")
    df["x"] = df["x"] - df["nose_x"].fillna(0)
    df["y"] = df["y"] - df["nose_y"].fillna(0)
    df = df.drop(columns=["nose_x", "nose_y"])

    pivot = df.pivot_table(index="frame", columns="UID", values=["x", "y"])
    pivot.columns = [f"{col[1]}_{col[0]}" for col in pivot.columns]
    pivot = pivot.reindex(columns=ALL_COLUMNS)

    return pivot.ffill().bfill().fillna(0).values.astype(np.float32)


@torch.no_grad()
def tokenize_video(
    video: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert video landmarks to factorized token sequence.

    Returns:
        Tuple of (pose_tokens, motion_tokens, dynamics_tokens)
        Each is shape (num_chunks,)
    """
    T = video.shape[0]

    if T < tok_config.chunk_size:
        # Pad short videos
        pad = np.zeros((tok_config.chunk_size - T, video.shape[1]), dtype=np.float32)
        video = np.vstack([video, pad])
        T = tok_config.chunk_size

    # Compute velocity and acceleration
    velocity = np.zeros_like(video)
    velocity[1:] = video[1:] - video[:-1]

    acceleration = np.zeros_like(video)
    acceleration[1:] = velocity[1:] - velocity[:-1]

    # Extract chunks
    pose_chunks = []
    motion_chunks = []
    dynamics_chunks = []

    for start in range(0, T - tok_config.chunk_size + 1, tok_config.chunk_stride):
        end = start + tok_config.chunk_size
        pose_chunks.append(video[start:end].flatten())
        motion_chunks.append(velocity[start:end].flatten())
        dynamics_chunks.append(acceleration[start:end].flatten())

    if len(pose_chunks) == 0:
        # Fallback for very short videos
        return (torch.tensor([0]), torch.tensor([0]), torch.tensor([0]))

    # Convert to tensors and tokenize
    pose_tensor = torch.tensor(np.array(pose_chunks), dtype=torch.float32).to(device)
    motion_tensor = torch.tensor(np.array(motion_chunks), dtype=torch.float32).to(
        device
    )
    dynamics_tensor = torch.tensor(np.array(dynamics_chunks), dtype=torch.float32).to(
        device
    )

    pose_idx, motion_idx, dynamics_idx = sign_encoder(
        pose_tensor, motion_tensor, dynamics_tensor
    )

    return pose_idx.cpu(), motion_idx.cpu(), dynamics_idx.cpu()


# ============== TOKENIZE ALL VIDEOS ==============

print("Tokenizing all videos with factorized tokens...")

train_df = pd.read_csv(TRAIN_FILE)
train_df["sign"] = train_df["sign"].map(SIGN2INDEX)

# Split
train_split, val_split = train_test_split(
    train_df, test_size=0.1, stratify=train_df["sign"], random_state=42
)

# Tokenize training set - now stores 3 token sequences per video
print(f"Tokenizing {len(train_split)} training videos...")
train_tokens = []  # List of (pose_tokens, motion_tokens, dynamics_tokens)
train_labels = []

for _, row in tqdm(train_split.iterrows(), total=len(train_split)):
    try:
        video = load_parquet(row["path"])
        pose_tok, motion_tok, dynamics_tok = tokenize_video(video)
        train_tokens.append((pose_tok, motion_tok, dynamics_tok))
        train_labels.append(row["sign"])
    except Exception as e:
        continue

# Tokenize validation set
print(f"Tokenizing {len(val_split)} validation videos...")
val_tokens = []
val_labels = []

for _, row in tqdm(val_split.iterrows(), total=len(val_split)):
    try:
        video = load_parquet(row["path"])
        pose_tok, motion_tok, dynamics_tok = tokenize_video(video)
        val_tokens.append((pose_tok, motion_tok, dynamics_tok))
        val_labels.append(row["sign"])
    except Exception as e:
        continue

print(f"\nTrain: {len(train_tokens)}, Val: {len(val_tokens)}")
avg_len = np.mean([len(t[0]) for t in train_tokens])
print(
    f"Average sequence length: {avg_len:.1f} chunks (each chunk = 3 factorized tokens)"
)

# ============== FACTORIZED TOKEN SEQUENCE DATASET ==============


class FactorizedTokenDataset(Dataset):
    """Dataset for factorized token sequences (pose, motion, dynamics)"""

    def __init__(self, tokens_list, labels, max_len=64):
        self.tokens = tokens_list  # List of (pose_tok, motion_tok, dynamics_tok) tuples
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        pose_tok, motion_tok, dynamics_tok = self.tokens[idx]
        label = self.labels[idx]

        seq_len = len(pose_tok)

        # Truncate or sample uniformly if too long
        if seq_len > self.max_len:
            indices = np.linspace(0, seq_len - 1, self.max_len).astype(int)
            pose_tok = pose_tok[indices]
            motion_tok = motion_tok[indices]
            dynamics_tok = dynamics_tok[indices]

        return pose_tok.long(), motion_tok.long(), dynamics_tok.long(), label


def collate_factorized(batch):
    """Collate function for factorized tokens with padding"""
    pose_list, motion_list, dynamics_list, labels = zip(*batch)

    lengths = [len(p) for p in pose_list]
    max_len = max(lengths)
    B = len(pose_list)

    # PAD tokens for each codebook
    PAD_POSE = tok_config.pose_codes
    PAD_MOTION = tok_config.motion_codes
    PAD_DYNAMICS = tok_config.dynamics_codes

    # Pad sequences
    pose_padded = torch.full((B, max_len), PAD_POSE, dtype=torch.long)
    motion_padded = torch.full((B, max_len), PAD_MOTION, dtype=torch.long)
    dynamics_padded = torch.full((B, max_len), PAD_DYNAMICS, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, (p, m, d) in enumerate(zip(pose_list, motion_list, dynamics_list)):
        L = len(p)
        pose_padded[i, :L] = p
        motion_padded[i, :L] = m
        dynamics_padded[i, :L] = d
        mask[i, :L] = True

    return pose_padded, motion_padded, dynamics_padded, mask, torch.tensor(labels)


# Create datasets
MAX_SEQ_LEN = 64
BATCH_SIZE = 64

train_dataset = FactorizedTokenDataset(train_tokens, train_labels, MAX_SEQ_LEN)
val_dataset = FactorizedTokenDataset(val_tokens, val_labels, MAX_SEQ_LEN)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_factorized,
    drop_last=True,
    num_workers=2 if IS_KAGGLE else 0,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_factorized,
    num_workers=2 if IS_KAGGLE else 0,
)

print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

# ============== FACTORIZED TOKEN CLASSIFIER MODEL ==============


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[: x.size(1)])


class FactorizedSignClassifier(nn.Module):
    """
    Transformer classifier for FACTORIZED sign language token sequences.

    Each time step has 3 tokens (pose, motion, dynamics).
    We embed each separately and combine them before the transformer.
    """

    def __init__(
        self,
        pose_codes,
        motion_codes,
        dynamics_codes,
        num_classes,
        d_model=256,
        n_heads=8,
        n_layers=6,
        dropout=0.1,
        max_len=128,
        pose_codebook=None,
        motion_codebook=None,
        dynamics_codebook=None,
    ):
        super().__init__()

        embed_dim = d_model // 3  # Split embedding dimension among 3 factors

        # Separate embeddings for each factor (+1 for PAD token each)
        self.pose_embed = nn.Embedding(
            pose_codes + 1, embed_dim, padding_idx=pose_codes
        )
        self.motion_embed = nn.Embedding(
            motion_codes + 1, embed_dim, padding_idx=motion_codes
        )
        self.dynamics_embed = nn.Embedding(
            dynamics_codes + 1, embed_dim, padding_idx=dynamics_codes
        )

        # Initialize with pre-trained codebooks if available
        if pose_codebook is not None:
            self._init_embedding(self.pose_embed, pose_codebook, embed_dim)
        if motion_codebook is not None:
            self._init_embedding(self.motion_embed, motion_codebook, embed_dim)
        if dynamics_codebook is not None:
            self._init_embedding(self.dynamics_embed, dynamics_codebook, embed_dim)

        # Project combined embeddings to d_model
        self.combine_proj = nn.Linear(embed_dim * 3, d_model)

        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def _init_embedding(self, embed_layer, codebook, target_dim):
        """Initialize embedding with pre-trained codebook, projecting if needed"""
        with torch.no_grad():
            if codebook.shape[1] != target_dim:
                # Project to target dimension
                proj = nn.Linear(codebook.shape[1], target_dim, bias=False)
                nn.init.orthogonal_(proj.weight)
                projected = proj(codebook)
                embed_layer.weight[:-1] = projected  # -1 to leave PAD token
            else:
                embed_layer.weight[:-1] = codebook

    def forward(self, pose_tok, motion_tok, dynamics_tok, mask):
        B = pose_tok.size(0)

        # Embed each factor
        pose_emb = self.pose_embed(pose_tok)
        motion_emb = self.motion_embed(motion_tok)
        dynamics_emb = self.dynamics_embed(dynamics_tok)

        # Combine: concatenate and project
        combined = torch.cat([pose_emb, motion_emb, dynamics_emb], dim=-1)
        x = self.combine_proj(combined)  # (B, T, d_model)
        x = self.pos_enc(x)

        # Add CLS token
        cls = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)

        # Update mask
        cls_mask = torch.ones(B, 1, device=mask.device, dtype=torch.bool)
        mask = torch.cat([cls_mask, mask], dim=1)

        # Transformer
        x = self.encoder(x, src_key_padding_mask=~mask)
        x = self.norm(x[:, 0])  # CLS token

        return self.classifier(x)


# Create classifier with pre-trained embeddings
model = FactorizedSignClassifier(
    pose_codes=tok_config.pose_codes,
    motion_codes=tok_config.motion_codes,
    dynamics_codes=tok_config.dynamics_codes,
    num_classes=NUM_CLASSES,
    d_model=256,
    n_heads=8,
    n_layers=6,
    dropout=0.1,
    max_len=MAX_SEQ_LEN + 1,
    pose_codebook=POSE_CODEBOOK,  # Use pre-trained codebook!
    motion_codebook=MOTION_CODEBOOK,
    dynamics_codebook=DYNAMICS_CODEBOOK,
).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Classifier parameters: {num_params:,}")
print(f"\nFactorized embeddings:")
print(f"  Pose: {tok_config.pose_codes} codes")
print(f"  Motion: {tok_config.motion_codes} codes")
print(f"  Dynamics: {tok_config.dynamics_codes} codes")

# Training setup
EPOCHS = 100
PATIENCE = 20

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=5e-4,
    epochs=EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,
    anneal_strategy="cos",
)
scaler = GradScaler(enabled=use_amp)


def train_epoch(loader):
    model.train()
    total_loss = 0

    for pose_tok, motion_tok, dynamics_tok, mask, labels in loader:
        pose_tok = pose_tok.to(device)
        motion_tok = motion_tok.to(device)
        dynamics_tok = dynamics_tok.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast(enabled=use_amp, device_type=device.type):
            logits = model(pose_tok, motion_tok, dynamics_tok, mask)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate_epoch(loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for pose_tok, motion_tok, dynamics_tok, mask, labels in loader:
        pose_tok = pose_tok.to(device)
        motion_tok = motion_tok.to(device)
        dynamics_tok = dynamics_tok.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        with autocast(enabled=use_amp, device_type=device.type):
            logits = model(pose_tok, motion_tok, dynamics_tok, mask)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


# ============== TRAINING ==============

best_acc = 0
patience_counter = 0
best_epoch = 0

print("Training Factorized Token Classifier...")
print("=" * 60)

for epoch in range(EPOCHS):
    train_loss = train_epoch(train_loader)
    val_loss, val_acc = validate_epoch(val_loader)

    lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
        f"Acc: {val_acc:.4f} | LR: {lr:.2e}"
    )

    if val_acc > best_acc:
        best_acc = val_acc
        best_epoch = epoch
        patience_counter = 0
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "config": {
                    "pose_codes": tok_config.pose_codes,
                    "motion_codes": tok_config.motion_codes,
                    "dynamics_codes": tok_config.dynamics_codes,
                    "num_classes": NUM_CLASSES,
                },
            },
            "token_classifier_best.pth",
        )
        print(f"  -> New best! Acc: {val_acc:.4f}")
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}")
        break

print(f"\nBest validation accuracy: {best_acc:.4f} at epoch {best_epoch}")

# Load best and evaluate
checkpoint = torch.load("token_classifier_best.pth")
model.load_state_dict(checkpoint["model_state_dict"])

val_loss, val_acc = validate_epoch(val_loader)
print(f"Final validation accuracy: {val_acc:.4f}")

# ## Summary
#
# This approach treats sign language recognition like NLP with **factorized tokens**:
#
# 1. **Pre-trained Factorized Tokenizer** (VQ-VAE): Converts landmark sequences to 3 discrete tokens per chunk
#    - **Pose token**: What handshape/position (512 codes)
#    - **Motion token**: Speed and direction (256 codes)
#    - **Dynamics token**: Sharp vs smooth movement (128 codes)
#
# 2. **Factorized Embeddings**: Each token type has its own embedding table (initialized from VQ-VAE codebook)
#
# 3. **Combined Representation**: The 3 embeddings are concatenated and projected before the transformer
#
# 4. **Transformer Classifier**: Processes combined token sequences with attention
#
# ### Why Factorization Helps
#
# | Aspect | Single Codebook | Factorized (Ours) |
# |--------|-----------------|-------------------|
# | Handshape | Mixed with motion | Separate pose codebook |
# | Speed | Mixed with pose | Separate motion codebook |
# | Rhythm | Lost in averaging | Separate dynamics codebook |
# | Vocabulary | 1024 tokens | 512 × 256 × 128 combinations |
#
# ### Advantages
# - **Disentangled features**: Pose, motion, and dynamics learned separately
# - **Transfer learning**: Pre-trained tokenizer captures general sign patterns
# - **Semantic tokens**: Each token represents a meaningful aspect of movement
# - **Efficient**: Shorter sequences (chunks vs frames), faster training
# - **Scalable**: Can pre-train tokenizer on large unlabeled datasets
