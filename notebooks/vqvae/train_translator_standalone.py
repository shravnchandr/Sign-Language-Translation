"""
Standalone training script for Sign Language Translator.

This file consolidates all dependencies from vqvae_seq2seq/ into a single file.
Run with: python train_translator_standalone.py --data-dir data/Isolated_ASL_Recognition --vqvae-checkpoint checkpoints/vqvae/best_model.pt

Requires: train_vqvae_standalone.py in the same directory for VQ-VAE model definitions.
"""

import os
import json
import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field, asdict
from collections import Counter
from tqdm import tqdm

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler


# =============================================================================
# Import VQ-VAE model from standalone training script
# =============================================================================
# We import the VQ-VAE model components to load the pre-trained checkpoint
from notebooks.training.train_vqvae_standalone import (
    ImprovedVQVAEConfig,
    ImprovedVQVAE,
    FastLandmarkProcessor,
    LandmarkConfig,
    FACE_LANDMARK_SUBSETS,
    HAND_LANDMARKS,
    POSE_LANDMARKS,
    FACE_LANDMARKS,
    POSE_INDICES,
    SignerIndependentSplitter,
    SplitConfig,
    create_signer_splits,
)


# =============================================================================
# Translation Configuration
# =============================================================================


@dataclass
class TranslationConfig:
    """
    Configuration for the sign language translation model.

    Uses Conformer encoder with hybrid CTC + Attention decoder.
    """

    # Model dimensions
    d_model: int = 512
    d_ff: int = 2048
    n_heads: int = 8

    # Encoder (Conformer)
    n_encoder_layers: int = 12
    encoder_kernel_size: int = 31
    encoder_dropout: float = 0.1

    # Decoder (Attention)
    n_decoder_layers: int = 6
    decoder_dropout: float = 0.1

    # Vocabulary
    vocab_size: int = 2500  # Combined gloss vocabulary
    pad_idx: int = 0
    bos_idx: int = 1
    eos_idx: int = 2
    unk_idx: int = 3

    # Input embedding (from VQ-VAE codebooks)
    pose_codebook_size: int = 1024
    motion_codebook_size: int = 512
    dynamics_codebook_size: int = 256
    face_codebook_size: int = 256
    embed_dim: int = 128  # VQ-VAE embedding dimension

    # CTC
    ctc_weight: float = 0.3
    ctc_blank_idx: int = 0  # Usually same as pad

    # Beam search
    beam_size: int = 5
    max_decode_len: int = 50
    length_penalty: float = 0.6
    ctc_prefix_weight: float = 0.4

    # Training
    learning_rate: float = 1e-4
    warmup_steps: int = 4000
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    max_epochs: int = 100
    batch_size: int = 32
    gradient_clip: float = 1.0

    # Regularization
    spec_augment: bool = True
    time_mask_max: int = 50
    time_mask_num: int = 2

    # Device
    device: str = "cuda"

    def get_total_input_tokens(self) -> int:
        """Total number of input tokens across all codebooks."""
        return (
            self.pose_codebook_size
            + self.motion_codebook_size
            + self.dynamics_codebook_size
            + self.face_codebook_size
        )


# =============================================================================
# Vocabulary
# =============================================================================


class GlossVocabulary:
    """
    Manages gloss vocabulary for sign language translation.

    Handles:
    - Building vocabulary from datasets
    - Special tokens (PAD, BOS, EOS, UNK)
    - Index <-> gloss mapping
    - Vocabulary persistence
    """

    # Special tokens
    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

    def __init__(
        self,
        glosses: Optional[List[str]] = None,
        min_count: int = 1,
    ):
        self.min_count = min_count
        self._gloss_to_idx: Dict[str, int] = {}
        self._idx_to_gloss: Dict[int, str] = {}

        # Initialize with special tokens
        for i, token in enumerate(self.SPECIAL_TOKENS):
            self._gloss_to_idx[token] = i
            self._idx_to_gloss[i] = token

        if glosses is not None:
            self.add_glosses(glosses)

    @property
    def pad_idx(self) -> int:
        return self._gloss_to_idx[self.PAD_TOKEN]

    @property
    def bos_idx(self) -> int:
        return self._gloss_to_idx[self.BOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self._gloss_to_idx[self.EOS_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self._gloss_to_idx[self.UNK_TOKEN]

    def __len__(self) -> int:
        return len(self._gloss_to_idx)

    def __contains__(self, gloss: str) -> bool:
        return gloss in self._gloss_to_idx

    def add_glosses(self, glosses: List[str]) -> None:
        """Add glosses to vocabulary, respecting min_count."""
        counts = Counter(glosses)
        for gloss, count in counts.items():
            if count >= self.min_count and gloss not in self._gloss_to_idx:
                idx = len(self._gloss_to_idx)
                self._gloss_to_idx[gloss] = idx
                self._idx_to_gloss[idx] = gloss

    def gloss_to_idx(self, gloss: str) -> int:
        """Convert gloss to index, returns UNK for unknown glosses."""
        return self._gloss_to_idx.get(gloss, self.unk_idx)

    def idx_to_gloss(self, idx: int) -> str:
        """Convert index to gloss."""
        return self._idx_to_gloss.get(idx, self.UNK_TOKEN)

    def encode(
        self, glosses: List[str], add_bos: bool = False, add_eos: bool = False
    ) -> List[int]:
        """Encode a sequence of glosses to indices."""
        indices = [self.gloss_to_idx(g) for g in glosses]
        if add_bos:
            indices = [self.bos_idx] + indices
        if add_eos:
            indices = indices + [self.eos_idx]
        return indices

    def decode(self, indices: List[int], remove_special: bool = True) -> List[str]:
        """Decode a sequence of indices to glosses."""
        glosses = [self.idx_to_gloss(i) for i in indices]
        if remove_special:
            special_set = set(self.SPECIAL_TOKENS)
            glosses = [g for g in glosses if g not in special_set]
        return glosses

    def get_all_glosses(self, include_special: bool = False) -> List[str]:
        """Get all glosses in vocabulary."""
        if include_special:
            return list(self._gloss_to_idx.keys())
        return [g for g in self._gloss_to_idx.keys() if g not in self.SPECIAL_TOKENS]

    def save(self, path: str) -> None:
        """Save vocabulary to JSON file."""
        data = {
            "gloss_to_idx": self._gloss_to_idx,
            "min_count": self.min_count,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GlossVocabulary":
        """Load vocabulary from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        vocab = cls(min_count=data.get("min_count", 1))
        vocab._gloss_to_idx = data["gloss_to_idx"]
        vocab._idx_to_gloss = {int(v): k for k, v in data["gloss_to_idx"].items()}
        return vocab

    @classmethod
    def from_sign_to_prediction_map(cls, path: str) -> "GlossVocabulary":
        """Create vocabulary from a sign_to_prediction_index_map.json file."""
        with open(path, "r") as f:
            sign_map = json.load(f)
        sorted_signs = sorted(sign_map.items(), key=lambda x: x[1])
        glosses = [sign for sign, _ in sorted_signs]
        return cls(glosses=glosses)


# =============================================================================
# Dataset
# =============================================================================


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
    ):
        self.df = df.reset_index(drop=True)
        self.base_path = Path(base_path)
        self.sign_to_idx = sign_to_idx
        self.processor = FastLandmarkProcessor(config)
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
        row = self.df.iloc[idx]
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
            [data["left_hand"], data["right_hand"], data["pose"], data["face"]],
            axis=1,
        )

        # Get label
        sign = row["sign"]
        label = self.sign_to_idx.get(sign, -1)
        if label == -1:
            raise ValueError(f"Unknown sign: {sign}")

        return {
            "landmarks": torch.tensor(landmarks, dtype=torch.float32),
            "left_hand": torch.tensor(data["left_hand"], dtype=torch.float32),
            "right_hand": torch.tensor(data["right_hand"], dtype=torch.float32),
            "pose": torch.tensor(data["pose"], dtype=torch.float32),
            "face": torch.tensor(data["face"], dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "length": torch.tensor(T, dtype=torch.long),
        }


def collate_translation(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Collate function for translation dataset."""
    lengths = torch.stack([item["length"] for item in batch])
    max_len = lengths.max().item()

    batch_size = len(batch)
    first = batch[0]
    n_landmarks = first["landmarks"].shape[1]
    n_coords = first["landmarks"].shape[2]

    landmarks = torch.zeros(batch_size, max_len, n_landmarks, n_coords)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    left_hand = torch.zeros(batch_size, max_len, 21, n_coords)
    right_hand = torch.zeros(batch_size, max_len, 21, n_coords)
    pose = torch.zeros(batch_size, max_len, 33, n_coords)

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
        "labels": torch.stack([item["label"] for item in batch]),
    }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn: Callable = collate_translation,
) -> DataLoader:
    """Create a DataLoader with appropriate settings."""
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
# Token Embedding
# =============================================================================


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class FactorizedTokenEmbedding(nn.Module):
    """
    Embedding layer for factorized VQ-VAE tokens.

    Each token consists of (pose_id, motion_id, dynamics_id, face_id).
    Embeddings are initialized from pre-trained VQ-VAE codebooks.
    """

    def __init__(
        self,
        pose_codebook_size: int = 1024,
        motion_codebook_size: int = 512,
        dynamics_codebook_size: int = 256,
        face_codebook_size: int = 256,
        embed_dim: int = 128,
        d_model: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_model = d_model

        self.pose_embed = nn.Embedding(pose_codebook_size, embed_dim)
        self.motion_embed = nn.Embedding(motion_codebook_size, embed_dim)
        self.dynamics_embed = nn.Embedding(dynamics_codebook_size, embed_dim)
        self.face_embed = nn.Embedding(face_codebook_size, embed_dim)

        self.projection = nn.Sequential(
            nn.Linear(embed_dim * 4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.scale = math.sqrt(d_model)

    def init_from_codebooks(self, codebooks: Dict[str, torch.Tensor]):
        """Initialize embeddings from pre-trained VQ-VAE codebooks."""
        if "pose" in codebooks:
            self.pose_embed.weight.data.copy_(codebooks["pose"])
        if "motion" in codebooks:
            self.motion_embed.weight.data.copy_(codebooks["motion"])
        if "dynamics" in codebooks:
            self.dynamics_embed.weight.data.copy_(codebooks["dynamics"])
        if "face" in codebooks:
            self.face_embed.weight.data.copy_(codebooks["face"])
        print("Initialized embeddings from VQ-VAE codebooks")

    def forward(
        self,
        pose_ids: torch.Tensor,
        motion_ids: torch.Tensor,
        dynamics_ids: torch.Tensor,
        face_ids: torch.Tensor,
    ) -> torch.Tensor:
        pose_emb = self.pose_embed(pose_ids)
        motion_emb = self.motion_embed(motion_ids)
        dynamics_emb = self.dynamics_embed(dynamics_ids)
        face_emb = self.face_embed(face_ids)

        combined = torch.cat([pose_emb, motion_emb, dynamics_emb, face_emb], dim=-1)
        projected = self.projection(combined)
        scaled = projected * self.scale
        output = self.pos_encoding(scaled)

        return output

    def forward_dict(self, token_indices: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with dictionary input."""
        return self.forward(
            pose_ids=token_indices["pose"],
            motion_ids=token_indices["motion"],
            dynamics_ids=token_indices["dynamics"],
            face_ids=token_indices["face"],
        )


class DirectLandmarkEmbedding(nn.Module):
    """
    Alternative embedding that works directly on landmark features.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        n_conv_layers: int = 2,
        kernel_size: int = 3,
        stride: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        conv_layers = []
        in_dim = input_dim
        out_dim = d_model // 4

        for i in range(n_conv_layers):
            conv_layers.extend(
                [
                    nn.Conv1d(in_dim, out_dim, kernel_size, stride, kernel_size // 2),
                    nn.BatchNorm1d(out_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = out_dim
            out_dim = min(out_dim * 2, d_model)

        self.conv_layers = nn.Sequential(*conv_layers)
        self.projection = nn.Linear(in_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        x = landmarks.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        x = x * self.scale
        x = self.pos_encoding(x)
        return x


class GlossEmbedding(nn.Module):
    """Embedding layer for gloss output tokens (decoder side)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        pad_idx: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens) * self.scale
        return self.pos_encoding(x)


# =============================================================================
# Conformer Encoder
# =============================================================================


class ConvolutionModule(nn.Module):
    """Convolution module in Conformer."""

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
        expansion_factor: int = 2,
    ):
        super().__init__()

        inner_dim = d_model * expansion_factor

        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, inner_dim * 2, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=inner_dim,
        )
        self.batch_norm = nn.BatchNorm1d(inner_dim)
        self.pointwise_conv2 = nn.Conv1d(inner_dim, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x.permute(0, 2, 1)


class FeedForwardModule(nn.Module):
    """Feed-forward module in Conformer."""

    def __init__(
        self,
        d_model: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.layer_norm(x))


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with relative positional encoding."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_norm = self.layer_norm(x)
        attn_out, _ = self.attention(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=mask,
            need_weights=False,
        )
        return self.dropout(attn_out)


class ConformerBlock(nn.Module):
    """Single Conformer block."""

    def __init__(
        self,
        d_model: int,
        d_ff: int = 2048,
        n_heads: int = 8,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.conv = ConvolutionModule(d_model, kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.attention(x, mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        x = self.layer_norm(x)
        return x


class ConformerEncoder(nn.Module):
    """Full Conformer encoder stack."""

    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        n_heads: int = 8,
        n_layers: int = 12,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    n_heads=n_heads,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class SpecAugment(nn.Module):
    """SpecAugment-like augmentation for sign language sequences."""

    def __init__(
        self,
        time_mask_max: int = 50,
        time_mask_num: int = 2,
        mask_value: float = 0.0,
    ):
        super().__init__()
        self.time_mask_max = time_mask_max
        self.time_mask_num = time_mask_num
        self.mask_value = mask_value

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        if not training:
            return x

        B, T, D = x.shape
        x = x.clone()

        for _ in range(self.time_mask_num):
            t_start = torch.randint(0, max(1, T - self.time_mask_max), (B,))
            t_length = torch.randint(1, self.time_mask_max + 1, (B,))

            for b in range(B):
                start = t_start[b].item()
                length = min(t_length[b].item(), T - start)
                x[b, start : start + length] = self.mask_value

        return x


# =============================================================================
# Decoder
# =============================================================================


class CTCHead(nn.Module):
    """CTC head for auxiliary loss and decoding."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        blank_idx: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blank_idx = blank_idx
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_size),
        )

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        logits = self.projection(encoder_output)
        return F.log_softmax(logits, dim=-1)

    def compute_loss(
        self,
        encoder_output: torch.Tensor,
        targets: torch.Tensor,
        encoder_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = self(encoder_output)
        log_probs = log_probs.permute(1, 0, 2)

        loss = F.ctc_loss(
            log_probs,
            targets,
            encoder_lengths,
            target_lengths,
            blank=self.blank_idx,
            zero_infinity=True,
        )
        return loss


class DecoderLayer(nn.Module):
    """Single decoder layer with self-attention and cross-attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn_dropout = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_norm = self.self_attn_norm(x)
        self_attn_out, _ = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=self_attn_mask,
            need_weights=False,
        )
        x = x + self.self_attn_dropout(self_attn_out)

        x_norm = self.cross_attn_norm(x)
        cross_attn_out, _ = self.cross_attn(
            x_norm,
            encoder_output,
            encoder_output,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        x = x + self.cross_attn_dropout(cross_attn_out)

        x_norm = self.ffn_norm(x)
        x = x + self.ffn(x_norm)

        return x


class AttentionDecoder(nn.Module):
    """Autoregressive attention decoder."""

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        vocab_size: int = 2500,
        pad_idx: int = 0,
        dropout: float = 0.1,
        max_len: int = 100,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        self.pos_encoding = self._create_pos_encoding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Tie embedding weights
        self.output_projection.weight = self.embedding.weight

        self.scale = math.sqrt(d_model)

        self.register_buffer("causal_mask", self._create_causal_mask(max_len))

    def _create_pos_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _create_causal_mask(self, size: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask

    def forward(
        self,
        decoder_input: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = decoder_input.shape

        x = self.embedding(decoder_input) * self.scale
        x = x + self.pos_encoding[:, :T].to(x.device)
        x = self.dropout(x)

        causal_mask = self.causal_mask[:T, :T]

        for layer in self.layers:
            x = layer(x, encoder_output, causal_mask, encoder_mask)

        x = self.layer_norm(x)
        logits = self.output_projection(x)

        return logits

    def compute_loss(
        self,
        encoder_output: torch.Tensor,
        targets: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
    ) -> torch.Tensor:
        decoder_input = targets[:, :-1]
        target_output = targets[:, 1:]

        logits = self(decoder_input, encoder_output, encoder_mask)

        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = target_output.reshape(-1)

        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.pad_idx,
            label_smoothing=label_smoothing,
        )

        return loss


class HybridDecoder(nn.Module):
    """Hybrid CTC + Attention decoder."""

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        vocab_size: int = 2500,
        pad_idx: int = 0,
        blank_idx: int = 0,
        ctc_weight: float = 0.3,
        dropout: float = 0.1,
        label_smoothing: float = 0.1,
    ):
        super().__init__()

        self.ctc_weight = ctc_weight
        self.label_smoothing = label_smoothing

        self.ctc = CTCHead(d_model, vocab_size, blank_idx, dropout)
        self.attention = AttentionDecoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            dropout=dropout,
        )

    def forward(
        self,
        encoder_output: torch.Tensor,
        decoder_input: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_logits = self.attention(decoder_input, encoder_output, encoder_mask)
        ctc_log_probs = self.ctc(encoder_output)
        return attn_logits, ctc_log_probs

    def compute_loss(
        self,
        encoder_output: torch.Tensor,
        targets: torch.Tensor,
        encoder_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        attn_loss = self.attention.compute_loss(
            encoder_output,
            targets,
            encoder_mask,
            self.label_smoothing,
        )

        ctc_targets = targets[:, 1:]
        ctc_loss = self.ctc.compute_loss(
            encoder_output,
            ctc_targets,
            encoder_lengths,
            target_lengths,
        )

        total_loss = (1 - self.ctc_weight) * attn_loss + self.ctc_weight * ctc_loss

        return {
            "total": total_loss,
            "ctc": ctc_loss,
            "attention": attn_loss,
        }


# =============================================================================
# Beam Search
# =============================================================================


@dataclass
class BeamHypothesis:
    """Single hypothesis in beam search."""

    tokens: List[int]
    score: float
    ctc_score: float
    attn_score: float
    finished: bool = False

    def __lt__(self, other: "BeamHypothesis") -> bool:
        return self.score < other.score


class CTCPrefixScorer:
    """Computes CTC prefix scores for beam search."""

    def __init__(
        self,
        ctc_log_probs: torch.Tensor,
        blank_idx: int = 0,
        eos_idx: int = 2,
    ):
        self.log_probs = ctc_log_probs
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.T = ctc_log_probs.shape[0]
        self.prefix_scores: Dict[tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_prefix_score(self, prefix: Tuple[int, ...]) -> float:
        if len(prefix) == 0:
            return 0.0

        T = self.T
        prefix_tensor = torch.tensor(prefix, device=self.log_probs.device)
        L = len(prefix)

        n_states = 2 * L + 1
        alpha = torch.full((T, n_states), float("-inf"), device=self.log_probs.device)

        alpha[0, 0] = self.log_probs[0, self.blank_idx]
        if L > 0:
            alpha[0, 1] = self.log_probs[0, prefix[0]]

        for t in range(1, T):
            for s in range(n_states):
                if s % 2 == 0:
                    char_idx = self.blank_idx
                else:
                    char_idx = prefix[s // 2]

                alpha[t, s] = alpha[t - 1, s]

                if s > 0:
                    alpha[t, s] = torch.logaddexp(alpha[t, s], alpha[t - 1, s - 1])

                if s > 1 and s % 2 == 1:
                    prev_char = prefix[(s - 2) // 2] if (s - 2) // 2 < L else -1
                    curr_char = prefix[s // 2]
                    if prev_char != curr_char:
                        alpha[t, s] = torch.logaddexp(alpha[t, s], alpha[t - 1, s - 2])

                alpha[t, s] = alpha[t, s] + self.log_probs[t, char_idx]

        final_score = torch.logaddexp(
            alpha[-1, -1], alpha[-1, -2] if n_states > 1 else alpha[-1, -1]
        )

        return final_score.item()

    def score_hypothesis(self, hypothesis: List[int]) -> float:
        prefix = tuple(hypothesis)
        return self._get_prefix_score(prefix)


class BeamSearch:
    """Beam search decoder with CTC prefix scoring."""

    def __init__(
        self,
        beam_size: int = 5,
        max_len: int = 50,
        eos_idx: int = 2,
        bos_idx: int = 1,
        pad_idx: int = 0,
        length_penalty: float = 0.6,
        ctc_weight: float = 0.4,
    ):
        self.beam_size = beam_size
        self.max_len = max_len
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx
        self.pad_idx = pad_idx
        self.length_penalty = length_penalty
        self.ctc_weight = ctc_weight

    def _length_normalize(self, score: float, length: int) -> float:
        return score / (length**self.length_penalty)

    def search(
        self,
        encoder_output: torch.Tensor,
        decoder: nn.Module,
        ctc_log_probs: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        device = encoder_output.device

        ctc_scorer = None
        if ctc_log_probs is not None and self.ctc_weight > 0:
            ctc_scorer = CTCPrefixScorer(ctc_log_probs[0], self.pad_idx, self.eos_idx)

        beams = [
            BeamHypothesis(
                tokens=[self.bos_idx],
                score=0.0,
                ctc_score=0.0,
                attn_score=0.0,
            )
        ]

        finished = []

        for step in range(self.max_len):
            all_candidates = []

            for beam in beams:
                if beam.finished:
                    finished.append(beam)
                    continue

                input_ids = torch.tensor([beam.tokens], device=device)
                logits = decoder(input_ids, encoder_output)
                log_probs = F.log_softmax(logits[0, -1], dim=-1)

                topk_log_probs, topk_ids = log_probs.topk(self.beam_size * 2)

                for log_prob, token_id in zip(
                    topk_log_probs.tolist(), topk_ids.tolist()
                ):
                    new_tokens = beam.tokens + [token_id]
                    new_attn_score = beam.attn_score + log_prob

                    if ctc_scorer is not None:
                        new_ctc_score = ctc_scorer.score_hypothesis(new_tokens[1:])
                    else:
                        new_ctc_score = 0.0

                    combined_score = (
                        1 - self.ctc_weight
                    ) * new_attn_score + self.ctc_weight * new_ctc_score

                    all_candidates.append(
                        BeamHypothesis(
                            tokens=new_tokens,
                            score=combined_score,
                            ctc_score=new_ctc_score,
                            attn_score=new_attn_score,
                            finished=(token_id == self.eos_idx),
                        )
                    )

            all_candidates.sort(
                key=lambda x: -self._length_normalize(x.score, len(x.tokens))
            )
            beams = all_candidates[: self.beam_size]

            if all(beam.finished for beam in beams):
                break

        finished.extend([b for b in beams if b.finished])
        if not finished:
            finished = beams

        finished.sort(key=lambda x: -self._length_normalize(x.score, len(x.tokens)))

        results = []
        for beam in finished[: self.beam_size]:
            tokens = beam.tokens[1:]
            if tokens and tokens[-1] == self.eos_idx:
                tokens = tokens[:-1]
            results.append(tokens)

        return results


class GreedyDecoder:
    """Simple greedy decoding (for fast inference)."""

    def __init__(
        self,
        max_len: int = 50,
        eos_idx: int = 2,
        bos_idx: int = 1,
    ):
        self.max_len = max_len
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx

    def decode(
        self,
        encoder_output: torch.Tensor,
        decoder: nn.Module,
    ) -> List[int]:
        device = encoder_output.device
        tokens = [self.bos_idx]

        for _ in range(self.max_len):
            input_ids = torch.tensor([tokens], device=device)
            logits = decoder(input_ids, encoder_output)
            next_token = logits[0, -1].argmax().item()

            if next_token == self.eos_idx:
                break

            tokens.append(next_token)

        return tokens[1:]


# =============================================================================
# Sign Translator Model
# =============================================================================


class SignTranslator(nn.Module):
    """
    Complete sign language translator model.

    Architecture:
    - Input: Factorized VQ-VAE tokens (pose, motion, dynamics, face)
    - Encoder: Conformer
    - Decoder: Hybrid CTC + Attention
    - Output: Gloss sequence
    """

    def __init__(self, config: Optional[TranslationConfig] = None):
        super().__init__()
        self.config = config or TranslationConfig()

        self.token_embedding = FactorizedTokenEmbedding(
            pose_codebook_size=self.config.pose_codebook_size,
            motion_codebook_size=self.config.motion_codebook_size,
            dynamics_codebook_size=self.config.dynamics_codebook_size,
            face_codebook_size=self.config.face_codebook_size,
            embed_dim=self.config.embed_dim,
            d_model=self.config.d_model,
            dropout=self.config.encoder_dropout,
        )

        self.spec_augment = (
            SpecAugment(
                time_mask_max=self.config.time_mask_max,
                time_mask_num=self.config.time_mask_num,
            )
            if self.config.spec_augment
            else None
        )

        self.encoder = ConformerEncoder(
            d_model=self.config.d_model,
            d_ff=self.config.d_ff,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_encoder_layers,
            kernel_size=self.config.encoder_kernel_size,
            dropout=self.config.encoder_dropout,
        )

        self.decoder = HybridDecoder(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_decoder_layers,
            d_ff=self.config.d_ff,
            vocab_size=self.config.vocab_size,
            pad_idx=self.config.pad_idx,
            blank_idx=self.config.ctc_blank_idx,
            ctc_weight=self.config.ctc_weight,
            dropout=self.config.decoder_dropout,
            label_smoothing=self.config.label_smoothing,
        )

        self.beam_search = BeamSearch(
            beam_size=self.config.beam_size,
            max_len=self.config.max_decode_len,
            eos_idx=self.config.eos_idx,
            bos_idx=self.config.bos_idx,
            pad_idx=self.config.pad_idx,
            length_penalty=self.config.length_penalty,
            ctc_weight=self.config.ctc_prefix_weight,
        )

        self.greedy_decoder = GreedyDecoder(
            max_len=self.config.max_decode_len,
            eos_idx=self.config.eos_idx,
            bos_idx=self.config.bos_idx,
        )

    def init_from_vqvae(self, vqvae_codebooks: Dict[str, torch.Tensor]):
        """Initialize token embeddings from pre-trained VQ-VAE codebooks."""
        self.token_embedding.init_from_codebooks(vqvae_codebooks)

    def encode(
        self,
        token_indices: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.token_embedding.forward_dict(token_indices)

        if self.training and self.spec_augment is not None:
            x = self.spec_augment(x)

        encoder_output = self.encoder(x, mask)
        return encoder_output

    def forward(
        self,
        token_indices: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        encoder_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoder_output = self.encode(token_indices, encoder_mask)

        if encoder_lengths is None:
            if encoder_mask is not None:
                encoder_lengths = (~encoder_mask).sum(dim=1)
            else:
                encoder_lengths = torch.full(
                    (encoder_output.shape[0],),
                    encoder_output.shape[1],
                    device=encoder_output.device,
                )

        if target_lengths is None:
            target_lengths = (targets != self.config.pad_idx).sum(dim=1) - 1

        losses = self.decoder.compute_loss(
            encoder_output,
            targets,
            encoder_lengths,
            target_lengths,
            encoder_mask,
        )

        return losses

    @torch.no_grad()
    def translate(
        self,
        token_indices: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        use_beam_search: bool = True,
    ) -> List[List[int]]:
        self.eval()
        B = token_indices["pose"].shape[0]

        encoder_output = self.encode(token_indices, mask)
        ctc_log_probs = self.decoder.ctc(encoder_output)

        results = []
        for b in range(B):
            enc_out_b = encoder_output[b : b + 1]
            ctc_probs_b = ctc_log_probs[b : b + 1]

            if use_beam_search:
                decoded = self.beam_search.search(
                    enc_out_b,
                    self.decoder.attention,
                    ctc_probs_b,
                )
                results.append(decoded[0] if decoded else [])
            else:
                decoded = self.greedy_decoder.decode(enc_out_b, self.decoder.attention)
                results.append(decoded)

        return results


# =============================================================================
# Trainer
# =============================================================================


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
):
    """Create learning rate schedule with linear warmup."""

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Training loop for Sign Translator."""

    def __init__(
        self,
        model: SignTranslator,
        vqvae: ImprovedVQVAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        vocabulary: GlossVocabulary,
        config: TranslationConfig,
        device: torch.device,
        output_dir: str,
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.vqvae = vqvae.to(device)
        self.vqvae.eval()

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocabulary = vocabulary
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.use_amp = use_amp and device.type == "cuda"

        # Scaler for AMP
        self.scaler = GradScaler("cuda") if self.use_amp else None

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        num_training_steps = len(train_loader) * config.max_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps,
        )

        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0

    def _tokenize_batch(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Tokenize landmarks using VQ-VAE."""
        with torch.no_grad():
            landmarks = batch["landmarks"].to(self.device)
            mask = batch["mask"].to(self.device)
            indices = self.vqvae.tokenize(landmarks, mask)
        return indices

    def _prepare_targets(self, labels: torch.Tensor) -> torch.Tensor:
        """Prepare target sequences with BOS token."""
        B = labels.shape[0]
        device = labels.device

        targets = torch.full(
            (B, 3),
            self.config.pad_idx,
            dtype=torch.long,
            device=device,
        )
        targets[:, 0] = self.config.bos_idx
        targets[:, 1] = labels
        targets[:, 2] = self.config.eos_idx

        return targets

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_losses = {"total": 0, "ctc": 0, "attention": 0}
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            token_indices = self._tokenize_batch(batch)

            labels = batch["labels"].to(self.device)
            targets = self._prepare_targets(labels)

            mask = batch["mask"].to(self.device)
            encoder_lengths = (~mask).sum(dim=1)
            chunk_size = 8
            encoder_lengths = (encoder_lengths // chunk_size).clamp(min=1)

            target_lengths = torch.ones(labels.shape[0], device=self.device, dtype=torch.long)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast(device_type="cuda"):
                    losses = self.model(
                        token_indices,
                        targets,
                        encoder_mask=None,
                        encoder_lengths=encoder_lengths,
                        target_lengths=target_lengths,
                    )

                self.scaler.scale(losses["total"]).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses = self.model(
                    token_indices,
                    targets,
                    encoder_mask=None,
                    encoder_lengths=encoder_lengths,
                    target_lengths=target_lengths,
                )
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )
                self.optimizer.step()

            self.scheduler.step()

            for key in total_losses:
                if key in losses:
                    total_losses[key] += losses[key].item()
            n_batches += 1

            pbar.set_postfix(
                {
                    "loss": f"{losses['total'].item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                }
            )

            if n_batches % 200 == 0:
                torch.cuda.empty_cache()

        return {k: v / n_batches for k, v in total_losses.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_losses = {"total": 0, "ctc": 0, "attention": 0}
        correct = 0
        total = 0
        n_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            token_indices = self._tokenize_batch(batch)

            labels = batch["labels"].to(self.device)
            targets = self._prepare_targets(labels)

            mask = batch["mask"].to(self.device)
            encoder_lengths = (~mask).sum(dim=1) // 8
            encoder_lengths = encoder_lengths.clamp(min=1)
            target_lengths = torch.ones(labels.shape[0], device=self.device, dtype=torch.long)

            if self.use_amp:
                with autocast(device_type="cuda"):
                    losses = self.model(
                        token_indices,
                        targets,
                        encoder_lengths=encoder_lengths,
                        target_lengths=target_lengths,
                    )
            else:
                losses = self.model(
                    token_indices,
                    targets,
                    encoder_lengths=encoder_lengths,
                    target_lengths=target_lengths,
                )

            for key in total_losses:
                if key in losses:
                    total_losses[key] += losses[key].item()

            predictions = self.model.translate(token_indices, use_beam_search=False)

            for pred, label in zip(predictions, labels.tolist()):
                if len(pred) > 0 and pred[0] == label:
                    correct += 1
                total += 1

            n_batches += 1

        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        avg_losses["accuracy"] = correct / total if total > 0 else 0

        return avg_losses

    def train(self):
        """Full training loop."""
        for epoch in range(self.config.max_epochs):
            train_losses = self.train_epoch(epoch)
            val_metrics = self.validate()

            print(f"\nEpoch {epoch}:")
            print(f"  Train - Loss: {train_losses['total']:.4f}")
            print(
                f"  Val   - Loss: {val_metrics['total']:.4f}, Acc: {val_metrics['accuracy']:.1%}"
            )

            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, val_metrics)

            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self._save_checkpoint(epoch, val_metrics, best=True)
                print(f"  New best accuracy: {self.best_val_acc:.1%}")

        print(f"\nTraining complete! Best accuracy: {self.best_val_acc:.1%}")

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        best: bool = False,
    ):
        """Save checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": asdict(self.config),
        }

        if best:
            path = os.path.join(self.output_dir, "best_model.pt")
        else:
            path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")

        torch.save(checkpoint, path)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train Sign Language Translator")
    parser.add_argument("--data-dir", type=str, default="data/Isolated_ASL_Recognition")
    parser.add_argument(
        "--vqvae-checkpoint",
        type=str,
        required=True,
        help="Path to trained VQ-VAE checkpoint",
    )
    parser.add_argument("--output-dir", type=str, default="checkpoints/translator")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--num-workers", type=int, default=0, help="DataLoader workers (0 for Kaggle)"
    )
    parser.add_argument(
        "--use-amp", action="store_true", default=True, help="Use mixed precision (AMP)"
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    use_amp = args.use_amp and not args.no_amp

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load VQ-VAE
    print("Loading VQ-VAE...")
    vqvae_checkpoint = torch.load(args.vqvae_checkpoint, map_location=device)
    vqvae_config = ImprovedVQVAEConfig(**vqvae_checkpoint["config"])
    vqvae = ImprovedVQVAE(vqvae_config)
    vqvae.load_state_dict(vqvae_checkpoint["model_state_dict"])
    vqvae = vqvae.to(device)
    vqvae.eval()
    print("VQ-VAE loaded successfully")

    # Load vocabulary
    sign_map_path = os.path.join(args.data_dir, "sign_to_prediction_index_map.json")
    vocabulary = GlossVocabulary.from_sign_to_prediction_map(sign_map_path)
    print(f"Vocabulary size: {len(vocabulary)}")

    # Create config
    config = TranslationConfig(
        vocab_size=len(vocabulary),
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        device=str(device),
    )

    # Save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Load data
    print("Loading data...")
    csv_path = os.path.join(args.data_dir, "train.csv")
    with open(sign_map_path) as f:
        sign_to_idx = json.load(f)

    splits, split_info = create_signer_splits(csv_path, args.data_dir)

    landmark_config = LandmarkConfig(
        include_z=True,
        face_subset=FACE_LANDMARK_SUBSETS["compact"],
    )

    train_dataset = TranslationDataset(
        df=splits["train"],
        base_path=args.data_dir,
        sign_to_idx=sign_to_idx,
        config=landmark_config,
        augment=True,
    )

    val_dataset = TranslationDataset(
        df=splits["val"],
        base_path=args.data_dir,
        sign_to_idx=sign_to_idx,
        config=landmark_config,
        augment=False,
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_translation,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_translation,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model
    model = SignTranslator(config)

    # Initialize from VQ-VAE codebooks
    codebooks = vqvae.get_codebook_embeddings()
    model.init_from_vqvae(codebooks)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if use_amp:
        print("Mixed precision training (AMP) enabled")

    # Create trainer
    trainer = Trainer(
        model=model,
        vqvae=vqvae,
        train_loader=train_loader,
        val_loader=val_loader,
        vocabulary=vocabulary,
        config=config,
        device=device,
        output_dir=args.output_dir,
        use_amp=use_amp,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
