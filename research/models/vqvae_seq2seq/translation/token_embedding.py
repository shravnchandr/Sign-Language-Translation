"""Token embedding layer for VQ-VAE tokens."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) tensor

        Returns:
            (B, T, D) tensor with positional encoding added
        """
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
        """
        Args:
            pose_codebook_size: Size of pose codebook
            motion_codebook_size: Size of motion codebook
            dynamics_codebook_size: Size of dynamics codebook
            face_codebook_size: Size of face codebook
            embed_dim: VQ-VAE embedding dimension
            d_model: Transformer model dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.d_model = d_model

        # Separate embedding layers for each factor
        self.pose_embed = nn.Embedding(pose_codebook_size, embed_dim)
        self.motion_embed = nn.Embedding(motion_codebook_size, embed_dim)
        self.dynamics_embed = nn.Embedding(dynamics_codebook_size, embed_dim)
        self.face_embed = nn.Embedding(face_codebook_size, embed_dim)

        # Projection to model dimension
        # 4 factors * embed_dim -> d_model
        self.projection = nn.Sequential(
            nn.Linear(embed_dim * 4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Scale factor for embedding
        self.scale = math.sqrt(d_model)

    def init_from_codebooks(self, codebooks: Dict[str, torch.Tensor]):
        """
        Initialize embeddings from pre-trained VQ-VAE codebooks.

        Args:
            codebooks: Dictionary with keys 'pose', 'motion', 'dynamics', 'face'
                      and values as (num_codes, embed_dim) tensors
        """
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
        """
        Embed factorized tokens.

        Args:
            pose_ids: (B, T) pose token indices
            motion_ids: (B, T) motion token indices
            dynamics_ids: (B, T) dynamics token indices
            face_ids: (B, T) face token indices

        Returns:
            (B, T, d_model) embedded sequence
        """
        # Get embeddings for each factor
        pose_emb = self.pose_embed(pose_ids)  # (B, T, embed_dim)
        motion_emb = self.motion_embed(motion_ids)
        dynamics_emb = self.dynamics_embed(dynamics_ids)
        face_emb = self.face_embed(face_ids)

        # Concatenate all factors
        combined = torch.cat([pose_emb, motion_emb, dynamics_emb, face_emb], dim=-1)

        # Project to model dimension
        projected = self.projection(combined)

        # Scale and add positional encoding
        scaled = projected * self.scale
        output = self.pos_encoding(scaled)

        return output

    def forward_dict(self, token_indices: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with dictionary input.

        Args:
            token_indices: Dictionary with 'pose', 'motion', 'dynamics', 'face' keys

        Returns:
            (B, T, d_model) embedded sequence
        """
        return self.forward(
            pose_ids=token_indices["pose"],
            motion_ids=token_indices["motion"],
            dynamics_ids=token_indices["dynamics"],
            face_ids=token_indices["face"],
        )


class DirectLandmarkEmbedding(nn.Module):
    """
    Alternative embedding that works directly on landmark features
    (before VQ-VAE tokenization).

    Useful for end-to-end training or when VQ-VAE tokens aren't available.
    """

    def __init__(
        self,
        input_dim: int,  # Total landmark dimension (e.g., 209 * 3 = 627)
        d_model: int = 512,
        n_conv_layers: int = 2,
        kernel_size: int = 3,
        stride: int = 2,  # Subsample time dimension
        dropout: float = 0.1,
    ):
        super().__init__()

        # Convolutional subsampling (like in speech recognition)
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

        # Final projection
        self.projection = nn.Linear(in_dim, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        self.scale = math.sqrt(d_model)

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Embed landmark sequence.

        Args:
            landmarks: (B, T, D) landmark features

        Returns:
            (B, T', d_model) embedded and subsampled sequence
        """
        # (B, T, D) -> (B, D, T)
        x = landmarks.permute(0, 2, 1)

        # Apply convolutions (subsamples time)
        x = self.conv_layers(x)

        # (B, D', T') -> (B, T', D')
        x = x.permute(0, 2, 1)

        # Project to model dimension
        x = self.projection(x)

        # Scale and add positional encoding
        x = x * self.scale
        x = self.pos_encoding(x)

        return x


class GlossEmbedding(nn.Module):
    """
    Embedding layer for gloss output tokens (decoder side).
    """

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
        """
        Embed gloss tokens.

        Args:
            tokens: (B, T) token indices

        Returns:
            (B, T, d_model) embedded sequence
        """
        x = self.embedding(tokens) * self.scale
        return self.pos_encoding(x)
