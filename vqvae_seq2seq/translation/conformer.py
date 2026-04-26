"""Conformer encoder for sign language translation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class ConvolutionModule(nn.Module):
    """
    Convolution module in Conformer.

    Consists of:
    1. Pointwise conv
    2. GLU activation
    3. Depthwise conv
    4. BatchNorm
    5. Swish activation
    6. Pointwise conv
    """

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

        # Pointwise expansion with GLU
        self.pointwise_conv1 = nn.Conv1d(d_model, inner_dim * 2, kernel_size=1)

        # Depthwise conv
        self.depthwise_conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=inner_dim,
        )
        self.batch_norm = nn.BatchNorm1d(inner_dim)

        # Pointwise projection
        self.pointwise_conv2 = nn.Conv1d(inner_dim, d_model, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) input tensor

        Returns:
            (B, T, D) output tensor
        """
        x = self.layer_norm(x)

        # (B, T, D) -> (B, D, T) for conv
        x = x.permute(0, 2, 1)

        # Pointwise with GLU
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)  # (B, inner_dim, T)

        # Depthwise conv
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)  # Swish activation

        # Pointwise projection
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        # (B, D, T) -> (B, T, D)
        return x.permute(0, 2, 1)


class FeedForwardModule(nn.Module):
    """
    Feed-forward module in Conformer.

    Uses Swish activation and layer norm.
    """

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
            nn.SiLU(),  # Swish
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.layer_norm(x))


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with relative positional encoding.
    """

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
        """
        Args:
            x: (B, T, D) input tensor
            mask: Optional (B, T) key padding mask (True = masked)

        Returns:
            (B, T, D) output tensor
        """
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
    """
    Single Conformer block.

    Architecture:
    1. Half feed-forward
    2. Self-attention
    3. Convolution
    4. Half feed-forward
    5. Layer norm
    """

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
        """
        Args:
            x: (B, T, D) input tensor
            mask: Optional (B, T) key padding mask

        Returns:
            (B, T, D) output tensor
        """
        # Half feed-forward
        x = x + 0.5 * self.ff1(x)

        # Self-attention
        x = x + self.attention(x, mask)

        # Convolution
        x = x + self.conv(x)

        # Half feed-forward
        x = x + 0.5 * self.ff2(x)

        # Final layer norm
        x = self.layer_norm(x)

        return x


class ConformerEncoder(nn.Module):
    """
    Full Conformer encoder stack.

    Better than pure Transformer for sequential sign data due to
    the convolution modules that capture local patterns.
    """

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
        """
        Args:
            x: (B, T, D) input tensor
            mask: Optional (B, T) key padding mask

        Returns:
            (B, T, D) encoded tensor
        """
        for layer in self.layers:
            x = layer(x, mask)

        return x


class SpecAugment(nn.Module):
    """
    SpecAugment-like augmentation for sign language sequences.

    Applies time masking to input sequences during training.
    """

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
        """
        Args:
            x: (B, T, D) input tensor
            training: Whether in training mode

        Returns:
            Augmented tensor
        """
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


class SubsamplingEncoder(nn.Module):
    """
    Encoder with initial subsampling to reduce sequence length.

    Useful for long sequences to reduce computation.
    """

    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        n_heads: int = 8,
        n_layers: int = 12,
        kernel_size: int = 31,
        dropout: float = 0.1,
        subsample_factor: int = 4,
    ):
        super().__init__()

        # Subsampling convolutions
        self.subsampling = (
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
            )
            if subsample_factor == 4
            else nn.Identity()
        )

        self.subsample_factor = subsample_factor

        # Conformer layers
        self.conformer = ConformerEncoder(
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, T, D) input tensor
            mask: Optional (B, T) key padding mask

        Returns:
            Tuple of (encoded tensor, updated mask)
        """
        # Subsample
        if self.subsample_factor > 1:
            x = x.permute(0, 2, 1)  # (B, D, T)
            x = self.subsampling(x)
            x = x.permute(0, 2, 1)  # (B, T', D)

            # Update mask
            if mask is not None:
                T_new = x.shape[1]
                mask = mask[:, :: self.subsample_factor][:, :T_new]

        # Encode
        x = self.conformer(x, mask)

        return x, mask
