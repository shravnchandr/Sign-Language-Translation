import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * torch.sigmoid(gate)


class ConformerConvModule(nn.Module):
    """Convolution module used in Conformer blocks."""

    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        # Pointwise Conv 1 (using Linear for T, D layout)
        self.pointwise_conv1 = nn.Linear(d_model, d_model * 2)
        self.glu = GLU(dim=-1)
        # Depthwise Conv
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = Swish()
        # Pointwise Conv 2
        self.pointwise_conv2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        x = self.layer_norm(x)
        x = self.pointwise_conv1(x)
        x = self.glu(x)  # (B, T, D)

        # Prepare for Depthwise Conv1d
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = x.transpose(1, 2)  # (B, T, D)

        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x


class ConformerBlock(nn.Module):
    """Combines Conv-style local modeling with Transformer global modeling."""

    def __init__(self, d_model, n_heads, kernel_size=31, dropout=0.1):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)

        self.conv = ConformerConvModule(d_model, kernel_size, dropout)

        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # x: (B, T, D)
        # 1. Feed Forward 1
        x = x + 0.5 * self.ff1(x)

        # 2. Multi-head Self Attention
        residual = x
        x = self.attn_norm(x)
        x, _ = self.attn(x, x, x, key_padding_mask=~mask if mask is not None else None)
        x = x + residual

        # 3. Convolution Module
        x = x + self.conv(x)

        # 4. Feed Forward 2
        x = x + 0.5 * self.ff2(x)

        return self.final_norm(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal PE — no max-length crash, no trainable parameters."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[: x.size(1)])
