import numpy as np
import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

_SDPA_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


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


class MaskedBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d whose batch statistics ignore padded frames.

    nn.BatchNorm1d averages over every (batch, time) position, so padding
    shifts both the training normalisation and the running statistics used at
    evaluation. With a (B, T) validity mask, mean / variance (and the running
    update) use valid frames only, and padded outputs are zeroed. Same
    parameters and buffers as nn.BatchNorm1d, so checkpoints are compatible.
    Statistics are computed in fp32 under autocast.
    """

    def forward(self, x, mask=None):
        if mask is None:
            return super().forward(x)
        m = mask[:, None, :].to(torch.float32)  # (B, 1, T)
        xf = x.float()
        if self.training:
            n = m.sum().clamp(min=2.0)
            mean = (xf * m).sum((0, 2)) / n
            var = ((xf - mean[None, :, None]).pow(2) * m).sum((0, 2)) / n
            with torch.no_grad():
                self.num_batches_tracked += 1
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(
                    self.momentum * var * n / (n - 1)
                )
        else:
            mean, var = self.running_mean, self.running_var
        y = (xf - mean[None, :, None]) * torch.rsqrt(var[None, :, None] + self.eps)
        if self.affine:
            y = y * self.weight[None, :, None] + self.bias[None, :, None]
        return (y * m).to(x.dtype)


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
        self.batch_norm = MaskedBatchNorm1d(d_model)
        self.swish = Swish()
        # Pointwise Conv 2
        self.pointwise_conv2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (B, T, D); mask: (B, T) True = valid
        x = self.layer_norm(x)
        x = self.pointwise_conv1(x)
        x = self.glu(x)  # (B, T, D)
        if mask is not None:
            # LayerNorm's bias and pointwise_conv1's bias make zeroed padding
            # nonzero again; re-zero it right before the temporal conv so it
            # cannot leak into valid boundary frames.
            x = x * mask.unsqueeze(-1)

        # Prepare for Depthwise Conv1d
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x, mask)
        x = self.swish(x)
        x = x.transpose(1, 2)  # (B, T, D)

        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x


class ConformerBlock(nn.Module):
    """Combines Conv-style local modeling with Transformer global modeling."""

    def __init__(
        self, d_model, n_heads, kernel_size=31, dropout=0.1, drop_path_rate=0.0
    ):
        super().__init__()
        self.drop_path_rate = drop_path_rate
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
        # Stochastic depth: skip the entire block with probability drop_path_rate.
        # Earlier blocks have a lower rate (passed by the caller); later blocks higher.
        if self.training and self.drop_path_rate > 0.0:
            if torch.rand(1).item() < self.drop_path_rate:
                return x

        # 1. Feed Forward 1
        x = x + 0.5 * self.ff1(x)

        # 2. Multi-head Self Attention
        residual = x
        x = self.attn_norm(x)
        key_padding_mask = ~mask if mask is not None else None
        # need_weights=False enables F.scaled_dot_product_attention (SDPA) dispatch;
        # sdpa_kernel selects Flash → MemEfficient → Math in priority order.
        with sdpa_kernel(_SDPA_BACKENDS):
            x, _ = self.attn(
                x, x, x, key_padding_mask=key_padding_mask, need_weights=False
            )
        x = x + residual

        # 3. Convolution Module
        # Zero padded positions BEFORE conv so they don't bleed into valid
        # boundary frames through the depthwise kernel, then zero again AFTER
        # so padded slots stay clean in the residual stream.
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        conv_out = self.conv(x, mask)
        if mask is not None:
            conv_out = conv_out * mask.unsqueeze(-1)
        x = x + conv_out

        # 4. Feed Forward 2
        x = x + 0.5 * self.ff2(x)

        x = self.final_norm(x)
        # Re-zero padded positions after the full block so accumulated padding
        # signal doesn't propagate into the next conformer layer's conv.
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return x


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
        T = x.size(1)
        if T <= self.pe.size(0):
            pe = self.pe[:T]
        else:
            # Generate extended PE on the fly — avoids crash on long sequences
            # without mutating the buffer (thread-safe).
            d = self.pe.size(1)
            pos = torch.arange(T, device=x.device).unsqueeze(1).float()
            div = torch.exp(
                torch.arange(0, d, 2, device=x.device).float() * (-np.log(10000.0) / d)
            )
            pe = torch.zeros(T, d, device=x.device)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div[: d // 2])
        return self.dropout(x + pe)
