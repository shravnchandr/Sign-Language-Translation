"""Multi-scale temporal encoder for handling variable motion speeds."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


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

        # Skip connection
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, 1)
        )

        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) tensor

        Returns:
            (B, C_out, T) tensor
        """
        residual = self.skip(x)

        # Handle stride in skip connection if needed
        if self.conv[0].stride[0] > 1:
            residual = F.avg_pool1d(residual, self.conv[0].stride[0])

        out = self.conv(x)

        # Handle size mismatch
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
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output latent dimension
            chunk_size: Temporal chunk size for this scale
            n_layers: Number of conv blocks
            dropout: Dropout rate
        """
        super().__init__()
        self.chunk_size = chunk_size

        # Build encoder layers
        layers = []
        in_dim = input_dim
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else output_dim
            layers.append(
                TemporalConvBlock(in_dim, out_dim, kernel_size=3, dropout=dropout)
            )
            in_dim = out_dim

        self.encoder = nn.Sequential(*layers)

        # Pooling to get per-chunk representations
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode input at this scale.

        Args:
            x: (B, T, D) input tensor
            mask: Optional (B, T) validity mask

        Returns:
            (B, n_chunks, output_dim) encoded chunks
        """
        B, T, D = x.shape

        # Pad to multiple of chunk_size
        pad_len = (self.chunk_size - T % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            if mask is not None:
                mask = F.pad(mask.float(), (0, pad_len)).bool()

        T_padded = x.shape[1]
        n_chunks = T_padded // self.chunk_size

        # Reshape into chunks: (B, n_chunks, chunk_size, D)
        x_chunks = x.reshape(B, n_chunks, self.chunk_size, D)

        # Process each chunk
        # Reshape to (B * n_chunks, D, chunk_size) for conv
        x_conv = x_chunks.reshape(B * n_chunks, self.chunk_size, D).permute(0, 2, 1)

        # Encode
        encoded = self.encoder(x_conv)  # (B * n_chunks, output_dim, chunk_size')

        # Pool to get single vector per chunk
        pooled = self.pool(encoded).squeeze(-1)  # (B * n_chunks, output_dim)

        # Reshape back
        output = pooled.reshape(B, n_chunks, -1)

        return output


class MultiScaleEncoder(nn.Module):
    """
    Multi-scale temporal encoder that processes at multiple chunk sizes.

    Addresses blindspot #3: Fixed chunk size.

    Captures both fine-grained (small chunks) and coarse (large chunks)
    temporal patterns, then fuses them.
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
        """
        Args:
            input_dim: Input feature dimension (e.g., 21*3 for a hand)
            hidden_dim: Hidden layer dimension
            output_dim: Output latent dimension per scale
            scales: Tuple of chunk sizes
            n_layers: Number of conv blocks per scale
            dropout: Dropout rate
        """
        super().__init__()
        self.scales = scales
        self.output_dim = output_dim

        # Create encoder for each scale
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

        # Fusion layer to combine multi-scale features
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * len(scales), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Cross-scale attention for better fusion
        self.cross_scale_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

    def _align_scales(
        self, scale_outputs: List[torch.Tensor], target_len: int
    ) -> torch.Tensor:
        """
        Align outputs from different scales to the same temporal length.

        Args:
            scale_outputs: List of (B, n_chunks_i, D) tensors
            target_len: Target number of chunks

        Returns:
            (B, target_len, D * n_scales) aligned and concatenated features
        """
        aligned = []
        for output in scale_outputs:
            # Interpolate to target length
            if output.shape[1] != target_len:
                # (B, D, n_chunks) -> interpolate -> (B, D, target_len) -> (B, target_len, D)
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
        """
        Encode input at multiple scales and fuse.

        Args:
            x: (B, T, D) input tensor
            mask: Optional (B, T) validity mask
            return_all_scales: If True, also return individual scale outputs

        Returns:
            (B, n_chunks, output_dim) fused multi-scale encoding
            Optionally: list of per-scale outputs
        """
        B, T, D = x.shape

        # Encode at each scale
        scale_outputs = []
        for encoder in self.scale_encoders:
            encoded = encoder(x, mask)
            scale_outputs.append(encoded)

        # Use the middle scale as reference for chunk count
        ref_idx = len(self.scales) // 2
        target_len = scale_outputs[ref_idx].shape[1]

        # Align and concatenate scales
        aligned = self._align_scales(
            scale_outputs, target_len
        )  # (B, target_len, D * n_scales)

        # Apply fusion MLP
        fused = self.fusion(aligned)  # (B, target_len, output_dim)

        # Apply cross-scale attention for refinement
        fused_attn, _ = self.cross_scale_attn(fused, fused, fused)
        fused = fused + fused_attn

        if return_all_scales:
            return fused, scale_outputs

        return fused


class MotionEncoder(nn.Module):
    """
    Encoder that explicitly models velocity and acceleration.

    Extracts motion features (velocity) and dynamics features (acceleration)
    in addition to pose features.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Pose encoder (position features)
        self.pose_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Motion encoder (velocity features)
        self.motion_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Dynamics encoder (acceleration features)
        self.dynamics_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode position, velocity, and acceleration.

        Args:
            x: (B, T, D) position tensor

        Returns:
            pose_features: (B, T, output_dim)
            motion_features: (B, T-1, output_dim)
            dynamics_features: (B, T-2, output_dim)
        """
        B, T, D = x.shape

        # Encode pose (position)
        pose_features = self.pose_encoder(x)

        # Compute and encode velocity
        velocity = x[:, 1:] - x[:, :-1]  # (B, T-1, D)
        motion_features = self.motion_encoder(velocity)

        # Compute and encode acceleration
        if T >= 3:
            acceleration = velocity[:, 1:] - velocity[:, :-1]  # (B, T-2, D)
            dynamics_features = self.dynamics_encoder(acceleration)
        else:
            dynamics_features = torch.zeros(
                B, 0, self.dynamics_encoder[-1].out_features, device=x.device
            )

        return pose_features, motion_features, dynamics_features


class MultiScaleMotionEncoder(nn.Module):
    """
    Combines multi-scale encoding with motion/dynamics extraction.

    This is the main encoder used in the VQ-VAE for body parts (hands, pose).
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

        # Multi-scale encoder for pose
        self.pose_encoder = MultiScaleEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            scales=scales,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Motion encoder (on velocity)
        self.motion_encoder = MultiScaleEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            scales=scales,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Dynamics encoder (on acceleration)
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
        """
        Encode at multiple scales with motion decomposition.

        Args:
            x: (B, T, D) position tensor
            mask: Optional validity mask

        Returns:
            pose_latent: (B, n_chunks, output_dim)
            motion_latent: (B, n_chunks, output_dim)
            dynamics_latent: (B, n_chunks, output_dim)
        """
        # Compute velocity and acceleration
        velocity = x[:, 1:] - x[:, :-1]  # (B, T-1, D)
        acceleration = velocity[:, 1:] - velocity[:, :-1]  # (B, T-2, D)

        # Pad to match original length
        velocity = F.pad(velocity, (0, 0, 0, 1))  # (B, T, D)
        acceleration = F.pad(acceleration, (0, 0, 0, 2))  # (B, T, D)

        # Adjust mask if needed
        vel_mask = mask
        acc_mask = mask

        # Encode each factor
        pose_latent = self.pose_encoder(x, mask)
        motion_latent = self.motion_encoder(velocity, vel_mask)
        dynamics_latent = self.dynamics_encoder(acceleration, acc_mask)

        return pose_latent, motion_latent, dynamics_latent
