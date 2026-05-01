"""Temporal augmentation for VQ-VAE training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class TemporalAugmentation(nn.Module):
    """
    Temporal augmentation module for landmark sequences.

    Addresses blindspot #9: No temporal augmentation.

    Includes:
    - Speed variation (time stretching/compression)
    - Frame dropout
    - Temporal jitter
    - Gaussian noise
    """

    def __init__(
        self,
        speed_range: Tuple[float, float] = (0.8, 1.2),
        frame_dropout_prob: float = 0.1,
        temporal_jitter_std: float = 0.02,
        noise_std: float = 0.01,
        spatial_noise_std: float = 0.005,
    ):
        """
        Args:
            speed_range: (min, max) speed multiplier range
            frame_dropout_prob: Probability of dropping a frame
            temporal_jitter_std: Standard deviation for temporal jitter
            noise_std: Standard deviation for coordinate noise
            spatial_noise_std: Standard deviation for per-landmark spatial noise
        """
        super().__init__()
        self.speed_range = speed_range
        self.frame_dropout_prob = frame_dropout_prob
        self.temporal_jitter_std = temporal_jitter_std
        self.noise_std = noise_std
        self.spatial_noise_std = spatial_noise_std

    def speed_augment(
        self, x: torch.Tensor, speed: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply speed variation by resampling the temporal dimension.

        Args:
            x: (B, T, N, C) tensor
            speed: Speed multiplier (if None, sample randomly)

        Returns:
            Resampled tensor
        """
        if speed is None:
            speed = np.random.uniform(*self.speed_range)

        B, T, N, C = x.shape
        new_T = int(T / speed)

        if new_T < 2:
            return x

        # Reshape for interpolation: (B, C*N, T)
        x_reshape = x.permute(0, 2, 3, 1).reshape(B, N * C, T)

        # Resample
        x_resampled = F.interpolate(
            x_reshape.float(), size=new_T, mode="linear", align_corners=True
        )

        # Reshape back: (B, new_T, N, C)
        x_out = x_resampled.reshape(B, N, C, new_T).permute(0, 3, 1, 2)

        return x_out

    def frame_dropout(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Randomly drop frames from the sequence.

        Args:
            x: (B, T, N, C) tensor
            mask: Optional (B, T) validity mask

        Returns:
            Tuple of (augmented tensor, updated mask)
        """
        B, T, N, C = x.shape
        device = x.device

        if T < 4:  # Don't drop if sequence is too short
            return x, mask

        # Generate drop mask
        keep_mask = torch.rand(B, T, device=device) > self.frame_dropout_prob

        # Always keep first and last frames
        keep_mask[:, 0] = True
        keep_mask[:, -1] = True

        # Ensure at least 50% of frames are kept
        min_frames = max(T // 2, 2)
        for b in range(B):
            if keep_mask[b].sum() < min_frames:
                # Randomly select more frames to keep
                drop_indices = (~keep_mask[b]).nonzero().squeeze(-1)
                n_to_add = min_frames - keep_mask[b].sum().item()
                if len(drop_indices) > 0:
                    add_indices = drop_indices[
                        torch.randperm(len(drop_indices))[: int(n_to_add)]
                    ]
                    keep_mask[b, add_indices] = True

        # Apply mask and interpolate to fill gaps
        # For simplicity, we use the mask to zero out dropped frames
        # and then interpolate
        x_masked = x * keep_mask.unsqueeze(-1).unsqueeze(-1)

        # Interpolate to restore original length
        # This creates smooth transitions where frames were dropped
        x_interp = self._interpolate_dropped(x_masked, keep_mask)

        # Update mask if provided
        if mask is not None:
            mask = mask & keep_mask

        return x_interp, mask

    def _interpolate_dropped(
        self, x: torch.Tensor, keep_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolate values for dropped frames using vectorized GPU ops.
        Replaces the original triple-nested Python loop + numpy round-trips.
        """
        B, T, N, C = x.shape
        # (B, N*C, T) — layout F.interpolate expects
        result = x.reshape(B, T, N * C).permute(0, 2, 1)

        for b in range(B):
            mask_b = keep_mask[b]
            if mask_b.all():
                continue
            kept_idx = mask_b.nonzero(as_tuple=True)[0]
            if kept_idx.numel() < 2:
                continue
            # Interpolate kept frames to full length entirely on GPU
            kept = result[b : b + 1, :, kept_idx]  # (1, N*C, n_kept)
            full = F.interpolate(kept, size=T, mode="linear", align_corners=True)
            result[b : b + 1, :, ~mask_b] = full[:, :, ~mask_b]

        return result.permute(0, 2, 1).reshape(B, T, N, C)

    def temporal_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add temporal jitter by slightly shifting coordinate values over time.

        This simulates small timing variations in sign production.

        Args:
            x: (B, T, N, C) tensor

        Returns:
            Jittered tensor
        """
        B, T, N, C = x.shape

        # Generate jitter offsets that vary smoothly over time
        # We use a low-frequency random signal
        jitter_freq = max(1, T // 8)

        # Generate base jitter at low frequency
        base_jitter = (
            torch.randn(B, jitter_freq, device=x.device) * self.temporal_jitter_std
        )

        # Interpolate to full length
        base_jitter = base_jitter.unsqueeze(1)  # (B, 1, jitter_freq)
        jitter = F.interpolate(base_jitter, size=T, mode="linear", align_corners=True)
        jitter = jitter.squeeze(1)  # (B, T)

        # Apply jitter as a temporal shift (approximate by adding to velocities)
        velocity = x[:, 1:] - x[:, :-1]
        jitter_scale = jitter[:, 1:].unsqueeze(-1).unsqueeze(-1)

        x_jittered = x.clone()
        x_jittered[:, 1:] = x[:, 1:] + velocity * jitter_scale

        return x_jittered

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to coordinates.

        Args:
            x: (B, T, N, C) tensor

        Returns:
            Noisy tensor
        """
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

    def add_spatial_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add per-landmark spatial noise that's consistent across time.

        This simulates slight differences in landmark detection.

        Args:
            x: (B, T, N, C) tensor

        Returns:
            Noisy tensor
        """
        B, T, N, C = x.shape

        # Generate noise that's the same for all time steps
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
        """
        Apply all augmentations.

        Args:
            x: (B, T, N, C) tensor
            mask: Optional (B, T) validity mask
            training: Whether in training mode

        Returns:
            Tuple of (augmented tensor, updated mask)
        """
        if not training:
            return x, mask

        # Apply augmentations with some randomness
        if torch.rand(1).item() < 0.5:
            x = self.speed_augment(x)
            # Update mask length if speed changed
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


def augment_data_dict(
    data: Dict[str, torch.Tensor],
    augmenter: TemporalAugmentation,
    training: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Apply augmentation to a dictionary of landmark data.

    WARNING: Each body part is augmented independently, so random decisions
    (speed, frame dropout) are sampled separately per key. This can de-sync
    body parts (e.g. one hand gets a speed boost while the other doesn't).
    Prefer augmenting the full concatenated landmarks tensor directly, as
    ImprovedVQVAE.forward() does. Only use this function if all body parts
    are guaranteed to receive the same augmentation parameters.

    Args:
        data: Dictionary with 'left_hand', 'right_hand', 'pose', 'face' tensors
        augmenter: TemporalAugmentation instance
        training: Whether in training mode

    Returns:
        Augmented data dictionary
    """
    if not training:
        return data

    result = {}

    # Get mask if present
    mask = data.get("mask")

    for key in ["left_hand", "right_hand", "pose", "face"]:
        if key in data:
            aug_data, new_mask = augmenter(data[key], mask, training)
            result[key] = aug_data

    # Update mask
    if mask is not None and new_mask is not None:
        result["mask"] = new_mask

    # Copy other keys
    for key in data:
        if key not in result:
            result[key] = data[key]

    return result
