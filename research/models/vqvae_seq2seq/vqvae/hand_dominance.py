"""Hand dominance detection and reordering module."""

import torch
import torch.nn as nn
from typing import Tuple


class HandDominanceModule(nn.Module):
    """
    Detects hand dominance and reorders hands consistently.

    Addresses blindspot #5: No hand dominance normalization.

    The dominant hand is determined by motion energy - the hand
    with more movement is considered dominant. This ensures
    consistent representation regardless of whether the signer
    is left or right-handed.
    """

    def __init__(
        self,
        hand_landmarks: int = 21,
        n_coords: int = 3,
        motion_smoothing: int = 3,
    ):
        """
        Args:
            hand_landmarks: Number of landmarks per hand
            n_coords: Number of coordinates per landmark
            motion_smoothing: Window size for motion smoothing
        """
        super().__init__()
        self.hand_landmarks = hand_landmarks
        self.n_coords = n_coords
        self.motion_smoothing = motion_smoothing

    def compute_motion_energy(self, hand: torch.Tensor) -> torch.Tensor:
        """
        Compute motion energy for a hand sequence.

        Args:
            hand: (B, T, N_hand, C) tensor

        Returns:
            (B,) tensor of total motion energy per sequence
        """
        # Compute frame-to-frame velocity
        velocity = hand[:, 1:] - hand[:, :-1]  # (B, T-1, N_hand, C)

        # Compute per-frame motion magnitude
        motion_magnitude = torch.norm(velocity, dim=-1)  # (B, T-1, N_hand)

        # Sum over landmarks and time to get total motion energy
        total_motion = motion_magnitude.sum(dim=(1, 2))  # (B,)

        return total_motion

    def compute_activity_ratio(
        self, hand: torch.Tensor, threshold: float = 0.01
    ) -> torch.Tensor:
        """
        Compute the ratio of frames where the hand is active (moving).

        Args:
            hand: (B, T, N_hand, C) tensor
            threshold: Motion threshold for considering a frame active

        Returns:
            (B,) tensor of activity ratios
        """
        velocity = hand[:, 1:] - hand[:, :-1]
        motion_magnitude = torch.norm(velocity, dim=-1).mean(dim=-1)  # (B, T-1)

        active_frames = (motion_magnitude > threshold).float()
        activity_ratio = active_frames.mean(dim=1)

        return activity_ratio

    def detect_dominant_hand(
        self, left_hand: torch.Tensor, right_hand: torch.Tensor
    ) -> torch.Tensor:
        """
        Detect which hand is dominant based on motion.

        Args:
            left_hand: (B, T, N_hand, C) tensor
            right_hand: (B, T, N_hand, C) tensor

        Returns:
            (B,) tensor with 0 for left dominant, 1 for right dominant
        """
        left_energy = self.compute_motion_energy(left_hand)
        right_energy = self.compute_motion_energy(right_hand)

        # Right is dominant if it has more motion
        dominant = (right_energy > left_energy).long()

        return dominant

    def forward(
        self,
        left_hand: torch.Tensor,
        right_hand: torch.Tensor,
        return_swap_mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reorder hands so dominant hand is always first.

        Args:
            left_hand: (B, T, N_hand, C) tensor
            right_hand: (B, T, N_hand, C) tensor
            return_swap_mask: If True, also return which samples were swapped

        Returns:
            Tuple of (dominant_hand, non_dominant_hand)
            Optionally: swap_mask indicating which samples had left as dominant
        """
        dominant_mask = self.detect_dominant_hand(left_hand, right_hand)

        # Create expanded mask for broadcasting
        B, T, N, C = left_hand.shape
        mask_expanded = dominant_mask.view(B, 1, 1, 1).expand_as(left_hand)

        # Select dominant and non-dominant based on mask
        # mask=0: left is dominant, mask=1: right is dominant
        dominant_hand = torch.where(mask_expanded == 0, left_hand, right_hand)
        non_dominant_hand = torch.where(mask_expanded == 0, right_hand, left_hand)

        if return_swap_mask:
            # swap_mask is True where left was dominant (i.e., we kept original order)
            swap_mask = dominant_mask == 0
            return dominant_hand, non_dominant_hand, swap_mask

        return dominant_hand, non_dominant_hand


class HandMirrorAugmentation(nn.Module):
    """
    Data augmentation that mirrors hand positions (swaps left/right).

    This helps the model learn to be invariant to handedness.
    """

    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of applying mirroring
        """
        super().__init__()
        self.p = p

    def forward(
        self, left_hand: torch.Tensor, right_hand: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly swap and mirror hands.

        Args:
            left_hand: (B, T, N, C) tensor
            right_hand: (B, T, N, C) tensor
            training: Whether in training mode

        Returns:
            Tuple of (possibly swapped left_hand, possibly swapped right_hand)
        """
        if not training:
            return left_hand, right_hand

        B = left_hand.shape[0]
        device = left_hand.device

        # Generate random mask for which samples to mirror
        mirror_mask = torch.rand(B, device=device) < self.p

        # Expand mask for broadcasting
        mask_expanded = mirror_mask.view(B, 1, 1, 1).expand_as(left_hand)

        # Swap hands where mask is True
        new_left = torch.where(mask_expanded, right_hand, left_hand)
        new_right = torch.where(mask_expanded, left_hand, right_hand)

        # Mirror x-coordinates (flip horizontally)
        # Assuming x is the first coordinate
        new_left_mirrored = new_left.clone()
        new_right_mirrored = new_right.clone()

        # Only mirror where we swapped
        new_left_mirrored[:, :, :, 0] = torch.where(
            mirror_mask.view(B, 1, 1), -new_left[:, :, :, 0], new_left[:, :, :, 0]
        )
        new_right_mirrored[:, :, :, 0] = torch.where(
            mirror_mask.view(B, 1, 1), -new_right[:, :, :, 0], new_right[:, :, :, 0]
        )

        return new_left_mirrored, new_right_mirrored


class TwoHandFusion(nn.Module):
    """
    Fuses information from both hands for signs that use both hands together.

    Some ASL signs require coordinated two-hand movements, so we also
    learn a joint representation.
    """

    def __init__(
        self,
        hand_dim: int = 63,  # 21 * 3
        hidden_dim: int = 128,
        output_dim: int = 64,
    ):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(hand_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Relative position encoder
        self.relative_encoder = nn.Sequential(
            nn.Linear(hand_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(
        self, dominant_hand: torch.Tensor, non_dominant_hand: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute fused two-hand representation.

        Args:
            dominant_hand: (B, T, N_hand, C) tensor
            non_dominant_hand: (B, T, N_hand, C) tensor

        Returns:
            (B, T, output_dim) tensor of fused features
        """
        B, T, N, C = dominant_hand.shape

        # Flatten hands
        dom_flat = dominant_hand.reshape(B, T, -1)  # (B, T, N*C)
        nondom_flat = non_dominant_hand.reshape(B, T, -1)

        # Concatenate and fuse
        concat = torch.cat([dom_flat, nondom_flat], dim=-1)
        fused = self.fusion(concat)

        # Add relative position encoding
        relative = dom_flat - nondom_flat
        relative_features = self.relative_encoder(relative)

        return fused + relative_features
