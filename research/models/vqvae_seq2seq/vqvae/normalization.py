"""Robust normalization with fallback chain for landmark data."""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class RobustNormalization(nn.Module):
    """
    Robust normalization with fallback chain: nose -> shoulder center -> hip center.

    Addresses blindspot #7: Fragile nose normalization.

    Expects input landmarks in the format where pose landmarks are
    at a known position in the tensor.
    """

    # Pose landmark indices (MediaPipe)
    NOSE_IDX = 0
    LEFT_SHOULDER_IDX = 11
    RIGHT_SHOULDER_IDX = 12
    LEFT_HIP_IDX = 23
    RIGHT_HIP_IDX = 24

    def __init__(
        self,
        pose_start_idx: int = 42,  # After left_hand (21) + right_hand (21)
        n_coords: int = 3,
        missing_threshold: float = 0.0,
    ):
        """
        Args:
            pose_start_idx: Starting index of pose landmarks in the landmark tensor
            n_coords: Number of coordinates per landmark (2 or 3)
            missing_threshold: Value below which a landmark is considered missing
        """
        super().__init__()
        self.pose_start_idx = pose_start_idx
        self.n_coords = n_coords
        self.missing_threshold = missing_threshold

    def _get_pose_landmark(
        self, landmarks: torch.Tensor, landmark_idx: int
    ) -> torch.Tensor:
        """
        Get a specific pose landmark.

        Args:
            landmarks: (B, T, N, C) tensor
            landmark_idx: Index within pose landmarks

        Returns:
            (B, T, C) tensor for the specific landmark
        """
        full_idx = self.pose_start_idx + landmark_idx
        return landmarks[:, :, full_idx, :]

    def _is_valid_landmark(self, landmark: torch.Tensor) -> torch.Tensor:
        """
        Check if a landmark is valid (not missing/zero).

        Args:
            landmark: (B, T, C) tensor

        Returns:
            (B, T) boolean mask
        """
        # Consider a landmark invalid if all coordinates are below threshold
        abs_vals = landmark.abs()
        return (abs_vals > self.missing_threshold).any(dim=-1)

    def _get_center(
        self, landmarks: torch.Tensor, idx1: int, idx2: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get center point between two landmarks.

        Returns:
            Tuple of (center coords, validity mask)
        """
        lm1 = self._get_pose_landmark(landmarks, idx1)
        lm2 = self._get_pose_landmark(landmarks, idx2)

        valid1 = self._is_valid_landmark(lm1)
        valid2 = self._is_valid_landmark(lm2)

        center = (lm1 + lm2) / 2
        valid = valid1 & valid2

        return center, valid

    def forward(
        self, landmarks: torch.Tensor, return_origins: bool = False
    ) -> torch.Tensor:
        """
        Apply robust normalization.

        Args:
            landmarks: (B, T, N, C) tensor of landmarks
            return_origins: If True, also return the origin points used

        Returns:
            Normalized landmarks (B, T, N, C)
            Optionally: origins (B, T, C) and origin_types (B, T) tensor
        """
        B, T, N, C = landmarks.shape

        # Get candidate origins
        nose = self._get_pose_landmark(landmarks, self.NOSE_IDX)
        nose_valid = self._is_valid_landmark(nose)

        shoulder_center, shoulder_valid = self._get_center(
            landmarks, self.LEFT_SHOULDER_IDX, self.RIGHT_SHOULDER_IDX
        )

        hip_center, hip_valid = self._get_center(
            landmarks, self.LEFT_HIP_IDX, self.RIGHT_HIP_IDX
        )

        # Initialize origin with zeros (fallback)
        origin = torch.zeros(B, T, C, device=landmarks.device, dtype=landmarks.dtype)
        origin_type = torch.zeros(B, T, device=landmarks.device, dtype=torch.long)

        # Apply fallback chain (reverse priority so nose gets priority)
        # 3: hip center
        origin = torch.where(hip_valid.unsqueeze(-1), hip_center, origin)
        origin_type = torch.where(
            hip_valid, torch.full_like(origin_type, 3), origin_type
        )

        # 2: shoulder center
        origin = torch.where(shoulder_valid.unsqueeze(-1), shoulder_center, origin)
        origin_type = torch.where(
            shoulder_valid, torch.full_like(origin_type, 2), origin_type
        )

        # 1: nose (highest priority)
        origin = torch.where(nose_valid.unsqueeze(-1), nose, origin)
        origin_type = torch.where(
            nose_valid, torch.full_like(origin_type, 1), origin_type
        )

        # Subtract origin from all landmarks
        normalized = landmarks - origin.unsqueeze(2)

        if return_origins:
            return normalized, origin, origin_type

        return normalized


class PerFrameNormalization(nn.Module):
    """
    Alternative normalization that normalizes each frame independently
    to a unit bounding box.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Normalize landmarks to [-1, 1] range per frame.

        Args:
            landmarks: (B, T, N, C) tensor

        Returns:
            Normalized landmarks
        """
        # Compute per-frame min and max
        mins = landmarks.min(dim=2, keepdim=True)[0]  # (B, T, 1, C)
        maxs = landmarks.max(dim=2, keepdim=True)[0]  # (B, T, 1, C)

        # Compute range and center
        ranges = maxs - mins + self.eps
        centers = (mins + maxs) / 2

        # Normalize to [-1, 1]
        normalized = (landmarks - centers) / (ranges / 2)

        return normalized


class ScaleNormalization(nn.Module):
    """
    Scale normalization based on body proportions (e.g., shoulder width).

    This makes the model invariant to different body sizes and camera distances.
    """

    def __init__(
        self,
        pose_start_idx: int = 42,
        target_shoulder_width: float = 0.4,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.pose_start_idx = pose_start_idx
        self.target_shoulder_width = target_shoulder_width
        self.eps = eps

        self.LEFT_SHOULDER_IDX = 11
        self.RIGHT_SHOULDER_IDX = 12

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Scale landmarks based on shoulder width.

        Args:
            landmarks: (B, T, N, C) tensor, assumed to already be origin-normalized

        Returns:
            Scale-normalized landmarks
        """
        # Get shoulder landmarks
        left_shoulder = landmarks[:, :, self.pose_start_idx + self.LEFT_SHOULDER_IDX, :]
        right_shoulder = landmarks[
            :, :, self.pose_start_idx + self.RIGHT_SHOULDER_IDX, :
        ]

        # Compute shoulder width (use only x,y for 2D distance)
        shoulder_diff = left_shoulder[:, :, :2] - right_shoulder[:, :, :2]
        shoulder_width = torch.norm(shoulder_diff, dim=-1, keepdim=True)  # (B, T, 1)

        # Compute per-sequence average shoulder width
        avg_width = shoulder_width.mean(dim=1, keepdim=True)  # (B, 1, 1)

        # Compute scale factor
        scale = self.target_shoulder_width / (avg_width + self.eps)

        # Apply scale to all landmarks
        scaled = landmarks * scale.unsqueeze(-1)

        return scaled
