import torch
import torch.nn as nn
from ..config import COORD_FEAT, COORDS_PER_LM, LH_START, RH_START


class RobustNormalization(nn.Module):
    """Fallback-based normalization: Nose -> Shoulder Center -> Hip Center."""

    def __init__(self, pose_start_idx: int, n_coords: int = 2):
        super().__init__()
        self.pose_start = pose_start_idx
        self.n_coords = n_coords
        self.L_SHOULDER, self.R_SHOULDER = 11, 12

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        p, c = self.pose_start, self.n_coords
        nose = x[:, :, p : p + c]
        ls = x[:, :, p + self.L_SHOULDER * c : p + (self.L_SHOULDER + 1) * c]
        rs = x[:, :, p + self.R_SHOULDER * c : p + (self.R_SHOULDER + 1) * c]
        shoulder_center = (ls + rs) / 2.0

        nose_valid = nose.abs().sum(dim=-1, keepdim=True) > 1e-6
        shoulder_valid = shoulder_center.abs().sum(dim=-1, keepdim=True) > 1e-6
        origin = torch.where(
            nose_valid,
            nose,
            torch.where(shoulder_valid, shoulder_center, torch.zeros_like(nose)),
        )

        x_pos = x[:, :, :COORD_FEAT].reshape(B, T, -1, c) - origin.unsqueeze(2)
        x[:, :, :COORD_FEAT] = x_pos.reshape(B, T, -1)
        return x


class WristNormalization(nn.Module):
    """
    Dual-stream hand normalization applied after body-relative normalization.

    For each hand:
      - Landmark 0 (wrist): kept nose-relative  → LOCATION stream
        (distinguishes "Father" at forehead from "Mother" at chin)
      - Landmarks 1–20 (fingers): wrist subtracted → SHAPE stream
        (pure hand configuration, invariant to arm position)

    Operates in-place on both the position and velocity halves.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        c = COORDS_PER_LM
        for hs in (LH_START, RH_START):
            # Position half: subtract wrist (lm 0) from fingers (lm 1-20)
            wrist = x[:, :, hs : hs + c]                              # (B, T, c)
            fingers = x[:, :, hs + c : hs + 21 * c].reshape(B, T, 20, c)
            x[:, :, hs + c : hs + 21 * c] = (
                (fingers - wrist.unsqueeze(2)).reshape(B, T, 20 * c)
            )
            # Velocity half: same subtraction offset by COORD_FEAT
            vs = COORD_FEAT + hs
            wrist_v = x[:, :, vs : vs + c]
            fingers_v = x[:, :, vs + c : vs + 21 * c].reshape(B, T, 20, c)
            x[:, :, vs + c : vs + 21 * c] = (
                (fingers_v - wrist_v.unsqueeze(2)).reshape(B, T, 20 * c)
            )
        return x
