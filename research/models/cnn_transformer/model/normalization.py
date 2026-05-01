import torch
import torch.nn as nn
from ..config import COORD_FEAT


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

        x_pos = x[:, :, :COORD_FEAT].reshape(B, T, -1, c)
        x_pos = x_pos - origin.unsqueeze(2)
        x[:, :, :COORD_FEAT] = x_pos.reshape(B, T, -1)
        return x
