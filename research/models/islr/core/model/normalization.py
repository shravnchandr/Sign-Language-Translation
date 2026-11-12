import torch
import torch.nn as nn
from ..config import COORD_FEAT, COORDS_PER_LM, LH_START, RH_START


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
            wrist = x[:, :, hs : hs + c]  # (B, T, c)
            fingers = x[:, :, hs + c : hs + 21 * c].reshape(B, T, 20, c)
            x[:, :, hs + c : hs + 21 * c] = (fingers - wrist.unsqueeze(2)).reshape(
                B, T, 20 * c
            )
            # Velocity half: same subtraction offset by COORD_FEAT
            vs = COORD_FEAT + hs
            wrist_v = x[:, :, vs : vs + c]
            fingers_v = x[:, :, vs + c : vs + 21 * c].reshape(B, T, 20, c)
            x[:, :, vs + c : vs + 21 * c] = (fingers_v - wrist_v.unsqueeze(2)).reshape(
                B, T, 20 * c
            )
        return x
