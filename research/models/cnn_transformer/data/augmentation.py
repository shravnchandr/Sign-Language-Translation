import numpy as np
import torch
import torch.nn.functional as F
from ..config import (
    COORDS_PER_LM,
    COORD_FEAT,
    LH_START,
    RH_START,
    N_LH,
    FINGER_LM_RANGES,
    FINGER_COORD_SLICES,
)


def augment_sample(
    video_coordinates: np.ndarray, noise_std: float = 3e-3, spatial_shift: float = 2e-2
) -> np.ndarray:
    video_coordinates = video_coordinates.copy()
    if np.random.random() > 0.5:
        video_coordinates += np.random.normal(0, noise_std, video_coordinates.shape)
    if np.random.random() > 0.5:
        video_coordinates += np.random.uniform(
            -spatial_shift, spatial_shift, (1, video_coordinates.shape[1])
        )
    return video_coordinates


class AdvancedAugmentation:
    """Advanced augmentation strategies for landmarks."""

    @staticmethod
    def temporal_cropping(x, mask, min_ratio=0.7, max_ratio=0.95):
        B, T, D = x.shape
        crop_len = np.random.randint(int(T * min_ratio), int(T * max_ratio))
        start = np.random.randint(0, T - crop_len + 1)
        x_cropped = x[:, start : start + crop_len, :]
        mask_cropped = mask[:, start : start + crop_len]
        if crop_len < T:
            pad_len = T - crop_len
            x_padded = F.pad(x_cropped, (0, 0, 0, pad_len), value=0)
            mask_padded = F.pad(mask_cropped, (0, pad_len), value=False)
            return x_padded, mask_padded
        return x_cropped, mask_cropped

    @staticmethod
    def random_flip(x, probability=0.3):
        """Negate all x-coords and swap left/right hand blocks for a mirror image."""
        if np.random.random() >= probability:
            return x
        x_flipped = x.clone()
        # One pass with step=COORDS_PER_LM covers all x-coords across the full
        # [position | velocity] tensor; a second pass would double-negate velocity.
        x_flipped[..., ::COORDS_PER_LM] = -x_flipped[..., ::COORDS_PER_LM]
        lh_size = N_LH * COORDS_PER_LM
        for half in (0, COORD_FEAT):
            lh_s = half + LH_START
            rh_s = half + RH_START
            lh_chunk = x_flipped[..., lh_s : lh_s + lh_size].clone()
            rh_chunk = x_flipped[..., rh_s : rh_s + lh_size].clone()
            x_flipped[..., lh_s : lh_s + lh_size] = rh_chunk
            x_flipped[..., rh_s : rh_s + lh_size] = lh_chunk
        return x_flipped

    @staticmethod
    def gaussian_noise(x, std=0.01):
        return x + torch.randn_like(x) * std

    @staticmethod
    def temporal_interpolation(x, mask):
        """Replace isolated invalid frames with the average of their neighbours."""
        if x.shape[1] < 3:
            return x, mask
        left_valid = mask[:, :-2]
        center_inv = ~mask[:, 1:-1]
        right_valid = mask[:, 2:]
        fill_mask = left_valid & center_inv & right_valid
        fill_mask_feat = fill_mask.unsqueeze(-1).expand_as(x[:, 1:-1])
        interpolated = (x[:, :-2] + x[:, 2:]) / 2.0
        x[:, 1:-1] = torch.where(fill_mask_feat, interpolated, x[:, 1:-1])
        mask[:, 1:-1] = mask[:, 1:-1] | fill_mask
        return x, mask

    @staticmethod
    def time_stretch(x, mask, min_stretch=0.8, max_stretch=1.3):
        B, T, D = x.shape
        new_len = int(T * np.random.uniform(min_stretch, max_stretch))
        if new_len == T:
            return x, mask
        # x.permute(0,2,1) is already (B, D, T) — no reshape needed
        x_stretched = F.interpolate(
            x.permute(0, 2, 1), size=new_len, mode="linear", align_corners=False
        ).permute(
            0, 2, 1
        )  # (B, new_len, D)
        mask_stretched = (
            F.interpolate(
                mask.float().unsqueeze(1),
                size=new_len,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
            > 0.5
        ).bool()
        if new_len < T:
            x_stretched = F.pad(x_stretched, (0, 0, 0, T - new_len))
            mask_stretched = F.pad(mask_stretched, (0, T - new_len))
        else:
            x_stretched = x_stretched[:, :T, :]
            mask_stretched = mask_stretched[:, :T]
        return x_stretched, mask_stretched

    @staticmethod
    def finger_dropout(x, mask=None, dropout_prob=0.3):
        """Randomly zero out entire fingers in both hands."""
        x = x.clone()
        n_fingers = len(FINGER_LM_RANGES)
        for hand_label in ("left", "right"):
            for fi in range(n_fingers):
                if np.random.random() < dropout_prob:
                    for feat_lo, feat_hi in FINGER_COORD_SLICES[(hand_label, fi)]:
                        x[:, :, feat_lo:feat_hi] = 0.0
        return x

    @staticmethod
    def spatial_rotation(x, max_angle=15):
        """Per-sample z-axis rotation via batched 2×2 matmul (no Python coord loop)."""
        B, T, D = x.shape
        angles = torch.tensor(
            np.radians(np.random.uniform(-max_angle, max_angle, B)),
            dtype=x.dtype,
            device=x.device,
        )  # (B,)
        cos_a, sin_a = torch.cos(angles), torch.sin(angles)
        # (B, 2, 2) rotation matrices, broadcast over T and K (landmark pairs)
        rot = torch.stack(
            [
                torch.stack([cos_a, -sin_a], dim=-1),
                torch.stack([sin_a, cos_a], dim=-1),
            ],
            dim=-2,
        )  # (B, 2, 2)
        # Reshape to expose (x,y) pairs: (B, T, K, C) where K = D // COORDS_PER_LM
        x_lm = x.reshape(B, T, D // COORDS_PER_LM, COORDS_PER_LM)
        xy = x_lm[..., :2]  # (B, T, K, 2) — covers all pairs when COORDS_PER_LM==2
        # Batched matmul: rot[:, None, None] is (B,1,1,2,2), xy[..., None] is (B,T,K,2,1)
        xy_rot = (rot[:, None, None] @ xy.unsqueeze(-1)).squeeze(-1)  # (B, T, K, 2)
        if COORDS_PER_LM == 2:
            return xy_rot.reshape(B, T, D)
        x_lm_out = x_lm.clone()
        x_lm_out[..., :2] = xy_rot
        return x_lm_out.reshape(B, T, D)

    @staticmethod
    def finger_dropout_batch(x, sample_prob=0.5, dropout_prob=0.25):
        """Vectorized finger dropout: batch mask replaces per-sample clone + loop."""
        B, T, D = x.shape
        device = x.device
        dmask = x.new_ones(B, D)
        sample_gate = torch.rand(B, device=device) < sample_prob  # (B,)
        for hand_label in ("left", "right"):
            for fi in range(len(FINGER_LM_RANGES)):
                drop = sample_gate & (torch.rand(B, device=device) < dropout_prob)
                if not drop.any():
                    continue
                for feat_lo, feat_hi in FINGER_COORD_SLICES[(hand_label, fi)]:
                    dmask[drop, feat_lo:feat_hi] = 0.0
        return x * dmask.unsqueeze(1)

    @staticmethod
    def random_scale(x, min_scale=0.9, max_scale=1.1):
        return x * np.random.uniform(min_scale, max_scale)


def mixup_batch(x, y, mask, alpha=0.2):
    # Dominance-aware pairing: HandDominanceModule reorders hands INSIDE the model
    # (after mixup), so mixing a lh-dominant sample with a rh-dominant sample
    # produces ambiguous hand slot assignments. Pair same-dominance samples only.
    B, device = x.size(0), x.device
    lh_wrist_vel = x[
        :, :, COORD_FEAT + LH_START : COORD_FEAT + LH_START + COORDS_PER_LM
    ]
    rh_wrist_vel = x[
        :, :, COORD_FEAT + RH_START : COORD_FEAT + RH_START + COORDS_PER_LM
    ]
    rh_dominant = (rh_wrist_vel**2).sum(-1).mean(1) > (lh_wrist_vel**2).sum(-1).mean(1)

    rh_idx = torch.where(rh_dominant)[0]
    lh_idx = torch.where(~rh_dominant)[0]

    index = torch.empty(B, dtype=torch.long, device=device)
    index[rh_idx] = rh_idx[torch.randperm(len(rh_idx), device=device)]
    index[lh_idx] = lh_idx[torch.randperm(len(lh_idx), device=device)]

    lam = np.random.beta(alpha, alpha)
    mixed_mask = mask | mask[index]
    return lam * x + (1 - lam) * x[index], y, y[index], lam, mixed_mask, index
