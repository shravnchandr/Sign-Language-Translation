import torch
import torch.nn as nn
from .conformer import ConformerBlock, SinusoidalPositionalEncoding
from .normalization import RobustNormalization, WristNormalization
from .grl import SignerDiscriminator
from ..config import (
    POSE_START,
    COORDS_PER_LM,
    COORD_FEAT,
    SELECTED_FACE_INDICES,
    LH_START,
    RH_START,
    FACE_START,
    INCLUDE_FACE,
    N_FACE,
)


class HandDominanceModule(nn.Module):
    """
    Reorder left/right hand channels so the dominant hand (higher wrist motion
    energy) always maps to the first hand slot before projection.

    This makes the model hand-agnostic: a left-handed signer doing "Hello" and
    a right-handed signer doing "Hello" both arrive at dominant_proj with the
    same semantics, halving the effective learning burden for one-handed signs.

    The LH↔RH swap is expressed as a precomputed gather permutation applied in
    a single indexing op. This is robust to any future change in landmark block
    layout — no manual slice-pair bookkeeping required.
    """

    def __init__(self):
        super().__init__()
        # Build a (2*COORD_FEAT,) permutation where:
        #   output[:, :, LH_START:POSE_START]  ← input[:, :, RH_START:FACE_START]
        #   output[:, :, RH_START:FACE_START]  ← input[:, :, LH_START:POSE_START]
        # All other feature indices remain identity-mapped.
        D = 2 * COORD_FEAT
        perm = torch.arange(D)
        for half in (0, COORD_FEAT):
            perm[half + LH_START : half + POSE_START] = torch.arange(
                half + RH_START, half + FACE_START
            )
            perm[half + RH_START : half + FACE_START] = torch.arange(
                half + LH_START, half + POSE_START
            )
        self.register_buffer("swap_perm", perm)  # moves to correct device with model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 2*COORD_FEAT) — caller owns x (already cloned upstream)
        lh_wrist_vel = x[
            :, :, COORD_FEAT + LH_START : COORD_FEAT + LH_START + COORDS_PER_LM
        ]
        rh_wrist_vel = x[
            :, :, COORD_FEAT + RH_START : COORD_FEAT + RH_START + COORDS_PER_LM
        ]
        lh_energy = (lh_wrist_vel**2).sum(dim=-1).mean(dim=1)  # (B,)
        rh_energy = (rh_wrist_vel**2).sum(dim=-1).mean(dim=1)  # (B,)

        swap_idx = torch.where(rh_energy > lh_energy)[0]
        if swap_idx.numel() == 0:
            return x

        # Single gather across the feature dimension — no tmp clone needed.
        x[swap_idx] = x[swap_idx][:, :, self.swap_perm]
        return x


class AnatomicalConformer(nn.Module):
    def __init__(self, num_classes, d_model=512, n_heads=8, n_layers=6, dropout=0.1, drop_path_max=0.1, n_signers=0):
        super().__init__()
        # Offsets are computed dynamically from COORDS_PER_LM, so toggling
        # INCLUDE_DEPTH is safe. Toggling INCLUDE_FACE is NOT safe — face_proj
        # assumes face data is present.
        assert INCLUDE_FACE, "AnatomicalConformer requires INCLUDE_FACE=True"

        self.robust_norm = RobustNormalization(
            pose_start_idx=POSE_START, n_coords=COORDS_PER_LM
        )
        self.hand_dominance = HandDominanceModule()
        self.wrist_norm = WristNormalization()

        # Position stream — per-part projections
        self.lh_proj = nn.Linear(21 * COORDS_PER_LM, d_model // 4)
        self.rh_proj = nn.Linear(21 * COORDS_PER_LM, d_model // 4)
        self.pose_proj = nn.Linear(33 * COORDS_PER_LM, d_model // 4)
        self.face_proj = nn.Linear(N_FACE * COORDS_PER_LM, d_model // 4)

        # Velocity stream — per-part projections mirroring position stream
        # 4 × (d_model // 8) = d_model // 2, same total width as before
        self.lh_vel_proj = nn.Linear(21 * COORDS_PER_LM, d_model // 8)
        self.rh_vel_proj = nn.Linear(21 * COORDS_PER_LM, d_model // 8)
        self.pose_vel_proj = nn.Linear(33 * COORDS_PER_LM, d_model // 8)
        self.face_vel_proj = nn.Linear(N_FACE * COORDS_PER_LM, d_model // 8)

        # Feature fusion: (d_model pos + d_model//2 vel) → d_model
        self.feat_fuse = nn.Sequential(
            nn.Linear(d_model + (d_model // 2), d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Linearly increasing drop-path: block 0 gets 0, block n_layers-1 gets drop_path_max.
        # Forces each layer to be independently useful (can't rely on later layers to rescue).
        drop_rates = [drop_path_max * i / max(n_layers - 1, 1) for i in range(n_layers)]
        self.blocks = nn.ModuleList(
            [ConformerBlock(d_model, n_heads, dropout=dropout, drop_path_rate=r) for r in drop_rates]
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self.signer_disc = SignerDiscriminator(d_model, n_signers) if n_signers > 0 else None

    def forward(self, x, mask, grl_lambda: float = 0.0):
        B, T, _ = x.shape

        x = self.robust_norm(x)
        x = self.hand_dominance(x)   # reorder so dominant hand is always in LH slot
        x = self.wrist_norm(x)       # landmark 0 = location, landmarks 1-20 = shape

        # Split position and velocity halves
        pos = x[:, :, :COORD_FEAT]
        vel = x[:, :, COORD_FEAT:]

        # Per-part position features
        lh = self.lh_proj(pos[:, :, LH_START:POSE_START])
        ps = self.pose_proj(pos[:, :, POSE_START:RH_START])
        rh = self.rh_proj(pos[:, :, RH_START:FACE_START])
        fc = self.face_proj(pos[:, :, FACE_START:])
        pos_feat = torch.cat([lh, ps, rh, fc], dim=-1)  # (B, T, d_model)

        # Per-part velocity features
        lh_v = self.lh_vel_proj(vel[:, :, LH_START:POSE_START])
        ps_v = self.pose_vel_proj(vel[:, :, POSE_START:RH_START])
        rh_v = self.rh_vel_proj(vel[:, :, RH_START:FACE_START])
        fc_v = self.face_vel_proj(vel[:, :, FACE_START:])
        vel_feat = torch.cat([lh_v, ps_v, rh_v, fc_v], dim=-1)  # (B, T, d_model//2)

        x = self.feat_fuse(torch.cat([pos_feat, vel_feat], dim=-1))

        # Sequence modeling
        x = self.pos_enc(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        cls_mask = torch.ones(B, 1, device=mask.device, dtype=torch.bool)
        mask = torch.cat([cls_mask, mask], dim=1)

        for block in self.blocks:
            x = block(x, mask)

        cls_out = x[:, 0]
        sign_logits = self.head(cls_out)
        if self.training and self.signer_disc is not None and grl_lambda > 0.0:
            return sign_logits, self.signer_disc(cls_out, lam=grl_lambda)
        return sign_logits
