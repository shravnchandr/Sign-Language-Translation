import torch
import torch.nn as nn
from .conformer import ConformerBlock, SinusoidalPositionalEncoding
from .normalization import WristNormalization
from .grl import SignerDiscriminator
from ..config import (
    POSE_START,
    COORDS_PER_LM,
    COORD_FEAT,
    LH_START,
    RH_START,
    FACE_START,
    INCLUDE_FACE,
    N_FACE,
    N_FACE_EYEBROW,
    N_FACE_MOUTH,
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

        self.hand_dominance = HandDominanceModule()
        self.wrist_norm = WristNormalization()

        # Position stream — per-part projections.
        # Face is split into eyebrows (grammatical) and mouth (phonological)
        # so each pathway can specialise independently.
        # Total: 3 × d_model//4 + 2 × d_model//8 = d_model
        self.lh_proj = nn.Linear(21 * COORDS_PER_LM, d_model // 4)
        self.rh_proj = nn.Linear(21 * COORDS_PER_LM, d_model // 4)
        self.pose_proj = nn.Linear(33 * COORDS_PER_LM, d_model // 4)
        self.eyebrow_proj = nn.Linear(N_FACE_EYEBROW * COORDS_PER_LM, d_model // 8)
        self.mouth_proj = nn.Linear(N_FACE_MOUTH * COORDS_PER_LM, d_model // 8)

        # Velocity stream — 3 temporal scales (Δ1, Δ2, Δ5) concatenated per part,
        # then projected. Equal budget to position stream: 4 × d_model//4 = d_model.
        # Δ2/Δ5 are computed inside forward() from body-relative positions so they
        # benefit from nose-subtraction without changing the dataset return shape.
        _V = 3 * COORDS_PER_LM  # features per landmark across all 3 vel scales
        self.lh_vel_proj = nn.Linear(21 * _V, d_model // 4)
        self.rh_vel_proj = nn.Linear(21 * _V, d_model // 4)
        self.pose_vel_proj = nn.Linear(33 * _V, d_model // 4)
        self.face_vel_proj = nn.Linear(N_FACE * _V, d_model // 4)

        # Geometry stream — explicit joint angles + fingertip distances per hand.
        # 15 joint-angle cosines + 10 fingertip pairwise distances = 25 per hand.
        # Equal budget across both hands: 2 × d_model//8 = d_model//4.
        self.lh_geo_proj = nn.Linear(25, d_model // 8)
        self.rh_geo_proj = nn.Linear(25, d_model // 8)

        # Feature fusion: (d_model pos + d_model vel + d_model//4 geo) → d_model
        self.feat_fuse = nn.Sequential(
            nn.Linear(2 * d_model + d_model // 4, d_model),
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

    @staticmethod
    def _hand_geometry(hand: torch.Tensor) -> torch.Tensor:
        """
        Compute rotation-invariant hand shape descriptors in the wrist frame.

        hand: (B, T, 21, c) — lm 0 = wrist at origin (zeros), lm 1-20 wrist-relative.
        Returns: (B, T, 25) = 15 joint-angle cosines + 10 fingertip pairwise distances.

        Joint angles capture finger curl (impossible to infer reliably from raw XYZ).
        Fingertip distances capture hand openness and inter-finger configuration.
        Both are invariant to wrist rotation and signer hand scale.
        """
        angles = []
        for chain in (
            (0, 1, 2, 3, 4),     # thumb:  wrist, CMC, MCP, IP, tip
            (0, 5, 6, 7, 8),     # index:  wrist, MCP, PIP, DIP, tip
            (0, 9, 10, 11, 12),  # middle
            (0, 13, 14, 15, 16), # ring
            (0, 17, 18, 19, 20), # pinky
        ):
            for i in range(3):   # angle at each of the 3 interior joints
                a = hand[:, :, chain[i]]
                b = hand[:, :, chain[i + 1]]
                c = hand[:, :, chain[i + 2]]
                v1, v2 = a - b, c - b
                cos_a = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-6)
                angles.append(cos_a.clamp(-1.0, 1.0))  # (B, T)

        tips = hand[:, :, [4, 8, 12, 16, 20]]  # (B, T, 5, c)
        dists = [
            (tips[:, :, i] - tips[:, :, j]).norm(dim=-1)
            for i in range(5) for j in range(i + 1, 5)
        ]
        return torch.stack(angles + dists, dim=-1)  # (B, T, 25)

    def forward(self, x, mask, grl_lambda: float = 0.0):
        B, T, _ = x.shape

        x = self.hand_dominance(x)   # reorder so dominant hand is always in LH slot
        x = self.wrist_norm(x)       # landmark 0 = location, landmarks 1-20 = shape

        # Split position and delta-1 velocity halves (dataset layout: [pos | vel1])
        pos = x[:, :, :COORD_FEAT]
        vel1 = x[:, :, COORD_FEAT:]

        # Geometry stream — computed from wrist-relative fingers (after WristNorm).
        # Wrist itself is at (0, 0, 0) in this frame; concatenate as lm 0.
        c = COORDS_PER_LM
        zeros = torch.zeros(B, T, 1, c, device=x.device, dtype=x.dtype)
        lh_hand = torch.cat([zeros, pos[:, :, LH_START + c:POSE_START].reshape(B, T, 20, c)], dim=2)
        rh_hand = torch.cat([zeros, pos[:, :, RH_START + c:FACE_START].reshape(B, T, 20, c)], dim=2)
        geo_feat = torch.cat([
            self.lh_geo_proj(self._hand_geometry(lh_hand)),
            self.rh_geo_proj(self._hand_geometry(rh_hand)),
        ], dim=-1)  # (B, T, d_model // 4)

        # Compute additional velocity scales from body-relative positions.
        # Divided by their time delta so all three scales share the same unit
        # (displacement per frame), preventing vel5's larger raw magnitude from
        # dominating vel_proj gradients and crowding out fine-grained Δ1 signal.
        # Boundary frames stay zero (correct: padded positions are zero).
        vel2 = torch.zeros_like(pos)
        vel2[:, 2:] = (pos[:, 2:] - pos[:, :-2]) / 2.0
        vel5 = torch.zeros_like(pos)
        vel5[:, 5:] = (pos[:, 5:] - pos[:, :-5]) / 5.0

        # Per-part position features — face split into eyebrow and mouth streams
        _eb = N_FACE_EYEBROW * COORDS_PER_LM  # eyebrow feature width
        lh = self.lh_proj(pos[:, :, LH_START:POSE_START])
        ps = self.pose_proj(pos[:, :, POSE_START:RH_START])
        rh = self.rh_proj(pos[:, :, RH_START:FACE_START])
        eb = self.eyebrow_proj(pos[:, :, FACE_START:FACE_START + _eb])
        mo = self.mouth_proj(pos[:, :, FACE_START + _eb:])
        pos_feat = torch.cat([lh, ps, rh, eb, mo], dim=-1)  # (B, T, d_model)

        # Per-part multi-scale velocity features (Δ1∥Δ2∥Δ5 concatenated per part)
        def _vcat(s):
            return torch.cat([vel1[:, :, s], vel2[:, :, s], vel5[:, :, s]], dim=-1)

        lh_v = self.lh_vel_proj(_vcat(slice(LH_START, POSE_START)))
        ps_v = self.pose_vel_proj(_vcat(slice(POSE_START, RH_START)))
        rh_v = self.rh_vel_proj(_vcat(slice(RH_START, FACE_START)))
        fc_v = self.face_vel_proj(_vcat(slice(FACE_START, None)))
        vel_feat = torch.cat([lh_v, ps_v, rh_v, fc_v], dim=-1)  # (B, T, d_model)

        x = self.feat_fuse(torch.cat([pos_feat, vel_feat, geo_feat], dim=-1))

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
