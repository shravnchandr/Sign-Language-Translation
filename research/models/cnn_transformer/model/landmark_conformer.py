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
    IN_FEAT,
    MIRROR_PERM,
    PRESENCE_START,
    N_FACE,
    N_FACE_EYEBROW,
    N_FACE_MOUTH,
)


class HandDominanceModule(nn.Module):
    """
    Canonicalise handedness: horizontally mirror any sequence whose right-hand
    slot moves more, so the dominant hand always arrives in the first (LH) slot
    with the same anatomy.

    A slot swap alone is not enough: a right hand moved into the LH slot is
    still a mirror-image hand shape on the other side of the body, so lh_proj
    would see two distributions for every one-handed sign, and pose/eyebrow
    landmarks would stay un-mirrored. Mirroring (negate x + swap every bilateral
    landmark via MIRROR_PERM) maps a right-dominant signer exactly onto a
    left-dominant one. The presence flags are swapped with their hands. Because
    every input is canonicalised here, a mirror-flip augmentation would be
    undone for all but near-tie sequences, so training uses none.
    """

    def __init__(self):
        super().__init__()
        # Derived constants, not learned state: non-persistent so checkpoints
        # don't depend on the input layout.
        self.register_buffer(
            "mirror_perm", torch.tensor(MIRROR_PERM), persistent=False
        )
        x_sign = torch.ones(IN_FEAT)
        x_sign[:PRESENCE_START:COORDS_PER_LM] = -1.0  # x of pos and vel only
        self.register_buffer("x_sign", x_sign, persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (x, dom_ratio).

        dom_ratio: (B,) in [0.5, 1] — dominant-hand share of wrist motion energy.
          ≈ 1   → clearly one-handed / dominant-hand sign
          ≈ 0.5 → both hands equally active (symmetric two-handed sign)
        Passed to dist_proj so the model can weight hand streams by ambiguity.
        """
        # x: (B, T, IN_FEAT) — caller owns x (already cloned upstream)
        lh_wrist_vel = x[
            :, :, COORD_FEAT + LH_START : COORD_FEAT + LH_START + COORDS_PER_LM
        ]
        rh_wrist_vel = x[
            :, :, COORD_FEAT + RH_START : COORD_FEAT + RH_START + COORDS_PER_LM
        ]
        lh_energy = (lh_wrist_vel**2).sum(dim=-1).mean(dim=1)  # (B,)
        rh_energy = (rh_wrist_vel**2).sum(dim=-1).mean(dim=1)  # (B,)

        dom_ratio = torch.maximum(lh_energy, rh_energy) / (
            lh_energy + rh_energy + 1e-6
        )  # (B,)

        swap_idx = torch.where(rh_energy > lh_energy)[0]
        if swap_idx.numel() > 0:
            x[swap_idx] = x[swap_idx][:, :, self.mirror_perm] * self.x_sign

        return x, dom_ratio


class LandmarkConformer(nn.Module):
    def __init__(
        self,
        num_classes,
        d_model=512,
        n_heads=8,
        n_layers=6,
        dropout=0.1,
        drop_path_max=0.1,
        n_signers=0,
        ctc_vocab_size=0,
    ):
        super().__init__()
        # Offsets are computed dynamically from COORDS_PER_LM, so toggling
        # INCLUDE_DEPTH is safe. Toggling INCLUDE_FACE is NOT safe — face_proj
        # assumes face data is present.
        assert INCLUDE_FACE, "LandmarkConformer requires INCLUDE_FACE=True"

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
        # 15 joint-angle cosines + 10 fingertip pairwise distances
        #   + 3 palm-normal components (3D only — encodes palm orientation)
        # = 28 (INCLUDE_DEPTH) or 25 (2D) per hand; projected 2 × d_model//8 = d_model//4.
        # Plus the distance stream (2 hand-nose distances, dom_ratio, 2 presence
        # flags) → d_model//8.
        _n_geo = 25 + (3 if COORDS_PER_LM == 3 else 0)
        self.lh_geo_proj = nn.Linear(_n_geo, d_model // 8)
        self.rh_geo_proj = nn.Linear(_n_geo, d_model // 8)
        # lh_dist, rh_dist, dom_ratio, lh_present, rh_present. The presence flags
        # disambiguate a never-detected hand (stored as 0 = the nose origin)
        # from a hand actually at the face.
        self.dist_proj = nn.Linear(5, d_model // 8)

        # Feature fusion: pos(d_model) + vel(d_model) + geo(d_model//4) + dist(d_model//8) → d_model
        self.feat_fuse = nn.Sequential(
            nn.Linear(2 * d_model + d_model // 4 + d_model // 8, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Linearly increasing drop-path: block 0 gets 0, block n_layers-1 gets drop_path_max.
        # Forces each layer to be independently useful (can't rely on later layers to rescue).
        drop_rates = [drop_path_max * i / max(n_layers - 1, 1) for i in range(n_layers)]
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(d_model, n_heads, dropout=dropout, drop_path_rate=r)
                for r in drop_rates
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self.signer_disc = (
            SignerDiscriminator(d_model, n_signers) if n_signers > 0 else None
        )
        # CTC pre-training head: per-frame projection → vocab + 1 blank.
        # When present, forward() skips the CLS token and returns (B, T, vocab+1)
        # instead of sign logits.  Not used during fine-tuning.
        self.ctc_head = (
            nn.Linear(d_model, ctc_vocab_size + 1) if ctc_vocab_size > 0 else None
        )

    @staticmethod
    def _hand_geometry(hand: torch.Tensor) -> torch.Tensor:
        """
        Compute rotation-invariant hand shape descriptors in the wrist frame.

        hand: (B, T, 21, c) — lm 0 = wrist at origin (zeros), lm 1-20 wrist-relative.
        Returns: (B, T, 28) in 3D or (B, T, 25) in 2D.
          15 joint-angle cosines: 3 per finger (at MCP, PIP, DIP joints)
          10 fingertip pairwise distances: hand openness and inter-finger spread
           3 palm-normal components (3D only): unit vector perpendicular to palm,
             encodes orientation (curl vs. toward-camera vs. left/right) that raw
             XYZ buries and joint angles cannot distinguish.
        All features are invariant to wrist rotation and signer hand scale.
        """
        angles = []
        for chain in (
            (0, 1, 2, 3, 4),  # thumb:  wrist, CMC, MCP, IP, tip
            (0, 5, 6, 7, 8),  # index:  wrist, MCP, PIP, DIP, tip
            (0, 9, 10, 11, 12),  # middle
            (0, 13, 14, 15, 16),  # ring
            (0, 17, 18, 19, 20),  # pinky
        ):
            for i in range(3):  # angle at each of the 3 interior joints
                a = hand[:, :, chain[i]]
                b = hand[:, :, chain[i + 1]]
                c = hand[:, :, chain[i + 2]]
                v1, v2 = a - b, c - b
                cos_a = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-6)
                angles.append(cos_a.clamp(-1.0, 1.0))  # (B, T)

        tips = hand[:, :, [4, 8, 12, 16, 20]]  # (B, T, 5, c)
        hand_scale = (
            hand[:, :, [5, 9, 13, 17]].norm(dim=-1).mean(dim=-1) + 1e-6
        )  # (B, T) mean MCP distance
        dists = [
            (tips[:, :, i] - tips[:, :, j]).norm(dim=-1) / hand_scale
            for i in range(5)
            for j in range(i + 1, 5)
        ]
        features = torch.stack(angles + dists, dim=-1)  # (B, T, 25)

        if hand.shape[-1] == 3:
            # Palm normal: cross product of (wrist→index_MCP) × (wrist→pinky_MCP).
            # Unit normal encodes which way the palm faces — the one distinction
            # that joint angles and raw XYZ both fail to capture reliably.
            n = torch.linalg.cross(hand[:, :, 5], hand[:, :, 17])  # (B, T, 3)
            n_unit = n / (n.norm(dim=-1, keepdim=True) + 1e-6)
            features = torch.cat([features, n_unit], dim=-1)  # (B, T, 28)

        return features

    def forward(self, x, mask, grl_lambda: float = 0.0):
        B, T, _ = x.shape

        x, dom_ratio = self.hand_dominance(x)  # mirror; dom_ratio (B,) = dom/total
        x = self.wrist_norm(x)  # landmark 0 = location, landmarks 1-20 = shape

        # Dataset layout: [pos | vel1 | presence (lh, rh)]
        pos = x[:, :, :COORD_FEAT]
        vel1 = x[:, :, COORD_FEAT:PRESENCE_START]
        presence = x[:, :, PRESENCE_START:]  # (B, T, 2), after dominance mirror
        c = COORDS_PER_LM

        # Shoulder-width normalization: scale all positional and velocity features by
        # the mean inter-shoulder distance over valid frames. Makes the model invariant
        # to camera distance and signer body size. Per-sequence mean (not per-frame)
        # avoids scale noise from shoulder movement within a sign.
        lshoulder = pos[:, :, POSE_START + 11 * c : POSE_START + 12 * c]
        rshoulder = pos[:, :, POSE_START + 12 * c : POSE_START + 13 * c]
        shoulder_w = (lshoulder - rshoulder).norm(dim=-1)  # (B, T)
        mean_sw = (shoulder_w * mask.float()).sum(1) / mask.float().sum(1).clamp(
            min=1
        )  # (B,)
        scale = mean_sw.clamp(min=1e-3).view(B, 1, 1)
        pos = pos / scale
        vel1 = vel1 / scale

        # Geometry stream — computed from wrist-relative fingers (after WristNorm).
        # Wrist itself is at (0, 0, 0) in this frame; concatenate as lm 0.
        zeros = torch.zeros(B, T, 1, c, device=x.device, dtype=x.dtype)
        lh_hand = torch.cat(
            [zeros, pos[:, :, LH_START + c : POSE_START].reshape(B, T, 20, c)], dim=2
        )
        rh_hand = torch.cat(
            [zeros, pos[:, :, RH_START + c : FACE_START].reshape(B, T, 20, c)], dim=2
        )
        geo_feat = torch.cat(
            [
                self.lh_geo_proj(self._hand_geometry(lh_hand)),
                self.rh_geo_proj(self._hand_geometry(rh_hand)),
            ],
            dim=-1,
        )  # (B, T, d_model // 4)

        # Hand-nose distances: dominant and non-dominant wrist to nose (origin).
        # Encodes when hands are in the face region — gates relevance of face NMMs.
        lh_dist = pos[:, :, LH_START : LH_START + c].norm(
            dim=-1, keepdim=True
        )  # (B, T, 1)
        rh_dist = pos[:, :, RH_START : RH_START + c].norm(
            dim=-1, keepdim=True
        )  # (B, T, 1)
        dom_ratio_feat = dom_ratio[:, None, None].expand(B, T, 1)  # (B, T, 1)
        dist_feat = self.dist_proj(
            torch.cat([lh_dist, rh_dist, dom_ratio_feat, presence], dim=-1)
        )  # (B, T, d_model // 8)

        # Compute additional velocity scales from body-relative positions.
        # Divided by their time delta so all three scales share the same unit
        # (displacement per frame), preventing vel5's larger raw magnitude from
        # dominating vel_proj gradients and crowding out fine-grained Δ1 signal.
        # Boundary frames stay zero (correct: padded positions are zero).
        vel2 = torch.zeros_like(pos)
        vel2[:, 2:] = (pos[:, 2:] - pos[:, :-2]) / 2.0
        vel5 = torch.zeros_like(pos)
        vel5[:, 5:] = (pos[:, 5:] - pos[:, :-5]) / 5.0

        # Per-part position features — face split into eyebrow and mouth streams.
        # Slice correctness depends on eyebrows being first in SELECTED_FACE_INDICES;
        # config.py enforces this ordering explicitly.
        _eb = N_FACE_EYEBROW * COORDS_PER_LM  # eyebrow feature width
        assert self.eyebrow_proj.in_features == _eb, (
            f"eyebrow_proj expects {self.eyebrow_proj.in_features} features "
            f"but SELECTED_FACE_INDICES gives {_eb} — check config.py ordering"
        )
        lh = self.lh_proj(pos[:, :, LH_START:POSE_START])
        ps = self.pose_proj(pos[:, :, POSE_START:RH_START])
        rh = self.rh_proj(pos[:, :, RH_START:FACE_START])
        eb = self.eyebrow_proj(pos[:, :, FACE_START : FACE_START + _eb])
        mo = self.mouth_proj(pos[:, :, FACE_START + _eb :])
        pos_feat = torch.cat([lh, ps, rh, eb, mo], dim=-1)  # (B, T, d_model)

        # Per-part multi-scale velocity features (Δ1∥Δ2∥Δ5 concatenated per part)
        def _vcat(s):
            return torch.cat([vel1[:, :, s], vel2[:, :, s], vel5[:, :, s]], dim=-1)

        lh_v = self.lh_vel_proj(_vcat(slice(LH_START, POSE_START)))
        ps_v = self.pose_vel_proj(_vcat(slice(POSE_START, RH_START)))
        rh_v = self.rh_vel_proj(_vcat(slice(RH_START, FACE_START)))
        fc_v = self.face_vel_proj(_vcat(slice(FACE_START, None)))
        vel_feat = torch.cat([lh_v, ps_v, rh_v, fc_v], dim=-1)  # (B, T, d_model)

        x = self.feat_fuse(torch.cat([pos_feat, vel_feat, geo_feat, dist_feat], dim=-1))

        x = self.pos_enc(x)

        if self.ctc_head is not None:
            # CTC pre-training: no CLS token; return per-frame logits (B, T, vocab+1)
            for block in self.blocks:
                x = block(x, mask)
            return self.ctc_head(x)

        # Classification: prepend CLS token
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
