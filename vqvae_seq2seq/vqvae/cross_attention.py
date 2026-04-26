"""Cross-factor attention for interaction between different body parts."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List


class CrossFactorAttention(nn.Module):
    """
    Cross-attention between different body part factors.

    Addresses blindspot #8: No cross-factor interaction.

    Allows hands, pose, and face representations to attend to each other,
    capturing coordinated movements and dependencies.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            embed_dim: Embedding dimension for all factors
            num_heads: Number of attention heads
            n_layers: Number of cross-attention layers
            dropout: Dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        # Cross-attention layers for each pair of factors
        # We use a shared architecture but separate weights for each layer
        self.cross_attn_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "hand_to_pose": nn.MultiheadAttention(
                            embed_dim, num_heads, dropout, batch_first=True
                        ),
                        "pose_to_hand": nn.MultiheadAttention(
                            embed_dim, num_heads, dropout, batch_first=True
                        ),
                        "hand_to_face": nn.MultiheadAttention(
                            embed_dim, num_heads, dropout, batch_first=True
                        ),
                        "face_to_hand": nn.MultiheadAttention(
                            embed_dim, num_heads, dropout, batch_first=True
                        ),
                        "pose_to_face": nn.MultiheadAttention(
                            embed_dim, num_heads, dropout, batch_first=True
                        ),
                        "face_to_pose": nn.MultiheadAttention(
                            embed_dim, num_heads, dropout, batch_first=True
                        ),
                    }
                )
                for _ in range(n_layers)
            ]
        )

        # Layer norms
        self.layer_norms = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "dominant_hand": nn.LayerNorm(embed_dim),
                        "non_dominant_hand": nn.LayerNorm(embed_dim),
                        "pose": nn.LayerNorm(embed_dim),
                        "face": nn.LayerNorm(embed_dim),
                    }
                )
                for _ in range(n_layers)
            ]
        )

        # FFN for each factor
        self.ffns = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        factor: nn.Sequential(
                            nn.Linear(embed_dim, embed_dim * 4),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(embed_dim * 4, embed_dim),
                            nn.Dropout(dropout),
                        )
                        for factor in [
                            "dominant_hand",
                            "non_dominant_hand",
                            "pose",
                            "face",
                        ]
                    }
                )
                for _ in range(n_layers)
            ]
        )

        # Final layer norms
        self.final_norms = nn.ModuleDict(
            {
                "dominant_hand": nn.LayerNorm(embed_dim),
                "non_dominant_hand": nn.LayerNorm(embed_dim),
                "pose": nn.LayerNorm(embed_dim),
                "face": nn.LayerNorm(embed_dim),
            }
        )

    def forward(
        self, factors: Dict[str, torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply cross-factor attention.

        Args:
            factors: Dictionary with keys:
                - 'dominant_hand': (B, T, D)
                - 'non_dominant_hand': (B, T, D)
                - 'pose': (B, T, D)
                - 'face': (B, T, D)
            mask: Optional (B, T) validity mask

        Returns:
            Updated factors dictionary with same shape
        """
        dom_hand = factors["dominant_hand"]
        nondom_hand = factors["non_dominant_hand"]
        pose = factors["pose"]
        face = factors["face"]

        # Create attention mask
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask  # True = masked

        for layer_idx in range(self.n_layers):
            cross_attn = self.cross_attn_layers[layer_idx]
            layer_norm = self.layer_norms[layer_idx]
            ffn = self.ffns[layer_idx]

            # ===== Hand-Pose Cross-Attention =====
            # Dominant hand attends to pose
            dom_to_pose, _ = cross_attn["hand_to_pose"](
                dom_hand, pose, pose, key_padding_mask=attn_mask
            )
            dom_hand = layer_norm["dominant_hand"](dom_hand + dom_to_pose)

            # Non-dominant hand attends to pose
            nondom_to_pose, _ = cross_attn["hand_to_pose"](
                nondom_hand, pose, pose, key_padding_mask=attn_mask
            )
            nondom_hand = layer_norm["non_dominant_hand"](nondom_hand + nondom_to_pose)

            # Pose attends to hands (combined)
            hands_combined = (dom_hand + nondom_hand) / 2
            pose_to_hand, _ = cross_attn["pose_to_hand"](
                pose, hands_combined, hands_combined, key_padding_mask=attn_mask
            )
            pose = layer_norm["pose"](pose + pose_to_hand)

            # ===== Hand-Face Cross-Attention =====
            # Hands attend to face (for signs near face)
            dom_to_face, _ = cross_attn["hand_to_face"](
                dom_hand, face, face, key_padding_mask=attn_mask
            )
            dom_hand = dom_hand + dom_to_face

            nondom_to_face, _ = cross_attn["hand_to_face"](
                nondom_hand, face, face, key_padding_mask=attn_mask
            )
            nondom_hand = nondom_hand + nondom_to_face

            # Face attends to hands
            face_to_hand, _ = cross_attn["face_to_hand"](
                face, hands_combined, hands_combined, key_padding_mask=attn_mask
            )
            face = layer_norm["face"](face + face_to_hand)

            # ===== Pose-Face Cross-Attention =====
            pose_to_face, _ = cross_attn["pose_to_face"](
                pose, face, face, key_padding_mask=attn_mask
            )
            pose = pose + pose_to_face

            face_to_pose, _ = cross_attn["face_to_pose"](
                face, pose, pose, key_padding_mask=attn_mask
            )
            face = face + face_to_pose

            # ===== FFN =====
            dom_hand = dom_hand + ffn["dominant_hand"](dom_hand)
            nondom_hand = nondom_hand + ffn["non_dominant_hand"](nondom_hand)
            pose = pose + ffn["pose"](pose)
            face = face + ffn["face"](face)

        # Final normalization
        return {
            "dominant_hand": self.final_norms["dominant_hand"](dom_hand),
            "non_dominant_hand": self.final_norms["non_dominant_hand"](nondom_hand),
            "pose": self.final_norms["pose"](pose),
            "face": self.final_norms["face"](face),
        }


class FactorFusion(nn.Module):
    """
    Fuses multiple factor representations into a single representation.

    Used after cross-attention to create a unified sign representation.
    """

    def __init__(
        self,
        factor_dim: int = 128,
        output_dim: int = 256,
        n_factors: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Learnable factor weights
        self.factor_weights = nn.Parameter(torch.ones(n_factors))

        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(factor_dim * n_factors, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
        )

        # Attention pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(factor_dim, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, factors: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse factor representations.

        Args:
            factors: Dictionary of factor tensors, each (B, T, D)

        Returns:
            (B, T, output_dim) fused representation
        """
        # Stack factors
        factor_list = [
            factors["dominant_hand"],
            factors["non_dominant_hand"],
            factors["pose"],
            factors["face"],
        ]
        stacked = torch.stack(factor_list, dim=2)  # (B, T, n_factors, D)

        # Apply learned weights
        weights = F.softmax(self.factor_weights, dim=0)
        weighted = stacked * weights.view(1, 1, -1, 1)

        # Concatenate weighted factors
        B, T, N, D = weighted.shape
        concat = weighted.reshape(B, T, N * D)

        # Fuse
        fused = self.fusion(concat)

        return fused


class HierarchicalCrossAttention(nn.Module):
    """
    Hierarchical cross-attention that operates at multiple temporal scales.

    Captures both local and global cross-factor dependencies.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        scales: List[int] = [1, 4, 8],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.scales = scales

        # Cross-attention at each scale
        self.scale_attention = nn.ModuleList(
            [
                CrossFactorAttention(embed_dim, num_heads, n_layers=1, dropout=dropout)
                for _ in scales
            ]
        )

        # Scale fusion
        self.scale_fusion = nn.ModuleDict(
            {
                factor: nn.Sequential(
                    nn.Linear(embed_dim * len(scales), embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.GELU(),
                )
                for factor in ["dominant_hand", "non_dominant_hand", "pose", "face"]
            }
        )

    def _pool_to_scale(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        """Pool temporal dimension to given scale."""
        if scale == 1:
            return x

        B, T, D = x.shape
        # Pad if needed
        pad = (scale - T % scale) % scale
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))

        T_new = (T + pad) // scale
        x = x.reshape(B, T_new, scale, D).mean(dim=2)
        return x

    def _upsample_from_scale(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Upsample back to original temporal length."""
        if x.shape[1] == target_len:
            return x

        x_t = x.permute(0, 2, 1)  # (B, D, T)
        x_t = F.interpolate(x_t, size=target_len, mode="linear", align_corners=True)
        return x_t.permute(0, 2, 1)

    def forward(
        self, factors: Dict[str, torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply hierarchical cross-attention.

        Args:
            factors: Dictionary of factor tensors
            mask: Optional validity mask

        Returns:
            Updated factors dictionary
        """
        T = factors["dominant_hand"].shape[1]

        # Process at each scale
        scale_outputs = {factor: [] for factor in factors}

        for scale, attn in zip(self.scales, self.scale_attention):
            # Pool to scale
            scaled_factors = {
                k: self._pool_to_scale(v, scale) for k, v in factors.items()
            }

            # Apply cross-attention
            attended = attn(scaled_factors, mask=None)  # No mask at coarse scales

            # Upsample back
            for factor in attended:
                upsampled = self._upsample_from_scale(attended[factor], T)
                scale_outputs[factor].append(upsampled)

        # Fuse scales
        result = {}
        for factor in factors:
            stacked = torch.cat(scale_outputs[factor], dim=-1)
            result[factor] = self.scale_fusion[factor](stacked)

        return result
