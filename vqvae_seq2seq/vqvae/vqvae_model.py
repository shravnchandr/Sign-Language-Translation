"""Complete Improved VQ-VAE model for sign language tokenization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from dataclasses import asdict

from .config import ImprovedVQVAEConfig
from .normalization import RobustNormalization, ScaleNormalization
from .hand_dominance import HandDominanceModule, TwoHandFusion
from .augmentation import TemporalAugmentation
from .multi_scale_encoder import MultiScaleMotionEncoder
from .face_encoder import FaceChunkEncoder
from .cross_attention import CrossFactorAttention, FactorFusion
from .vector_quantizer import (
    EMAVectorQuantizer,
    FactorizedVectorQuantizer,
)


class Decoder(nn.Module):
    """Decoder to reconstruct landmarks from quantized representations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        chunk_size: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.output_dim = output_dim

        # Upsample from chunk to frames
        self.upsample = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * chunk_size),
            nn.GELU(),
        )

        # Temporal refinement
        self.temporal_refine = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Decode quantized representations back to landmarks.

        Args:
            z: (B, n_chunks, input_dim) quantized representation
            target_len: Target sequence length

        Returns:
            (B, target_len, output_dim) reconstructed landmarks
        """
        B, n_chunks, D = z.shape

        # Upsample each chunk
        upsampled = self.upsample(z)  # (B, n_chunks, hidden * chunk_size)
        upsampled = upsampled.reshape(
            B, n_chunks * self.chunk_size, -1
        )  # (B, T', hidden)

        # Temporal refinement
        upsampled_t = upsampled.permute(0, 2, 1)  # (B, hidden, T')
        refined = self.temporal_refine(upsampled_t)
        refined = refined.permute(0, 2, 1)  # (B, T', hidden)

        # Interpolate to target length
        if refined.shape[1] != target_len:
            refined_t = refined.permute(0, 2, 1)
            refined_t = F.interpolate(
                refined_t, size=target_len, mode="linear", align_corners=True
            )
            refined = refined_t.permute(0, 2, 1)

        # Output projection
        output = self.output_proj(refined)

        return output


class ImprovedVQVAE(nn.Module):
    """
    Improved VQ-VAE for sign language tokenization.

    Addresses all 10 blindspots:
    1. Z-coordinate: Full 3D landmarks
    2. Face underweighted: Dedicated FaceNMMEncoder
    3. Fixed chunk size: Multi-scale encoders
    4. Signer-dependent: Handled in data pipeline
    5. Hand dominance: HandDominanceModule
    6. Domain shift: Trained on combined datasets
    7. Fragile normalization: RobustNormalization
    8. No cross-factor: CrossFactorAttention
    9. No temporal augmentation: TemporalAugmentation
    10. Suboptimal codebooks: Increased sizes
    """

    def __init__(self, config: Optional[ImprovedVQVAEConfig] = None):
        super().__init__()
        self.config = config or ImprovedVQVAEConfig()

        # Normalization modules
        self.robust_norm = RobustNormalization(
            pose_start_idx=42,  # After both hands
            n_coords=self.config.n_coords,
        )
        self.scale_norm = ScaleNormalization(pose_start_idx=42)

        # Hand dominance
        self.hand_dominance = HandDominanceModule(
            hand_landmarks=self.config.hand_landmarks,
            n_coords=self.config.n_coords,
        )

        # Two-hand fusion
        self.two_hand_fusion = TwoHandFusion(
            hand_dim=self.config.hand_dim,
            hidden_dim=self.config.encoder_hidden_dim,
            output_dim=self.config.embed_dim,
        )

        # Augmentation
        self.augmentation = TemporalAugmentation(
            speed_range=self.config.augment_speed_range,
            frame_dropout_prob=self.config.augment_frame_dropout_prob,
            temporal_jitter_std=self.config.augment_temporal_jitter_std,
        )

        # Multi-scale encoders for hands and pose
        self.dominant_hand_encoder = MultiScaleMotionEncoder(
            input_dim=self.config.hand_dim,
            hidden_dim=self.config.encoder_hidden_dim,
            output_dim=self.config.embed_dim,
            scales=self.config.temporal_scales,
            n_layers=self.config.encoder_n_layers,
            dropout=self.config.encoder_dropout,
        )

        self.non_dominant_hand_encoder = MultiScaleMotionEncoder(
            input_dim=self.config.hand_dim,
            hidden_dim=self.config.encoder_hidden_dim,
            output_dim=self.config.embed_dim,
            scales=self.config.temporal_scales,
            n_layers=self.config.encoder_n_layers,
            dropout=self.config.encoder_dropout,
        )

        self.pose_encoder = MultiScaleMotionEncoder(
            input_dim=self.config.pose_dim,
            hidden_dim=self.config.encoder_hidden_dim,
            output_dim=self.config.embed_dim,
            scales=self.config.temporal_scales,
            n_layers=self.config.encoder_n_layers,
            dropout=self.config.encoder_dropout,
        )

        # Face encoder
        self.face_encoder = FaceChunkEncoder(
            n_face_landmarks=self.config.face_landmarks,
            n_coords=self.config.n_coords,
            hidden_dim=self.config.encoder_hidden_dim,
            output_dim=self.config.embed_dim,
            chunk_size=self.config.base_chunk_size,
            dropout=self.config.encoder_dropout,
        )

        # Cross-factor attention
        if self.config.use_cross_attention:
            self.cross_attention = CrossFactorAttention(
                embed_dim=self.config.embed_dim,
                num_heads=self.config.cross_attention_heads,
                n_layers=self.config.cross_attention_layers,
                dropout=self.config.encoder_dropout,
            )

        # Factor fusion
        self.factor_fusion = FactorFusion(
            factor_dim=self.config.embed_dim,
            output_dim=self.config.latent_dim,
            n_factors=4,
            dropout=self.config.encoder_dropout,
        )

        # Vector quantizers
        self.quantizers = FactorizedVectorQuantizer(
            codebook_configs={
                "pose": (self.config.pose_codebook_size, self.config.embed_dim),
                "motion": (self.config.motion_codebook_size, self.config.embed_dim),
                "dynamics": (self.config.dynamics_codebook_size, self.config.embed_dim),
                "face": (self.config.face_codebook_size, self.config.embed_dim),
            },
            commitment_weight=self.config.commitment_weight,
            ema_decay=self.config.ema_decay,
            reset_threshold=self.config.codebook_reset_threshold,
            reset_patience=self.config.codebook_reset_patience,
        )

        # Diversity loss

        # Decoder
        self.decoder = Decoder(
            input_dim=self.config.latent_dim,
            hidden_dim=self.config.decoder_hidden_dim,
            output_dim=self.config.get_total_input_dim(),
            chunk_size=self.config.base_chunk_size,
            n_layers=self.config.decoder_n_layers,
            dropout=self.config.encoder_dropout,
        )

    def _extract_body_parts(self, landmarks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract different body parts from combined landmarks tensor.

        Args:
            landmarks: (B, T, N, C) where N = 21+21+33+134 = 209

        Returns:
            Dictionary with 'left_hand', 'right_hand', 'pose', 'face'
        """
        return {
            "left_hand": landmarks[:, :, :21],  # (B, T, 21, C)
            "right_hand": landmarks[:, :, 21:42],  # (B, T, 21, C)
            "pose": landmarks[:, :, 42:75],  # (B, T, 33, C)
            "face": landmarks[:, :, 75:],  # (B, T, 134, C)
        }

    def encode(
        self,
        landmarks: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Encode landmarks to quantized representations.

        Args:
            landmarks: (B, T, N, C) landmarks tensor
            mask: Optional (B, T) validity mask

        Returns:
            quantized: Dictionary of quantized latents
            indices: Dictionary of codebook indices
        """
        B, T, N, C = landmarks.shape

        # Apply normalization
        landmarks = self.robust_norm(landmarks)
        landmarks = self.scale_norm(landmarks)

        # Extract body parts
        parts = self._extract_body_parts(landmarks)

        # Apply hand dominance reordering
        dom_hand, nondom_hand = self.hand_dominance(
            parts["left_hand"], parts["right_hand"]
        )

        # Flatten landmark dimensions for encoding
        dom_hand_flat = dom_hand.reshape(B, T, -1)  # (B, T, 63)
        nondom_hand_flat = nondom_hand.reshape(B, T, -1)
        pose_flat = parts["pose"].reshape(B, T, -1)  # (B, T, 99)

        # Encode each factor
        dom_pose, dom_motion, dom_dynamics = self.dominant_hand_encoder(
            dom_hand_flat, mask
        )
        nondom_pose, nondom_motion, nondom_dynamics = self.non_dominant_hand_encoder(
            nondom_hand_flat, mask
        )
        body_pose, body_motion, body_dynamics = self.pose_encoder(pose_flat, mask)
        face_features = self.face_encoder(parts["face"], mask)

        # Aggregate pose features (combine hands and body)
        # Take mean across body parts for pose codebook
        pose_combined = (dom_pose + nondom_pose + body_pose) / 3
        motion_combined = (dom_motion + nondom_motion + body_motion) / 3
        dynamics_combined = (dom_dynamics + nondom_dynamics + body_dynamics) / 3

        # Apply cross-factor attention if enabled
        if self.config.use_cross_attention:
            n_chunks = pose_combined.shape[1]

            # Align face features to same chunk count
            if face_features.shape[1] != n_chunks:
                face_t = face_features.permute(0, 2, 1)
                face_t = F.interpolate(
                    face_t, size=n_chunks, mode="linear", align_corners=True
                )
                face_features = face_t.permute(0, 2, 1)

            factors = {
                "dominant_hand": dom_pose,
                "non_dominant_hand": nondom_pose,
                "pose": body_pose,
                "face": face_features,
            }

            attended_factors = self.cross_attention(factors)

            # Update with attended versions
            pose_combined = (
                attended_factors["dominant_hand"]
                + attended_factors["non_dominant_hand"]
                + attended_factors["pose"]
            ) / 3

        # Prepare latents for quantization
        latents = {
            "pose": pose_combined,
            "motion": motion_combined,
            "dynamics": dynamics_combined,
            "face": face_features,
        }

        # Quantize
        quantized, indices, vq_losses = self.quantizers(latents, training=self.training)

        return quantized, indices, vq_losses

    def decode(
        self, quantized: Dict[str, torch.Tensor], target_len: int
    ) -> torch.Tensor:
        """
        Decode quantized representations back to landmarks.

        Args:
            quantized: Dictionary of quantized latents
            target_len: Target sequence length

        Returns:
            (B, T, N*C) reconstructed landmarks
        """
        # Map all 4 quantized codebooks into the 4 fusion slots so every
        # codebook receives gradient from the reconstruction loss.
        factors = {
            "dominant_hand": quantized["pose"],
            "non_dominant_hand": quantized["motion"],
            "pose": quantized["dynamics"],
            "face": quantized["face"],
        }
        fused = self.factor_fusion(factors)

        # Decode
        reconstructed = self.decoder(fused, target_len)

        return reconstructed

    def forward(
        self,
        landmarks: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: encode, quantize, decode.

        Args:
            landmarks: (B, T, N, C) landmarks tensor
            mask: Optional (B, T) validity mask

        Returns:
            Dictionary with:
            - 'reconstructed': (B, T, N*C) reconstructed landmarks
            - 'indices': Dictionary of codebook indices
            - 'losses': Dictionary of loss components
        """
        B, T, N, C = landmarks.shape

        # Apply augmentation during training
        if self.training:
            landmarks, mask = self.augmentation(landmarks, mask, training=True)
            T = landmarks.shape[1]  # May have changed due to speed augmentation

        # Encode and quantize
        quantized, indices, vq_losses = self.encode(landmarks, mask)

        # Decode
        reconstructed = self.decode(quantized, T)

        # Compute reconstruction loss
        landmarks_flat = landmarks.reshape(B, T, -1)
        recon_loss = F.mse_loss(reconstructed, landmarks_flat)

        # Compute velocity reconstruction loss
        target_velocity = landmarks_flat[:, 1:] - landmarks_flat[:, :-1]
        pred_velocity = reconstructed[:, 1:] - reconstructed[:, :-1]
        velocity_loss = F.mse_loss(pred_velocity, target_velocity)

        # Soft diversity loss — propagated from EMAVectorQuantizer where it is
        # computed on continuous distances, giving real gradients back to the encoder.
        soft_diversity = vq_losses["total"]["soft_diversity"]

        # Aggregate losses
        losses = {
            "reconstruction": recon_loss * self.config.reconstruction_weight,
            "velocity_reconstruction": velocity_loss
            * self.config.velocity_reconstruction_weight,
            "vq": vq_losses["total"]["vq_loss"],
            "diversity": soft_diversity * self.config.codebook_diversity_weight,
        }
        losses["total"] = sum(losses.values())

        return {
            "reconstructed": reconstructed,
            "indices": indices,
            "quantized": quantized,
            "losses": losses,
        }

    def tokenize(
        self, landmarks: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize landmarks into discrete codes.

        Args:
            landmarks: (B, T, N, C) landmarks tensor
            mask: Optional (B, T) validity mask

        Returns:
            Dictionary with indices for each factor
        """
        self.eval()
        with torch.no_grad():
            _, indices, _ = self.encode(landmarks, mask)
        return indices

    def get_codebook_embeddings(self) -> Dict[str, torch.Tensor]:
        """Get all codebook embeddings for downstream use."""
        return {
            name: quantizer.embeddings.data
            for name, quantizer in self.quantizers.quantizers.items()
        }


def create_vqvae(config: Optional[ImprovedVQVAEConfig] = None) -> ImprovedVQVAE:
    """Factory function to create VQ-VAE model."""
    return ImprovedVQVAE(config)
