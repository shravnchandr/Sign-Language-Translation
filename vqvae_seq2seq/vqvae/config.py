"""Configuration for Improved VQ-VAE."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ImprovedVQVAEConfig:
    """
    Configuration for the improved VQ-VAE tokenizer.

    Addresses blindspots:
    - #1: Z-coordinate included (3D input)
    - #2: Dedicated face encoder
    - #3: Multi-scale temporal encoding
    - #10: Increased codebook sizes
    """

    # Input dimensions (3D coordinates)
    hand_dim: int = 63  # 21 landmarks * 3 coords
    pose_dim: int = 99  # 33 landmarks * 3 coords
    face_dim: int = 402  # 134 landmarks * 3 coords (compact subset)
    n_coords: int = 3  # x, y, z

    # Landmark counts
    hand_landmarks: int = 21
    pose_landmarks: int = 33
    face_landmarks: int = 134  # Compact subset

    # Multi-scale temporal encoding (blindspot #3)
    temporal_scales: Tuple[int, ...] = (4, 8, 16)
    base_chunk_size: int = 8  # Reference chunk size

    # Encoder architecture
    encoder_hidden_dim: int = 256
    encoder_n_layers: int = 3
    encoder_dropout: float = 0.1

    # Codebook sizes — sized to ~2x observed utilization at epoch 3
    pose_codebook_size: int = 256
    motion_codebook_size: int = 256
    dynamics_codebook_size: int = 128
    face_codebook_size: int = 128

    # Embedding dimensions
    embed_dim: int = 128  # Codebook embedding dimension
    latent_dim: int = 256  # Encoder output dimension

    # Vector quantizer settings
    commitment_weight: float = 0.25
    ema_decay: float = 0.97
    codebook_reset_threshold: float = 0.001
    codebook_reset_patience: int = 50
    codebook_reset_warmdown_ratio: float = 0.8  # disable resets after this fraction of training

    # Cross-factor attention (blindspot #8)
    use_cross_attention: bool = True
    cross_attention_heads: int = 4
    cross_attention_layers: int = 2

    # Face NMM encoder (blindspot #2)
    face_region_dims: dict = field(
        default_factory=lambda: {
            "eyebrows": 16 * 3,  # 16 landmarks * 3 coords
            "eyes": 32 * 3,
            "nose": 10 * 3,
            "mouth": 40 * 3,
            "face_oval": 36 * 3,
        }
    )

    # Decoder architecture
    decoder_hidden_dim: int = 256
    decoder_n_layers: int = 3

    # Training settings
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 100

    # Loss weights
    reconstruction_weight: float = 1.0
    velocity_reconstruction_weight: float = 0.5
    codebook_diversity_weight: float = 0.2

    # Augmentation (blindspot #9)
    augment_speed_range: Tuple[float, float] = (0.8, 1.2)
    augment_frame_dropout_prob: float = 0.1
    augment_temporal_jitter_std: float = 0.02

    # Device
    device: str = "cuda"

    def get_total_landmarks(self) -> int:
        """Get total number of landmarks across all body parts."""
        return (
            self.hand_landmarks * 2  # Both hands
            + self.pose_landmarks
            + self.face_landmarks
        )

    def get_total_input_dim(self) -> int:
        """Get total input dimension (flattened)."""
        return self.get_total_landmarks() * self.n_coords
