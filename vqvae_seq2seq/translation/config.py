"""Configuration for Sign Language Translation Model."""

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class TranslationConfig:
    """
    Configuration for the sign language translation model.

    Uses Conformer encoder with hybrid CTC + Attention decoder.
    """

    # Model dimensions
    d_model: int = 512
    d_ff: int = 2048
    n_heads: int = 8

    # Encoder (Conformer)
    n_encoder_layers: int = 12
    encoder_kernel_size: int = 31
    encoder_dropout: float = 0.1

    # Decoder (Attention)
    n_decoder_layers: int = 6
    decoder_dropout: float = 0.1

    # Vocabulary
    vocab_size: int = 2500  # Combined gloss vocabulary
    pad_idx: int = 0
    bos_idx: int = 1
    eos_idx: int = 2
    unk_idx: int = 3

    # Input embedding (from VQ-VAE codebooks) — must match ImprovedVQVAEConfig
    pose_codebook_size: int = 256
    motion_codebook_size: int = 256
    dynamics_codebook_size: int = 128
    face_codebook_size: int = 128
    embed_dim: int = 128  # VQ-VAE embedding dimension

    # CTC
    ctc_weight: float = 0.3
    ctc_blank_idx: int = 0  # Usually same as pad

    # Beam search
    beam_size: int = 5
    max_decode_len: int = 50
    length_penalty: float = 0.6
    ctc_prefix_weight: float = 0.4

    # Training
    learning_rate: float = 1e-4
    warmup_steps: int = 4000
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    max_epochs: int = 100
    batch_size: int = 32
    gradient_clip: float = 1.0

    # Regularization
    spec_augment: bool = True
    time_mask_max: int = 50
    time_mask_num: int = 2

    # Device
    device: str = "cuda"

    def get_total_input_tokens(self) -> int:
        """Total number of input tokens across all codebooks."""
        return (
            self.pose_codebook_size
            + self.motion_codebook_size
            + self.dynamics_codebook_size
            + self.face_codebook_size
        )
