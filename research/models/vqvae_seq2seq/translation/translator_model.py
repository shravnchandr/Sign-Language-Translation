"""Complete Sign Language Translator model."""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple

from .config import TranslationConfig
from .token_embedding import (
    FactorizedTokenEmbedding,
    DirectLandmarkEmbedding,
    GlossEmbedding,
)
from .conformer import ConformerEncoder, SpecAugment
from .decoder import HybridDecoder
from .beam_search import BeamSearch, GreedyDecoder


class SignTranslator(nn.Module):
    """
    Complete sign language translator model.

    Architecture:
    - Input: Factorized VQ-VAE tokens (pose, motion, dynamics, face)
    - Encoder: Conformer
    - Decoder: Hybrid CTC + Attention
    - Output: Gloss sequence
    """

    def __init__(self, config: Optional[TranslationConfig] = None):
        super().__init__()
        self.config = config or TranslationConfig()

        # Input embedding (from VQ-VAE tokens)
        self.token_embedding = FactorizedTokenEmbedding(
            pose_codebook_size=self.config.pose_codebook_size,
            motion_codebook_size=self.config.motion_codebook_size,
            dynamics_codebook_size=self.config.dynamics_codebook_size,
            face_codebook_size=self.config.face_codebook_size,
            embed_dim=self.config.embed_dim,
            d_model=self.config.d_model,
            dropout=self.config.encoder_dropout,
        )

        # SpecAugment for training
        self.spec_augment = (
            SpecAugment(
                time_mask_max=self.config.time_mask_max,
                time_mask_num=self.config.time_mask_num,
            )
            if self.config.spec_augment
            else None
        )

        # Conformer encoder
        self.encoder = ConformerEncoder(
            d_model=self.config.d_model,
            d_ff=self.config.d_ff,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_encoder_layers,
            kernel_size=self.config.encoder_kernel_size,
            dropout=self.config.encoder_dropout,
        )

        # Hybrid decoder
        self.decoder = HybridDecoder(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_decoder_layers,
            d_ff=self.config.d_ff,
            vocab_size=self.config.vocab_size,
            pad_idx=self.config.pad_idx,
            blank_idx=self.config.ctc_blank_idx,
            ctc_weight=self.config.ctc_weight,
            dropout=self.config.decoder_dropout,
            label_smoothing=self.config.label_smoothing,
        )

        # Beam search decoder
        self.beam_search = BeamSearch(
            beam_size=self.config.beam_size,
            max_len=self.config.max_decode_len,
            eos_idx=self.config.eos_idx,
            bos_idx=self.config.bos_idx,
            pad_idx=self.config.pad_idx,
            length_penalty=self.config.length_penalty,
            ctc_weight=self.config.ctc_prefix_weight,
        )

        # Greedy decoder for fast inference
        self.greedy_decoder = GreedyDecoder(
            max_len=self.config.max_decode_len,
            eos_idx=self.config.eos_idx,
            bos_idx=self.config.bos_idx,
        )

    def init_from_vqvae(self, vqvae_codebooks: Dict[str, torch.Tensor]):
        """
        Initialize token embeddings from pre-trained VQ-VAE codebooks.

        Args:
            vqvae_codebooks: Dictionary with 'pose', 'motion', 'dynamics', 'face' codebooks
        """
        self.token_embedding.init_from_codebooks(vqvae_codebooks)

    def encode(
        self,
        token_indices: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode input tokens.

        Args:
            token_indices: Dictionary with 'pose', 'motion', 'dynamics', 'face' indices
            mask: Optional (B, T) padding mask

        Returns:
            (B, T, d_model) encoder output
        """
        # Embed tokens
        x = self.token_embedding.forward_dict(token_indices)

        # Apply SpecAugment during training
        if self.training and self.spec_augment is not None:
            x = self.spec_augment(x)

        # Encode
        encoder_output = self.encoder(x, mask)

        return encoder_output

    def forward(
        self,
        token_indices: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        encoder_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            token_indices: Dictionary with 'pose', 'motion', 'dynamics', 'face' indices
            targets: (B, T_dec) target tokens with BOS
            encoder_mask: Optional (B, T_enc) encoder padding mask
            encoder_lengths: (B,) encoder output lengths for CTC
            target_lengths: (B,) target lengths for CTC

        Returns:
            Dictionary with losses
        """
        # Encode
        encoder_output = self.encode(token_indices, encoder_mask)

        # Infer lengths if not provided
        if encoder_lengths is None:
            if encoder_mask is not None:
                encoder_lengths = (~encoder_mask).sum(dim=1)
            else:
                encoder_lengths = torch.full(
                    (encoder_output.shape[0],),
                    encoder_output.shape[1],
                    device=encoder_output.device,
                )

        if target_lengths is None:
            # Assume targets are padded with pad_idx
            target_lengths = (targets != self.config.pad_idx).sum(
                dim=1
            ) - 1  # Exclude BOS

        # Compute loss
        losses = self.decoder.compute_loss(
            encoder_output,
            targets,
            encoder_lengths,
            target_lengths,
            encoder_mask,
        )

        return losses

    @torch.no_grad()
    def translate(
        self,
        token_indices: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        use_beam_search: bool = True,
    ) -> List[List[int]]:
        """
        Translate input tokens to gloss sequences.

        Args:
            token_indices: Dictionary with 'pose', 'motion', 'dynamics', 'face' indices
            mask: Optional (B, T) padding mask
            use_beam_search: Whether to use beam search (slower but better)

        Returns:
            List of decoded sequences (one per batch item)
        """
        self.eval()
        B = token_indices["pose"].shape[0]

        # Encode
        encoder_output = self.encode(token_indices, mask)

        # Get CTC log probs for beam search rescoring
        ctc_log_probs = self.decoder.ctc(encoder_output)

        # Decode each sequence
        results = []
        for b in range(B):
            enc_out_b = encoder_output[b : b + 1]
            ctc_probs_b = ctc_log_probs[b : b + 1]

            if use_beam_search:
                decoded = self.beam_search.search(
                    enc_out_b,
                    self.decoder.attention,
                    ctc_probs_b,
                )
                results.append(decoded[0] if decoded else [])
            else:
                decoded = self.greedy_decoder.decode(enc_out_b, self.decoder.attention)
                results.append(decoded)

        return results

    @torch.no_grad()
    def translate_batch_greedy(
        self,
        token_indices: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Greedy translation for a batch (faster for evaluation).

        Args:
            token_indices: Dictionary with token indices
            mask: Optional padding mask

        Returns:
            (B, max_len) decoded token indices
        """
        self.eval()
        B = token_indices["pose"].shape[0]
        device = token_indices["pose"].device

        # Encode
        encoder_output = self.encode(token_indices, mask)

        # Initialize output
        outputs = torch.full(
            (B, self.config.max_decode_len),
            self.config.pad_idx,
            dtype=torch.long,
            device=device,
        )
        outputs[:, 0] = self.config.bos_idx

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(1, self.config.max_decode_len):
            # Get predictions
            logits = self.decoder.attention(outputs[:, :t], encoder_output, mask)
            next_tokens = logits[:, -1].argmax(dim=-1)

            # Update outputs
            outputs[:, t] = next_tokens

            # Check for EOS
            finished = finished | (next_tokens == self.config.eos_idx)
            if finished.all():
                break

        return outputs


class SignTranslatorFromLandmarks(nn.Module):
    """
    End-to-end translator that works directly from landmarks.

    Useful when VQ-VAE tokenization isn't available.
    """

    def __init__(
        self,
        config: Optional[TranslationConfig] = None,
        input_dim: int = 627,  # 209 landmarks * 3 coords
    ):
        super().__init__()
        self.config = config or TranslationConfig()

        # Direct landmark embedding
        self.input_embedding = DirectLandmarkEmbedding(
            input_dim=input_dim,
            d_model=self.config.d_model,
            dropout=self.config.encoder_dropout,
        )

        # Conformer encoder
        self.encoder = ConformerEncoder(
            d_model=self.config.d_model,
            d_ff=self.config.d_ff,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_encoder_layers,
            kernel_size=self.config.encoder_kernel_size,
            dropout=self.config.encoder_dropout,
        )

        # Hybrid decoder
        self.decoder = HybridDecoder(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_decoder_layers,
            d_ff=self.config.d_ff,
            vocab_size=self.config.vocab_size,
            pad_idx=self.config.pad_idx,
            blank_idx=self.config.ctc_blank_idx,
            ctc_weight=self.config.ctc_weight,
            dropout=self.config.decoder_dropout,
        )

    def forward(
        self,
        landmarks: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        encoder_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            landmarks: (B, T, D) landmark features
            targets: (B, T_dec) target tokens
            mask: Optional padding mask
            encoder_lengths: Encoder lengths for CTC
            target_lengths: Target lengths for CTC

        Returns:
            Dictionary with losses
        """
        # Embed and encode
        x = self.input_embedding(landmarks)
        encoder_output = self.encoder(x, mask)

        # Handle lengths
        if encoder_lengths is None:
            encoder_lengths = torch.full(
                (landmarks.shape[0],),
                encoder_output.shape[1],
                device=landmarks.device,
            )

        if target_lengths is None:
            target_lengths = (targets != self.config.pad_idx).sum(dim=1) - 1

        # Compute loss
        losses = self.decoder.compute_loss(
            encoder_output,
            targets,
            encoder_lengths,
            target_lengths,
            mask,
        )

        return losses


def create_translator(
    config: Optional[TranslationConfig] = None,
    from_landmarks: bool = False,
    input_dim: int = 627,
) -> nn.Module:
    """Factory function to create translator model."""
    if from_landmarks:
        return SignTranslatorFromLandmarks(config, input_dim)
    return SignTranslator(config)
