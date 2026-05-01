"""Hybrid CTC + Attention decoder for sign language translation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class CTCHead(nn.Module):
    """
    CTC head for auxiliary loss and decoding.

    Provides alignment information that helps attention decoder.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        blank_idx: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blank_idx = blank_idx

        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_size),
        )

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_output: (B, T, D) encoder output

        Returns:
            (B, T, vocab_size) log probabilities
        """
        logits = self.projection(encoder_output)
        return F.log_softmax(logits, dim=-1)

    def compute_loss(
        self,
        encoder_output: torch.Tensor,
        targets: torch.Tensor,
        encoder_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CTC loss.

        Args:
            encoder_output: (B, T, D)
            targets: (B, S) target sequences (without special tokens)
            encoder_lengths: (B,) lengths of encoder outputs
            target_lengths: (B,) lengths of targets

        Returns:
            CTC loss scalar
        """
        log_probs = self(encoder_output)  # (B, T, V)
        log_probs = log_probs.permute(1, 0, 2)  # (T, B, V) for CTC

        # Cast to float32 — F.ctc_loss is numerically unstable in FP16 under AMP
        loss = F.ctc_loss(
            log_probs.float(),
            targets,
            encoder_lengths,
            target_lengths,
            blank=self.blank_idx,
            zero_infinity=True,
        )

        return loss


class DecoderLayer(nn.Module):
    """
    Single decoder layer with self-attention and cross-attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention (masked)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn_dropout = nn.Dropout(dropout)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_dropout = nn.Dropout(dropout)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T_dec, D) decoder input
            encoder_output: (B, T_enc, D) encoder output
            self_attn_mask: (T_dec, T_dec) causal mask
            memory_key_padding_mask: (B, T_enc) encoder padding mask

        Returns:
            (B, T_dec, D) decoder output
        """
        # Self-attention with causal mask
        x_norm = self.self_attn_norm(x)
        self_attn_out, _ = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=self_attn_mask,
            need_weights=False,
        )
        x = x + self.self_attn_dropout(self_attn_out)

        # Cross-attention
        x_norm = self.cross_attn_norm(x)
        cross_attn_out, _ = self.cross_attn(
            x_norm,
            encoder_output,
            encoder_output,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        x = x + self.cross_attn_dropout(cross_attn_out)

        # Feed-forward
        x_norm = self.ffn_norm(x)
        x = x + self.ffn(x_norm)

        return x


class AttentionDecoder(nn.Module):
    """
    Autoregressive attention decoder.

    Generates output tokens one by one, attending to encoder output.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        vocab_size: int = 2500,
        pad_idx: int = 0,
        dropout: float = 0.1,
        max_len: int = 100,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # Positional encoding
        self.pos_encoding = self._create_pos_encoding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Decoder layers
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        self.layer_norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Tie embedding weights
        self.output_projection.weight = self.embedding.weight

        self.scale = math.sqrt(d_model)

        # Register causal mask buffer
        self.register_buffer("causal_mask", self._create_causal_mask(max_len))

    def _create_pos_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _create_causal_mask(self, size: int) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask

    def forward(
        self,
        decoder_input: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            decoder_input: (B, T_dec) target tokens (shifted right)
            encoder_output: (B, T_enc, D) encoder output
            encoder_mask: (B, T_enc) encoder padding mask

        Returns:
            (B, T_dec, vocab_size) output logits
        """
        B, T = decoder_input.shape

        # Embed tokens
        x = self.embedding(decoder_input) * self.scale

        # Add positional encoding
        x = x + self.pos_encoding[:, :T].to(x.device)
        x = self.dropout(x)

        # Get causal mask
        causal_mask = self.causal_mask[:T, :T]

        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, causal_mask, encoder_mask)

        x = self.layer_norm(x)

        # Project to vocabulary
        logits = self.output_projection(x)

        return logits

    def compute_loss(
        self,
        encoder_output: torch.Tensor,
        targets: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for training.

        Args:
            encoder_output: (B, T_enc, D)
            targets: (B, T_dec) target tokens including BOS
            encoder_mask: Optional encoder padding mask
            label_smoothing: Label smoothing factor

        Returns:
            Cross-entropy loss scalar
        """
        # Input is targets[:-1], output should predict targets[1:]
        decoder_input = targets[:, :-1]
        target_output = targets[:, 1:]

        logits = self(decoder_input, encoder_output, encoder_mask)

        # Flatten for loss computation
        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = target_output.reshape(-1)

        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.pad_idx,
            label_smoothing=label_smoothing,
        )

        return loss


class HybridDecoder(nn.Module):
    """
    Hybrid CTC + Attention decoder.

    Combines CTC (for alignment) and attention (for quality).
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        vocab_size: int = 2500,
        pad_idx: int = 0,
        blank_idx: int = 0,
        ctc_weight: float = 0.3,
        dropout: float = 0.1,
        label_smoothing: float = 0.1,
    ):
        super().__init__()

        self.ctc_weight = ctc_weight
        self.label_smoothing = label_smoothing

        # CTC head
        self.ctc = CTCHead(d_model, vocab_size, blank_idx, dropout)

        # Attention decoder
        self.attention = AttentionDecoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            dropout=dropout,
        )

    def forward(
        self,
        encoder_output: torch.Tensor,
        decoder_input: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_output: (B, T_enc, D)
            decoder_input: (B, T_dec) target tokens for teacher forcing
            encoder_mask: Optional encoder padding mask

        Returns:
            Tuple of (attention logits, ctc log probs)
        """
        attn_logits = self.attention(decoder_input, encoder_output, encoder_mask)
        ctc_log_probs = self.ctc(encoder_output)

        return attn_logits, ctc_log_probs

    def compute_loss(
        self,
        encoder_output: torch.Tensor,
        targets: torch.Tensor,
        encoder_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hybrid loss.

        Args:
            encoder_output: (B, T_enc, D)
            targets: (B, T_dec) targets with BOS token
            encoder_lengths: (B,) encoder output lengths
            target_lengths: (B,) target lengths (without BOS/EOS)
            encoder_mask: Optional encoder padding mask

        Returns:
            Dictionary with 'total', 'ctc', 'attention' losses
        """
        # Attention loss (targets include BOS)
        attn_loss = self.attention.compute_loss(
            encoder_output,
            targets,
            encoder_mask,
            self.label_smoothing,
        )

        # CTC loss (targets without BOS/EOS)
        # Remove BOS from targets for CTC
        ctc_targets = targets[:, 1:]  # Remove BOS
        # Also need to adjust lengths and remove padding/EOS
        ctc_loss = self.ctc.compute_loss(
            encoder_output,
            ctc_targets,
            encoder_lengths,
            target_lengths,
        )

        # Combine losses
        total_loss = (1 - self.ctc_weight) * attn_loss + self.ctc_weight * ctc_loss

        return {
            "total": total_loss,
            "ctc": ctc_loss,
            "attention": attn_loss,
        }
