"""Vector Quantizer with EMA updates, codebook reset, and diversity loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class VectorQuantizer(nn.Module):
    """
    Basic Vector Quantizer with straight-through gradient estimation.

    Maps continuous latent vectors to discrete codebook entries.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_weight: float = 0.25,
    ):
        """
        Args:
            num_embeddings: Size of the codebook
            embedding_dim: Dimension of each codebook vector
            commitment_weight: Weight for commitment loss
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight

        # Initialize codebook
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(
            -1.0 / num_embeddings, 1.0 / num_embeddings
        )

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Quantize input latent vectors.

        Args:
            z: (B, ..., D) input latent vectors

        Returns:
            z_q: Quantized vectors (same shape as z)
            indices: Codebook indices (B, ...)
            losses: Dictionary with 'codebook_loss' and 'commitment_loss'
        """
        # Save input shape
        input_shape = z.shape
        B = input_shape[0]
        D = input_shape[-1]

        # Flatten to (N, D)
        z_flat = z.reshape(-1, D)
        N = z_flat.shape[0]

        # Compute distances to all codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z.e
        z_sq = (z_flat**2).sum(dim=1, keepdim=True)  # (N, 1)
        e_sq = (self.embeddings.weight**2).sum(dim=1)  # (K,)
        ze = torch.matmul(z_flat, self.embeddings.weight.t())  # (N, K)

        distances = z_sq + e_sq - 2 * ze  # (N, K)

        # Get nearest codebook entries
        indices = distances.argmin(dim=1)  # (N,)

        # Get quantized vectors
        z_q_flat = self.embeddings(indices)  # (N, D)

        # Reshape back to original shape
        z_q = z_q_flat.reshape(input_shape)
        indices = indices.reshape(input_shape[:-1])

        # Compute losses
        codebook_loss = F.mse_loss(z_q.detach(), z)  # Move z towards codebook
        commitment_loss = F.mse_loss(z_q, z.detach())  # Move codebook towards z

        # Straight-through gradient
        z_q = z + (z_q - z).detach()

        losses = {
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss * self.commitment_weight,
            "vq_loss": codebook_loss + commitment_loss * self.commitment_weight,
        }

        return z_q, indices, losses

    def get_codebook_usage(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute histogram of codebook usage."""
        return torch.bincount(indices.flatten(), minlength=self.num_embeddings).float()

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode indices back to embeddings."""
        return self.embeddings(indices)


class EMAVectorQuantizer(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average updates.

    More stable than gradient-based updates and doesn't require
    the codebook loss term.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
        reset_threshold: float = 0.01,
        reset_patience: int = 100,
        epsilon: float = 1e-5,
    ):
        """
        Args:
            num_embeddings: Size of the codebook
            embedding_dim: Dimension of each codebook vector
            commitment_weight: Weight for commitment loss
            ema_decay: Decay rate for EMA updates
            reset_threshold: Usage threshold below which codes are reset
            reset_patience: Steps before considering resetting codes
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        self.ema_decay = ema_decay
        self.reset_threshold = reset_threshold
        self.reset_patience = reset_patience
        self.epsilon = epsilon

        # Initialize codebook
        self.register_buffer("embeddings", torch.randn(num_embeddings, embedding_dim))
        nn.init.xavier_uniform_(self.embeddings)

        # EMA tracking
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_embed_sum", self.embeddings.clone())

        # Usage tracking for reset
        self.register_buffer("usage_count", torch.zeros(num_embeddings))
        self.register_buffer("steps_since_reset", torch.tensor(0))

    def forward(
        self, z: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Quantize input latent vectors with EMA updates.

        Args:
            z: (B, ..., D) input latent vectors
            training: Whether in training mode (enables EMA updates)

        Returns:
            z_q: Quantized vectors
            indices: Codebook indices
            losses: Dictionary with loss terms
        """
        input_shape = z.shape
        D = input_shape[-1]

        # Flatten
        z_flat = z.reshape(-1, D)
        N = z_flat.shape[0]

        # Compute distances
        z_sq = (z_flat**2).sum(dim=1, keepdim=True)
        e_sq = (self.embeddings**2).sum(dim=1)
        ze = torch.matmul(z_flat, self.embeddings.t())
        distances = z_sq + e_sq - 2 * ze

        # Get nearest codes
        indices = distances.argmin(dim=1)
        z_q_flat = self.embeddings[indices]

        # EMA update during training
        if training and self.training:
            self._ema_update(z_flat, indices)

        # Reshape outputs
        z_q = z_q_flat.reshape(input_shape)
        indices = indices.reshape(input_shape[:-1])

        # Commitment loss only (codebook updated via EMA)
        commitment_loss = F.mse_loss(z_q.detach(), z)

        # Straight-through gradient
        z_q = z + (z_q - z).detach()

        losses = {
            "commitment_loss": commitment_loss * self.commitment_weight,
            "vq_loss": commitment_loss * self.commitment_weight,
        }

        return z_q, indices, losses

    def _ema_update(self, z_flat: torch.Tensor, indices: torch.Tensor):
        """Update codebook using EMA."""
        N = z_flat.shape[0]

        # One-hot encode indices
        one_hot = F.one_hot(indices, self.num_embeddings).float()  # (N, K)

        # Update cluster sizes
        cluster_size = one_hot.sum(dim=0)
        self.ema_cluster_size.mul_(self.ema_decay).add_(
            cluster_size, alpha=1 - self.ema_decay
        )

        # Update embedding sums
        embed_sum = torch.matmul(one_hot.t(), z_flat)  # (K, D)
        self.ema_embed_sum.mul_(self.ema_decay).add_(
            embed_sum, alpha=1 - self.ema_decay
        )

        # Normalize to get new embeddings
        n = self.ema_cluster_size.unsqueeze(1)
        self.embeddings.copy_(self.ema_embed_sum / (n + self.epsilon))

        # Track usage
        self.usage_count.add_(cluster_size)
        self.steps_since_reset.add_(1)

        # Check for reset
        if self.steps_since_reset >= self.reset_patience:
            self._maybe_reset_codes(z_flat)

    def _maybe_reset_codes(self, z_flat: torch.Tensor):
        """Reset underused codes."""
        # Normalize usage counts
        total_usage = self.usage_count.sum()
        if total_usage == 0:
            return

        usage_rate = self.usage_count / total_usage

        # Find underused codes
        underused = usage_rate < self.reset_threshold

        if underused.any():
            # Sample random vectors from the input to reinitialize
            num_reset = underused.sum().item()
            random_indices = torch.randint(
                0, z_flat.shape[0], (num_reset,), device=z_flat.device
            )
            new_embeddings = z_flat[random_indices]

            # Add small noise for diversity
            noise = torch.randn_like(new_embeddings) * 0.01
            new_embeddings = new_embeddings + noise

            # Reset underused codes
            self.embeddings[underused] = new_embeddings
            self.ema_embed_sum[underused] = new_embeddings
            self.ema_cluster_size[underused] = 1.0

        # Reset tracking
        self.usage_count.zero_()
        self.steps_since_reset.zero_()

    def get_codebook_usage(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute histogram of codebook usage."""
        return torch.bincount(indices.flatten(), minlength=self.num_embeddings).float()

    def get_perplexity(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute perplexity (measure of codebook utilization)."""
        usage = self.get_codebook_usage(indices)
        probs = usage / usage.sum()
        probs = probs + 1e-10  # Avoid log(0)
        entropy = -(probs * probs.log()).sum()
        perplexity = entropy.exp()
        return perplexity

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode indices back to embeddings."""
        return self.embeddings[indices]


class DiversityLoss(nn.Module):
    """
    Loss to encourage diverse codebook usage.

    Penalizes uneven distribution of codes.
    """

    def __init__(self, target_perplexity_ratio: float = 0.8):
        """
        Args:
            target_perplexity_ratio: Target perplexity as ratio of codebook size
        """
        super().__init__()
        self.target_perplexity_ratio = target_perplexity_ratio

    def forward(self, indices: torch.Tensor, num_embeddings: int) -> torch.Tensor:
        """
        Compute diversity loss.

        Args:
            indices: Codebook indices from quantization
            num_embeddings: Size of the codebook

        Returns:
            Diversity loss (higher when usage is uneven)
        """
        # Compute usage distribution
        usage = torch.bincount(indices.flatten(), minlength=num_embeddings).float()
        probs = usage / (usage.sum() + 1e-10)

        # Target is uniform distribution
        target_probs = torch.ones_like(probs) / num_embeddings

        # KL divergence from uniform
        kl_div = F.kl_div((probs + 1e-10).log(), target_probs, reduction="sum")

        return kl_div


class FactorizedVectorQuantizer(nn.Module):
    """
    Factorized quantizer with multiple codebooks.

    Uses separate codebooks for different factors (pose, motion, dynamics, face).
    """

    def __init__(
        self,
        codebook_configs: Dict[str, Tuple[int, int]],  # {name: (num_codes, dim)}
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()

        self.codebook_names = list(codebook_configs.keys())
        self.quantizers = nn.ModuleDict()

        for name, (num_codes, dim) in codebook_configs.items():
            self.quantizers[name] = EMAVectorQuantizer(
                num_embeddings=num_codes,
                embedding_dim=dim,
                commitment_weight=commitment_weight,
                ema_decay=ema_decay,
            )

    def forward(
        self, latents: Dict[str, torch.Tensor], training: bool = True
    ) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]
    ]:
        """
        Quantize multiple latent factors.

        Args:
            latents: Dictionary mapping factor names to latent tensors
            training: Whether in training mode

        Returns:
            quantized: Dictionary of quantized tensors
            indices: Dictionary of codebook indices
            losses: Dictionary of losses per factor
        """
        quantized = {}
        indices = {}
        losses = {}

        for name in self.codebook_names:
            if name in latents:
                z_q, idx, loss = self.quantizers[name](latents[name], training)
                quantized[name] = z_q
                indices[name] = idx
                losses[name] = loss

        # Aggregate losses
        total_loss = sum(l["vq_loss"] for l in losses.values())
        losses["total"] = {"vq_loss": total_loss}

        return quantized, indices, losses

    def get_all_codebook_usage(
        self, indices: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Get usage statistics for all codebooks."""
        usage = {}
        for name, idx in indices.items():
            if name in self.quantizers:
                usage[name] = self.quantizers[name].get_codebook_usage(idx)
        return usage
