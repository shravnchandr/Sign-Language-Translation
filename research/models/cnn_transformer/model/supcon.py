import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossSignerSupCon(nn.Module):
    """Supervised contrastive loss whose positives are the same sign performed
    by a *different* signer, with a cross-batch memory of recent embeddings.

    Classification only needs every clip on the right side of a decision
    boundary, so the model can keep one tight cluster per (sign, signer) — it
    reached 97% on its training signers vs 68% on new ones. This term pulls
    together executions of a sign by different people and pushes different
    signs apart, so new signers' clips are more likely to fall inside a shared
    per-sign region. Same-sign/same-signer pairs are neither positives nor
    negatives (pulling them together teaches nothing about other people).

    A batch of 64 clips over 250 signs rarely holds the same sign from two
    signers, so a FIFO queue of the last `queue_size` detached embeddings
    (with labels and signer ids) supplies extra candidates — "cross-batch
    memory" — without changing the length-bucketed batching. Gradients flow
    through the current batch (anchors and in-batch keys) only.

    Signer id −1 (unknown) never forms a positive. Computed in fp32.
    """

    def __init__(self, dim: int, queue_size: int = 4096, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.queue_size = queue_size
        self.register_buffer("q_z", torch.zeros(queue_size, dim))
        self.register_buffer("q_y", torch.full((queue_size,), -1, dtype=torch.long))
        self.register_buffer("q_s", torch.full((queue_size,), -1, dtype=torch.long))
        self.ptr = 0
        self.filled = 0

    @torch.no_grad()
    def _enqueue(self, z, y, s):
        n = min(z.shape[0], self.queue_size)
        z, y, s = z[-n:], y[-n:], s[-n:]
        idx = (self.ptr + torch.arange(n, device=z.device)) % self.queue_size
        self.q_z[idx], self.q_y[idx], self.q_s[idx] = z, y, s
        self.ptr = (self.ptr + n) % self.queue_size
        self.filled = min(self.filled + n, self.queue_size)

    def forward(self, z, labels, signers):
        """z: (B, dim) projections; labels, signers: (B,) long.
        Returns (loss, fraction of anchors that had ≥1 positive)."""
        z = F.normalize(z.float(), dim=-1)
        B = z.shape[0]
        keys = torch.cat([z, self.q_z[: self.filled]])
        ky = torch.cat([labels, self.q_y[: self.filled]])
        ks = torch.cat([signers, self.q_s[: self.filled]])

        sim = z @ keys.T / self.temperature  # (B, B + filled)
        is_self = torch.zeros_like(sim, dtype=torch.bool)
        is_self[:, :B] = torch.eye(B, dtype=torch.bool, device=z.device)
        same_sign = labels[:, None] == ky[None, :]
        known = (signers[:, None] >= 0) & (ks[None, :] >= 0)
        same_signer = known & (signers[:, None] == ks[None, :])
        pos = same_sign & known & ~same_signer & ~is_self
        valid = ~is_self & ~(same_sign & same_signer)

        sim = sim.masked_fill(~valid, float("-inf"))
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        has_pos = pos.any(1)
        self._enqueue(z.detach(), labels, signers)
        if not has_pos.any():
            return z.sum() * 0.0, 0.0
        per_anchor = -torch.where(pos, log_prob, torch.zeros_like(log_prob)).sum(1)
        loss = (per_anchor[has_pos] / pos.sum(1)[has_pos]).mean()
        return loss, has_pos.float().mean().item()
