import torch
import torch.nn as nn
import torch.nn.functional as F


class EMAQuantizer(nn.Module):
    """Vector quantizer with an EMA-updated codebook (no codebook gradient).

    Straight-through estimator for the encoder, commitment loss
    beta·||z − sg(q)||², data-dependent initialisation from the first training
    batch, and periodic reset of dead codes to random encoder outputs. All
    codebook state is buffers, so it is checkpointed and — like BatchNorm
    running stats — updated only in training mode while `update` is True.
    Distances and EMA updates run in fp32 regardless of autocast.
    """

    def __init__(self, dim, n_codes=256, beta=0.25, decay=0.99, eps=1e-5,
                 reset_every=200, dead_threshold=1e-2):
        super().__init__()
        self.n_codes, self.beta, self.decay, self.eps = n_codes, beta, decay, eps
        self.reset_every, self.dead_threshold = reset_every, dead_threshold
        self.update = True
        # Zeros, not random: the first training batch initialises the codebook,
        # and drawing from the global RNG here would shift every later layer's
        # initial weights relative to a run without VQ.
        self.register_buffer("codebook", torch.zeros(n_codes, dim))
        self.register_buffer("ema_count", torch.ones(n_codes))
        self.register_buffer("ema_sum", self.codebook.clone())
        self.register_buffer("ema_steps", torch.zeros((), dtype=torch.long))

    def forward(self, z):
        """z: (N, dim) valid vectors → (z_q straight-through, commit loss, idx, perplexity)."""
        with torch.autocast(device_type=z.device.type, enabled=False):
            zf = z.float()
            if self.training and self.update and zf.shape[0] > 0:
                self._maybe_init(zf)
            cb = self.codebook
            d = zf.pow(2).sum(1, keepdim=True) - 2 * zf @ cb.T + cb.pow(2).sum(1)
            idx = d.argmin(1)
            q = cb[idx]
            onehot = F.one_hot(idx, self.n_codes).float()
            usage = onehot.mean(0)
            perplexity = torch.exp(-(usage * torch.log(usage + 1e-10)).sum())
            if self.training and self.update and zf.shape[0] > 0:
                self._ema_update(zf, onehot)
            commit = self.beta * F.mse_loss(zf, q.detach())
        # Straight-through in training; in eval the code itself, so the output is
        # exactly a function of the code (z + (q − z) differs from q by rounding).
        z_q = z + (q.to(z.dtype) - z).detach() if self.training else q.to(z.dtype)
        return z_q, commit, idx, perplexity

    @torch.no_grad()
    def _maybe_init(self, zf):
        if self.ema_steps.item() == 0:
            # Distinct frames when the batch has enough (no exact duplicate codes);
            # near-duplicates (adjacent frames) go unused and are reset as dead codes.
            n = zf.shape[0]
            sel = (torch.randperm(n, device=zf.device)[: self.n_codes] if n >= self.n_codes
                   else torch.randint(0, n, (self.n_codes,), device=zf.device))
            pick = zf[sel]
            self.codebook.copy_(pick)
            self.ema_sum.copy_(pick)
            self.ema_count.fill_(1.0)

    @torch.no_grad()
    def _ema_update(self, zf, onehot):
        self.ema_count.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
        self.ema_sum.mul_(self.decay).add_(onehot.T @ zf, alpha=1 - self.decay)
        n = self.ema_count.sum()
        count = (self.ema_count + self.eps) / (n + self.n_codes * self.eps) * n  # Laplace
        self.codebook.copy_(self.ema_sum / count.unsqueeze(1))
        self.ema_steps.add_(1)
        if self.ema_steps.item() % self.reset_every == 0:
            dead = self.ema_count < self.dead_threshold
            if dead.any():
                pick = zf[torch.randint(0, zf.shape[0], (int(dead.sum()),), device=zf.device)]
                self.codebook[dead] = pick
                self.ema_sum[dead] = pick
                self.ema_count[dead] = 1.0


class PartVQ(nn.Module):
    """Factorised per-frame bottleneck: one EMAQuantizer per body-part group.

    groups: {name: list of feature indices into the (B, T, D) input}. Only valid
    frames (mask True) are quantised and update the codebooks; padded frames are
    passed through untouched (they are masked downstream). After each forward,
    `aux_loss` holds the summed commitment loss and `stats` the per-group code
    perplexity (effective number of codes used in the batch).
    """

    def __init__(self, groups: dict, n_codes=256, beta=0.25, decay=0.99):
        super().__init__()
        self.names = list(groups)
        for name, idx in groups.items():
            self.register_buffer(f"idx_{name}", torch.tensor(idx, dtype=torch.long), persistent=False)
        self.quantizers = nn.ModuleDict(
            {n: EMAQuantizer(len(i), n_codes, beta, decay) for n, i in groups.items()}
        )
        self.aux_loss = None
        self.stats = {}

    def forward(self, x, mask):
        out = x.clone()
        valid = x[mask]  # (N, D)
        aux = x.new_zeros((), dtype=torch.float32)
        for n in self.names:
            idx = getattr(self, f"idx_{n}")
            z_q, commit, _, ppl = self.quantizers[n](valid[:, idx])
            valid[:, idx] = z_q
            aux = aux + commit
            self.stats[n] = ppl.detach()
        out[mask] = valid
        self.aux_loss = aux
        return out
