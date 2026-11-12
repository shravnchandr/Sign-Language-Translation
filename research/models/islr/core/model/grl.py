import math
import torch
import torch.nn as nn


class _GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lam, None


def grad_reverse(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    return _GradRev.apply(x, lam)


class SignerDiscriminator(nn.Module):
    """MLP discriminator head with gradient reversal for signer-invariant training.

    GRL negates gradients flowing back through the feature extractor, pushing it
    to produce representations that are maximally uninformative about signer identity
    while remaining discriminative for sign classification.
    """

    def __init__(self, d_model: int, n_signers: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, n_signers),
        )

    def forward(self, x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        return self.net(grad_reverse(x, lam))


def ganin_lambda(epoch: int, total_epochs: int, max_lambda: float = 1.0) -> float:
    """Ganin et al. 2016 ramp: near 0 at start, ~max_lambda by mid-training."""
    p = epoch / max(total_epochs - 1, 1)
    return max_lambda * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
