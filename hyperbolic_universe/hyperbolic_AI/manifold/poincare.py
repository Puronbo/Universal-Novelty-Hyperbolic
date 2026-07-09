"""
Poincare disk geometry, implemented with PyTorch so that gradients are
computed by autograd instead of hand-rolled finite differences.
"""

from __future__ import annotations

import torch

EPS = 1e-7


def project_to_disk(x: torch.Tensor, max_norm: float = 0.999) -> torch.Tensor:
    """Clamp points to stay strictly inside the open unit disk."""
    norm = x.norm(dim=-1, keepdim=True).clamp_min(EPS)
    factor = torch.clamp(max_norm / norm, max=1.0)
    return x * factor


def geodesic_distance(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Batched geodesic distance in the Poincare disk model.
    u: (..., 2), v: (..., 2) or broadcastable to u's shape.
    Returns a tensor of shape u.shape[:-1].
    """
    sq_norm_u = (u ** 2).sum(-1)
    sq_norm_v = (v ** 2).sum(-1)
    sq_dist = ((u - v) ** 2).sum(-1)
    denom = ((1.0 - sq_norm_u) * (1.0 - sq_norm_v)).clamp_min(EPS)
    arg = (1.0 + 2.0 * sq_dist / denom).clamp_min(1.0 + EPS)
    return torch.acosh(arg)


def pairwise_geodesic_distance(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """All-pairs geodesic distance. u: (N, 2), v: (M, 2) -> (N, M)."""
    u_exp = u.unsqueeze(1)
    v_exp = v.unsqueeze(0)
    return geodesic_distance(u_exp, v_exp)


def riemannian_scale(u: torch.Tensor) -> torch.Tensor:
    """Conformal factor scaling a Euclidean gradient into the Poincare metric."""
    sq_norm = (u ** 2).sum(-1, keepdim=True)
    return ((1.0 - sq_norm) ** 2) / 4.0
