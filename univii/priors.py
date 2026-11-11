"""Learned priors over the latent space.

    MixturePrior   mixture of components with learned locations and widths, in the Euclidean
                   block, the hyperbolic block, or both; optionally with components tied to
                   labels and arranged like a label tree

The model owns one as ``model.mixture_prior`` when ``UniVIConfig(prior="mog")``.
In the hyperbolic block a component is a wrapped normal centred at a learned
point, so the prototypes themselves live in, and are learned in, hyperbolic space.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOG_2PI = math.log(2.0 * math.pi)


class MixturePrior(nn.Module):
    """``p(z) = sum_k pi_k p_k(z)`` with learned weights, component centres and widths.

    ``dim`` Euclidean coordinates come first, then ``hyperbolic_dim`` hyperbolic
    ones (tangent coordinates at the origin), matching the model's latent layout.
    This class evaluates the Euclidean part itself; the model supplies the
    hyperbolic part, which needs the latent geometry.
    """

    def __init__(self, n_components: int, dim: int, hyperbolic_dim: int = 0, init_scale: float = 1.0):
        super().__init__()
        self.n_components, self.dim, self.hyperbolic_dim = int(n_components), int(dim), int(hyperbolic_dim)
        total = self.dim + self.hyperbolic_dim
        means = init_scale * torch.randn(self.n_components, total)
        means[:, self.dim:] *= 0.3                      # hyperbolic centres start near the origin
        self.means = nn.Parameter(means)
        self.logvars = nn.Parameter(torch.zeros(self.n_components, total))
        self.logits = nn.Parameter(torch.zeros(self.n_components))

    def component_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Euclidean part of ``log p_k(z)`` for every component: ``(B, K)``. ``z`` holds the
        Euclidean coordinates only."""
        if self.dim == 0:
            return torch.zeros(z.shape[0], self.n_components, dtype=z.dtype, device=z.device)
        diff = z[:, :self.dim].unsqueeze(1) - self.means[:, :self.dim].unsqueeze(0)
        lv = self.logvars[:, :self.dim].unsqueeze(0)
        return (-0.5 * (diff ** 2 / lv.exp() + lv + _LOG_2PI)).sum(-1)

    def mix(self, comp: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Combine per-component log densities ``(B, K)`` into ``log p(z)`` per cell.

        Cells with a valid label (``0 <= label < K``) are scored under that
        component alone, which ties components to classes; all others are
        scored under the full mixture.
        """
        mix = torch.logsumexp(comp + F.log_softmax(self.logits, dim=0), dim=1)
        if labels is None:
            return mix
        labels = torch.as_tensor(labels).long().view(-1).to(comp.device)
        valid = (labels >= 0) & (labels < self.n_components)
        own = comp.gather(1, labels.clamp(0, self.n_components - 1).unsqueeze(1)).squeeze(1)
        return torch.where(valid, own, mix)

    def log_prob(self, z: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``log p(z)`` for a purely Euclidean prior."""
        if self.hyperbolic_dim:
            raise ValueError("this prior has a hyperbolic block; use the model, which knows the geometry")
        return self.mix(self.component_log_prob(z), labels)

    @torch.no_grad()
    def responsibilities(self, z: torch.Tensor) -> torch.Tensor:
        """Posterior probability of each component for a purely Euclidean prior: ``(B, K)``."""
        if self.hyperbolic_dim:
            raise ValueError("this prior has a hyperbolic block; use model.prior_responsibilities")
        return F.softmax(self.component_log_prob(z) + F.log_softmax(self.logits, dim=0), dim=1)

    def tree_penalty(self, tree_dist: torch.Tensor, scale: torch.Tensor,
                     pairwise: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> torch.Tensor:
        """Mean squared gap between distances among component centres and ``scale * tree_dist``.

        ``pairwise`` maps the centres to their ``(K, K)`` distance matrix; the
        default is Euclidean distance. The model passes its geodesic distance.
        """
        if tree_dist.shape != (self.n_components, self.n_components):
            raise ValueError(f"prior_tree_weight needs a hierarchy with {self.n_components} labels (one per prior "
                             f"component), got a {tuple(tree_dist.shape)} distance matrix")
        if pairwise is None:
            d = torch.sqrt(((self.means.unsqueeze(1) - self.means.unsqueeze(0)) ** 2).sum(-1) + 1e-12)
        else:
            d = pairwise(self.means)
        off = ~torch.eye(self.n_components, dtype=torch.bool, device=d.device)
        return ((d - scale * tree_dist.to(d.dtype)) ** 2)[off].mean()


__all__ = ["MixturePrior"]
