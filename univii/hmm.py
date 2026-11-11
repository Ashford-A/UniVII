"""A small differentiable hidden Markov model for segmenting ordered data.

Use it wherever cells or features have a natural order and you expect runs of a
few discrete states: latent coordinates along pseudotime, a signal along the
genome, a score across a time course.

    GaussianHMM          states with diagonal-Gaussian emissions; exact inference in log space
        log_likelihood   forward algorithm (differentiable)
        posterior        per-position state probabilities (forward-backward)
        viterbi          most probable state path
        fit              maximum likelihood by gradient ascent
    segment_by_order     fit an HMM along an ordering and return a state per cell
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_LOG_2PI = float(np.log(2.0 * np.pi))


class GaussianHMM(nn.Module):
    """HMM with ``n_states`` states and diagonal-Gaussian emissions in ``dim`` dimensions.

    ``stickiness`` sets the initial probability of staying in the same state;
    values near 1 favour long runs. All parameters are learned by :meth:`fit`.
    Sequences are ``(T, dim)`` tensors; computations run in float64.
    """

    def __init__(self, n_states: int, dim: int, stickiness: float = 0.9):
        super().__init__()
        if n_states < 1 or dim < 1 or not 0.0 < stickiness < 1.0:
            raise ValueError("need n_states >= 1, dim >= 1 and 0 < stickiness < 1")
        self.n_states, self.dim = int(n_states), int(dim)
        off = (1.0 - stickiness) / max(n_states - 1, 1)
        trans = torch.full((n_states, n_states), off) + torch.eye(n_states) * (stickiness - off)
        self.start_logits = nn.Parameter(torch.zeros(n_states, dtype=torch.float64))
        self.trans_logits = nn.Parameter(torch.log(trans.double()))
        self.means = nn.Parameter(torch.zeros(n_states, dim, dtype=torch.float64))
        self.logvars = nn.Parameter(torch.zeros(n_states, dim, dtype=torch.float64))

    # -- pieces ---------------------------------------------------------------
    def _x(self, x) -> torch.Tensor:
        x = torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x, dtype=torch.float64)
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"expected a (T, {self.dim}) sequence, got {tuple(x.shape)}")
        return x

    def emission_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """``log p(x_t | state k)``: ``(T, K)``."""
        diff = x.unsqueeze(1) - self.means.unsqueeze(0)
        lv = self.logvars.unsqueeze(0)
        return (-0.5 * (diff ** 2 / lv.exp() + lv + _LOG_2PI)).sum(-1)

    def _forward(self, log_b: torch.Tensor) -> torch.Tensor:
        log_a = F.log_softmax(self.trans_logits, dim=1)
        alpha = [F.log_softmax(self.start_logits, dim=0) + log_b[0]]
        for t in range(1, log_b.shape[0]):
            alpha.append(torch.logsumexp(alpha[-1].unsqueeze(1) + log_a, dim=0) + log_b[t])
        return torch.stack(alpha)

    def _backward(self, log_b: torch.Tensor) -> torch.Tensor:
        log_a = F.log_softmax(self.trans_logits, dim=1)
        beta = [torch.zeros(self.n_states, dtype=log_b.dtype)]
        for t in range(log_b.shape[0] - 1, 0, -1):
            beta.append(torch.logsumexp(log_a + (log_b[t] + beta[-1]).unsqueeze(0), dim=1))
        return torch.stack(beta[::-1])

    # -- inference ------------------------------------------------------------
    def log_likelihood(self, x) -> torch.Tensor:
        """``log p(x_1..T)`` of one sequence."""
        return torch.logsumexp(self._forward(self.emission_log_prob(self._x(x)))[-1], dim=0)

    @torch.no_grad()
    def posterior(self, x) -> torch.Tensor:
        """State probabilities at every position given the whole sequence: ``(T, K)``."""
        log_b = self.emission_log_prob(self._x(x))
        g = self._forward(log_b) + self._backward(log_b)
        return torch.softmax(g, dim=1)

    @torch.no_grad()
    def viterbi(self, x) -> torch.Tensor:
        """Most probable state path: ``(T,)`` integers."""
        log_b = self.emission_log_prob(self._x(x))
        log_a = F.log_softmax(self.trans_logits, dim=1)
        delta = F.log_softmax(self.start_logits, dim=0) + log_b[0]
        back = []
        for t in range(1, log_b.shape[0]):
            score, arg = (delta.unsqueeze(1) + log_a).max(dim=0)
            back.append(arg)
            delta = score + log_b[t]
        path = [int(delta.argmax())]
        for arg in reversed(back):
            path.append(int(arg[path[-1]]))
        return torch.tensor(path[::-1], dtype=torch.long)

    # -- fitting --------------------------------------------------------------
    def initialise(self, x, seed: int = 0) -> None:
        """Start the means at quantiles along the first principal direction and the
        variances at the overall variance."""
        x = self._x(x)
        g = torch.Generator().manual_seed(int(seed))
        centred = x - x.mean(0)
        direction = torch.linalg.svd(centred, full_matrices=False)[2][0] if x.shape[0] > 1 else torch.ones(self.dim, dtype=x.dtype)
        order = torch.argsort(centred @ direction)
        cuts = torch.linspace(0, x.shape[0], self.n_states + 1).long()
        with torch.no_grad():
            for k in range(self.n_states):
                members = order[cuts[k]:max(int(cuts[k + 1]), int(cuts[k]) + 1)]
                self.means[k] = x[members].mean(0) + 1e-3 * torch.randn(self.dim, generator=g, dtype=x.dtype)
            self.logvars.copy_(torch.log(x.var(0).clamp_min(1e-6)).expand_as(self.logvars))

    def fit(self, x, n_iter: int = 200, lr: float = 0.05, seed: int = 0, initialise: bool = True) -> "GaussianHMM":
        """Maximise the log-likelihood of one sequence by gradient ascent (Adam)."""
        x = self._x(x)
        if initialise:
            self.initialise(x, seed)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(int(n_iter)):
            loss = -self.log_likelihood(x) / x.shape[0]
            opt.zero_grad()
            loss.backward()
            opt.step()
        return self


def segment_by_order(values, order, n_states: int = 3, stickiness: float = 0.95, n_iter: int = 200,
                     seed: int = 0) -> Tuple[np.ndarray, np.ndarray, GaussianHMM]:
    """Segment cells into runs of states along an ordering.

    ``values`` is ``(cells, dims)`` (or ``(cells,)``), for example latent
    coordinates or a score; ``order`` is one number per cell, for example
    pseudotime or genomic position. The cells are sorted by ``order``, an HMM is
    fitted to the sorted sequence, and the result is returned in the original
    cell order as ``(states, probabilities, model)``. State numbers are
    arbitrary labels.
    """
    values = np.asarray(values.detach().cpu() if torch.is_tensor(values) else values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    order = np.asarray(order, dtype=float)
    if order.shape != (values.shape[0],):
        raise ValueError("order must have one entry per cell")
    idx = np.argsort(order, kind="stable")
    hmm = GaussianHMM(n_states, values.shape[1], stickiness).fit(values[idx], n_iter=n_iter, seed=seed)
    states = np.empty(len(idx), dtype=int)
    probs = np.empty((len(idx), n_states))
    states[idx] = hmm.viterbi(values[idx]).numpy()
    probs[idx] = hmm.posterior(values[idx]).numpy()
    return states, probs, hmm


__all__ = ["GaussianHMM", "segment_by_order"]
