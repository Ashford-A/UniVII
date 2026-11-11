"""Reference-anchored analysis: place query cells on a frozen model and score
how far they sit from what the model can represent.

Typical use: train on healthy cells, freeze, then for tumor cells ask (1) where
on the healthy manifold each cell lands and (2) how badly the frozen decoders
explain it.

    z0 = model.encode_fused(x_query)[0]                    # amortised placement
    z = refine_latent(model, x_query, z_init=z0)           # optional, decoder-only refinement
    nll, per_mod = reconstruction_nll(model, x_query, z)
    nll_null, _ = reconstruction_nll(model, x_healthy_heldout, z_healthy)
    score = deviation_scores(nll, nll_null)                # robust z-score against healthy cells

Call hierarchy::

    refine_latent            gradient descent on z through frozen decoders
        reconstruction_nll   the model's own per-cell reconstruction loss at a given z
    deviation_scores         robust z-score of query NLL against a null population
    anchor_on_label          set the latent anchor from cells carrying one label
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


def reconstruction_nll(model, x_dict: Mapping[str, torch.Tensor], z: torch.Tensor, *,
                       recon_targets: Optional[Mapping[str, Any]] = None,
                       modalities: Optional[Sequence[str]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Per-cell reconstruction loss of ``x_dict`` decoded from latent ``z``.

    Uses each modality's configured likelihood, so the unit is that modality's
    negative log-likelihood (or squared error for Gaussian modalities).
    ``z`` must be in raw, un-anchored coordinates. Returns ``(total, per_modality)``.
    """
    per_mod: Dict[str, torch.Tensor] = {}
    total: Optional[torch.Tensor] = None
    for name, m_cfg in model.mod_cfg_by_name.items():
        if modalities is not None and name not in modalities:
            continue
        if name not in x_dict or x_dict[name] is None:
            continue
        target = x_dict[name]
        if recon_targets is not None and recon_targets.get(name, None) is not None:
            target = recon_targets[name]
        loss = model._recon_loss(x=target, raw_dec_out=model.decoders[name](z),
                                 likelihood=m_cfg.likelihood, mod_name=name)
        per_mod[name] = loss
        total = loss if total is None else total + loss
    if total is None:
        raise ValueError("no modality in x_dict could be scored")
    return total, per_mod


def refine_latent(model, x_dict: Mapping[str, torch.Tensor], *, z_init: Optional[torch.Tensor] = None,
                  steps: int = 100, lr: float = 0.05, prior_weight: float = 0.0,
                  recon_targets: Optional[Mapping[str, Any]] = None,
                  modalities: Optional[Sequence[str]] = None) -> torch.Tensor:
    """Refine latent coordinates by gradient descent through the frozen decoders.

    The encoder is trusted only for the starting point (``z_init``, default the
    fused posterior mean). This matters for cells unlike the training data,
    where an amortised encoder has no guarantee of landing on the best-fitting
    latent. ``prior_weight`` adds ``prior_weight * radius ** 2`` to keep cells
    from drifting far from the origin. Model weights are not changed.
    """
    was_training = model.training
    model.eval()
    params = list(model.parameters())
    flags = [p.requires_grad for p in params]
    try:
        with torch.no_grad():
            z0 = model.encode_fused(dict(x_dict))[0] if z_init is None else torch.as_tensor(z_init)
        for p in params:
            p.requires_grad_(False)
        with torch.enable_grad():
            z = z0.detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([z], lr=lr)
            for _ in range(int(steps)):
                zc = model.latent.constrain(z)
                loss, _ = reconstruction_nll(model, x_dict, zc, recon_targets=recon_targets, modalities=modalities)
                if prior_weight > 0.0:
                    loss = loss + float(prior_weight) * (zc ** 2).sum(-1)
                opt.zero_grad()
                loss.sum().backward()
                opt.step()
        return model.latent.constrain(z.detach())
    finally:
        for p, f in zip(params, flags):
            p.requires_grad_(f)
        model.train(was_training)


def deviation_scores(nll_query, nll_null) -> np.ndarray:
    """Robust z-score of query reconstruction loss against a null population
    (for example held-out healthy cells): ``(x - median) / (1.4826 * MAD)``."""
    q = np.asarray(nll_query.detach().cpu() if torch.is_tensor(nll_query) else nll_query, dtype=float)
    n = np.asarray(nll_null.detach().cpu() if torch.is_tensor(nll_null) else nll_null, dtype=float)
    med = np.median(n)
    mad = 1.4826 * np.median(np.abs(n - med))
    if not mad > 0:
        raise ValueError("null population has zero spread; cannot standardise")
    return (q - med) / mad


def anchor_on_label(model, z, labels, root_label) -> int:
    """Anchor the model's latent space at the centroid of cells whose label equals
    ``root_label``. Returns the number of cells used."""
    labels = np.asarray(labels)
    mask = labels == root_label
    if not mask.any():
        raise ValueError(f"no cells carry the label {root_label!r}")
    model.set_latent_anchor(torch.as_tensor(z)[torch.as_tensor(mask)])
    return int(mask.sum())


__all__ = ["reconstruction_nll", "refine_latent", "deviation_scores", "anchor_on_label"]
