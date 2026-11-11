"""Latent-space geometry for UniVI: Euclidean, hyperbolic and product spaces.

Coordinate convention
---------------------
Every latent vector UniVI passes around stays a plain ``(B, latent_dim)``
tensor. The first ``euclidean_dim`` columns are ordinary Euclidean
coordinates. The remaining ``hyperbolic_dim`` columns are *tangent
coordinates at the origin* of a hyperbolic space of curvature ``-c``: the
point they denote is ``exp_0(t)`` on the hyperboloid. Decoders, prediction
heads, checkpoints and ``.obsm`` exports therefore keep their shapes, while
sampling, KL, alignment and distances use the true geometry.

Consequences worth knowing:

* The norm of the hyperbolic block is the geodesic distance from the origin.
* Euclidean distance between hyperbolic blocks is *not* the geodesic distance
  (except from the origin). Use :func:`latent_distance`,
  :func:`pairwise_latent_distance` or the model methods of the same name.
* Posteriors on the hyperbolic block are wrapped normals (Nagano et al.,
  2019): a diagonal Gaussian in the tangent space at the mean, pushed onto
  the manifold. ``mu`` is the mean in tangent coordinates at the origin and
  ``logvar`` is the log-variance in the tangent space at the mean.
* Mixture-of-experts fusion still combines ``mu`` and ``logvar`` by precision
  weighting. On the hyperbolic block this is tangent-space fusion at the
  origin, an approximation that is exact for a single expert.
* KL terms on the hyperbolic block have no closed form and are estimated
  with one Monte Carlo sample per cell, so they are noisier than the
  Euclidean KL and can be slightly negative for a single cell.

Numerics: hyperbolic computations run in float64 internally and are cast
back, except on Apple MPS, which has no float64. There they run in float32
and ``hyperbolic_max_radius`` should be kept at or below about 4.

Module layout (top = public entry points)::

    LatentGeometry                 nn.Module owned by the model as ``model.latent``
        rsample / kl / log_prob    wrapped-normal posterior on the hyperbolic block
        sqdist / dist / cdist      geodesic distances on mixed coordinates
        set_anchor / anchor        isometry that moves a root population to the origin
        set_hierarchy / hierarchy_loss   weak supervision from a label tree
    latent_distance, pairwise_latent_distance, latent_radius, to_poincare
                                   the same operations for arrays, without a model
    tree_distance_matrix, hierarchy_metrics
                                   build a label tree and score an embedding against it
    expmap0, logmap0, lorentz_distance, boost, wrapped-normal helpers
                                   unit-curvature primitives used by everything above
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

GEOMETRIES = ("euclidean", "hyperbolic", "product")
_EPS = 1e-12
_LOG_2PI = math.log(2.0 * math.pi)

ArrayLike = Union[np.ndarray, torch.Tensor]


# ---------------------------------------------------------------------------
# dimension bookkeeping
# ---------------------------------------------------------------------------

def split_latent_dims(latent_dim: int, geometry: str = "euclidean",
                      hyperbolic_dim: Optional[int] = None) -> Tuple[int, int]:
    """Return ``(euclidean_dim, hyperbolic_dim)`` for a latent of size ``latent_dim``.

    ``euclidean``: all Euclidean. ``hyperbolic``: all hyperbolic. ``product``:
    the last ``hyperbolic_dim`` columns are hyperbolic (default: half, rounded up).
    """
    geometry = str(geometry).lower().strip()
    latent_dim = int(latent_dim)
    if geometry not in GEOMETRIES:
        raise ValueError(f"latent_geometry must be one of {GEOMETRIES}, got {geometry!r}")
    if geometry == "euclidean":
        if hyperbolic_dim not in (None, 0):
            raise ValueError("hyperbolic_dim must be None or 0 when latent_geometry='euclidean'")
        return latent_dim, 0
    if geometry == "hyperbolic":
        if hyperbolic_dim not in (None, latent_dim):
            raise ValueError("hyperbolic_dim must be None or latent_dim when latent_geometry='hyperbolic'")
        return 0, latent_dim
    d_h = latent_dim - latent_dim // 2 if hyperbolic_dim is None else int(hyperbolic_dim)
    if not 0 < d_h < latent_dim:
        raise ValueError(
            f"latent_geometry='product' needs 0 < hyperbolic_dim < latent_dim, got {d_h} with latent_dim={latent_dim}")
    return latent_dim - d_h, d_h


# ---------------------------------------------------------------------------
# unit-curvature primitives (Lorentz / hyperboloid model)
# ---------------------------------------------------------------------------

def expmap0(v: torch.Tensor) -> torch.Tensor:
    """Tangent coordinates at the origin ``(..., d)`` -> hyperboloid ``(..., d + 1)``."""
    r = v.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    return torch.cat([torch.cosh(r), torch.sinh(r) * v / r], dim=-1)


def logmap0(x: torch.Tensor) -> torch.Tensor:
    """Hyperboloid ``(..., d + 1)`` -> tangent coordinates at the origin ``(..., d)``."""
    xb = x[..., 1:]
    n = xb.norm(dim=-1, keepdim=True)
    # asinh of the spatial norm is better conditioned near the origin than acosh(x0)
    return torch.asinh(n) * xb / n.clamp_min(_EPS)


def lorentz_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Geodesic distance on the unit hyperboloid.

    Uses ``cosh(d) = 1 + <x - y, x - y>_L / 2``, which loses far less precision
    for nearby points than ``acosh(-<x, y>_L)``.
    """
    diff = x - y
    s = (-diff[..., 0] ** 2 + (diff[..., 1:] ** 2).sum(-1)).clamp_min(1e-24)
    return 2.0 * torch.asinh(0.5 * torch.sqrt(s))


def boost(center: torch.Tensor, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Apply the Lorentz boost that maps ``center`` to the origin (or its inverse).

    The boost is an isometry and its differential is parallel transport along
    the geodesic between ``center`` and the origin, so it carries wrapped
    normals centred at ``center`` to wrapped normals centred at the origin.
    """
    c0, cb = center[..., :1], center[..., 1:]
    if inverse:
        cb = -cb
    x0, xb = x[..., :1], x[..., 1:]
    dot = (cb * xb).sum(-1, keepdim=True)
    y0 = c0 * x0 - dot
    yb = xb - cb * x0 + cb * dot / (1.0 + c0)
    # re-project the time coordinate so rounding error does not accumulate
    y0 = torch.sqrt(1.0 + (yb ** 2).sum(-1, keepdim=True))
    return torch.cat([y0, yb], dim=-1)


def lorentz_centroid(x: torch.Tensor) -> torch.Tensor:
    """Lorentzian centroid of points ``(n, d + 1)``. Close to, but not exactly, the Frechet mean."""
    s = x.sum(0)
    norm = torch.sqrt((s[0] ** 2 - (s[1:] ** 2).sum()).clamp_min(_EPS))
    return s / norm


def log_sinhc(r: torch.Tensor) -> torch.Tensor:
    """``log(sinh(r) / r)``, stable for small and large ``r``."""
    small = r < 1e-3
    rs = torch.where(small, torch.ones_like(r), r)
    big = rs + torch.log1p(-torch.exp(-2.0 * rs)) - math.log(2.0) - torch.log(rs)
    return torch.where(small, r ** 2 / 6.0, big)


def _work_dtype(x: torch.Tensor) -> torch.dtype:
    return x.dtype if x.device.type == "mps" else torch.float64


# ---------------------------------------------------------------------------
# the geometry module owned by the model
# ---------------------------------------------------------------------------

class LatentGeometry(nn.Module):
    """Geometry of a UniVI latent space. See the module docstring for conventions.

    Parameters
    ----------
    latent_dim, geometry, hyperbolic_dim
        See :func:`split_latent_dims`.
    curvature
        Magnitude ``c > 0``; the hyperbolic block has curvature ``-c``. Larger
        ``c`` means stronger curvature at a given coordinate scale.
    learn_curvature
        Make ``c`` a trainable parameter (stored as ``log c``).
    max_radius
        Cap on ``sqrt(c) * ||mu_h||``, the mean's distance from the origin in
        curvature units.
    kl_hyperbolic_weight
        Multiplier on the hyperbolic part of the prior KL (the Euclidean part
        has weight 1). Both are then scaled by the model's ``beta``.
    hierarchy_scale
        Factor applied to label-tree distances in :meth:`hierarchy_loss`.
        A number fixes it; ``None`` makes it a trainable parameter.
    """

    def __init__(self, latent_dim: int, geometry: str = "euclidean", hyperbolic_dim: Optional[int] = None,
                 curvature: float = 1.0, learn_curvature: bool = False, max_radius: float = 8.0,
                 kl_hyperbolic_weight: float = 1.0, hierarchy_scale: Optional[float] = 1.0):
        super().__init__()
        self.geometry = str(geometry).lower().strip()
        self.latent_dim = int(latent_dim)
        self.euclidean_dim, self.hyperbolic_dim = split_latent_dims(latent_dim, self.geometry, hyperbolic_dim)
        if not float(curvature) > 0.0:
            raise ValueError(f"curvature must be > 0, got {curvature}")
        if not float(max_radius) > 0.0:
            raise ValueError(f"hyperbolic_max_radius must be > 0, got {max_radius}")
        self.max_radius = float(max_radius)
        self.kl_hyperbolic_weight = float(kl_hyperbolic_weight)
        self.learn_curvature = bool(learn_curvature) and self.hyperbolic_dim > 0

        log_c = torch.tensor(math.log(float(curvature)), dtype=torch.get_default_dtype())
        if self.learn_curvature:
            self.log_curvature = nn.Parameter(log_c)
        else:
            self.register_buffer("log_curvature", log_c)

        if hierarchy_scale is None:
            self.hierarchy_log_scale = nn.Parameter(torch.zeros(()))
        else:
            if not float(hierarchy_scale) > 0.0:
                raise ValueError(f"hierarchy_scale must be > 0 or None, got {hierarchy_scale}")
            self.register_buffer("hierarchy_log_scale", torch.tensor(math.log(float(hierarchy_scale))))

        self.register_buffer("anchor_point", torch.zeros(self.latent_dim))
        self.register_buffer("has_anchor", torch.zeros((), dtype=torch.bool))
        self.register_buffer("hierarchy_dist", torch.zeros(0, 0))
        self.register_buffer("hierarchy_root", torch.tensor(-1, dtype=torch.long))

        if self.geometry == "product" and self.hyperbolic_dim == 1:
            warnings.warn("A 1-dimensional hyperbolic block is isometric to a line; it adds no curvature.",
                          stacklevel=2)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Any) -> "LatentGeometry":
        """Build from a :class:`univii.UniVIConfig` (missing fields fall back to Euclidean defaults)."""
        return cls(
            latent_dim=int(cfg.latent_dim),
            geometry=getattr(cfg, "latent_geometry", "euclidean"),
            hyperbolic_dim=getattr(cfg, "hyperbolic_dim", None),
            curvature=float(getattr(cfg, "curvature", 1.0)),
            learn_curvature=bool(getattr(cfg, "learn_curvature", False)),
            max_radius=float(getattr(cfg, "hyperbolic_max_radius", 8.0)),
            kl_hyperbolic_weight=float(getattr(cfg, "kl_hyperbolic_weight", 1.0)),
            hierarchy_scale=getattr(cfg, "hierarchy_scale", 1.0),
        )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Checkpoints written before this module existed have none of its keys;
        # fill them with the current values so strict loading still works.
        for name, value in list(self.named_buffers(recurse=False)) + list(self.named_parameters(recurse=False)):
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = value.detach().clone()
        # The hierarchy matrix is sized by the number of labels.
        key = prefix + "hierarchy_dist"
        if state_dict[key].shape != self.hierarchy_dist.shape:
            self.hierarchy_dist = torch.zeros_like(state_dict[key])
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def extra_repr(self) -> str:
        return (f"geometry={self.geometry!r}, euclidean_dim={self.euclidean_dim}, "
                f"hyperbolic_dim={self.hyperbolic_dim}, curvature={float(self.curvature):.4g}, "
                f"learn_curvature={self.learn_curvature}")

    # -- basic properties ---------------------------------------------------
    @property
    def is_euclidean(self) -> bool:
        return self.hyperbolic_dim == 0

    @property
    def curvature(self) -> torch.Tensor:
        """Curvature magnitude ``c`` (the space has curvature ``-c``)."""
        return self.log_curvature.exp()

    @property
    def hierarchy_scale(self) -> torch.Tensor:
        return self.hierarchy_log_scale.exp()

    def _sqrt_c(self, like: torch.Tensor) -> torch.Tensor:
        return torch.exp(0.5 * self.log_curvature).to(dtype=like.dtype, device=like.device)

    def split(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split ``(..., latent_dim)`` into its Euclidean and hyperbolic blocks."""
        return t[..., :self.euclidean_dim], t[..., self.euclidean_dim:]

    def constrain(self, mu: torch.Tensor) -> torch.Tensor:
        """Clip the hyperbolic block of ``mu`` to ``max_radius`` (no-op for Euclidean)."""
        if self.is_euclidean:
            return mu
        e, h = self.split(mu)
        limit = self.max_radius / self._sqrt_c(h)
        n = h.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        h = h * torch.clamp(limit / n, max=1.0)
        return torch.cat([e, h], dim=-1) if self.euclidean_dim else h

    # -- wrapped normal on the hyperbolic block -----------------------------
    def _h_rsample(self, mu_h: torch.Tensor, logvar_h: torch.Tensor):
        """Sample tangent coordinates and return them with ``log q`` of the sample."""
        out_dtype, wd = mu_h.dtype, _work_dtype(mu_h)
        mu, lv = mu_h.to(wd), logvar_h.to(wd)
        sc = self._sqrt_c(mu)
        v = torch.randn_like(lv) * torch.exp(0.5 * lv)
        z = boost(expmap0(sc * mu), expmap0(sc * v), inverse=True)
        t = logmap0(z) / sc
        d = mu.shape[-1]
        log_q = (-0.5 * (v ** 2 / lv.exp() + lv + _LOG_2PI)).sum(-1) - (d - 1) * log_sinhc(sc * v.norm(dim=-1))
        return t.to(out_dtype), log_q

    def _h_log_prob(self, t_h: torch.Tensor, mu_h: torch.Tensor, logvar_h: torch.Tensor) -> torch.Tensor:
        wd = _work_dtype(t_h)
        t, mu, lv = t_h.to(wd), mu_h.to(wd), logvar_h.to(wd)
        sc = self._sqrt_c(t)
        v = logmap0(boost(expmap0(sc * mu), expmap0(sc * t))) / sc
        d = t.shape[-1]
        return (-0.5 * (v ** 2 / lv.exp() + lv + _LOG_2PI)).sum(-1) - (d - 1) * log_sinhc(sc * v.norm(dim=-1))

    def rsample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterised sample in mixed coordinates."""
        mu = self.constrain(mu)
        e_mu, h_mu = self.split(mu)
        e_lv, h_lv = self.split(logvar)
        parts = []
        if self.euclidean_dim:
            parts.append(e_mu + torch.randn_like(e_mu) * torch.exp(0.5 * e_lv))
        if self.hyperbolic_dim:
            parts.append(self._h_rsample(h_mu, h_lv)[0])
        return torch.cat(parts, dim=-1)

    def log_prob(self, t: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Log-density of ``t`` under the posterior ``(mu, logvar)``, per cell."""
        mu = self.constrain(mu)
        (e_t, h_t), (e_mu, h_mu), (e_lv, h_lv) = self.split(t), self.split(mu), self.split(logvar)
        out = torch.zeros(t.shape[:-1], dtype=t.dtype, device=t.device)
        if self.euclidean_dim:
            out = out + (-0.5 * ((e_t - e_mu) ** 2 / e_lv.exp() + e_lv + _LOG_2PI)).sum(-1)
        if self.hyperbolic_dim:
            out = out + self._h_log_prob(h_t, h_mu, h_lv).to(t.dtype)
        return out

    def kl(self, mu_q: torch.Tensor, logvar_q: torch.Tensor,
           mu_p: torch.Tensor, logvar_p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``KL(q || p)`` per cell as ``(euclidean part, hyperbolic part)``.

        The Euclidean part is analytic. The hyperbolic part is a one-sample
        Monte Carlo estimate with a sample drawn inside this call.
        """
        mu_q, mu_p = self.constrain(mu_q), self.constrain(mu_p)
        (e_mq, h_mq), (e_lq, h_lq) = self.split(mu_q), self.split(logvar_q)
        (e_mp, h_mp), (e_lp, h_lp) = self.split(mu_p), self.split(logvar_p)
        zeros = torch.zeros(mu_q.shape[:-1], dtype=mu_q.dtype, device=mu_q.device)
        kl_e, kl_h = zeros, zeros
        if self.euclidean_dim:
            kl_e = 0.5 * (e_lp - e_lq + (e_lq.exp() + (e_mq - e_mp) ** 2) / e_lp.exp() - 1.0).sum(-1)
        if self.hyperbolic_dim:
            t, log_q = self._h_rsample(h_mq, h_lq)
            kl_h = (log_q - self._h_log_prob(t, h_mp, h_lp)).to(mu_q.dtype)
        return kl_e, kl_h

    # -- distances ----------------------------------------------------------
    def sqdist(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Squared geodesic distance between rows of ``a`` and ``b`` (broadcasts)."""
        (e_a, h_a), (e_b, h_b) = self.split(a), self.split(b)
        out = ((e_a - e_b) ** 2).sum(-1) if self.euclidean_dim else 0.0
        if self.hyperbolic_dim:
            wd = _work_dtype(h_a)
            sc = self._sqrt_c(h_a.to(wd))
            d = lorentz_distance(expmap0(sc * h_a.to(wd)), expmap0(sc * h_b.to(wd))) / sc
            out = out + (d ** 2).to(a.dtype)
        return out

    def dist(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Geodesic distance between rows of ``a`` and ``b`` (broadcasts)."""
        return torch.sqrt(self.sqdist(a, b) + _EPS)

    def cdist(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """All-pairs ``(n, m)`` geodesic distance."""
        return self.dist(a.unsqueeze(1), b.unsqueeze(0))

    def radius(self, t: torch.Tensor) -> torch.Tensor:
        """Geodesic distance from the origin (the anchor, for anchored coordinates)."""
        return t.norm(dim=-1)

    def to_poincare(self, t: torch.Tensor) -> torch.Tensor:
        """Hyperbolic block as Poincare-ball coordinates in the unit ball, for plotting."""
        if self.is_euclidean:
            raise ValueError("to_poincare needs a hyperbolic block; this latent space is Euclidean")
        h = self.split(t)[1]
        wd = _work_dtype(h)
        z = expmap0(self._sqrt_c(h.to(wd)) * h.to(wd))
        return (z[..., 1:] / (1.0 + z[..., :1])).to(t.dtype)

    # -- anchoring ----------------------------------------------------------
    @torch.no_grad()
    def set_anchor(self, points: torch.Tensor) -> None:
        """Store the centroid of ``points`` (raw coordinates, ``(n, latent_dim)``) as the anchor.

        :meth:`anchor` then maps any coordinates through the isometry that
        sends this centroid to the origin. Distances are unchanged; radius
        becomes distance from the anchored population.
        """
        points = torch.as_tensor(points, dtype=self.anchor_point.dtype, device=self.anchor_point.device)
        if points.ndim != 2 or points.shape[1] != self.latent_dim or points.shape[0] == 0:
            raise ValueError(f"expected a non-empty (n, {self.latent_dim}) array of latent coordinates")
        e, h = self.split(points)
        parts = []
        if self.euclidean_dim:
            parts.append(e.mean(0))
        if self.hyperbolic_dim:
            wd = _work_dtype(h)
            sc = self._sqrt_c(h.to(wd))
            parts.append((logmap0(lorentz_centroid(expmap0(sc * h.to(wd)))) / sc).to(points.dtype))
        self.anchor_point.copy_(torch.cat(parts))
        self.has_anchor.fill_(True)

    @torch.no_grad()
    def clear_anchor(self) -> None:
        self.anchor_point.zero_()
        self.has_anchor.fill_(False)

    def anchor(self, t: torch.Tensor) -> torch.Tensor:
        """Raw coordinates -> anchored coordinates (identity if no anchor is set)."""
        if not bool(self.has_anchor):
            return t
        (e, h), (a_e, a_h) = self.split(t), self.split(self.anchor_point.to(t.device))
        parts = []
        if self.euclidean_dim:
            parts.append(e - a_e.to(e.dtype))
        if self.hyperbolic_dim:
            wd = _work_dtype(h)
            sc = self._sqrt_c(h.to(wd))
            parts.append((logmap0(boost(expmap0(sc * a_h.to(wd)), expmap0(sc * h.to(wd)))) / sc).to(t.dtype))
        return torch.cat(parts, dim=-1)

    # -- hierarchy supervision ----------------------------------------------
    @torch.no_grad()
    def set_hierarchy(self, dist: ArrayLike, root_index: Optional[int] = None) -> None:
        """Register a label-by-label tree distance matrix (see :func:`tree_distance_matrix`)."""
        dist = torch.as_tensor(np.asarray(dist), dtype=self.anchor_point.dtype, device=self.anchor_point.device)
        if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
            raise ValueError("hierarchy distance matrix must be square")
        if not torch.allclose(dist, dist.T, atol=1e-6) or bool((dist < 0).any()):
            raise ValueError("hierarchy distance matrix must be symmetric and non-negative")
        if root_index is not None and not 0 <= int(root_index) < dist.shape[0]:
            raise ValueError(f"root_index {root_index} is outside 0..{dist.shape[0] - 1}")
        self.hierarchy_dist = dist
        self.hierarchy_root.fill_(-1 if root_index is None else int(root_index))

    @property
    def has_hierarchy(self) -> bool:
        return self.hierarchy_dist.numel() > 0

    def hierarchy_loss(self, mu: torch.Tensor, y: torch.Tensor, *, root_weight: float = 0.0,
                       ignore_index: int = -1) -> torch.Tensor:
        """Per-cell loss tying latent distances to label-tree distances.

        Each labelled cell is paired with one random labelled cell from the
        batch and penalised by ``(d_latent - scale * d_tree) ** 2``. With
        ``root_weight > 0``, cells of the root label are also pulled to the
        origin by ``root_weight * radius ** 2``. Unlabelled cells get 0.
        """
        if not self.has_hierarchy:
            raise RuntimeError("no hierarchy registered; call set_hierarchy first")
        mu = self.constrain(mu)
        y = y.long().view(-1)
        k = self.hierarchy_dist.shape[0]
        out = torch.zeros(mu.shape[0], dtype=mu.dtype, device=mu.device)
        valid = (y >= 0) & (y < k) & (y != int(ignore_index))
        idx = torch.nonzero(valid, as_tuple=False).view(-1)
        if idx.numel() >= 2:
            partner = idx[torch.randperm(idx.numel(), device=idx.device)]
            target = self.hierarchy_scale.to(mu.dtype) * self.hierarchy_dist.to(mu.dtype)[y[idx], y[partner]]
            out = out.index_add(0, idx, (self.dist(mu[idx], mu[partner]) - target) ** 2)
        root = int(self.hierarchy_root)
        if root_weight > 0.0 and root >= 0:
            out = out + float(root_weight) * (mu ** 2).sum(-1) * (valid & (y == root)).to(mu.dtype)
        return out


# ---------------------------------------------------------------------------
# array-level helpers (no model needed)
# ---------------------------------------------------------------------------

def _geometry_for(z: ArrayLike, geometry: str, hyperbolic_dim: Optional[int], curvature: float) -> LatentGeometry:
    return LatentGeometry(int(np.shape(z)[-1]), geometry, hyperbolic_dim, curvature=curvature).double()


def _as_tensor(z: ArrayLike) -> torch.Tensor:
    return torch.as_tensor(np.asarray(z.detach().cpu() if torch.is_tensor(z) else z), dtype=torch.float64)


def latent_distance(a: ArrayLike, b: ArrayLike, geometry: str = "euclidean",
                    hyperbolic_dim: Optional[int] = None, curvature: float = 1.0) -> np.ndarray:
    """Row-wise geodesic distance between two ``(n, latent_dim)`` arrays."""
    return _geometry_for(a, geometry, hyperbolic_dim, curvature).dist(_as_tensor(a), _as_tensor(b)).numpy()


def pairwise_latent_distance(a: ArrayLike, b: Optional[ArrayLike] = None, geometry: str = "euclidean",
                             hyperbolic_dim: Optional[int] = None, curvature: float = 1.0,
                             chunk: int = 1024) -> np.ndarray:
    """All-pairs ``(n, m)`` geodesic distance; ``b=None`` compares ``a`` with itself."""
    geom = _geometry_for(a, geometry, hyperbolic_dim, curvature)
    ta = _as_tensor(a)
    tb = ta if b is None else _as_tensor(b)
    return torch.cat([geom.cdist(ta[i:i + chunk], tb) for i in range(0, len(ta), chunk)]).numpy()


def latent_radius(z: ArrayLike) -> np.ndarray:
    """Distance from the origin. Valid in every geometry because the hyperbolic
    block is stored in tangent coordinates at the origin."""
    return np.linalg.norm(np.asarray(z.detach().cpu() if torch.is_tensor(z) else z), axis=-1)


def to_poincare(z: ArrayLike, geometry: str = "hyperbolic", hyperbolic_dim: Optional[int] = None,
                curvature: float = 1.0) -> np.ndarray:
    """Hyperbolic block as Poincare-ball coordinates (unit ball), for plotting."""
    return _geometry_for(z, geometry, hyperbolic_dim, curvature).to_poincare(_as_tensor(z)).numpy()


def tree_distance_matrix(parents: Mapping[str, Optional[str]],
                         lengths: Optional[Mapping[str, float]] = None,
                         order: Optional[Sequence[str]] = None):
    """Path-length distances between the nodes of a rooted tree.

    Parameters
    ----------
    parents
        ``{node: parent}`` with ``None`` for the single root.
    lengths
        Optional ``{node: length of the edge to its parent}`` (default 1).
    order
        Node order for the rows and columns (default: the keys of ``parents``).
        Use the label order of the classification head that feeds the
        hierarchy loss.

    Returns
    -------
    ``(dist, names, depth)``: the ``(K, K)`` matrix, the node names in row
    order, and each node's distance from the root.
    """
    names = list(order) if order is not None else list(parents)
    if set(names) != set(parents):
        raise ValueError("order must contain exactly the nodes in parents")
    roots = [n for n in names if parents[n] is None]
    if len(roots) != 1:
        raise ValueError(f"tree needs exactly one root (parent None), found {len(roots)}")

    def path(n: str):
        out, seen = [], set()
        while n is not None:
            if n in seen:
                raise ValueError("parents contains a cycle")
            if n not in parents:
                raise ValueError(f"parent {n!r} is not itself a node")
            seen.add(n)
            out.append(n)
            n = parents[n]
        return out

    edge = {n: 0.0 if parents[n] is None else float((lengths or {}).get(n, 1.0)) for n in names}
    paths = {n: path(n) for n in names}
    depth = {n: sum(edge[a] for a in paths[n]) for n in names}
    dist = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        anc = set(paths[a])
        for j, b in enumerate(names):
            lca = next(n for n in paths[b] if n in anc)
            dist[i, j] = depth[a] + depth[b] - 2.0 * depth[lca]
    return dist, names, np.array([depth[n] for n in names])


def hierarchy_metrics(z: ArrayLike, labels: Sequence[int], dist: ArrayLike, depth: Optional[ArrayLike] = None,
                      geometry: str = "euclidean", hyperbolic_dim: Optional[int] = None, curvature: float = 1.0,
                      n_pairs: int = 20000, seed: int = 0) -> Dict[str, float]:
    """Score an embedding against a label tree.

    Returns a dict with:

    - ``dist_spearman``: rank agreement of latent distance with label-tree distance over random cell pairs.
    - ``rel_distortion``: mean relative error after the best single rescaling (pairs with distinct labels).
    - ``radius_depth``: rank agreement of latent radius with label depth. Only present if ``depth`` is
      given, and meaningful for anchored coordinates.
    """
    from scipy.stats import spearmanr

    z_np = np.asarray(z.detach().cpu() if torch.is_tensor(z) else z)
    labels = np.asarray(labels)
    dist = np.asarray(dist, dtype=float)
    keep = np.flatnonzero((labels >= 0) & (labels < dist.shape[0]))
    rng = np.random.default_rng(seed)
    i, j = rng.choice(keep, n_pairs), rng.choice(keep, n_pairs)
    d_lat = latent_distance(z_np[i], z_np[j], geometry, hyperbolic_dim, curvature)
    d_tree = dist[labels[i], labels[j]]
    out = {"dist_spearman": float(spearmanr(d_lat, d_tree).statistic)}
    ok = d_tree > 0
    s = (d_lat[ok] * d_tree[ok]).sum() / (d_tree[ok] ** 2).sum()
    out["rel_distortion"] = float(np.mean(np.abs(d_lat[ok] - s * d_tree[ok]) / (s * d_tree[ok])))
    if depth is not None:
        out["radius_depth"] = float(spearmanr(latent_radius(z_np[keep]), np.asarray(depth)[labels[keep]]).statistic)
    return out


__all__ = [
    "GEOMETRIES", "LatentGeometry", "split_latent_dims",
    "expmap0", "logmap0", "lorentz_distance", "boost", "lorentz_centroid", "log_sinhc",
    "latent_distance", "pairwise_latent_distance", "latent_radius", "to_poincare",
    "tree_distance_matrix", "hierarchy_metrics",
]
