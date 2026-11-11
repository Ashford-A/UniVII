"""Graph encoder: each cell is encoded together with its in-batch neighbours.

    GraphGaussianEncoder   GraphSAGE-style rounds over a k-nearest-neighbour graph -> (mu, logvar)
    knn_in_batch           neighbour indices from the coordinates a batch carries

Importing this module registers ``encoder_type="graph"``. The model passes the
batch's coordinates (``x_dict[UniVIConfig.spatial_key]``) to encoders that set
``uses_coords = True``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..config import GraphConfig
from .registry import register_encoder


def knn_in_batch(coords: torch.Tensor, k: int) -> torch.Tensor:
    """Indices ``(B, k)`` of each row's nearest other rows. ``k`` is capped at ``B - 1``."""
    b = coords.shape[0]
    k = min(int(k), b - 1)
    d = torch.cdist(coords, coords)
    d = d + torch.diag(torch.full((b,), float("inf"), device=d.device, dtype=d.dtype))
    return torch.topk(d, k, dim=1, largest=False).indices


class GraphGaussianEncoder(nn.Module):
    """GraphSAGE-style encoder.

    Round ``l``: ``h_i <- act(W_l [h_i, agg_{j in N(i)} h_j])``, with ``N(i)`` the
    ``k`` nearest in-batch neighbours of cell ``i`` in coordinate space. Without
    coordinates, or in a batch of one, the neighbourhood term is ``h_i`` itself.
    """

    uses_coords = True

    def __init__(self, input_dim: int, cfg: GraphConfig, latent_dim: int):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        dims = [int(input_dim)] + [int(h) for h in cfg.hidden]
        self.layers = nn.ModuleList(nn.Linear(2 * dims[i], dims[i + 1]) for i in range(len(dims) - 1))
        self.norms = nn.ModuleList(nn.LayerNorm(d) for d in dims[1:])
        self.act = nn.SiLU()
        self.drop = nn.Dropout(float(cfg.dropout))
        self.head = nn.Linear(dims[-1], 2 * int(latent_dim))

    def _aggregate(self, h: torch.Tensor, idx: Optional[torch.Tensor]) -> torch.Tensor:
        if idx is None:
            return h
        neigh = h[idx]                                           # (B, k, d)
        return neigh.max(dim=1).values if self.cfg.aggregate == "max" else neigh.mean(dim=1)

    def forward(self, x: torch.Tensor, coords: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = None
        if coords is not None and x.shape[0] > 1:
            idx = knn_in_batch(torch.as_tensor(coords).to(device=x.device, dtype=x.dtype), self.cfg.k)
        h = x
        for layer, norm in zip(self.layers, self.norms):
            h = self.drop(self.act(norm(layer(torch.cat([h, self._aggregate(h, idx)], dim=-1)))))
        mu, logvar = self.head(h).chunk(2, dim=-1)
        return mu, logvar


@register_encoder("graph", overwrite=True)
def _build_graph_encoder(uni_cfg, mod_cfg) -> nn.Module:
    if mod_cfg.graph is None:
        raise ValueError(f"Modality {mod_cfg.name!r}: encoder_type='graph' requires graph=GraphConfig(...)")
    return GraphGaussianEncoder(int(mod_cfg.input_dim), mod_cfg.graph, int(uni_cfg.latent_dim))


__all__ = ["GraphGaussianEncoder", "knn_in_batch"]
