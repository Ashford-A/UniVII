"""Exploring hierarchy in a latent space: trees over clusters, pseudotime, tree comparison.

All functions take latent coordinates as arrays, with the geometry they came from
(``geometry``, ``hyperbolic_dim``, ``curvature``), and use geodesic distances.

    cluster_centroids   centre of each label's cells (geometry-aware)
    cluster_tree        minimum spanning tree over the centroids, rooted
    graph_pseudotime    shortest-path distance from root cells on a k-nearest-neighbour graph
    tree_agreement      how closely two trees over the same nodes agree
"""
from __future__ import annotations

from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components, dijkstra, minimum_spanning_tree

from .geometry import LatentGeometry, pairwise_latent_distance, tree_distance_matrix


def _np(z) -> np.ndarray:
    return np.asarray(z.detach().cpu() if torch.is_tensor(z) else z, dtype=float)


def cluster_centroids(z, labels, geometry: str = "euclidean", hyperbolic_dim: Optional[int] = None,
                      curvature: float = 1.0):
    """Centroid of each label's cells. Returns ``(centroids, names)``.

    The Euclidean block is averaged; the hyperbolic block uses the Lorentzian
    centroid, which stays on the manifold.
    """
    z, labels = _np(z), np.asarray(labels)
    names = list(dict.fromkeys(labels.tolist()))
    geom = LatentGeometry(z.shape[1], geometry, hyperbolic_dim, curvature=curvature).double()
    out = []
    for name in names:
        geom.set_anchor(torch.as_tensor(z[labels == name], dtype=torch.float64))
        out.append(geom.anchor_point.clone().numpy())
    return np.vstack(out), names


def cluster_tree(z, labels, root: Optional[Hashable] = None, geometry: str = "euclidean",
                 hyperbolic_dim: Optional[int] = None, curvature: float = 1.0) -> Dict[str, Any]:
    """Minimum spanning tree over label centroids, rooted at ``root``.

    ``root=None`` picks the label whose centroid is closest to the origin,
    which is the anchored population if the coordinates are anchored.

    Returns a dict with ``parents`` (``{label: parent label or None}``),
    ``edge_length`` (``{label: distance to its parent}``), ``names``,
    ``centroids`` and ``distances`` (the full centroid distance matrix).
    Two limits to keep in mind. A spanning tree always exists, so this never
    reports "no hierarchy". And it joins each cluster to its nearest
    neighbour, so two sibling clusters that sit closer to each other than to
    their parent are linked to each other, not to the parent. Judge the
    result against a reference with :func:`tree_agreement`.
    """
    centroids, names = cluster_centroids(z, labels, geometry, hyperbolic_dim, curvature)
    D = pairwise_latent_distance(centroids, geometry=geometry, hyperbolic_dim=hyperbolic_dim, curvature=curvature)
    np.fill_diagonal(D, 0.0)
    mst = minimum_spanning_tree(sp.csr_matrix(D)).toarray()
    adj = (mst + mst.T) > 0
    if root is None:
        r = int(np.argmin(np.linalg.norm(centroids, axis=1)))
    else:
        if root not in names:
            raise ValueError(f"root {root!r} is not one of the labels")
        r = names.index(root)
    parents: Dict[Hashable, Optional[Hashable]] = {names[r]: None}
    lengths: Dict[Hashable, float] = {names[r]: 0.0}
    stack, seen = [r], {r}
    while stack:
        i = stack.pop()
        for j in np.flatnonzero(adj[i]):
            if j not in seen:
                seen.add(int(j))
                parents[names[j]] = names[i]
                lengths[names[j]] = float(D[i, j])
                stack.append(int(j))
    return {"parents": parents, "edge_length": lengths, "names": names, "centroids": centroids, "distances": D}


def graph_pseudotime(z, root_mask, k: int = 15, geometry: str = "euclidean", hyperbolic_dim: Optional[int] = None,
                     curvature: float = 1.0, chunk: int = 1024) -> np.ndarray:
    """Shortest-path distance from the root cells along a k-nearest-neighbour graph.

    Edges are weighted by geodesic latent distance. If the graph falls into
    several pieces, each piece is joined to the rest by its single shortest
    connecting edge, so every cell gets a finite value. Memory is
    ``O(cells * chunk)``.
    """
    z = _np(z)
    root_mask = np.asarray(root_mask, dtype=bool)
    n = z.shape[0]
    if root_mask.shape != (n,) or not root_mask.any():
        raise ValueError("root_mask must be a boolean array over cells with at least one True")
    k = min(int(k), n - 1)
    rows, cols, vals = [], [], []
    for start in range(0, n, chunk):
        D = pairwise_latent_distance(z[start:start + chunk], z, geometry, hyperbolic_dim, curvature)
        D[np.arange(D.shape[0]), np.arange(start, start + D.shape[0])] = np.inf
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
        rows.append(np.repeat(np.arange(start, start + D.shape[0]), k))
        cols.append(idx.ravel())
        vals.append(np.take_along_axis(D, idx, 1).ravel())
    G = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    G = G.maximum(G.T).tolil()
    n_comp, comp = connected_components(G.tocsr(), directed=False)
    while n_comp > 1:
        inside = comp == comp[np.flatnonzero(root_mask)[0]]
        a, b = np.flatnonzero(inside), np.flatnonzero(~inside)
        best = (np.inf, -1, -1)
        for start in range(0, len(a), chunk):
            D = pairwise_latent_distance(z[a[start:start + chunk]], z[b], geometry, hyperbolic_dim, curvature)
            i, j = np.unravel_index(np.argmin(D), D.shape)
            if D[i, j] < best[0]:
                best = (float(D[i, j]), int(a[start + i]), int(b[j]))
        G[best[1], best[2]] = G[best[2], best[1]] = max(best[0], 1e-12)
        n_comp, comp = connected_components(G.tocsr(), directed=False)
    return dijkstra(G.tocsr(), directed=False, indices=np.flatnonzero(root_mask), min_only=True)


def tree_agreement(parents_a: Mapping[Hashable, Optional[Hashable]],
                   parents_b: Mapping[Hashable, Optional[Hashable]]) -> Dict[str, float]:
    """Compare two rooted trees over the same nodes.

    ``edge_jaccard``     shared undirected edges / all edges in either tree (1 = same shape)
    ``path_spearman``    rank agreement of node-to-node path lengths, counting each edge as 1
    ``parent_accuracy``  fraction of nodes with the same parent (depends on the choice of root)
    """
    from scipy.stats import spearmanr

    if set(parents_a) != set(parents_b):
        raise ValueError("both trees must have the same nodes")
    names = list(parents_a)

    def edges(par):
        return {frozenset((n, p)) for n, p in par.items() if p is not None}

    ea, eb = edges(parents_a), edges(parents_b)
    da = tree_distance_matrix({str(n): (None if p is None else str(p)) for n, p in parents_a.items()},
                              order=[str(n) for n in names])[0]
    db = tree_distance_matrix({str(n): (None if p is None else str(p)) for n, p in parents_b.items()},
                              order=[str(n) for n in names])[0]
    iu = np.triu_indices(len(names), 1)
    return {
        "edge_jaccard": len(ea & eb) / max(len(ea | eb), 1),
        "path_spearman": float(spearmanr(da[iu], db[iu]).statistic) if len(iu[0]) > 2 else float("nan"),
        "parent_accuracy": float(np.mean([parents_a[n] == parents_b[n] for n in names])),
    }


__all__ = ["cluster_centroids", "cluster_tree", "graph_pseudotime", "tree_agreement"]
