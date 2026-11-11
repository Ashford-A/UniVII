"""Spatial omics utilities: turn coordinates and tissue images into UniVI inputs.

Four ways to use spatial information, which combine freely:

1. **Niche modality.** :func:`niche_modality` averages each spot's neighbours
   into a second expression-like modality (``{"rna": adata, "niche": niche}``).
2. **Image modality.** :func:`tiles_from_visium` / :func:`extract_tiles` cut a
   tile around every spot; pair it with ``encoder_type="conv"`` and
   ``decoder_type="conv"`` (see :class:`univii.config.ConvConfig`).
3. **Embedding modality.** Features from a pretrained image model are just a
   matrix: put them in an AnnData and use the default MLP encoder. No helper
   is needed.
4. **Spatial smoothness prior.** ``UniVIConfig(spatial_weight=..., spatial_sigma=...)``
   pulls the latents of nearby spots together. It needs coordinates in every
   batch, which :func:`make_spatial_loader` supplies.

Call hierarchy::

    make_spatial_loader            loader that carries coordinates and (optionally) spatially local batches
        coords_modality            coordinates as an AnnData the loader can pair with the data
        SpatialBlockSampler        batches made of neighbouring spots
    niche_modality                 neighbourhood-averaged copy of an AnnData
        neighborhood_aggregate     the same for a plain matrix
            knn_graph              k nearest spatial neighbours
    tiles_from_visium              image tiles for a Scanpy/Squidpy-style Visium AnnData
        extract_tiles              tiles around arbitrary pixel centres in any image array
    unflatten_tiles                flat tile matrix -> (n, C, H, W) for plotting
    median_neighbor_distance       typical spot spacing; a starting value for spatial_sigma
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Sampler


# ---------------------------------------------------------------------------
# neighbour graphs and aggregation
# ---------------------------------------------------------------------------

def knn_graph(coords, k: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """Indices and distances of each point's ``k`` nearest neighbours (self excluded).

    Returns ``(idx, dist)``, both ``(n, k)``.
    """
    from sklearn.neighbors import NearestNeighbors

    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2:
        raise ValueError("coords must be (n, dims)")
    if not 1 <= int(k) < coords.shape[0]:
        raise ValueError(f"k must be between 1 and n - 1 = {coords.shape[0] - 1}, got {k}")
    dist, idx = NearestNeighbors(n_neighbors=int(k) + 1).fit(coords).kneighbors(coords)
    # drop each point itself (robust to duplicate coordinates, where self may not come first)
    rows = np.arange(coords.shape[0])[:, None]
    is_self = idx == rows
    keep = np.argsort(is_self, axis=1, kind="stable")[:, :int(k)]
    missing = ~is_self.any(axis=1)
    if missing.any():                      # self absent from the k + 1 hits: keep the k closest
        keep[missing] = np.arange(int(k))[None, :]
    keep.sort(axis=1)
    return np.take_along_axis(idx, keep, 1), np.take_along_axis(dist, keep, 1)


def median_neighbor_distance(coords, k: int = 1) -> float:
    """Median distance to the ``k``-th nearest neighbour: the typical spot spacing."""
    return float(np.median(knn_graph(coords, k)[1][:, -1]))


def neighborhood_weights(coords, k: int = 6, kernel: str = "uniform", sigma: Optional[float] = None,
                         include_self: bool = False) -> sp.csr_matrix:
    """Row-normalised ``(n, n)`` sparse weights over each point's ``k`` nearest neighbours.

    ``kernel="uniform"`` weights neighbours equally; ``"gaussian"`` uses
    ``exp(-d^2 / (2 sigma^2))`` with ``sigma`` defaulting to the median
    nearest-neighbour distance.
    """
    idx, dist = knn_graph(coords, k)
    n = idx.shape[0]
    if kernel == "uniform":
        w = np.ones_like(dist)
    elif kernel == "gaussian":
        s = float(sigma) if sigma is not None else float(np.median(dist[:, 0]))
        if not s > 0:
            raise ValueError("sigma must be > 0 (duplicate coordinates give a zero default; pass sigma)")
        w = np.exp(-dist ** 2 / (2.0 * s ** 2))
    else:
        raise ValueError(f"kernel must be 'uniform' or 'gaussian', got {kernel!r}")
    W = sp.csr_matrix((w.ravel(), (np.repeat(np.arange(n), idx.shape[1]), idx.ravel())), shape=(n, n))
    if include_self:
        W = W + sp.identity(n, format="csr")
    row_sum = np.asarray(W.sum(1)).ravel()
    row_sum[row_sum == 0] = 1.0
    return sp.diags(1.0 / row_sum) @ W


def neighborhood_aggregate(X, coords, k: int = 6, kernel: str = "uniform", sigma: Optional[float] = None,
                           include_self: bool = False):
    """Weighted mean of each row's spatial neighbours. Sparse in, sparse out."""
    W = neighborhood_weights(coords, k, kernel, sigma, include_self)
    out = W @ X
    return out.tocsr() if sp.issparse(out) else np.asarray(out)


def niche_modality(adata, spatial_key: str = "spatial", k: int = 6, layer: Optional[str] = None,
                   kernel: str = "uniform", sigma: Optional[float] = None, include_self: bool = False,
                   group_key: Optional[str] = None):
    """AnnData whose ``X`` is the neighbourhood mean of ``adata`` (same cells, same features).

    Use it as a second modality so the model sees each spot together with its
    surroundings. With ``group_key`` (for example a slide or sample column)
    neighbours are searched within each group only, so slides do not mix.
    """
    import anndata as ad

    X = adata.layers[layer] if layer is not None else adata.X
    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    if group_key is None:
        agg = neighborhood_aggregate(X, coords, k, kernel, sigma, include_self)
    else:
        groups = adata.obs[group_key].astype(str).values
        agg = sp.lil_matrix(X.shape) if sp.issparse(X) else np.zeros(X.shape, dtype=float)
        for g in np.unique(groups):
            rows = np.flatnonzero(groups == g)
            agg[rows] = neighborhood_aggregate(X[rows], coords[rows], k, kernel, sigma, include_self)
        if sp.issparse(X):
            agg = agg.tocsr()
    out = ad.AnnData(X=agg, obs=adata.obs.copy(), var=adata.var.copy())
    out.obsm[spatial_key] = coords
    out.uns["niche"] = {"k": int(k), "kernel": kernel, "include_self": bool(include_self), "layer": layer,
                        "group_key": group_key}
    return out


# ---------------------------------------------------------------------------
# image tiles
# ---------------------------------------------------------------------------

def extract_tiles(image, centers_xy, tile_size: int, out_size: Optional[int] = None,
                  flatten: bool = True) -> np.ndarray:
    """Square tiles centred on pixel coordinates.

    Parameters
    ----------
    image
        ``(H, W)`` or ``(H, W, C)`` array. Integer images are scaled to ``[0, 1]``;
        float images are used as they are.
    centers_xy
        ``(n, 2)`` pixel coordinates as ``(x, y)`` = ``(column, row)``, the
        convention of ``adata.obsm["spatial"]``.
    tile_size
        Side length in image pixels. Tiles overlapping the border are
        completed by reflecting the image.
    out_size
        Resize tiles to this side length (area averaging); ``None`` keeps ``tile_size``.
    flatten
        Return ``(n, C * S * S)`` in channel-first order, the layout conv
        modalities expect. Otherwise ``(n, C, S, S)``.
    """
    img = np.asarray(image)
    if img.ndim == 2:
        img = img[:, :, None]
    if img.ndim != 3:
        raise ValueError("image must be (H, W) or (H, W, C)")
    tile_size = int(tile_size)
    if tile_size < 1:
        raise ValueError("tile_size must be >= 1")
    img = img.astype(np.float32) / 255.0 if np.issubdtype(img.dtype, np.integer) else img.astype(np.float32)
    centers = np.asarray(centers_xy, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers_xy must be (n, 2) as (x, y)")
    H, W, C = img.shape
    if (centers[:, 0] < 0).any() or (centers[:, 0] > W).any() or (centers[:, 1] < 0).any() or (centers[:, 1] > H).any():
        raise ValueError("some centres lie outside the image; check the coordinate scale factor")
    pad = tile_size
    mode = "reflect" if min(H, W) > pad else "edge"
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode=mode)
    x0 = np.rint(centers[:, 0] - tile_size / 2.0).astype(int) + pad
    y0 = np.rint(centers[:, 1] - tile_size / 2.0).astype(int) + pad
    tiles = np.stack([padded[y:y + tile_size, x:x + tile_size] for x, y in zip(x0, y0)])  # (n, S, S, C)
    tiles = np.ascontiguousarray(tiles.transpose(0, 3, 1, 2))
    if out_size is not None and int(out_size) != tile_size:
        t = torch.from_numpy(tiles)
        if tile_size % int(out_size) == 0:
            t = torch.nn.functional.avg_pool2d(t, tile_size // int(out_size))
        else:
            t = torch.nn.functional.interpolate(t, size=(int(out_size), int(out_size)), mode="bilinear",
                                                align_corners=False, antialias=True)
        tiles = t.numpy()
    return tiles.reshape(tiles.shape[0], -1) if flatten else tiles


def unflatten_tiles(X, shape: Sequence[int]) -> np.ndarray:
    """Flat ``(n, C * H * W)`` matrix -> ``(n, C, H, W)``. ``shape`` is ``(C, H, W)``,
    stored by :func:`tiles_from_visium` in ``adata.uns["tile_shape"]``."""
    X = np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)
    return X.reshape(X.shape[0], *[int(v) for v in shape])


def tiles_from_visium(adata, library_id: Optional[str] = None, img_key: str = "hires",
                      tile_size_fullres: Optional[float] = None, out_size: int = 32,
                      spatial_key: str = "spatial"):
    """Image-tile AnnData for a Visium-style AnnData (Scanpy / Squidpy layout).

    Reads ``adata.uns["spatial"][library_id]["images"][img_key]`` and its
    ``tissue_<img_key>_scalef`` scale factor, and cuts one tile per spot.
    ``tile_size_fullres`` is the tile side in full-resolution pixels (default:
    the spot diameter). Returns an AnnData with the same ``obs`` whose ``X``
    is the flat tile matrix in ``[0, 1]`` and whose ``uns["tile_shape"]`` is
    ``(C, out_size, out_size)``; use ``ConvConfig(in_channels=C,
    spatial_shape=(out_size, out_size))`` and ``input_dim = X.shape[1]``.

    Memory: ``n_spots * C * out_size**2 * 4`` bytes (5,000 spots of 3 x 32 x 32
    is about 61 MB).
    """
    import anndata as ad

    if "spatial" not in adata.uns:
        raise KeyError("adata.uns['spatial'] is missing; this does not look like a Visium AnnData")
    libs = list(adata.uns["spatial"])
    if library_id is None:
        if len(libs) != 1:
            raise ValueError(f"several libraries found ({libs}); pass library_id")
        library_id = libs[0]
    entry = adata.uns["spatial"][library_id]
    image = entry["images"][img_key]
    scales = entry["scalefactors"]
    scalef = float(scales[f"tissue_{img_key}_scalef"])
    diameter = float(tile_size_fullres) if tile_size_fullres is not None else float(scales["spot_diameter_fullres"])
    tile_px = max(2, int(round(diameter * scalef)))
    centers = np.asarray(adata.obsm[spatial_key], dtype=float)[:, :2] * scalef
    X = extract_tiles(image, centers, tile_px, out_size=out_size, flatten=True)
    channels = 1 if np.asarray(image).ndim == 2 else int(np.asarray(image).shape[2])
    out = ad.AnnData(X=X, obs=adata.obs.copy())
    out.obsm[spatial_key] = np.asarray(adata.obsm[spatial_key])
    out.uns["tile_shape"] = [channels, int(out_size), int(out_size)]
    out.uns["tile_source"] = {"library_id": library_id, "img_key": img_key, "tile_px_in_image": tile_px,
                              "scalef": scalef}
    return out


# ---------------------------------------------------------------------------
# coordinates in the batch, and spatially local batches
# ---------------------------------------------------------------------------

def coords_modality(adata, spatial_key: str = "spatial"):
    """Coordinates as an AnnData (``X`` = coordinates) that a loader can pair with the data."""
    import anndata as ad
    import pandas as pd

    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
    var = pd.DataFrame(index=[f"coord_{i}" for i in range(coords.shape[1])])
    return ad.AnnData(X=coords, obs=adata.obs[[]].copy(), var=var)


class SpatialBlockSampler(Sampler[List[int]]):
    """Batch sampler whose batches are spatial neighbourhoods.

    Each batch is a random unvisited seed spot plus its nearest spots, so most
    of a spot's neighbours share its batch and the spatial smoothness prior
    has real pairs to act on. Batches are drawn until every spot has been
    seen; because neighbourhoods overlap, a spot can appear in more than one
    batch per epoch. With ``groups`` (for example a slide label) batches never
    span groups.
    """

    def __init__(self, coords, batch_size: int = 256, groups: Optional[Sequence[Any]] = None, seed: int = 0):
        from sklearn.neighbors import NearestNeighbors

        self.coords = np.asarray(coords, dtype=float)
        self.batch_size = int(batch_size)
        if self.batch_size < 2:
            raise ValueError("batch_size must be >= 2")
        self.seed, self._epoch = int(seed), 0
        labels = np.zeros(len(self.coords), dtype=int) if groups is None else np.unique(np.asarray(groups).astype(str), return_inverse=True)[1]
        self._members = [np.flatnonzero(labels == g) for g in np.unique(labels)]
        self._trees = [NearestNeighbors().fit(self.coords[m]) for m in self._members]
        self._group_of = labels

    def __len__(self) -> int:
        return int(sum(np.ceil(len(m) / self.batch_size) for m in self._members))

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        unseen = np.ones(len(self.coords), dtype=bool)
        order = rng.permutation(len(self.coords))
        for seed_cell in order:
            if not unseen[seed_cell]:
                continue
            g = self._group_of[seed_cell]
            members = self._members[g]
            k = min(self.batch_size, len(members))
            if k < 2:
                unseen[members] = False
                continue
            local = self._trees[g].kneighbors(self.coords[seed_cell][None, :], n_neighbors=k, return_distance=False)[0]
            batch = members[local]
            unseen[batch] = False
            yield batch.tolist()


def make_spatial_loader(adata_by_mod: Mapping[str, Any], *, spatial_key: str = "spatial",
                        coords_from: Optional[str] = None, batch_size: int = 256, contiguous: bool = True,
                        group_key: Optional[str] = None, shuffle: bool = True, labels=None,
                        recon_targets_spec=None, seed: int = 0) -> DataLoader:
    """Loader whose batches carry spot coordinates under ``x_dict[spatial_key]``.

    Coordinates are read from ``adata_by_mod[coords_from].obsm[spatial_key]``
    (default: the first modality that has them). ``spatial_key`` must match
    ``UniVIConfig.spatial_key`` and must not be a modality name.

    ``contiguous=True`` draws spatially local batches with
    :class:`SpatialBlockSampler` (recommended for the smoothness prior, and for
    training only); ``False`` gives an ordinary loader, for example for
    evaluation. ``group_key`` names an ``obs`` column (slide, sample) that
    batches must not span.
    """
    from .data import MultiModalDataset, collate_multimodal_xy_recon

    if spatial_key in adata_by_mod:
        raise ValueError(f"{spatial_key!r} is already a modality name; choose another spatial_key")
    source = coords_from or next((k for k, a in adata_by_mod.items() if spatial_key in a.obsm), None)
    if source is None:
        raise KeyError(f"no modality has .obsm[{spatial_key!r}]")
    ref = adata_by_mod[source]
    with_coords: Dict[str, Any] = dict(adata_by_mod)
    with_coords[spatial_key] = coords_modality(ref, spatial_key)
    dataset = MultiModalDataset(with_coords, labels=labels, device=None, recon_targets_spec=recon_targets_spec)
    if contiguous:
        groups = None if group_key is None else ref.obs[group_key].astype(str).values
        sampler = SpatialBlockSampler(np.asarray(ref.obsm[spatial_key]), batch_size, groups=groups, seed=seed)
        return DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=collate_multimodal_xy_recon)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed), collate_fn=collate_multimodal_xy_recon)


__all__ = [
    "knn_graph", "median_neighbor_distance", "neighborhood_weights", "neighborhood_aggregate", "niche_modality",
    "extract_tiles", "unflatten_tiles", "tiles_from_visium",
    "coords_modality", "SpatialBlockSampler", "make_spatial_loader",
]
