"""Stain normalisation for H&E image tiles (Macenko's method).

Slides stained in different batches or scanned on different machines differ
in colour for reasons unrelated to the tissue. This estimates each slide's two
stain colours, separates every pixel into stain amounts, and re-renders the
tiles with one common pair of colours and a common intensity.

    MacenkoNormalizer   fit a target appearance, then transform the tiles of other slides
    rgb_to_od, od_to_rgb   conversion between pixel intensity and optical density

Fit and transform one slide (one staining batch) at a time: the stain colours
are estimated from the pixels passed in, so they should share a stain.
Reference: Macenko et al., IEEE ISBI 2009.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

_FLOOR = 1.0 / 255.0


def rgb_to_od(rgb: np.ndarray) -> np.ndarray:
    """Optical density ``-log(I)`` of intensities in ``[0, 1]`` (floored at 1/255)."""
    return -np.log(np.clip(np.asarray(rgb, dtype=np.float64), _FLOOR, 1.0))


def od_to_rgb(od: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rgb_to_od`, clipped to ``[0, 1]``."""
    return np.clip(np.exp(-np.asarray(od, dtype=np.float64)), 0.0, 1.0)


def _pixels(tiles) -> Tuple[np.ndarray, tuple, Optional[object]]:
    """Return pixels as ``(n_pixels, 3)``, the original array shape, and the AnnData if one was given."""
    adata = None
    if hasattr(tiles, "uns") and hasattr(tiles, "X"):
        adata = tiles
        shape = tuple(int(v) for v in adata.uns["tile_shape"])
        arr = np.asarray(adata.X, dtype=np.float64).reshape((adata.n_obs,) + shape)
    else:
        arr = np.asarray(tiles, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[None]
    if arr.ndim != 4 or arr.shape[1] != 3:
        raise ValueError(f"expected RGB tiles as (n, 3, H, W) or a tile AnnData, got shape {arr.shape}")
    if arr.min() < 0.0 or arr.max() > 1.0 + 1e-6:
        raise ValueError("tile intensities must lie in [0, 1]")
    return arr.transpose(0, 2, 3, 1).reshape(-1, 3), arr.shape, adata


class MacenkoNormalizer:
    """Macenko stain normalisation.

    After :meth:`fit`, ``stain_matrix`` holds the target's two stain colours as columns of a
    ``(3, 2)`` matrix (unit optical-density vectors, haematoxylin first) and ``max_concentration``
    the target's stain intensities.

    Parameters
    ----------
    od_threshold
        Pixels whose optical density is below this in every channel are treated as background
        (glass) and ignored when estimating stain colours.
    angle_percentile
        Robustness of the stain-colour estimate: the two colours are taken at this percentile and
        its complement of the pixel directions in the stain plane.
    intensity_percentile
        Stain amounts are scaled so that this percentile matches the target's.
    max_pixels
        Estimate from at most this many randomly chosen tissue pixels.
    seed
        Seed for that random choice.
    """

    def __init__(self, od_threshold: float = 0.15, angle_percentile: float = 1.0,
                 intensity_percentile: float = 99.0, max_pixels: int = 200_000, seed: int = 0):
        self.od_threshold, self.angle_percentile = float(od_threshold), float(angle_percentile)
        self.intensity_percentile, self.max_pixels, self.seed = float(intensity_percentile), int(max_pixels), int(seed)
        self.stain_matrix: Optional[np.ndarray] = None
        self.max_concentration: Optional[np.ndarray] = None

    def estimate(self, tiles) -> Tuple[np.ndarray, np.ndarray]:
        """Stain colours ``(3, 2)`` and stain intensities ``(2,)`` of one slide's tiles."""
        od = rgb_to_od(_pixels(tiles)[0])
        tissue = od[(od > self.od_threshold).any(axis=1)]
        if tissue.shape[0] < 100:
            raise ValueError("too few tissue pixels to estimate stain colours (is this background?)")
        if tissue.shape[0] > self.max_pixels:
            tissue = tissue[np.random.default_rng(self.seed).choice(tissue.shape[0], self.max_pixels, replace=False)]
        # plane spanned by the two strongest directions of optical density
        _, vecs = np.linalg.eigh(np.cov(tissue.T))
        plane = vecs[:, [2, 1]]
        if plane[:, 0].sum() < 0:
            plane[:, 0] *= -1
        if plane[:, 1].sum() < 0:
            plane[:, 1] *= -1
        proj = tissue @ plane
        angle = np.arctan2(proj[:, 1], proj[:, 0])
        lo, hi = np.percentile(angle, [self.angle_percentile, 100.0 - self.angle_percentile])
        v1 = plane @ np.array([np.cos(lo), np.sin(lo)])
        v2 = plane @ np.array([np.cos(hi), np.sin(hi)])
        # haematoxylin absorbs more red than eosin does, so it has the larger first component
        stains = np.stack([v1, v2], axis=1) if v1[0] > v2[0] else np.stack([v2, v1], axis=1)
        stains = stains / np.linalg.norm(stains, axis=0, keepdims=True)
        conc = np.linalg.lstsq(stains, tissue.T, rcond=None)[0]
        return stains, np.percentile(conc, self.intensity_percentile, axis=1)

    def fit(self, tiles) -> "MacenkoNormalizer":
        """Take the target appearance from the tiles of a reference slide."""
        self.stain_matrix, self.max_concentration = self.estimate(tiles)
        return self

    def transform(self, tiles):
        """Re-render one slide's tiles with the target stain colours and intensity.

        Returns the same kind of object that was passed in: an array of the same
        shape, or a copy of the tile AnnData.
        """
        if self.stain_matrix is None:
            raise RuntimeError("call fit(reference_tiles) first")
        pixels, shape, adata = _pixels(tiles)
        stains, max_c = self.estimate(tiles)
        od = rgb_to_od(pixels)
        conc = np.linalg.lstsq(stains, od.T, rcond=None)[0]
        conc = np.clip(conc, 0.0, None) * (self.max_concentration / np.maximum(max_c, 1e-8))[:, None]
        out = od_to_rgb((self.stain_matrix @ conc).T).reshape(shape[0], shape[2], shape[3], 3).transpose(0, 3, 1, 2)
        if adata is not None:
            new = adata.copy()
            new.X = out.reshape(adata.n_obs, -1).astype(np.float32)
            return new
        return out.astype(np.float32).reshape(np.asarray(tiles).shape)


__all__ = ["MacenkoNormalizer", "rgb_to_od", "od_to_rgb"]
