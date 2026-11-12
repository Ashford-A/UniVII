"""Atlas, mosaic and bridge workflows.

A *mosaic* is a set of cohorts that measured different modalities, for example
RNA-only, ATAC-only and a smaller paired "bridge" cohort. The model already
accepts batches with any subset of its modalities; this module adds the pieces
around that:

    make_mosaic_loader   one training loader over several cohorts, with a cohort label per cell
    adapt_to_cohort      fit a new cohort onto a trained atlas while leaving the atlas unchanged
    encode_cohort        latent means for one cohort's cells
    MosaicLoader         the interleaving loader itself
    cohort_mixing        how well cohorts are mixed in a latent space (1 = fully mixed)
    label_purity         how well labels stay together (1 = every neighbour shares the label)
    transfer_labels      label, confidence and novelty for query cells from a labelled reference

To pull cohorts together during training, set ``UniVIConfig(cohort_mmd_weight=...)``;
the loader from :func:`make_mosaic_loader` supplies the cohort labels that term needs.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .geometry import pairwise_latent_distance, rows_per_chunk


class MosaicLoader:
    """Interleave batches from several loaders.

    Each step picks one loader at random (probability ``ratios``; by default
    proportional to its number of batches) and yields its next batch,
    restarting a loader that runs out. One pass yields ``epoch_length``
    batches (default: the total over all loaders). The order differs between
    epochs and is reproducible from ``seed``.

    A low ratio lets a small or noisy cohort trickle in without dominating;
    a high ratio oversamples a small bridge cohort.
    """

    def __init__(self, loaders, ratios: Optional[Sequence[float]] = None, epoch_length: Optional[int] = None,
                 seed: int = 0):
        if isinstance(loaders, Mapping):
            self.names, self.loaders = list(loaders), list(loaders.values())
        else:
            self.loaders = list(loaders)
            self.names = [str(i) for i in range(len(self.loaders))]
        if not self.loaders:
            raise ValueError("MosaicLoader needs at least one loader")
        sizes = np.array([len(l) for l in self.loaders], dtype=float)
        if (sizes < 1).any():
            raise ValueError("every loader must yield at least one batch")
        r = sizes if ratios is None else np.asarray(ratios, dtype=float)
        if r.shape != sizes.shape or (r < 0).any() or r.sum() <= 0:
            raise ValueError("ratios must be one non-negative number per loader, not all zero")
        self.ratios = r / r.sum()
        self.epoch_length = int(epoch_length) if epoch_length is not None else int(sizes.sum())
        self.seed, self._epoch = int(seed), 0

    def __len__(self) -> int:
        return self.epoch_length

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        iters = [iter(l) for l in self.loaders]
        for _ in range(self.epoch_length):
            i = int(rng.choice(len(self.loaders), p=self.ratios))
            try:
                batch = next(iters[i])
            except StopIteration:
                iters[i] = iter(self.loaders[i])
                batch = next(iters[i])
            yield batch


def make_mosaic_loader(cohorts: Mapping[str, Mapping[str, Any]], *, batch_size: int = 256,
                       labels: Optional[Mapping[str, Mapping[str, Any]]] = None, cohort_head: str = "cohort",
                       ratios: Optional[Sequence[float]] = None, epoch_length: Optional[int] = None,
                       shuffle: bool = True, drop_last: bool = False, seed: int = 0) -> MosaicLoader:
    """Build one training loader over cohorts that measured different modalities.

    Parameters
    ----------
    cohorts
        ``{cohort name: {modality: AnnData}}``. Within a cohort the modalities must hold the same
        cells in the same order; across cohorts the modality sets may differ. Modalities shared by
        several cohorts must use the same features, prepared the same way.
    labels
        Optional ``{cohort name: {head: integer codes}}`` for supervised heads, for cohorts that
        have labels.
    cohort_head
        Every cell also gets an integer cohort code under this name in ``y``, which is what
        ``UniVIConfig(cohort_mmd_weight=...)`` and adversarial heads read. The codes are in
        ``loader.cohort_codes``.
    ratios, epoch_length, seed
        Passed to :class:`MosaicLoader`.
    """
    from .workflows import make_loader

    labels = labels or {}
    codes = {name: i for i, name in enumerate(cohorts)}
    loaders = {}
    for name, adatas in cohorts.items():
        n = next(iter(adatas.values())).n_obs
        y = dict(labels.get(name, {}))
        if cohort_head in y:
            raise ValueError(f"labels for cohort {name!r} already use the name {cohort_head!r}")
        y[cohort_head] = np.full(n, codes[name], dtype=np.int64)
        loaders[name] = make_loader(dict(adatas), labels=y, batch_size=batch_size, shuffle=shuffle,
                                    drop_last=drop_last)
    out = MosaicLoader(loaders, ratios=ratios, epoch_length=epoch_length, seed=seed)
    out.cohort_codes = codes
    return out


# ----------------------------------------------------------------------------- new cohorts

def _dense(x) -> np.ndarray:
    x = x.X if hasattr(x, "X") else x
    x = x.toarray() if hasattr(x, "toarray") else x
    return np.asarray(x.detach().cpu() if torch.is_tensor(x) else x, dtype=np.float32)


def encode_cohort(model, data: Mapping[str, Any], cohort: Optional[int] = None, batch_size: int = 1024) -> np.ndarray:
    """Fused latent means for the cells of one cohort.

    ``data`` maps modality name to an AnnData, array or tensor. ``cohort`` is
    the cohort's code; it matters for cohorts added with ``model.add_cohort()``,
    whose input shift is applied only when the model is told which cohort the
    cells belong to. The model's cohort setting is restored afterwards.
    """
    arrays = {k: _dense(v) for k, v in data.items()}
    n = next(iter(arrays.values())).shape[0]
    device = next(model.parameters()).device
    previous, was_training = model._decode_cohort, model.training
    model.eval()
    try:
        if cohort is not None:
            model.set_cohort(int(cohort))
        out = []
        with torch.no_grad():
            for start in range(0, n, int(batch_size)):
                x = {k: torch.as_tensor(v[start:start + batch_size], device=device) for k, v in arrays.items()}
                out.append(model.encode_fused(x)[0].cpu().numpy())
        return np.vstack(out)
    finally:
        model.set_cohort(previous)
        model.train(was_training)


def adapt_to_cohort(model, loader, *, reference_z=None, code: Optional[int] = None, n_epochs: int = 50,
                    lr: float = 1e-2, mmd_weight: float = 10.0, mmd_batch: int = 512, seed: int = 0) -> Dict[str, Any]:
    """Fit a new cohort onto a trained atlas without changing the atlas.

    Everything the model has learned is frozen. Only two small sets of numbers
    are trained, both specific to the new cohort: a shift per feature that is
    added to the cohort's data, and the cohort's row in each decoder's cohort
    embedding. The embeddings of existing cohorts are therefore exactly what
    they were.

    The shift is a learned correction that moves the new cohort's data into the
    feature space the atlas was trained on. For modalities with a Gaussian-type
    likelihood the shifted data are both what the encoder sees and what the
    decoder has to reconstruct, so the shift is pulled toward wherever the
    frozen model explains the data best. For other likelihoods (counts) only
    the encoder input is shifted.

    The objective is the model's usual loss on the new cells plus, if
    ``reference_z`` is given, an MMD penalty pulling the new cells' latent means
    toward those reference latent means (use the fused means of the atlas, for
    example from :func:`encode_cohort`).

    Parameters
    ----------
    loader
        Batches of the new cohort (any loader the trainer accepts). Cohort labels in the batches
        are ignored; every cell is treated as the new cohort.
    code
        Code of the cohort to train. By default a new one is created with ``model.add_cohort()``.
    reference_z
        ``(cells, latent_dim)`` latent means to align with.

    Returns
    -------
    dict
        ``{"code": int, "history": [mean loss per epoch]}``. Afterwards, embed the new cells with
        ``encode_cohort(model, data, cohort=code)``.

    Notes
    -----
    A shift corrects differences that are roughly constant per feature. It cannot fix a cohort
    whose features relate to biology differently; if alignment stays poor, fine-tune the encoders.
    The learned shifts are in ``model.input_shifts[modality].weight[code]``.
    """
    if int(model.cfg.n_cohorts) < 1:
        raise ValueError("adapt_to_cohort needs a model built with n_cohorts >= 1 (cohort-conditioned decoders)")
    code = model.add_cohort() if code is None else int(code)
    if not model.input_shifts:
        raise ValueError("this model has no input shifts; add the cohort with model.add_cohort()")
    device = next(model.parameters()).device
    head = model.cfg.cohort_head
    trainable = [d.embed.weight for d in model.decoders.values() if hasattr(d, "embed")]
    trainable += [s.weight for s in model.input_shifts.values()]
    saved = {p: p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    for p in trainable:
        p.requires_grad_(True)
    ref = None if reference_z is None else torch.as_tensor(_np(reference_z), dtype=torch.float32, device=device)
    gen = torch.Generator().manual_seed(int(seed))
    gaussian = ("gaussian", "normal", "mse", "gaussian_diag")
    harmonised = {m.name for m in model.cfg.modalities if (m.likelihood or "gaussian").lower() in gaussian
                  and m.name in model.input_shifts}
    was_training = model.training
    model.eval()                                    # frozen parts keep their statistics; no dropout
    model._skip_input_shift = set(harmonised)       # applied here instead, to input and target alike
    opt = torch.optim.Adam(trainable, lr=lr)
    history = []
    try:
        for _ in range(int(n_epochs)):
            total, count = 0.0, 0
            for batch in loader:
                x_dict = batch[0] if isinstance(batch, (tuple, list)) else batch
                x_dict = {k: torch.as_tensor(v, device=device) for k, v in x_dict.items() if v is not None}
                n = next(iter(x_dict.values())).shape[0]
                for m in harmonised:                # shift input and reconstruction target together
                    if m in x_dict:
                        x_dict[m] = x_dict[m] + model.input_shifts[m].weight[code].to(x_dict[m].dtype)
                out = model(x_dict, epoch=10 ** 6, y={head: torch.full((n,), code, dtype=torch.long, device=device)})
                loss = out["loss"]
                if ref is not None and n >= 2:
                    pick = torch.randint(0, ref.shape[0], (min(int(mmd_batch), ref.shape[0]),), generator=gen)
                    loss = loss + float(mmd_weight) * model._mmd(out["mu_z"], ref[pick.to(device)].to(out["mu_z"].dtype))
                opt.zero_grad()
                loss.backward()
                for p in trainable:                 # only the new cohort's row may move
                    if p.grad is None:              # a modality the new cohort did not measure
                        continue
                    keep = torch.zeros(p.shape[0], 1, dtype=p.dtype, device=p.device)
                    keep[code] = 1.0
                    p.grad.mul_(keep)
                opt.step()
                total, count = total + float(loss.detach()) * n, count + n
            history.append(total / max(count, 1))
    finally:
        model._skip_input_shift = set()
        for p, flag in saved.items():
            p.requires_grad_(flag)
        model.train(was_training)
    return {"code": code, "history": history}


# ----------------------------------------------------------------------------- neighbours

def _np(z) -> np.ndarray:
    return np.asarray(z.detach().cpu() if torch.is_tensor(z) else z, dtype=float)


def _knn(query: np.ndarray, ref: np.ndarray, k: int, geo: Mapping[str, Any], exclude_self: bool,
         chunk: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """Indices and distances of the k nearest reference rows for every query row.

    Euclidean spaces use a tree or brute-force index from scikit-learn. Other
    geometries compute geodesic distances in blocks sized to the reference, so
    memory stays bounded however many reference cells there are. With
    ``exclude_self`` the query must be the reference itself.
    """
    k = min(int(k), ref.shape[0] - (1 if exclude_self else 0))
    if k < 1:
        raise ValueError("not enough cells for a neighbour search")
    if geo.get("geometry", "euclidean") == "euclidean":
        from sklearn.neighbors import NearestNeighbors
        dist, idx = NearestNeighbors(n_neighbors=k + int(exclude_self)).fit(ref).kneighbors(query)
        if exclude_self:                                       # drop each row's own index; if absent (duplicates), drop the farthest
            own = idx == np.arange(query.shape[0])[:, None]
            order = np.argsort(own, axis=1, kind="stable")
            idx, dist = np.take_along_axis(idx, order, 1)[:, :k], np.take_along_axis(dist, order, 1)[:, :k]
        return idx, dist
    step = rows_per_chunk(ref.shape[0], chunk)
    idx_out, dist_out = [], []
    for start in range(0, query.shape[0], step):
        D = pairwise_latent_distance(query[start:start + step], ref, **geo)
        if exclude_self:
            D[np.arange(D.shape[0]), np.arange(start, start + D.shape[0])] = np.inf
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
        idx_out.append(idx)
        dist_out.append(np.take_along_axis(D, idx, 1))
    return np.vstack(idx_out), np.vstack(dist_out)


def _geo(geometry, hyperbolic_dim, curvature) -> Dict[str, Any]:
    return {"geometry": geometry, "hyperbolic_dim": hyperbolic_dim, "curvature": curvature}


def cohort_mixing(z, cohort, k: int = 15, geometry: str = "euclidean", hyperbolic_dim: Optional[int] = None,
                  curvature: float = 1.0) -> float:
    """How well cohorts are mixed, from 0 (separate) to about 1 (as mixed as their sizes allow).

    For each cell, the share of its ``k`` nearest neighbours that come from a
    *different* cohort is divided by the share expected if cohorts were
    perfectly mixed; the result is averaged over cells. Values can slightly
    exceed 1 by chance. Read it next to :func:`label_purity`: a latent that
    has collapsed mixes cohorts perfectly and is useless.
    """
    z, cohort = _np(z), np.asarray(cohort)
    idx, _ = _knn(z, z, k, _geo(geometry, hyperbolic_dim, curvature), exclude_self=True)
    other = (cohort[idx] != cohort[:, None]).mean(1)
    names, counts = np.unique(cohort, return_counts=True)
    if len(names) < 2:
        raise ValueError("cohort_mixing needs at least two cohorts")
    share = dict(zip(names.tolist(), (counts / counts.sum()).tolist()))
    expected = np.array([1.0 - share[c] for c in cohort.tolist()])
    return float(np.mean(other / expected))


def label_purity(z, labels, k: int = 15, geometry: str = "euclidean", hyperbolic_dim: Optional[int] = None,
                 curvature: float = 1.0) -> float:
    """Average share of each cell's ``k`` nearest neighbours that carry its own label."""
    z, labels = _np(z), np.asarray(labels)
    idx, _ = _knn(z, z, k, _geo(geometry, hyperbolic_dim, curvature), exclude_self=True)
    return float((labels[idx] == labels[:, None]).mean())


def transfer_labels(z_ref, labels_ref, z_query, k: int = 15, geometry: str = "euclidean",
                    hyperbolic_dim: Optional[int] = None, curvature: float = 1.0) -> pd.DataFrame:
    """Give query cells labels from a labelled reference embedded in the same latent space.

    Returns one row per query cell:

    - ``label``: the most common label among the ``k`` nearest reference cells.
    - ``confidence``: the share of those neighbours carrying that label.
    - ``novelty``: mean distance to those neighbours, divided by the median of the same
      quantity for reference cells among themselves. Values near 1 mean the query cell sits
      as close to the reference as reference cells sit to each other; values well above 1
      flag cells the reference does not cover, whatever their ``confidence``.
    """
    z_ref, z_query, labels_ref = _np(z_ref), _np(z_query), np.asarray(labels_ref)
    geo = _geo(geometry, hyperbolic_dim, curvature)
    idx, dist = _knn(z_query, z_ref, k, geo, exclude_self=False)
    names, codes = np.unique(labels_ref, return_inverse=True)
    votes = np.zeros((z_query.shape[0], len(names)))
    np.add.at(votes, (np.repeat(np.arange(z_query.shape[0]), idx.shape[1]), codes[idx].ravel()), 1.0)
    best = votes.argmax(1)
    _, ref_dist = _knn(z_ref, z_ref, k, geo, exclude_self=True)
    typical = max(float(np.median(ref_dist.mean(1))), 1e-12)
    return pd.DataFrame({"label": names[best], "confidence": votes.max(1) / idx.shape[1],
                         "novelty": dist.mean(1) / typical})


__all__ = ["MosaicLoader", "make_mosaic_loader", "encode_cohort", "adapt_to_cohort", "cohort_mixing",
           "label_purity", "transfer_labels"]
