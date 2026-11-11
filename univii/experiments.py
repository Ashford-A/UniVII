"""Run the same experiment across several model configurations and seeds.

    compare_models   train each variant for each seed, evaluate, return one row per run
    summarize        mean and standard deviation per variant

Example::

    variants = {
        "euclidean": presets.paired_config(dims),
        "product":   presets.paired_config(dims, latent_geometry="product"),
    }
    runs = compare_models(variants, train_loader, evaluate=lambda model: {"my_metric": ...}, seeds=(0, 1, 2))
    print(summarize(runs))
"""
from __future__ import annotations

import copy
import time
from dataclasses import replace
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
import torch

from .config import TrainingConfig, UniVIConfig

Variant = Union[UniVIConfig, Tuple[UniVIConfig, Mapping[str, Any]], Callable[[], torch.nn.Module]]


def _build(variant: Variant) -> torch.nn.Module:
    from .models import UniVIMultiModalVAE

    if isinstance(variant, UniVIConfig):
        return UniVIMultiModalVAE(copy.deepcopy(variant))
    if isinstance(variant, tuple):
        cfg, kwargs = variant
        return UniVIMultiModalVAE(copy.deepcopy(cfg), **dict(kwargs))
    return variant()


def compare_models(variants: Mapping[str, Variant], train_loader, evaluate: Callable[[torch.nn.Module], Mapping[str, float]],
                   seeds: Sequence[int] = (0,), train_cfg: Optional[TrainingConfig] = None, val_loader=None,
                   setup: Optional[Callable[[torch.nn.Module], None]] = None, keep_models: bool = False):
    """Train and evaluate every variant for every seed.

    Parameters
    ----------
    variants
        ``{name: variant}``. A variant is a ``UniVIConfig``, a ``(config, model_kwargs)``
        tuple (for example ``(cfg, {"loss_mode": "v2"})``), or a function returning a model.
    train_loader, val_loader
        Loaders, or functions ``seed -> loader`` if the data order should change with the seed.
    evaluate
        ``model -> {metric: value}``, called on the trained model in eval mode. Compute your
        metrics on held-out data here.
    setup
        Optional ``model -> None`` run after construction and before training, for steps such as
        ``model.set_hierarchy(...)``.
    keep_models
        Also return ``{(name, seed): model}``.

    Returns
    -------
    pandas.DataFrame
        One row per run: ``variant``, ``seed``, ``n_parameters``, ``train_seconds``,
        ``final_train_loss`` and the metrics. With ``keep_models=True`` the result is the tuple
        ``(frame, models)``.
    """
    from .trainer import UniVITrainer

    train_cfg = train_cfg or TrainingConfig(n_epochs=50, log_every=10 ** 9)
    rows, models = [], {}
    for name, variant in variants.items():
        for seed in seeds:
            torch.manual_seed(int(seed))
            model = _build(variant)
            if setup is not None:
                setup(model)
            tl = train_loader(seed) if callable(train_loader) else train_loader
            vl = val_loader(seed) if callable(val_loader) else val_loader
            start = time.time()
            history = UniVITrainer(model, tl, val_loader=vl, train_cfg=replace(train_cfg, seed=int(seed))).fit()
            seconds = time.time() - start
            model.eval()
            with torch.no_grad():
                metrics = dict(evaluate(model))
            rows.append({"variant": name, "seed": int(seed),
                         "n_parameters": sum(p.numel() for p in model.parameters()),
                         "train_seconds": seconds, "final_train_loss": float(history["train_loss"][-1]), **metrics})
            if keep_models:
                models[(name, int(seed))] = model
    frame = pd.DataFrame(rows)
    return (frame, models) if keep_models else frame


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of every numeric column per variant, in the order given."""
    numeric = [c for c in runs.columns if c not in ("variant", "seed") and pd.api.types.is_numeric_dtype(runs[c])]
    out = runs.groupby("variant", sort=False)[numeric].agg(["mean", "std"])
    out.insert(0, "n_seeds", runs.groupby("variant", sort=False)["seed"].nunique())
    return out


__all__ = ["compare_models", "summarize"]
