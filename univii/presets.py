"""Ready-made configurations for common experiment types.

Each function returns a :class:`univii.UniVIConfig` you can pass straight to
``UniVIMultiModalVAE`` or adjust first; any keyword you add overrides the preset.

    paired_config          any set of matrix modalities
    clustered_config       ... with a learned mixture prior
    contrastive_config     ... with contrastive alignment between modalities
    hierarchy_config       ... with a hyperbolic or product latent and label-tree supervision
    atlas_config           several cohorts, possibly with different modalities, aligned in one latent
    spatial_image_config   expression plus image tiles (conv or ViT), with a spatial prior
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from .config import ClassHeadConfig, ConvConfig, ModalityConfig, UniVIConfig, ViTConfig


def _hidden(n_features: int) -> list:
    if n_features > 2000:
        return [512, 256]
    if n_features > 200:
        return [256, 128]
    return [64]


def _modalities(input_dims: Mapping[str, int], likelihoods: Optional[Mapping[str, str]] = None) -> list:
    likelihoods = dict(likelihoods or {})
    return [ModalityConfig(name, int(n), _hidden(int(n)), _hidden(int(n))[::-1],
                           likelihood=likelihoods.get(name, "gaussian")) for name, n in input_dims.items()]


def paired_config(input_dims: Mapping[str, int], likelihoods: Optional[Mapping[str, str]] = None,
                  latent_dim: int = 30, **overrides: Any) -> UniVIConfig:
    """Baseline: one MLP encoder and decoder per modality, Euclidean latent.

    ``input_dims`` maps modality name to feature count, for example
    ``{"rna": 2000, "adt": 130}``; ``likelihoods`` defaults to Gaussian.
    """
    return UniVIConfig(**{"latent_dim": latent_dim, "modalities": _modalities(input_dims, likelihoods), **overrides})


def clustered_config(input_dims: Mapping[str, int], n_components: int = 16, **kw: Any) -> UniVIConfig:
    """Mixture-of-Gaussians prior, for data expected to form discrete groups."""
    return paired_config(input_dims, **{"prior": "mog", "prior_components": n_components, **kw})


def contrastive_config(input_dims: Mapping[str, int], weight: float = 1.0, temperature: float = 1.0,
                       **kw: Any) -> UniVIConfig:
    """Adds an InfoNCE term that makes each cell's modalities retrieve each other."""
    return paired_config(input_dims, **{"contrastive_weight": weight, "contrastive_temperature": temperature, **kw})


def hierarchy_config(input_dims: Mapping[str, int], n_labels: int, label_head: str = "celltype",
                     geometry: str = "product", hierarchy_weight: float = 5.0, hierarchy_scale: float = 1.0,
                     **kw: Any) -> UniVIConfig:
    """Hyperbolic or product latent with label-tree supervision.

    After building the model, register the tree with
    ``model.set_hierarchy(dist, root_index=...)`` and pass labels as
    ``y={label_head: ...}``.
    """
    base = {"latent_geometry": geometry, "hierarchy_head": label_head, "hierarchy_weight": hierarchy_weight,
            "hierarchy_root_weight": 1.0, "hierarchy_scale": hierarchy_scale,
            "class_heads": [ClassHeadConfig(name=label_head, n_classes=int(n_labels))]}
    return paired_config(input_dims, **{**base, **kw})


def atlas_config(input_dims: Mapping[str, int], n_cohorts: int, likelihoods: Optional[Mapping[str, str]] = None,
                 cohort_mmd_weight: float = 10.0, **kw: Any) -> UniVIConfig:
    """Several cohorts, possibly measuring different modalities, in one latent space.

    Decoders are conditioned on the cohort, the latent means of different
    cohorts are pulled together, and encoder batch normalisation is off
    because training batches usually hold one cohort each. Train with
    ``univii.atlas.make_mosaic_loader``.
    """
    base = {"n_cohorts": int(n_cohorts), "cohort_mmd_weight": cohort_mmd_weight, "encoder_batchnorm": False}
    return paired_config(input_dims, likelihoods, **{**base, **kw})


def spatial_image_config(n_genes: int, tile_shape: Sequence[int], spatial_sigma: float, encoder: str = "conv",
                         rna_likelihood: str = "gaussian", latent_dim: int = 32, spatial_weight: float = 1.0,
                         **kw: Any) -> UniVIConfig:
    """Expression plus image tiles with a spatial smoothness prior.

    ``tile_shape`` is ``(C, H, W)``, as stored in ``tiles.uns["tile_shape"]`` by
    ``univii.spatial.tiles_from_visium``. ``encoder`` is ``"conv"`` or ``"vit"``;
    the decoder is convolutional either way. Use with ``loss_mode="v1"``, or
    pass ``recon_normalize_by_dim=False`` to the model with ``loss_mode="v2"``.
    """
    c, h, w = (int(v) for v in tile_shape)
    stages = 3 if min(h, w) >= 32 else 2
    conv = ConvConfig(in_channels=c, spatial_shape=(h, w), channels=(32, 64, 128)[:stages],
                      output_activation="sigmoid", augment=("flip", "rot90") if h == w else ("flip",))
    image = dict(likelihood="gaussian", decoder_type="conv", conv=conv)
    if encoder == "conv":
        image.update(encoder_type="conv")
    elif encoder == "vit":
        patch = 8 if min(h, w) >= 32 else 4
        image.update(encoder_type="vit", vit=ViTConfig(in_channels=c, image_size=(h, w), patch_size=patch,
                                                       augment=conv.augment))
    else:
        raise ValueError("encoder must be 'conv' or 'vit'")
    mods = [ModalityConfig("rna", int(n_genes), _hidden(int(n_genes)), _hidden(int(n_genes))[::-1],
                           likelihood=rna_likelihood),
            ModalityConfig("image", c * h * w, [], [], **image)]
    return UniVIConfig(**{"latent_dim": latent_dim, "modalities": mods, "spatial_weight": spatial_weight,
                          "spatial_sigma": float(spatial_sigma), **kw})


__all__ = ["paired_config", "clustered_config", "contrastive_config", "hierarchy_config", "atlas_config",
           "spatial_image_config"]
