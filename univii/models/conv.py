"""Convolutional encoders and decoders for grid-shaped modalities.

A conv modality is stored and passed around as a flat ``(cells, input_dim)``
matrix in channel-first order (``input_dim = C * H * W`` for image tiles,
``C * L`` for 1-D signals). The modules here reshape internally, so the data
loaders, likelihoods, losses and exports need no special cases.

    ConvGaussianEncoder       flat grid -> (mu, logvar)
    ConvDecoder               z -> flat grid (mean, mean + logvar, or logits)
    BackboneGaussianEncoder   wrap any feature extractor (for example a pretrained
                              image model) with a trainable head -> (mu, logvar)

Importing this module registers ``encoder_type="conv"`` and
``decoder_type="conv"`` in :mod:`univii.models.registry`.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from ..config import ConvConfig
from .registry import register_decoder, register_encoder

_ACT = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "leakyrelu": lambda: nn.LeakyReLU(0.2)}
#: likelihoods a conv decoder can emit, mapped to its output kind
CONV_LIKELIHOODS = {"gaussian": "mean", "normal": "mean", "mse": "mean",
                    "gaussian_diag": "mean_logvar", "bernoulli": "logits"}


def _conv(ndim: int):
    return nn.Conv1d if ndim == 1 else nn.Conv2d


def _conv_t(ndim: int):
    return nn.ConvTranspose1d if ndim == 1 else nn.ConvTranspose2d


def _norm(kind: str, channels: int, groups: int, ndim: int) -> nn.Module:
    if kind == "none":
        return nn.Identity()
    if kind == "batch":
        return (nn.BatchNorm1d if ndim == 1 else nn.BatchNorm2d)(channels)
    g = math.gcd(int(groups), int(channels))        # largest group count that divides the channels
    return nn.GroupNorm(g, channels)


class _ResBlock(nn.Module):
    def __init__(self, ch: int, cfg: ConvConfig):
        super().__init__()
        nd, k = cfg.ndim, int(cfg.kernel_size)
        self.body = nn.Sequential(
            _norm(cfg.norm, ch, cfg.groups, nd), _ACT[cfg.activation](), _conv(nd)(ch, ch, k, padding=k // 2),
            _norm(cfg.norm, ch, cfg.groups, nd), _ACT[cfg.activation](), _conv(nd)(ch, ch, k, padding=k // 2),
        )

    def forward(self, x):
        return x + self.body(x)


def _stage_tail(ch: int, cfg: ConvConfig) -> nn.Sequential:
    layers = [_norm(cfg.norm, ch, cfg.groups, cfg.ndim), _ACT[cfg.activation]()]
    layers += [_ResBlock(ch, cfg) for _ in range(int(cfg.blocks_per_stage))]
    if float(cfg.dropout) > 0:
        layers.append(nn.Dropout(float(cfg.dropout)))
    return nn.Sequential(*layers)


def _as_grid(x: torch.Tensor, cfg: ConvConfig) -> torch.Tensor:
    """Accept a flat ``(B, C * prod(shape))`` matrix or an already shaped ``(B, C, *shape)`` tensor."""
    shape = (int(cfg.in_channels), *[int(v) for v in cfg.spatial_shape])
    if x.ndim == 2:
        if x.shape[1] != cfg.flat_dim:
            raise ValueError(f"expected {cfg.flat_dim} features (C * prod(spatial_shape)), got {x.shape[1]}")
        return x.reshape(x.shape[0], *shape)
    if tuple(x.shape[1:]) != shape:
        raise ValueError(f"expected input of shape (B, {', '.join(map(str, shape))}), got {tuple(x.shape)}")
    return x


class ConvGaussianEncoder(nn.Module):
    """Strided residual conv encoder: grid -> ``(mu, logvar)``."""

    def __init__(self, cfg: ConvConfig, latent_dim: int):
        super().__init__()
        cfg.validate()
        self.cfg, self.latent_dim = cfg, int(latent_dim)
        nd, k = cfg.ndim, int(cfg.kernel_size)
        chans = [int(c) for c in cfg.channels]
        layers = [_conv(nd)(int(cfg.in_channels), chans[0], k, padding=k // 2)]
        prev = chans[0]
        for ch in chans:
            layers += [_conv(nd)(prev, ch, k, stride=2, padding=k // 2), _stage_tail(ch, cfg)]
            prev = ch
        self.features_net = nn.Sequential(*layers)
        self.out_shape = tuple(int(v) // 2 ** len(chans) for v in cfg.spatial_shape)
        self.feature_dim = prev if cfg.pool == "mean" else prev * int(math.prod(self.out_shape))
        self.head = nn.Sequential(nn.Linear(self.feature_dim, int(cfg.hidden_dim)), _ACT[cfg.activation](),
                                  nn.Linear(int(cfg.hidden_dim), 2 * self.latent_dim))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled feature vector before the latent head, ``(B, feature_dim)``."""
        grid = _as_grid(x, self.cfg)
        if self.training and self.cfg.augment:
            from .vision import augment_images
            grid = augment_images(grid, self.cfg.augment)
        h = self.features_net(grid)
        return h.flatten(2).mean(-1) if self.cfg.pool == "mean" else h.flatten(1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.head(self.features(x)).chunk(2, dim=-1)
        return mu, logvar


class ConvDecoder(nn.Module):
    """Upsampling residual conv decoder: ``z`` -> flat grid.

    ``output`` selects what is returned, matching the built-in decoders:
    ``"mean"`` a tensor, ``"mean_logvar"`` ``{"mean", "logvar"}``, ``"logits"``
    ``{"logits"}``. ``cfg.output_activation`` applies to the mean only.
    """

    def __init__(self, cfg: ConvConfig, latent_dim: int, output: str = "mean"):
        super().__init__()
        cfg.validate()
        if output not in ("mean", "mean_logvar", "logits"):
            raise ValueError(f"output must be 'mean', 'mean_logvar' or 'logits', got {output!r}")
        self.cfg, self.output = cfg, output
        nd, k = cfg.ndim, int(cfg.kernel_size)
        chans = [int(c) for c in cfg.channels]
        self.base_shape = tuple(int(v) // 2 ** len(chans) for v in cfg.spatial_shape)
        self.base_ch = chans[-1]
        self.fc = nn.Sequential(nn.Linear(int(latent_dim), int(cfg.hidden_dim)), _ACT[cfg.activation](),
                                nn.Linear(int(cfg.hidden_dim), self.base_ch * int(math.prod(self.base_shape))),
                                _ACT[cfg.activation]())
        layers, prev = [], self.base_ch
        for ch in list(reversed(chans[:-1])) + [chans[0]]:
            if cfg.upsample == "transpose":
                layers.append(_conv_t(nd)(prev, ch, 4, stride=2, padding=1))
            else:
                layers += [nn.Upsample(scale_factor=2, mode="nearest"), _conv(nd)(prev, ch, k, padding=k // 2)]
            layers.append(_stage_tail(ch, cfg))
            prev = ch
        out_ch = int(cfg.in_channels) * (2 if output == "mean_logvar" else 1)
        layers.append(_conv(nd)(prev, out_ch, k, padding=k // 2))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        h = self.fc(z).reshape(z.shape[0], self.base_ch, *self.base_shape)
        out = self.net(h)
        if self.output == "logits":
            return {"logits": out.flatten(1)}
        if self.output == "mean_logvar":
            mean, logvar = out.chunk(2, dim=1)
            if self.cfg.output_activation == "sigmoid":
                mean = torch.sigmoid(mean)
            return {"mean": mean.flatten(1), "logvar": logvar.flatten(1)}
        if self.cfg.output_activation == "sigmoid":
            out = torch.sigmoid(out)
        return out.flatten(1)


class BackboneGaussianEncoder(nn.Module):
    """Wrap any feature extractor with a trainable head that emits ``(mu, logvar)``.

    Parameters
    ----------
    backbone
        Module mapping the (reshaped) input to features of any shape; they are
        flattened per cell and must total ``feature_dim``.
    feature_dim, latent_dim
        Feature width out of the backbone and the model's latent size.
    input_shape
        Per-cell shape the backbone expects, for example ``(3, 224, 224)``.
        Flat inputs are reshaped to it; ``None`` passes inputs through unchanged.
    freeze
        Keep the backbone in eval mode with gradients off (the default, for
        pretrained models). The head always trains.
    """

    def __init__(self, backbone: nn.Module, feature_dim: int, latent_dim: int,
                 input_shape: Optional[Sequence[int]] = None, hidden_dim: int = 256, freeze: bool = True):
        super().__init__()
        self.backbone, self.freeze = backbone, bool(freeze)
        self.input_shape = None if input_shape is None else tuple(int(v) for v in input_shape)
        self.head = nn.Sequential(nn.Linear(int(feature_dim), int(hidden_dim)), nn.SiLU(),
                                  nn.Linear(int(hidden_dim), 2 * int(latent_dim)))
        if self.freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def features(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_shape is not None and x.ndim == 2:
            x = x.reshape(x.shape[0], *self.input_shape)
        if self.freeze:
            with torch.no_grad():
                return self.backbone(x).flatten(1)
        return self.backbone(x).flatten(1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.head(self.features(x)).chunk(2, dim=-1)
        return mu, logvar


@register_encoder("conv", overwrite=True)
def _build_conv_encoder(uni_cfg, mod_cfg) -> nn.Module:
    if mod_cfg.conv is None:
        raise ValueError(f"Modality {mod_cfg.name!r}: encoder_type='conv' requires conv=ConvConfig(...)")
    return ConvGaussianEncoder(mod_cfg.conv, int(uni_cfg.latent_dim))


@register_decoder("conv", overwrite=True)
def _build_conv_decoder(uni_cfg, mod_cfg, likelihood: str) -> nn.Module:
    if mod_cfg.conv is None:
        raise ValueError(f"Modality {mod_cfg.name!r}: decoder_type='conv' requires conv=ConvConfig(...)")
    lk = (likelihood or "gaussian").lower().strip()
    if lk not in CONV_LIKELIHOODS:
        raise ValueError(f"Modality {mod_cfg.name!r}: decoder_type='conv' supports likelihoods "
                         f"{sorted(CONV_LIKELIHOODS)}, got {likelihood!r}")
    return ConvDecoder(mod_cfg.conv, int(uni_cfg.latent_dim), output=CONV_LIKELIHOODS[lk])


__all__ = ["ConvGaussianEncoder", "ConvDecoder", "BackboneGaussianEncoder", "CONV_LIKELIHOODS"]
