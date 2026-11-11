"""Vision-transformer encoder for image modalities, and shared image augmentation.

    ViTGaussianEncoder     flat image -> patches -> transformer -> (mu, logvar)
    augment_images         random flips / quarter turns for a batch of images

Importing this module registers ``encoder_type="vit"``.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn

from ..config import ViTConfig
from .registry import register_encoder


def augment_images(x: torch.Tensor, augment: Sequence[str]) -> torch.Tensor:
    """Apply one random flip and/or quarter turn to a whole batch ``(B, C, H, W)``.

    The same transform is used for every image in the batch; batches differ.
    Tile orientation carries no meaning in histology, so this teaches the
    encoder to ignore it.
    """
    if "flip" in augment and bool(torch.rand(()) < 0.5):
        x = torch.flip(x, dims=(-1,))
    if "rot90" in augment:
        x = torch.rot90(x, int(torch.randint(0, 4, ())), dims=(-2, -1))
    return x


class ViTGaussianEncoder(nn.Module):
    """Patch-embedding transformer encoder: image -> ``(mu, logvar)``."""

    def __init__(self, cfg: ViTConfig, latent_dim: int):
        super().__init__()
        cfg.validate()
        self.cfg, self.latent_dim = cfg, int(latent_dim)
        d = int(cfg.d_model)
        self.patch = nn.Conv2d(int(cfg.in_channels), d, kernel_size=int(cfg.patch_size), stride=int(cfg.patch_size))
        n_tokens = cfg.n_patches + (1 if cfg.pool == "cls" else 0)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d)) if cfg.pool == "cls" else None
        layer = nn.TransformerEncoderLayer(d, int(cfg.num_heads), int(cfg.dim_feedforward), float(cfg.dropout),
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, int(cfg.num_layers), enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2 * self.latent_dim)

    def _as_image(self, x: torch.Tensor) -> torch.Tensor:
        shape = (int(self.cfg.in_channels), int(self.cfg.image_size[0]), int(self.cfg.image_size[1]))
        if x.ndim == 2:
            if x.shape[1] != self.cfg.flat_dim:
                raise ValueError(f"expected {self.cfg.flat_dim} features (C * H * W), got {x.shape[1]}")
            return x.reshape(x.shape[0], *shape)
        if tuple(x.shape[1:]) != shape:
            raise ValueError(f"expected input of shape (B, {shape[0]}, {shape[1]}, {shape[2]}), got {tuple(x.shape)}")
        return x

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Encoded patch tokens ``(B, n_tokens, d_model)`` (with the class token first, if used)."""
        img = self._as_image(x)
        if self.training and self.cfg.augment:
            img = augment_images(img, self.cfg.augment)
        h = self.patch(img).flatten(2).transpose(1, 2)
        if self.cls is not None:
            h = torch.cat([self.cls.expand(h.shape[0], -1, -1), h], dim=1)
        return self.norm(self.encoder(h + self.pos))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tokens(x)
        return h[:, 0] if self.cls is not None else h.mean(dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.head(self.features(x)).chunk(2, dim=-1)
        return mu, logvar


@register_encoder("vit", overwrite=True)
def _build_vit_encoder(uni_cfg, mod_cfg) -> nn.Module:
    if mod_cfg.vit is None:
        raise ValueError(f"Modality {mod_cfg.name!r}: encoder_type='vit' requires vit=ViTConfig(...)")
    return ViTGaussianEncoder(mod_cfg.vit, int(uni_cfg.latent_dim))


__all__ = ["ViTGaussianEncoder", "augment_images"]
