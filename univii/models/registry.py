"""Plug-in registry for per-modality encoders and decoders.

Add your own architecture without editing UniVI::

    from univii.models.registry import register_encoder, register_decoder

    @register_encoder("my_encoder")
    def build(uni_cfg, mod_cfg):
        # must return an nn.Module whose forward(x) gives (mu, logvar), each (cells, uni_cfg.latent_dim).
        # x arrives as a flat (cells, mod_cfg.input_dim) float tensor.
        return MyEncoder(mod_cfg.input_dim, uni_cfg.latent_dim, **(mod_cfg.encoder_kwargs or {}))

    @register_decoder("my_decoder")
    def build(uni_cfg, mod_cfg, likelihood):
        # forward(z) must return what the built-in decoder for `likelihood` returns:
        #   gaussian / normal / mse  -> a (cells, input_dim) tensor of means
        #   gaussian_diag            -> {"mean": ..., "logvar": ...}
        #   bernoulli                -> {"logits": ...}
        # (see univii/models/decoders.py for the other likelihoods)
        return MyDecoder(uni_cfg.latent_dim, mod_cfg.input_dim, **(mod_cfg.decoder_kwargs or {}))

    ModalityConfig("image", input_dim=..., encoder_hidden=[], decoder_hidden=[],
                   encoder_type="my_encoder", decoder_type="my_decoder")

Register before building the config's model, in every process that loads a
checkpoint using the custom type. Built-in names: encoders "mlp",
"transformer", "conv"; decoders "mlp", "conv".
"""
from __future__ import annotations

from typing import Any, Callable, Dict

import torch.nn as nn

EncoderBuilder = Callable[[Any, Any], nn.Module]
DecoderBuilder = Callable[[Any, Any, str], nn.Module]

ENCODER_BUILDERS: Dict[str, EncoderBuilder] = {}
DECODER_BUILDERS: Dict[str, DecoderBuilder] = {}
_RESERVED_ENCODERS = ("mlp", "transformer")
_RESERVED_DECODERS = ("mlp",)


def _register(table: Dict[str, Any], reserved, name: str, overwrite: bool):
    key = str(name).lower().strip()
    if not key:
        raise ValueError("builder name must be a non-empty string")
    if key in reserved:
        raise ValueError(f"{key!r} is a built-in name and cannot be re-registered")

    def deco(fn):
        if key in table and table[key] is not fn and not overwrite:
            raise ValueError(f"{key!r} is already registered; pass overwrite=True to replace it")
        table[key] = fn
        return fn

    return deco


def register_encoder(name: str, *, overwrite: bool = False):
    """Decorator registering ``fn(uni_cfg, mod_cfg) -> nn.Module`` as ``encoder_type=name``."""
    return _register(ENCODER_BUILDERS, _RESERVED_ENCODERS, name, overwrite)


def register_decoder(name: str, *, overwrite: bool = False):
    """Decorator registering ``fn(uni_cfg, mod_cfg, likelihood) -> nn.Module`` as ``decoder_type=name``."""
    return _register(DECODER_BUILDERS, _RESERVED_DECODERS, name, overwrite)


def available_encoders():
    return sorted(set(_RESERVED_ENCODERS) | set(ENCODER_BUILDERS))


def available_decoders():
    return sorted(set(_RESERVED_DECODERS) | set(DECODER_BUILDERS))


__all__ = ["register_encoder", "register_decoder", "available_encoders", "available_decoders",
           "ENCODER_BUILDERS", "DECODER_BUILDERS"]
