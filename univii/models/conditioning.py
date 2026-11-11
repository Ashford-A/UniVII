"""Cohort-conditioned decoders.

With ``UniVIConfig(n_cohorts=K)`` every modality decoder is wrapped so that it
receives a learned embedding of the cell's cohort next to the latent vector.
Cohort-specific technical differences can then be explained by the embedding,
so the latent does not have to encode them.
"""
from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn


class CohortConditionedDecoder(nn.Module):
    """Wrap a decoder built for ``latent_dim + dim`` inputs and feed it ``[z, embedding(cohort)]``.

    ``codes`` selects the cohort: ``None`` means cohort 0 for every row, an
    ``int`` means that cohort for every row, and a tensor gives one code per
    row (codes outside ``0..K-1``, such as ``-1`` for unknown, use cohort 0).
    The model sets it before each pass; see ``UniVIMultiModalVAE.set_decode_cohort``.
    """

    def __init__(self, base: nn.Module, n_cohorts: int, dim: int):
        super().__init__()
        self.base = base
        self.embed = nn.Embedding(int(n_cohorts), int(dim))
        self.codes: Optional[Union[int, torch.Tensor]] = None

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("base"), name)      # attributes of the wrapped decoder

    def forward(self, z: torch.Tensor):
        n, k = z.shape[0], self.embed.num_embeddings
        codes = self.codes
        if codes is None or isinstance(codes, int):
            code = 0 if codes is None else int(codes)
            if not 0 <= code < k:
                raise ValueError(f"cohort code {code} is outside 0..{k - 1}")
            idx = torch.full((n,), code, dtype=torch.long, device=z.device)
        else:
            idx = torch.as_tensor(codes).long().view(-1).to(z.device)
            if idx.shape[0] != n:
                raise ValueError(f"got {idx.shape[0]} cohort codes for {n} cells")
            idx = torch.where((idx >= 0) & (idx < k), idx, torch.zeros_like(idx))
        return self.base(torch.cat([z, self.embed(idx).to(z.dtype)], dim=-1))


__all__ = ["CohortConditionedDecoder"]
