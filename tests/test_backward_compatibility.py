"""UniVII must read what UniVI 1.2.1 wrote, and behave exactly like it when no new option is used.

``tests/fixtures/univi_1_2_1`` holds two small reference bundles and their outputs, written by the
UniVI 1.2.1 code (package ``univi``). One bundle includes a fitted preprocessor, pickled under the
module name ``univi.preprocessing``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

import univii
from univii import load_reference

FIXTURE = Path(__file__).parent / "fixtures" / "univi_1_2_1"


@pytest.mark.parametrize("mode", ["v1", "v2"])
def test_reference_bundle_from_univi_1_2_1_loads_and_reproduces_outputs(mode):
    expected = torch.load(FIXTURE / "expected.pt", weights_only=True)
    model = load_reference(FIXTURE / f"reference_{mode}")[0].eval()
    assert model.loss_mode == mode and model.latent_geometry == "euclidean"
    x, want = expected["x"], expected[mode]
    with torch.no_grad():
        out = model(x, epoch=0)
        mu, logvar, _ = model.encode_fused(x)
    for key in ("mu_z", "logvar_z", "kl", "align"):
        assert torch.allclose(out[key], want[key], atol=1e-6, rtol=1e-5), key
    assert torch.allclose(mu, want["encode_mu"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(logvar, want["encode_logvar"], atol=1e-6, rtol=1e-5)
    assert float(out["kl_h"]) == 0.0 and "hierarchy" not in out and "spatial" not in out


@pytest.mark.skipif(importlib.util.find_spec("univi") is not None,
                    reason="UniVI is installed, so its own preprocessor classes are used")
def test_preprocessors_pickled_by_univi_load_without_univi_installed():
    expected = torch.load(FIXTURE / "expected.pt", weights_only=True)
    _, prep, _ = load_reference(FIXTURE / "reference_v1")
    assert type(prep["rna"]).__module__ == "univii.preprocessing"
    assert not any(n == "univi" or n.startswith("univi.") for n in sys.modules)     # the redirect is gone
    counts = expected["prep_input"].numpy()
    a = ad.AnnData(sp.csr_matrix(counts), obs=pd.DataFrame(index=[f"cell{i}" for i in range(30)]),
                   var=pd.DataFrame(index=[f"f{i}" for i in range(20)]))
    a.layers["counts"] = a.X.copy()
    out = prep["rna"].transform(a).X
    out = out.toarray() if sp.issparse(out) else np.asarray(out)
    assert np.allclose(out, expected["prep_output"].numpy(), atol=1e-5)


def test_new_options_default_to_off_and_aliases_point_to_the_same_classes():
    from univii import ModalityConfig, UniVIConfig

    cfg = UniVIConfig(latent_dim=4, modalities=[ModalityConfig("rna", 20, [8], [8])])
    assert cfg.latent_geometry == "euclidean" and cfg.hierarchy_weight == 0.0 and cfg.spatial_weight == 0.0
    assert cfg.modalities[0].encoder_type == "mlp" and cfg.modalities[0].decoder_type == "mlp"
    assert univii.UniVIIConfig is univii.UniVIConfig and univii.UniVIIMultiModalVAE is univii.UniVIMultiModalVAE
    assert univii.UniVIITrainer is univii.UniVITrainer and univii.UniVIIRefiner is univii.UniVIRefiner
