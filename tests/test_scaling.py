"""Distances, neighbour searches and HMM inference must not build cells x cells x dimensions arrays,
and their memory-light versions must give the same answers as the straightforward ones."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from univii import ModalityConfig, UniVIConfig, UniVIMultiModalVAE
from univii.atlas import cohort_mixing, label_purity, transfer_labels
from univii.geometry import LatentGeometry, pairwise_latent_distance, rows_per_chunk
from univii.hmm import GaussianHMM


@pytest.mark.parametrize("geometry,hyperbolic_dim", [("euclidean", None), ("hyperbolic", None), ("product", 3)])
def test_all_pairs_distance_matches_the_broadcast_distance(geometry, hyperbolic_dim):
    torch.manual_seed(0)
    geom = LatentGeometry(8, geometry, hyperbolic_dim).double()
    a, b = 0.6 * torch.randn(40, 8, dtype=torch.float64), 0.6 * torch.randn(30, 8, dtype=torch.float64)
    assert torch.allclose(geom.cdist_sq(a, b), geom.sqdist(a.unsqueeze(1), b.unsqueeze(0)), atol=1e-8)
    assert geom.cdist(a, b).shape == (40, 30)
    assert float(geom.cdist_sq(a, a).diagonal().abs().max()) < 1e-8
    a.requires_grad_(True)
    geom.cdist_sq(a, a).sum().backward()                       # zero distances on the diagonal must not break the gradient
    assert torch.isfinite(a.grad).all()


def test_pairwise_terms_in_the_model_stay_small_in_memory():
    """The cohort-alignment and contrastive terms on a large batch: would need gigabytes with an (n, m, d) array."""
    torch.manual_seed(0)
    mods = [ModalityConfig("rna", 20, [16], [16]), ModalityConfig("adt", 6, [8], [8])]
    model = UniVIMultiModalVAE(UniVIConfig(latent_dim=64, modalities=mods, cohort_mmd_weight=1.0, cohort_memory=4000,
                                           contrastive_weight=1.0, encoder_batchnorm=False))
    a, b = torch.randn(3000, 64, requires_grad=True), torch.randn(4000, 64)
    value = model._mmd(a, b)
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(a.grad).all()
    x = {"rna": torch.randn(64, 20), "adt": torch.randn(64, 6)}
    out = model(x, epoch=0, y={"cohort": torch.tensor([0] * 32 + [1] * 32)})
    assert torch.isfinite(out["loss"]) and "contrastive" in out and "cohort_mmd" in out


def test_neighbour_tools_handle_wide_embeddings_and_agree_across_code_paths():
    rng = np.random.default_rng(0)
    centres = 4.0 * rng.normal(size=(3, 2048))
    y_ref, y_query = rng.integers(0, 3, 600), rng.integers(0, 3, 200)
    ref, query = centres[y_ref] + rng.normal(size=(600, 2048)), centres[y_query] + rng.normal(size=(200, 2048))
    out = transfer_labels(ref, y_ref, query)                   # 2,048 dimensions: previously tens of gigabytes
    assert (out["label"].to_numpy() == y_query).mean() > 0.95 and 0.5 < out["novelty"].median() < 2.0
    assert label_purity(ref, y_ref) > 0.95
    assert 0.8 < cohort_mixing(np.vstack([ref, query]), np.r_[["a"] * 600, ["b"] * 200]) < 1.2
    low = 0.2 * rng.normal(size=(300, 4)); labels = rng.integers(0, 2, 300)
    # a product space whose hyperbolic part is switched off by zero coordinates must agree with the Euclidean path
    euclid = transfer_labels(low, labels, low[:50])
    product = transfer_labels(np.c_[low, np.zeros((300, 2))], labels, np.c_[low[:50], np.zeros((50, 2))], geometry="product", hyperbolic_dim=2)
    assert (euclid["label"].to_numpy() == product["label"].to_numpy()).all()
    assert np.allclose(euclid["novelty"], product["novelty"], rtol=1e-4)
    assert rows_per_chunk(100) == 1024 and rows_per_chunk(10_000_000) == 8 and rows_per_chunk(100_000) == 500
    d = pairwise_latent_distance(low[:10], low[:7])
    expected_sq = ((low[:10, None] - low[None, :7]) ** 2).sum(-1)
    # Match the distance API's epsilon stabilization at zero.
    expected = np.sqrt(expected_sq + 1e-12)
    np.testing.assert_allclose(d, expected, rtol=1e-7, atol=1e-8)


def test_hmm_batch_inference_equals_one_sequence_at_a_time():
    torch.manual_seed(0)
    hmm = GaussianHMM(3, 2, stickiness=0.8)
    with torch.no_grad():
        hmm.means.copy_(torch.randn(3, 2, dtype=torch.float64)); hmm.start_logits.copy_(torch.randn(3, dtype=torch.float64))
    batch = torch.randn(20, 30, 2, dtype=torch.float64)
    assert torch.allclose(hmm.posterior(batch), torch.stack([hmm.posterior(s) for s in batch]), atol=1e-10)
    assert torch.equal(hmm.viterbi(batch), torch.stack([hmm.viterbi(s) for s in batch]))
    assert torch.allclose(hmm.log_likelihood(batch), torch.stack([hmm.log_likelihood(s) for s in batch]), atol=1e-10)
    before = float(hmm.log_likelihood(batch).sum())
    hmm.fit(batch, n_iter=20, initialise=False)                # fitting accepts a batch as well
    assert float(hmm.log_likelihood(batch).sum()) > before
    with pytest.raises(ValueError):
        hmm.posterior(torch.zeros(4, 30, 5))
