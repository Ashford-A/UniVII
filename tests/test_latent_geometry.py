"""Tests for the latent-geometry options (Euclidean / hyperbolic / product)."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from univii import ClassHeadConfig, ModalityConfig, UniVIConfig, UniVIMultiModalVAE
from univii.config import TokenizerConfig, TransformerConfig
from univii.geometry import (LatentGeometry, boost, expmap0, hierarchy_metrics, latent_distance, logmap0,
                            lorentz_distance, pairwise_latent_distance, split_latent_dims, to_poincare,
                            tree_distance_matrix)
from univii.reference import anchor_on_label, deviation_scores, reconstruction_nll, refine_latent

GEOS = [("hyperbolic", None), ("product", 3)]


def _model(geometry="euclidean", hyperbolic_dim=None, loss_mode="v2", **cfg_kw):
    cfg = UniVIConfig(
        latent_dim=6,
        modalities=[ModalityConfig("rna", 40, [32], [32], likelihood="gaussian"),
                    ModalityConfig("adt", 10, [16], [16], likelihood="gaussian")],
        latent_geometry=geometry, hyperbolic_dim=hyperbolic_dim,
        class_heads=[ClassHeadConfig(name="ct", n_classes=5)], **cfg_kw)
    return UniVIMultiModalVAE(cfg, loss_mode=loss_mode)


def _batch(n=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 5, (n,), generator=g)
    centers = torch.randn(5, 40, generator=g)
    x = {"rna": centers[y] + 0.3 * torch.randn(n, 40, generator=g),
         "adt": centers[y][:, :10] + 0.3 * torch.randn(n, 10, generator=g)}
    return x, {"ct": y}


# ---------------------------------------------------------------- primitives

def test_split_latent_dims():
    assert split_latent_dims(8, "euclidean") == (8, 0)
    assert split_latent_dims(8, "hyperbolic") == (0, 8)
    assert split_latent_dims(7, "product") == (3, 4)
    assert split_latent_dims(8, "product", 2) == (6, 2)
    for bad in (dict(geometry="product", hyperbolic_dim=8), dict(geometry="euclidean", hyperbolic_dim=2),
                dict(geometry="spherical")):
        with pytest.raises(ValueError):
            split_latent_dims(8, **bad)


def test_maps_and_boost_are_consistent():
    torch.manual_seed(0)
    v = torch.randn(200, 5, dtype=torch.float64) * 2
    x = expmap0(v)
    assert torch.allclose(logmap0(x), v, atol=1e-9)
    origin = torch.zeros(200, 6, dtype=torch.float64)
    origin[:, 0] = 1
    assert torch.allclose(lorentz_distance(x, origin), v.norm(dim=-1), atol=1e-6)
    c = expmap0(torch.randn(1, 5, dtype=torch.float64))
    y = boost(c, x)
    assert torch.allclose(boost(c, y, inverse=True), x, atol=1e-8)
    assert torch.allclose(lorentz_distance(x[:100], x[100:]), lorentz_distance(y[:100], y[100:]), atol=1e-7)


def test_wrapped_normal_is_normalised_and_kl_matches_closed_forms():
    torch.manual_seed(0)
    n = 200_000
    z = torch.zeros(n, 3, dtype=torch.float64)
    g = LatentGeometry(3, "hyperbolic", curvature=0.7).double()
    t, log_p = g._h_rsample(z, z)
    mq = torch.tensor([0.4, -0.3, 0.2], dtype=torch.float64).expand(n, 3)
    lq = torch.full((n, 3), -0.6, dtype=torch.float64)
    assert abs(torch.exp(g._h_log_prob(t, mq, lq) - log_p).mean().item() - 1.0) < 0.02
    # a 1-D hyperbolic space is a line, so the estimate must match the Gaussian KL
    g1 = LatentGeometry(1, "hyperbolic").double()
    mq1, lq1 = torch.full((n, 1), 0.7, dtype=torch.float64), torch.full((n, 1), -1.0, dtype=torch.float64)
    mp1, lp1 = torch.full((n, 1), -0.2, dtype=torch.float64), torch.full((n, 1), 0.3, dtype=torch.float64)
    exact = 0.5 * (lp1 - lq1 + (lq1.exp() + (mq1 - mp1) ** 2) / lp1.exp() - 1).sum(-1).mean().item()
    assert abs(g1.kl(mq1, lq1, mp1, lp1)[1].mean().item() - exact) < 0.01
    # KL of a distribution with itself is exactly zero sample by sample
    assert g.kl(mq, lq, mq, lq)[1].abs().max().item() < 1e-9


def test_flat_limit_recovers_euclidean_distance():
    a, b = np.random.default_rng(0).normal(size=(2, 50, 4))
    d = latent_distance(a, b, "hyperbolic", curvature=1e-8)
    assert np.allclose(d, np.linalg.norm(a - b, axis=1), atol=1e-4)
    assert np.allclose(latent_distance(a, b, "euclidean"), np.linalg.norm(a - b, axis=1), atol=1e-5)


def test_distance_is_a_metric_and_anchor_is_an_isometry():
    torch.manual_seed(0)
    g = LatentGeometry(6, "product", hyperbolic_dim=4).double()
    p = torch.randn(120, 6, dtype=torch.float64)
    D = g.cdist(p, p)
    assert torch.allclose(D, D.T, atol=1e-8) and D.diagonal().max() < 1e-5
    i, j, k = np.random.default_rng(0).integers(0, 120, (3, 2000))
    assert bool((D[i, j] + D[j, k] - D[i, k] > -1e-8).all())
    g.set_anchor(p[:30] * 0.2 + 1.0)
    q = g.anchor(p)
    assert torch.allclose(g.cdist(q, q), D, atol=1e-6)
    assert g.anchor(p[:30] * 0.2 + 1.0).mean(0).norm().item() < 0.05
    assert np.allclose(pairwise_latent_distance(p, geometry="product", hyperbolic_dim=4), D.numpy(), atol=1e-6)
    assert np.abs(to_poincare(p, "product", 4)).max() < 1.0


def test_tree_distance_matrix():
    dist, names, depth = tree_distance_matrix(
        {"HSC": None, "MPP": "HSC", "GMP": "MPP", "CLP": "MPP", "B": "CLP"}, lengths={"B": 2.0})
    assert dist[names.index("GMP"), names.index("B")] == 4.0 and depth[names.index("B")] == 4.0
    with pytest.raises(ValueError):
        tree_distance_matrix({"a": None, "b": None})


# ------------------------------------------------------------------- model

def test_euclidean_default_reports_zero_hyperbolic_kl():
    model = _model()
    x, y = _batch()
    out = model(x, epoch=0, y=y)
    assert model.latent.is_euclidean and float(out["kl_h"]) == 0.0
    assert torch.allclose(out["kl_e"], out["kl"])


@pytest.mark.parametrize("geometry,hd", GEOS)
@pytest.mark.parametrize("loss_mode", ["v1", "v2"])
def test_forward_backward_and_training_step(geometry, hd, loss_mode):
    torch.manual_seed(0)
    model = _model(geometry, hd, loss_mode, use_moe_gating=True)
    x, y = _batch()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for step in range(40):
        out = model(x, epoch=step, y=y)
        assert torch.isfinite(out["loss"])
        opt.zero_grad()
        out["loss"].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        opt.step()
        losses.append(float(out["loss"]))
    assert losses[-1] < losses[0]
    assert out["z"].shape == (48, 6) and out["mu_z"].dtype == torch.float32
    assert float(out["kl_h"]) > 0.0
    mu, logvar, z = model.eval().encode_fused(x)
    assert mu.shape == (48, 6) and torch.isfinite(mu).all()


@pytest.mark.parametrize("geometry,hd", [("hyperbolic", None), ("product", 2)])
def test_fused_transformer_path(geometry, hd):
    T = TransformerConfig(d_model=16, num_heads=2, num_layers=1, dim_feedforward=32, dropout=0.0, attn_dropout=0.0)
    tok = lambda n: TokenizerConfig(mode="topk_channels", n_tokens=6, channels=("value", "rank"), n_features=n)
    cfg = UniVIConfig(latent_dim=4, fused_encoder_type="multimodal_transformer", fused_transformer=T,
                      latent_geometry=geometry, hyperbolic_dim=hd,
                      modalities=[ModalityConfig("rna", 20, [16], [16], tokenizer=tok(20)),
                                  ModalityConfig("atac", 30, [16], [16], tokenizer=tok(30))])
    model = UniVIMultiModalVAE(cfg, loss_mode="v2", recon_normalize_by_dim=False)
    out = model({"rna": torch.randn(8, 20), "atac": torch.randn(8, 30)}, epoch=0)
    out["loss"].backward()
    assert torch.isfinite(out["loss"]) and float(out["used_fused_encoder"]) == 1.0


def test_means_respect_max_radius_and_curvature_can_be_learned():
    model = _model("hyperbolic", hyperbolic_max_radius=1.5, curvature=2.0, learn_curvature=True)
    x, y = _batch()
    out = model({k: 50 * v for k, v in x.items()}, epoch=0, y=y)
    limit = 1.5 / np.sqrt(model.curvature)
    assert float(out["mu_z"].norm(dim=-1).max()) <= limit + 1e-4
    assert abs(model.curvature - 2.0) < 1e-6
    out["loss"].backward()
    assert model.latent.log_curvature.grad is not None and torch.isfinite(model.latent.log_curvature.grad)
    assert not any(n.endswith("log_curvature") for n, _ in _model("hyperbolic").named_parameters())


def test_hyperbolic_kl_weight_scales_only_the_hyperbolic_part():
    x, y = _batch()
    outs = []
    for w in (1.0, 0.0):
        torch.manual_seed(0)
        model = _model("product", 3, kl_hyperbolic_weight=w)
        torch.manual_seed(1)
        outs.append(model(x, epoch=0, y=y))
    assert torch.allclose(outs[0]["kl"] - outs[0]["kl_h"], outs[1]["kl"], atol=1e-5)
    assert torch.allclose(outs[1]["kl"], outs[1]["kl_e"], atol=1e-5)


@pytest.mark.parametrize("geometry,hd", [("euclidean", None)] + GEOS)
def test_hierarchy_loss_improves_tree_agreement(geometry, hd):
    torch.manual_seed(0)
    dist, names, depth = tree_distance_matrix({"a": None, "b": "a", "c": "a", "d": "b", "e": "b"})
    x, y = _batch(n=256)
    scores = {}
    for weight in (0.0, 5.0):
        torch.manual_seed(0)
        model = _model(geometry, hd, hierarchy_weight=weight, hierarchy_root_weight=1.0, hierarchy_head="ct")
        model.set_hierarchy(dist, root_index=0)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        for step in range(150):
            out = model(x, epoch=step, y=y)
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
        assert ("hierarchy" in out) == (weight > 0)
        z = model.eval().encode_fused(x)[0].detach()
        scores[weight] = hierarchy_metrics(z, y["ct"].numpy(), dist, depth, geometry, hd)
    assert scores[5.0]["dist_spearman"] > scores[0.0]["dist_spearman"]
    assert scores[5.0]["dist_spearman"] > 0.8
    # unlabelled cells contribute nothing
    lone = model.latent.hierarchy_loss(torch.randn(4, 6), torch.full((4,), -1))
    assert float(lone.abs().sum()) == 0.0


@pytest.mark.parametrize("geometry,hd", [("euclidean", None)] + GEOS)
def test_anchoring_moves_root_to_origin_without_changing_distances(geometry, hd):
    torch.manual_seed(0)
    model = _model(geometry, hd).eval()
    x, y = _batch(n=96)
    with torch.no_grad():
        mu = model.encode_fused(x)[0]
        n_root = anchor_on_label(model, mu, y["ct"].numpy(), 0)
        anchored = model.encode_fused(x, anchored=True)[0]
    assert n_root == int((y["ct"] == 0).sum())
    assert torch.allclose(anchored, model.anchor_latent(mu), atol=1e-6)
    assert torch.allclose(model.pairwise_latent_distance(mu), model.pairwise_latent_distance(anchored), atol=1e-3)
    root = (y["ct"] == 0)
    assert float(model.latent_radius(anchored[root]).mean()) < float(model.latent_radius(anchored[~root]).mean())
    model.clear_latent_anchor()
    assert torch.equal(model.anchor_latent(mu), mu)


# -------------------------------------------------------------- persistence

def test_old_state_dicts_load_and_new_state_round_trips():
    old = _model()
    legacy = {k: v for k, v in old.state_dict().items() if not k.startswith("latent.")}
    fresh = _model()
    fresh.load_state_dict(legacy, strict=True)          # checkpoints from before this feature

    dist, _, _ = tree_distance_matrix({"a": None, "b": "a", "c": "a", "d": "b", "e": "b"})
    src = _model("product", 3, curvature=0.5)
    src.set_hierarchy(dist, root_index=0)
    src.set_latent_anchor(torch.randn(10, 6))
    dst = _model("product", 3)                           # different curvature, no hierarchy, no anchor
    dst.load_state_dict(copy.deepcopy(src.state_dict()), strict=True)
    assert abs(dst.curvature - 0.5) < 1e-6 and bool(dst.latent.has_anchor)
    assert torch.equal(dst.latent.hierarchy_dist, src.latent.hierarchy_dist) and int(dst.latent.hierarchy_root) == 0
    assert torch.equal(dst.latent.anchor_point, src.latent.anchor_point)


def test_config_validation_and_dataclass_round_trip():
    from dataclasses import asdict
    for bad in (dict(geometry="product", hyperbolic_dim=6), dict(curvature=0.0),
                dict(kl_hyperbolic_weight=-1.0), dict(hierarchy_scale=0.0)):
        with pytest.raises(ValueError):
            _model(**bad)
    cfg = _model("product", 2, hierarchy_scale=None).cfg
    d = asdict(cfg)
    assert d["latent_geometry"] == "product" and d["hyperbolic_dim"] == 2 and d["hierarchy_scale"] is None
    assert any(n.endswith("hierarchy_log_scale") for n, _ in _model(hierarchy_scale=None).named_parameters())


# ---------------------------------------------------------------- reference

@pytest.mark.parametrize("geometry,hd", [("euclidean", None)] + GEOS)
def test_refine_latent_lowers_nll_and_leaves_weights_alone(geometry, hd):
    torch.manual_seed(0)
    model = _model(geometry, hd)
    x, y = _batch()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(60):
        out = model(x, epoch=step, y=y)
        opt.zero_grad(); out["loss"].backward(); opt.step()
    before = copy.deepcopy(model.state_dict())
    g = torch.Generator().manual_seed(7)                  # cells pushed off the training manifold
    shifted = {k: v + 3.0 * torch.randn(1, v.shape[1], generator=g) for k, v in x.items()}
    with torch.no_grad():
        z0 = model.eval().encode_fused(shifted)[0]
        nll0, per_mod = reconstruction_nll(model, shifted, z0)
    z1 = refine_latent(model, shifted, steps=60)
    with torch.no_grad():
        nll1, _ = reconstruction_nll(model, shifted, z1)
    z_null = refine_latent(model, x, steps=60)            # same treatment for the null population
    with torch.no_grad():
        nll_null, _ = reconstruction_nll(model, x, z_null)
    assert set(per_mod) == {"rna", "adt"} and float(nll1.mean()) < float(nll0.mean())
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(p.requires_grad for p in model.parameters()) and not model.training
    assert np.median(deviation_scores(nll1, nll_null)) > 3.0
    assert abs(np.median(deviation_scores(nll_null, nll_null))) < 1e-9


# ------------------------------------------------------------------ trainer

@pytest.mark.parametrize("geometry,hd", GEOS)
def test_trainer_loop_with_hierarchy_labels(geometry, hd):
    """The hierarchy loss must be reachable through the standard loader and trainer."""
    import anndata as ad
    from univii import TrainingConfig, UniVITrainer, make_loader

    torch.manual_seed(0)
    x, y = _batch(n=120)
    names = [f"c{i}" for i in range(120)]
    rna = ad.AnnData(x["rna"].numpy()); rna.obs_names = names
    adt = ad.AnnData(x["adt"].numpy()); adt.obs_names = names
    labels = y["ct"].numpy().copy()
    labels[:10] = -1                                        # some cells unlabelled
    loader = make_loader({"rna": rna, "adt": adt}, labels={"ct": labels}, batch_size=40, shuffle=True)

    dist, _, depth = tree_distance_matrix({"a": None, "b": "a", "c": "a", "d": "b", "e": "b"})
    model = _model(geometry, hd, hierarchy_weight=5.0, hierarchy_root_weight=1.0, hierarchy_head="ct")
    model.set_hierarchy(dist, root_index=0)
    seen = []
    handle = model.register_forward_hook(lambda m, i, out: seen.append("hierarchy" in out))
    trainer = UniVITrainer(model, loader, val_loader=loader,
                           train_cfg=TrainingConfig(n_epochs=25, batch_size=40, lr=3e-3, log_every=1000))
    history = trainer.fit()
    handle.remove()
    assert seen and all(seen)
    assert np.isfinite(history["train_loss"]).all() and history["train_loss"][-1] < history["train_loss"][0]
    z = model.eval().encode_fused({k: v for k, v in x.items()})[0].detach()
    score = hierarchy_metrics(z[10:], labels[10:], dist, depth, geometry, hd)
    assert score["dist_spearman"] > 0.7
