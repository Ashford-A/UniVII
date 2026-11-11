"""Tests for conv encoders/decoders, the plug-in registry and the spatial utilities."""
from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import torch
import torch.nn as nn

from univii import (ConvConfig, ModalityConfig, TrainingConfig, UniVIConfig, UniVIMultiModalVAE, UniVITrainer,
                   make_loader, register_decoder, register_encoder)
from univii.models import BackboneGaussianEncoder, ConvDecoder, ConvGaussianEncoder, available_encoders
from univii.spatial import (SpatialBlockSampler, extract_tiles, knn_graph, make_spatial_loader,
                           median_neighbor_distance, neighborhood_aggregate, niche_modality, tiles_from_visium,
                           unflatten_tiles)

IMG = ConvConfig(in_channels=3, spatial_shape=(16, 16), channels=(8, 16), hidden_dim=32)


def _image_modality(likelihood="gaussian", conv=IMG, **kw):
    return ModalityConfig("image", conv.flat_dim, [], [], likelihood=likelihood,
                          encoder_type="conv", decoder_type="conv", conv=conv, **kw)


def _model(likelihood="gaussian", loss_mode="v2", conv=IMG, **cfg_kw):
    cfg = UniVIConfig(latent_dim=6, modalities=[ModalityConfig("rna", 30, [32], [32]),
                                                 _image_modality(likelihood, conv)], **cfg_kw)
    # loss_mode="v2" divides each modality's reconstruction loss by its dimension by default, which
    # starves a 768-pixel image next to the KL term; switch that off for image modalities.
    extra = {"recon_normalize_by_dim": False} if loss_mode == "v2" else {}
    return UniVIMultiModalVAE(cfg, loss_mode=loss_mode, **extra)


def _paired_batch(n=32, seed=0, binary=False):
    """Images whose brightness pattern is tied to the expression profile."""
    g = torch.Generator().manual_seed(seed)
    code = torch.rand(n, 3, generator=g)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 16), torch.linspace(0, 1, 16), indexing="ij")
    img = code[:, :, None, None] * (0.5 + 0.5 * xx)[None, None] + 0.05 * torch.rand(n, 3, 16, 16, generator=g)
    img = img.clamp(0, 1)
    if binary:
        img = (img > 0.5).float()
    rna = torch.cat([code.repeat(1, 10)], dim=1) + 0.05 * torch.randn(n, 30, generator=g)
    return {"rna": rna, "image": img.flatten(1)}


# ------------------------------------------------------------------ conv modules

@pytest.mark.parametrize("conv", [
    IMG,
    ConvConfig(in_channels=1, spatial_shape=(32, 16), channels=(4, 8, 8), pool="mean", norm="batch",
               upsample="transpose", blocks_per_stage=2, activation="gelu"),
    ConvConfig(in_channels=2, spatial_shape=(64,), channels=(8, 16), norm="none", dropout=0.1),   # 1-D signal
])
def test_conv_encoder_decoder_shapes(conv):
    enc, n = ConvGaussianEncoder(conv, latent_dim=5), 4
    x = torch.randn(n, conv.flat_dim)
    mu, logvar = enc(x)
    assert mu.shape == logvar.shape == (n, 5)
    shaped = x.reshape(n, conv.in_channels, *conv.spatial_shape)
    assert torch.allclose(enc.eval()(shaped)[0], enc.eval()(x)[0])            # flat and shaped inputs agree
    assert ConvDecoder(conv, 5)(mu).shape == (n, conv.flat_dim)
    out = ConvDecoder(conv, 5, output="mean_logvar")(mu)
    assert out["mean"].shape == out["logvar"].shape == (n, conv.flat_dim)
    assert ConvDecoder(conv, 5, output="logits")(mu)["logits"].shape == (n, conv.flat_dim)
    with pytest.raises(ValueError):
        enc(torch.randn(n, conv.flat_dim + 1))


def test_sigmoid_output_stays_in_unit_interval():
    conv = ConvConfig(in_channels=3, spatial_shape=(16, 16), channels=(8, 16), output_activation="sigmoid")
    out = ConvDecoder(conv, 4)(10 * torch.randn(8, 4))
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_conv_config_validation():
    for bad in (dict(spatial_shape=(30, 30)), dict(spatial_shape=(16, 16, 16)), dict(kernel_size=4),
                dict(channels=()), dict(pool="max"), dict(dropout=1.0)):
        with pytest.raises(ValueError):
            ConvConfig(**{**dict(in_channels=3, spatial_shape=(16, 16), channels=(8, 16)), **bad}).validate()
    with pytest.raises(ValueError, match="input_dim"):
        UniVIConfig(latent_dim=4, modalities=[ModalityConfig("image", 100, [], [], encoder_type="conv", conv=IMG)]).validate()
    with pytest.raises(ValueError, match="requires conv"):
        UniVIConfig(latent_dim=4, modalities=[ModalityConfig("image", 768, [], [], decoder_type="conv")]).validate()
    with pytest.raises(ValueError, match="supports likelihoods"):
        _model(likelihood="nb")


# ------------------------------------------------------------------ in the model

@pytest.mark.parametrize("likelihood", ["gaussian", "gaussian_diag", "bernoulli"])
@pytest.mark.parametrize("loss_mode", ["v1", "v2"])
def test_image_modality_trains_and_translates(likelihood, loss_mode):
    torch.manual_seed(0)
    model = _model(likelihood, loss_mode)
    assert isinstance(model.encoders["image"], ConvGaussianEncoder) and isinstance(model.decoders["image"], ConvDecoder)
    x = _paired_batch(binary=(likelihood == "bernoulli"))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for step in range(60):
        out = model(x, epoch=step)
        opt.zero_grad(); out["loss"].backward(); opt.step()
        losses.append(float(out["loss"].detach()))
    assert np.isfinite(losses).all() and losses[-1] < losses[0]
    model.eval()
    with torch.no_grad():                                   # RNA-only cells -> predicted image
        z = model.encode_fused({"rna": x["rna"]})[0]
        pred = model.decode_modalities(z)["image"]
    pred = pred["mean"] if isinstance(pred, dict) and "mean" in pred else pred
    pred = torch.sigmoid(pred["logits"]) if isinstance(pred, dict) else pred
    assert pred.shape == x["image"].shape
    if likelihood == "gaussian":                            # cross-modal prediction beats the mean image
        err = ((pred - x["image"]) ** 2).mean()
        base = ((x["image"].mean(0, keepdim=True) - x["image"]) ** 2).mean()
        assert float(err) < float(base)


def test_conv_modality_with_hyperbolic_latent_and_mlp_decoder():
    cfg = UniVIConfig(latent_dim=6, latent_geometry="product", hyperbolic_dim=3, modalities=[
        ModalityConfig("rna", 30, [32], [32]),
        ModalityConfig("image", IMG.flat_dim, [], [64], encoder_type="conv", conv=IMG)])   # conv in, MLP out
    model = UniVIMultiModalVAE(cfg)
    assert not isinstance(model.decoders["image"], ConvDecoder)
    out = model(_paired_batch(), epoch=0)
    out["loss"].backward()
    assert torch.isfinite(out["loss"]) and float(out["kl_h"]) > 0


def test_default_models_gain_no_parameters_or_state():
    plain = UniVIMultiModalVAE(UniVIConfig(latent_dim=6, modalities=[ModalityConfig("rna", 30, [32], [32])]))
    assert not any("conv" in k for k in plain.state_dict())
    assert {"conv", "mlp", "transformer"} <= set(available_encoders())


# ------------------------------------------------------------------ registry

def test_custom_encoder_and_decoder_plug_in():
    class TinyEnc(nn.Module):
        def __init__(self, d_in, d_lat, width=8):
            super().__init__()
            self.net = nn.Linear(d_in, 2 * d_lat)
            self.width = width

        def forward(self, x):
            return self.net(x).chunk(2, dim=-1)

    @register_encoder("tiny_test_encoder", overwrite=True)
    def _enc(uni_cfg, mod_cfg):
        return TinyEnc(mod_cfg.input_dim, uni_cfg.latent_dim, **(mod_cfg.encoder_kwargs or {}))

    @register_decoder("tiny_test_decoder", overwrite=True)
    def _dec(uni_cfg, mod_cfg, likelihood):
        assert likelihood == "gaussian"
        return nn.Linear(uni_cfg.latent_dim, mod_cfg.input_dim)

    cfg = UniVIConfig(latent_dim=4, modalities=[
        ModalityConfig("rna", 30, [16], [16]),
        ModalityConfig("x", 12, [], [], encoder_type="tiny_test_encoder", decoder_type="tiny_test_decoder",
                       encoder_kwargs={"width": 3})])
    model = UniVIMultiModalVAE(cfg)
    assert isinstance(model.encoders["x"], TinyEnc) and model.encoders["x"].width == 3
    out = model({"rna": torch.randn(8, 30), "x": torch.randn(8, 12)}, epoch=0)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    with pytest.raises(ValueError, match="unknown encoder_type"):
        UniVIConfig(latent_dim=4, modalities=[ModalityConfig("x", 12, [], [], encoder_type="nope")]).validate()
    with pytest.raises(ValueError, match="built-in"):
        register_encoder("mlp")(lambda u, m: None)
    with pytest.raises(ValueError, match="already registered"):
        register_encoder("tiny_test_encoder")(lambda u, m: None)


def test_backbone_encoder_keeps_a_frozen_backbone_frozen():
    backbone = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.AdaptiveAvgPool2d(1))

    @register_encoder("frozen_backbone_test", overwrite=True)
    def _enc(uni_cfg, mod_cfg):
        return BackboneGaussianEncoder(backbone, feature_dim=4, latent_dim=uni_cfg.latent_dim,
                                       input_shape=(3, 16, 16), hidden_dim=16, freeze=True)

    cfg = UniVIConfig(latent_dim=4, modalities=[
        ModalityConfig("rna", 30, [16], [16]),
        ModalityConfig("image", 768, [], [32], encoder_type="frozen_backbone_test")])
    model = UniVIMultiModalVAE(cfg).train()
    before = {k: v.clone() for k, v in backbone.state_dict().items()}
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    for step in range(3):
        out = model(_paired_batch(), epoch=step)
        opt.zero_grad(); out["loss"].backward(); opt.step()
    assert not backbone.training                                           # stays in eval under model.train()
    assert all(torch.equal(before[k], v) for k, v in backbone.state_dict().items())   # incl. BatchNorm stats
    assert all(p.grad is None for p in backbone.parameters())
    assert any(p.grad is not None for p in model.encoders["image"].head.parameters())


# ------------------------------------------------------------------ spatial utilities

def _grid(side=12):
    yy, xx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], 1).astype(float)


def test_knn_graph_and_aggregation_match_brute_force():
    rng = np.random.default_rng(0)
    coords = rng.uniform(size=(80, 2))
    idx, dist = knn_graph(coords, k=5)
    D = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    assert np.array_equal(np.sort(idx, 1), np.sort(np.argsort(D, 1)[:, :5], 1))
    assert np.allclose(np.sort(dist, 1), np.sort(D, 1)[:, :5])
    X = rng.normal(size=(80, 7))
    brute = np.stack([X[np.argsort(D[i])[:5]].mean(0) for i in range(80)])
    assert np.allclose(neighborhood_aggregate(X, coords, k=5), brute)
    sparse = neighborhood_aggregate(sp.csr_matrix(X), coords, k=5)
    assert sp.issparse(sparse) and np.allclose(sparse.toarray(), brute)
    with_self = neighborhood_aggregate(X, coords, k=5, include_self=True)
    assert np.allclose(with_self, (5 * brute + X) / 6)
    assert abs(median_neighbor_distance(_grid()) - 1.0) < 1e-9
    assert knn_graph(np.zeros((6, 2)), k=2)[0].shape == (6, 2)             # duplicate coordinates do not break it


def test_niche_modality_respects_groups():
    rng = np.random.default_rng(0)
    a = ad.AnnData(rng.normal(size=(60, 5)).astype(np.float32))
    a.obsm["spatial"] = rng.uniform(size=(60, 2))
    a.obs["slide"] = ["s1"] * 30 + ["s2"] * 30
    a.X[30:] += 100.0                                                      # slides differ a lot in level
    niche = niche_modality(a, k=4, group_key="slide")
    assert niche.shape == a.shape and list(niche.obs_names) == list(a.obs_names)
    assert niche.X[:30].max() < 50 and niche.X[30:].min() > 50             # no leakage across slides
    mixed = niche_modality(a, k=4)
    assert not np.allclose(mixed.X, niche.X)


def test_extract_tiles_crops_resizes_and_pads():
    img = np.arange(40 * 50, dtype=np.float32).reshape(40, 50) / 2000.0
    rgb = np.stack([img, img * 0.5, img * 0.25], -1)
    tiles = extract_tiles(rgb, [[10, 20], [25, 8]], tile_size=8, flatten=False)       # (x, y) centres
    assert tiles.shape == (2, 3, 8, 8)
    assert np.allclose(tiles[0, 0], img[16:24, 6:14]) and np.allclose(tiles[1, 1], 0.5 * img[4:12, 21:29])
    small = extract_tiles(rgb, [[10, 20]], tile_size=8, out_size=4, flatten=False)
    assert np.allclose(small[0, 0], img[16:24, 6:14].reshape(4, 2, 4, 2).mean((1, 3)), atol=1e-6)
    flat = extract_tiles(rgb, [[10, 20]], tile_size=8)
    assert flat.shape == (1, 3 * 8 * 8) and np.allclose(unflatten_tiles(flat, (3, 8, 8)), tiles[:1])
    edge = extract_tiles(rgb, [[0, 0], [50, 40]], tile_size=8, flatten=False)         # border tiles are padded
    assert edge.shape == (2, 3, 8, 8) and np.isfinite(edge).all()
    u8 = extract_tiles((rgb * 255).astype(np.uint8), [[10, 20]], tile_size=8, flatten=False)
    assert u8.max() <= 1.0 and np.allclose(u8, tiles[:1], atol=1 / 255)
    with pytest.raises(ValueError, match="outside the image"):
        extract_tiles(rgb, [[500, 20]], tile_size=8)


def test_tiles_from_visium_layout():
    rng = np.random.default_rng(0)
    hires = rng.uniform(size=(200, 200, 3)).astype(np.float32)
    a = ad.AnnData(rng.normal(size=(25, 4)).astype(np.float32))
    a.obsm["spatial"] = rng.uniform(100, 1900, size=(25, 2))                 # full-resolution pixels
    a.uns["spatial"] = {"lib1": {"images": {"hires": hires},
                                 "scalefactors": {"tissue_hires_scalef": 0.1, "spot_diameter_fullres": 160.0}}}
    tiles = tiles_from_visium(a, out_size=8)
    assert tiles.shape == (25, 3 * 8 * 8) and tiles.uns["tile_shape"] == [3, 8, 8]
    assert tiles.uns["tile_source"]["tile_px_in_image"] == 16 and list(tiles.obs_names) == list(a.obs_names)
    cx, cy = (a.obsm["spatial"][0] * 0.1)
    x0, y0 = int(np.rint(cx - 8)), int(np.rint(cy - 8))
    expected = hires[y0:y0 + 16, x0:x0 + 16].transpose(2, 0, 1).reshape(3, 8, 2, 8, 2).mean((2, 4))
    assert np.allclose(unflatten_tiles(tiles.X, tiles.uns["tile_shape"])[0], expected, atol=1e-5)


def test_block_sampler_covers_all_spots_with_local_batches():
    coords = _grid(20)                                                        # 400 spots
    groups = np.where(coords[:, 0] < 10, "left", "right")
    sampler = SpatialBlockSampler(coords, batch_size=40, groups=groups, seed=0)
    batches = list(sampler)
    assert set(np.concatenate(batches)) == set(range(400))
    assert all(len(set(groups[b])) == 1 for b in batches)                     # batches never span groups
    spread = np.mean([coords[b].std(0).mean() for b in batches])
    assert spread < 0.6 * coords.std(0).mean()                                # much tighter than random batches
    assert [b for b in sampler] != batches                                    # a new epoch reshuffles


def _spatial_adatas(side=14, seed=0):
    """Expression varies smoothly across the tissue, plus spot-level noise."""
    rng = np.random.default_rng(seed)
    coords = _grid(side)
    smooth = np.stack([np.sin(coords[:, 0] / 3.0), np.cos(coords[:, 1] / 3.0), coords[:, 0] / side], 1)
    X = smooth @ rng.normal(size=(3, 30)) + 1.0 * rng.normal(size=(side * side, 30))
    a = ad.AnnData(X.astype(np.float32))
    a.obsm["spatial"] = coords
    return a


def test_spatial_prior_through_loader_and_trainer():
    a = _spatial_adatas()
    coords = a.obsm["spatial"]
    idx, _ = knn_graph(coords, 4)
    ratios = {}
    for weight in (0.0, 5.0):
        torch.manual_seed(0)
        cfg = UniVIConfig(latent_dim=4, modalities=[ModalityConfig("rna", 30, [32], [32])],
                          spatial_weight=weight, spatial_sigma=median_neighbor_distance(coords) if weight else None)
        model = UniVIMultiModalVAE(cfg)
        loader = make_spatial_loader({"rna": a}, batch_size=49, seed=0)
        seen = []
        hook = model.register_forward_hook(lambda m, i, out: seen.append("spatial" in out))
        UniVITrainer(model, loader, train_cfg=TrainingConfig(n_epochs=40, lr=3e-3, log_every=1000)).fit()
        hook.remove()
        assert seen and all(s == (weight > 0) for s in seen)
        z = model.eval().encode_fused({"rna": torch.as_tensor(a.X)})[0].detach().numpy()
        near = np.linalg.norm(z[:, None, :] - z[idx], axis=-1).mean()
        ratios[weight] = near / np.linalg.norm(z - z[np.random.default_rng(0).permutation(len(z))], axis=1).mean()
    assert ratios[5.0] < ratios[0.0]                 # neighbours end up relatively closer with the prior on

    plain = make_spatial_loader({"rna": a}, batch_size=49, contiguous=False, shuffle=False)
    x_dict = next(iter(plain))[0] if isinstance(next(iter(plain)), (tuple, list)) else next(iter(plain))
    assert "spatial" in x_dict and x_dict["spatial"].shape[1] == 2
    with pytest.raises(ValueError, match="requires spatial_sigma"):
        UniVIConfig(latent_dim=4, modalities=[ModalityConfig("rna", 30, [32], [32])], spatial_weight=1.0).validate()
    with pytest.raises(ValueError, match="already a modality"):
        make_spatial_loader({"spatial": a})


def test_far_apart_spots_are_not_pulled_together():
    cfg = UniVIConfig(latent_dim=4, modalities=[ModalityConfig("rna", 30, [32], [32])],
                      spatial_weight=1.0, spatial_sigma=1.0)
    model = UniVIMultiModalVAE(cfg)
    x = torch.randn(16, 30)
    far = model({"rna": x, "spatial": 1000.0 * torch.randn(16, 2)}, epoch=0)
    near = model({"rna": x, "spatial": 0.1 * torch.randn(16, 2)}, epoch=0)
    assert float(far["spatial"]) < 1e-6 < float(near["spatial"])
    assert "spatial" not in model({"rna": x}, epoch=0)          # batches without coordinates are fine


def test_end_to_end_tiles_niche_and_expression_with_trainer():
    """The intended workflow: expression + niche + image tiles, conv image branch, spatial prior."""
    rng = np.random.default_rng(0)
    a = _spatial_adatas(side=10)
    a.obsm["spatial"] = a.obsm["spatial"] * 100.0 + 100.0
    a.uns["spatial"] = {"lib": {"images": {"hires": rng.uniform(size=(130, 130, 3)).astype(np.float32)},
                                "scalefactors": {"tissue_hires_scalef": 0.1, "spot_diameter_fullres": 160.0}}}
    tiles = tiles_from_visium(a, out_size=16)
    niche = niche_modality(a, k=4)
    conv = ConvConfig(in_channels=3, spatial_shape=(16, 16), channels=(8, 16), hidden_dim=32, output_activation="sigmoid")
    cfg = UniVIConfig(latent_dim=6, spatial_weight=1.0, spatial_sigma=median_neighbor_distance(a.obsm["spatial"]),
                      modalities=[ModalityConfig("rna", 30, [32], [32]), ModalityConfig("niche", 30, [32], [32]),
                                  ModalityConfig("image", tiles.n_vars, [], [], encoder_type="conv",
                                                 decoder_type="conv", conv=conv)])
    model = UniVIMultiModalVAE(cfg, loss_mode="v1")
    loader = make_spatial_loader({"rna": a, "niche": niche, "image": tiles}, batch_size=25)
    history = UniVITrainer(model, loader, train_cfg=TrainingConfig(n_epochs=6, lr=2e-3, log_every=1000)).fit()
    assert np.isfinite(history["train_loss"]).all() and history["train_loss"][-1] < history["train_loss"][0]
    eval_loader = make_loader({"rna": a, "niche": niche, "image": tiles}, batch_size=50)
    batch = next(iter(eval_loader))
    x_dict = batch[0] if isinstance(batch, (tuple, list)) else batch
    mu = model.eval().encode_fused({k: torch.as_tensor(v) for k, v in x_dict.items()})[0]
    assert mu.shape == (50, 6) and torch.isfinite(mu).all()


def test_reference_bundle_round_trips_conv_geometry_and_anchor(tmp_path):
    from univii import load_reference, save_reference
    from univii.geometry import tree_distance_matrix

    torch.manual_seed(0)
    model = _model("gaussian", "v1", latent_geometry="product", hyperbolic_dim=3, curvature=0.5,
                   spatial_weight=1.0, spatial_sigma=2.0)
    dist, _, _ = tree_distance_matrix({"a": None, "b": "a", "c": "a"})
    model.set_hierarchy(dist, root_index=0)
    model.set_latent_anchor(torch.randn(12, 6))
    x = _paired_batch()
    with torch.no_grad():
        before = model.eval().encode_fused(x, anchored=True)[0]
    save_reference(tmp_path, model, preprocessors={})
    loaded = load_reference(tmp_path)[0]
    assert isinstance(loaded.cfg.modalities[1].conv, ConvConfig)
    assert isinstance(loaded.encoders["image"], ConvGaussianEncoder) and loaded.latent_geometry == "product"
    assert abs(loaded.curvature - 0.5) < 1e-6 and float(loaded.cfg.spatial_sigma) == 2.0
    assert torch.equal(loaded.latent.hierarchy_dist, model.latent.hierarchy_dist)
    with torch.no_grad():
        after = loaded.eval().encode_fused(x, anchored=True)[0]
    assert torch.allclose(before, after, atol=1e-6)
