"""Tests for the 0.2.0 toolkit: priors, contrastive alignment, linear decoders, input masking,
ViT and graph encoders, hierarchy tools, the HMM, presets and the experiment runner."""
from __future__ import annotations

import itertools

import anndata as ad
import numpy as np
import pytest
import torch

from univii import (ClassHeadConfig, ConvConfig, GraphConfig, ModalityConfig, TrainingConfig, UniVIConfig,
                    UniVIMultiModalVAE, UniVITrainer, ViTConfig, presets)
from univii.experiments import compare_models, summarize
from univii.geometry import tree_distance_matrix
from univii.hierarchy import cluster_centroids, cluster_tree, graph_pseudotime, tree_agreement
from univii.hmm import GaussianHMM, segment_by_order
from univii.models import GraphGaussianEncoder, ViTGaussianEncoder, augment_images
from univii.models.graph import knn_in_batch
from univii.priors import MixturePrior
from univii.spatial import encode_spatial, make_spatial_loader


def _mods(**rna_kw):
    return [ModalityConfig("rna", 40, [32], [32], **rna_kw), ModalityConfig("adt", 10, [16], [16])]


def _batch(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 5, (n,), generator=g)
    centers = 2.0 * torch.randn(5, 40, generator=g)
    x = {"rna": centers[y] + 0.3 * torch.randn(n, 40, generator=g),
         "adt": centers[y][:, :10] + 0.3 * torch.randn(n, 10, generator=g)}
    return x, {"ct": y}


def _train(model, x, y=None, steps=120, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    first = last = None
    for step in range(steps):
        out = model(x, epoch=step, y=y)
        opt.zero_grad(); out["loss"].backward(); opt.step()
        first = out if first is None else first
        last = out
    return first, last


# ------------------------------------------------------------------ hidden Markov model

def _brute_force(hmm, x):
    """Enumerate every state path of a short sequence."""
    log_b = hmm.emission_log_prob(hmm._x(x))
    log_a = torch.log_softmax(hmm.trans_logits, 1)
    log_pi = torch.log_softmax(hmm.start_logits, 0)
    T, K = log_b.shape
    scores = {}
    for path in itertools.product(range(K), repeat=T):
        s = log_pi[path[0]] + log_b[0, path[0]]
        for t in range(1, T):
            s = s + log_a[path[t - 1], path[t]] + log_b[t, path[t]]
        scores[path] = s
    total = torch.logsumexp(torch.stack(list(scores.values())), 0)
    marg = torch.zeros(T, K, dtype=torch.float64)
    for path, s in scores.items():
        for t, k in enumerate(path):
            marg[t, k] += torch.exp(s - total)
    return total, marg, max(scores, key=lambda p: float(scores[p]))


def test_hmm_matches_brute_force_enumeration():
    torch.manual_seed(0)
    hmm = GaussianHMM(3, 2, stickiness=0.7)
    with torch.no_grad():
        hmm.means.copy_(torch.randn(3, 2, dtype=torch.float64))
        hmm.logvars.copy_(0.3 * torch.randn(3, 2, dtype=torch.float64))
        hmm.start_logits.copy_(torch.randn(3, dtype=torch.float64))
    x = torch.randn(5, 2, dtype=torch.float64)
    total, marg, best = _brute_force(hmm, x)
    assert torch.allclose(hmm.log_likelihood(x), total, atol=1e-9)
    post = hmm.posterior(x)
    assert torch.allclose(post.sum(1), torch.ones(5, dtype=torch.float64)) and torch.allclose(post, marg, atol=1e-9)
    assert tuple(hmm.viterbi(x).tolist()) == best


def test_hmm_recovers_segments_and_respects_cell_order():
    rng = np.random.default_rng(0)
    truth = np.repeat([0, 1, 2, 1], 60)
    signal = np.array([-3.0, 0.0, 3.0])[truth] + 0.5 * rng.normal(size=truth.size)
    shuffle = rng.permutation(truth.size)                       # cells arrive in arbitrary order
    states, probs, hmm = segment_by_order(signal[shuffle], order=shuffle, n_states=3)
    assert probs.shape == (240, 3) and np.allclose(probs.sum(1), 1.0)
    ordered = np.empty(240, dtype=int)
    ordered[shuffle] = states                                   # back to position order
    relabel = {s: int(np.round(np.median(truth[ordered == s]))) for s in np.unique(ordered)}
    assert np.mean(np.vectorize(relabel.get)(ordered) == truth) > 0.95
    assert (np.diff(ordered) != 0).sum() <= 6                   # runs, not noise
    with pytest.raises(ValueError):
        segment_by_order(signal, order=np.arange(10))


# ------------------------------------------------------------------ mixture prior

def test_mixture_prior_basics():
    torch.manual_seed(0)
    prior = MixturePrior(1, 3)
    with torch.no_grad():
        prior.means.zero_()
    z = torch.randn(50, 3)
    standard = (-0.5 * (z ** 2 + np.log(2 * np.pi))).sum(-1)
    assert torch.allclose(prior.log_prob(z), standard, atol=1e-5)          # one unit component = N(0, I)
    prior = MixturePrior(4, 3)
    r = prior.responsibilities(z)
    assert r.shape == (50, 4) and torch.allclose(r.sum(1), torch.ones(50), atol=1e-6)
    labels = torch.tensor([2] * 25 + [-1] * 25)
    lp = prior.log_prob(z, labels)
    assert torch.allclose(lp[:25], prior.component_log_prob(z)[:25, 2]) and torch.allclose(lp[25:], prior.log_prob(z)[25:])
    line = MixturePrior(3, 2)
    with torch.no_grad():
        line.means.copy_(torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
    dist = torch.tensor([[0.0, 1, 2], [1, 0, 1], [2, 1, 0]])
    assert float(line.tree_penalty(dist, torch.tensor(1.0))) < 1e-9
    with pytest.raises(ValueError):
        line.tree_penalty(torch.zeros(4, 4), torch.tensor(1.0))


@pytest.mark.parametrize("geometry,hd", [("euclidean", None), ("product", 3), ("hyperbolic", None)])
@pytest.mark.parametrize("loss_mode", ["v1", "v2"])
def test_label_tied_mixture_prior_learns_one_component_per_class(geometry, hd, loss_mode):
    torch.manual_seed(0)
    cfg = UniVIConfig(latent_dim=6, modalities=_mods(), latent_geometry=geometry, hyperbolic_dim=hd,
                      prior="mog", prior_components=5, prior_label_head="ct", prior_tree_weight=1.0)
    model = UniVIMultiModalVAE(cfg, loss_mode=loss_mode)
    dist, _, _ = tree_distance_matrix({"a": None, "b": "a", "c": "a", "d": "b", "e": "b"})
    model.set_hierarchy(dist, root_index=0)
    x, y = _batch(n=128)
    first, last = _train(model, x, y, steps=200)
    assert "prior_tree" in last and float(last["prior_tree"]) < float(first["prior_tree"])
    assert torch.isfinite(last["loss"]) and float(last["loss"]) < float(first["loss"])
    with torch.no_grad():
        resp = model.prior_responsibilities(model.eval().encode_fused(x)[0])
    assert resp.shape == (128, 5) and float((resp.argmax(1) == y["ct"]).float().mean()) > 0.9


def test_mixture_prior_validation_and_unlabelled_use():
    with pytest.raises(ValueError, match="require prior='mog'"):
        UniVIConfig(latent_dim=6, modalities=_mods(), prior_tree_weight=1.0).validate()
    model = UniVIMultiModalVAE(UniVIConfig(latent_dim=6, modalities=_mods(), prior="mog", prior_components=3))
    x, _ = _batch()
    out = model(x, epoch=0)                                     # no labels at all
    out["loss"].backward()
    assert torch.isfinite(out["loss"]) and model.mixture_prior.means.grad is not None
    with pytest.raises(ValueError, match="prior='mog'"):
        UniVIMultiModalVAE(UniVIConfig(latent_dim=6, modalities=_mods())).prior_responsibilities(torch.zeros(2, 6))


# ------------------------------------------------------------------ contrastive alignment

def _retrieval(model, x):
    with torch.no_grad():
        mu, _ = model.eval().encode_modalities(x)
        d = model.pairwise_latent_distance(mu["rna"], mu["adt"])
    return float((d.argmin(1) == torch.arange(d.shape[0])).float().mean())


@pytest.mark.parametrize("geometry", ["euclidean", "hyperbolic"])
def test_contrastive_term_makes_modalities_retrieve_each_other(geometry):
    x, _ = _batch(n=64)
    scores = {}
    for weight in (0.0, 5.0):
        torch.manual_seed(0)
        cfg = UniVIConfig(latent_dim=6, modalities=_mods(), gamma=0.0, latent_geometry=geometry,
                          contrastive_weight=weight, contrastive_temperature=0.5)
        model = UniVIMultiModalVAE(cfg, loss_mode="v2")
        _, last = _train(model, x, steps=150)
        assert ("contrastive" in last) == (weight > 0)
        scores[weight] = _retrieval(model, x)
    assert scores[5.0] > scores[0.0] + 0.2                     # far above the unaligned baseline
    if geometry == "euclidean":
        assert scores[5.0] > 0.5
    single = UniVIMultiModalVAE(UniVIConfig(latent_dim=6, modalities=_mods(), contrastive_weight=1.0))
    assert "contrastive" not in single({"rna": x["rna"]}, epoch=0)          # needs two modalities


# ------------------------------------------------------------------ decoders and input masking

@pytest.mark.parametrize("likelihood", ["gaussian", "nb"])
def test_linear_decoder_exposes_loadings(likelihood):
    torch.manual_seed(0)
    mods = [ModalityConfig("rna", 40, [32], [], likelihood=likelihood, decoder_type="linear"),
            ModalityConfig("adt", 10, [16], [16])]
    model = UniVIMultiModalVAE(UniVIConfig(latent_dim=6, modalities=mods))
    x, _ = _batch()
    if likelihood == "nb":
        x = {"rna": torch.poisson(torch.exp(0.3 * x["rna"]).clamp(max=50)), "adt": x["adt"]}
    first, last = _train(model, x, steps=60)
    assert float(last["loss"]) < float(first["loss"])
    loadings = model.decoder_loadings("rna")
    assert loadings.shape == (40, 6) and torch.isfinite(loadings).all()
    assert sum(isinstance(m, torch.nn.Linear) for m in model.decoders["rna"].modules()) == 1
    with pytest.raises(ValueError, match="does not use"):
        model.decoder_loadings("adt")


def test_input_dropout_masks_only_in_training():
    x, _ = _batch()
    torch.manual_seed(0)
    plain = UniVIMultiModalVAE(UniVIConfig(latent_dim=6, modalities=_mods()))
    torch.manual_seed(0)
    masked = UniVIMultiModalVAE(UniVIConfig(latent_dim=6, modalities=_mods(input_dropout=0.5)))
    with torch.no_grad():
        assert torch.equal(plain.eval().encode_modalities(x)[0]["rna"], masked.eval().encode_modalities(x)[0]["rna"])
        a, b = masked.train().encode_modalities(x)[0], masked.encode_modalities(x)[0]
        assert not torch.allclose(a["rna"], b["rna"]) and torch.equal(a["adt"], b["adt"])
        c, d = plain.train().encode_modalities(x)[0], plain.encode_modalities(x)[0]
        assert torch.equal(c["rna"], d["rna"])
    with pytest.raises(ValueError, match="input_dropout"):
        UniVIConfig(latent_dim=6, modalities=_mods(input_dropout=1.0)).validate()


# ------------------------------------------------------------------ image encoders

def test_vit_encoder_shapes_pooling_and_augmentation():
    torch.manual_seed(0)
    x = torch.rand(4, 3 * 16 * 16)
    for pool in ("mean", "cls"):
        enc = ViTGaussianEncoder(ViTConfig(image_size=(16, 16), patch_size=4, d_model=32, num_heads=2,
                                           num_layers=2, dim_feedforward=64, pool=pool), latent_dim=5).eval()
        mu, logvar = enc(x)
        assert mu.shape == logvar.shape == (4, 5)
        assert torch.allclose(enc(x.reshape(4, 3, 16, 16))[0], mu, atol=1e-6)
        assert enc.tokens(x).shape == (4, 16 + (pool == "cls"), 32)
    img = torch.arange(2 * 1 * 4 * 4, dtype=torch.float32).reshape(2, 1, 4, 4)
    variants = [torch.rot90(f, k, dims=(-2, -1)) for f in (img, torch.flip(img, dims=(-1,))) for k in range(4)]
    seen = set()
    for _ in range(60):
        out = augment_images(img, ("flip", "rot90"))
        match = [i for i, v in enumerate(variants) if torch.equal(out, v)]
        assert len(match) == 1                                   # always one of the 8 symmetries
        seen.add(match[0])
    assert len(seen) >= 6
    for bad in (dict(image_size=(30, 30)), dict(d_model=33), dict(augment=("blur",)), dict(image_size=(16, 8), augment=("rot90",))):
        with pytest.raises(ValueError):
            ViTConfig(**{**dict(image_size=(16, 16), patch_size=4, d_model=32, num_heads=2), **bad}).validate()


@pytest.mark.parametrize("encoder", ["vit", "conv"])
def test_image_model_with_augmentation_trains_and_is_deterministic_in_eval(encoder):
    torch.manual_seed(0)
    conv = ConvConfig(in_channels=3, spatial_shape=(16, 16), channels=(8, 16), hidden_dim=32, augment=("flip", "rot90"))
    image = dict(decoder_type="conv", conv=conv)
    if encoder == "vit":
        image.update(encoder_type="vit", vit=ViTConfig(image_size=(16, 16), patch_size=4, d_model=32, num_heads=2,
                                                       num_layers=2, dim_feedforward=64, augment=("flip", "rot90")))
    else:
        image.update(encoder_type="conv")
    cfg = UniVIConfig(latent_dim=6, modalities=[ModalityConfig("rna", 30, [32], [32]),
                                                 ModalityConfig("image", 768, [], [], **image)])
    model = UniVIMultiModalVAE(cfg, loss_mode="v1")
    g = torch.Generator().manual_seed(1)
    code = torch.rand(32, 3, generator=g)
    x = {"rna": code.repeat(1, 10) + 0.05 * torch.randn(32, 30, generator=g),
         "image": (code[:, :, None, None] * torch.ones(32, 3, 16, 16)).flatten(1)}
    first, last = _train(model, x, steps=60)
    assert float(last["loss"]) < float(first["loss"])
    with torch.no_grad():
        a, b = model.eval().encode_fused(x)[0], model.encode_fused(x)[0]
    assert torch.equal(a, b)


# ------------------------------------------------------------------ graph encoder

def test_graph_encoder_aggregates_in_batch_neighbours():
    coords = torch.tensor([[0.0, 0], [1, 0], [10, 0], [11, 0]])
    assert knn_in_batch(coords, 1).view(-1).tolist() == [1, 0, 3, 2]
    assert knn_in_batch(coords, 99).shape == (4, 3)                           # capped at B - 1
    torch.manual_seed(0)
    enc = GraphGaussianEncoder(5, GraphConfig(k=1, hidden=(8,)), latent_dim=3).eval()
    x = torch.randn(4, 5)
    with torch.no_grad():
        mu = enc(x, coords=coords)[0]
        manual = enc.head(enc.act(enc.norms[0](enc.layers[0](torch.cat([x, x[[1, 0, 3, 2]]], dim=-1))))).chunk(2, -1)[0]
        alone = enc(x)[0]
        self_only = enc.head(enc.act(enc.norms[0](enc.layers[0](torch.cat([x, x], dim=-1))))).chunk(2, -1)[0]
    assert torch.allclose(mu, manual, atol=1e-6) and torch.allclose(alone, self_only, atol=1e-6)
    assert not torch.allclose(mu, alone)
    assert enc(x[:1], coords=coords[:1])[0].shape == (1, 3)                   # a batch of one is fine


def _spatial_adata(side=10, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    coords = np.stack([xx.ravel(), yy.ravel()], 1).astype(float)
    smooth = np.stack([np.sin(coords[:, 0] / 3.0), np.cos(coords[:, 1] / 3.0)], 1)
    a = ad.AnnData((smooth @ rng.normal(size=(2, 30)) + 0.5 * rng.normal(size=(side * side, 30))).astype(np.float32))
    a.obsm["spatial"] = coords
    return a


def test_graph_encoder_through_spatial_loader_trainer_and_encode_spatial():
    torch.manual_seed(0)
    a = _spatial_adata()
    cfg = UniVIConfig(latent_dim=4, modalities=[ModalityConfig("rna", 30, [], [32], encoder_type="graph",
                                                                graph=GraphConfig(k=4, hidden=(32, 16)))])
    model = UniVIMultiModalVAE(cfg)
    assert isinstance(model.encoders["rna"], GraphGaussianEncoder)
    history = UniVITrainer(model, make_spatial_loader({"rna": a}, batch_size=25),
                           train_cfg=TrainingConfig(n_epochs=15, lr=3e-3, log_every=10 ** 9)).fit()
    assert history["train_loss"][-1] < history["train_loss"][0]
    x = torch.as_tensor(a.X[:25])
    with torch.no_grad():
        with_coords = model.eval().encode_fused({"rna": x, "spatial": torch.as_tensor(a.obsm["spatial"][:25])})[0]
        without = model.encode_fused({"rna": x})[0]
    assert not torch.allclose(with_coords, without)              # the coordinates reach the encoder
    z1 = encode_spatial(model, {"rna": a}, batch_size=25, seed=0)
    z2 = encode_spatial(model, {"rna": a}, batch_size=25, seed=0)
    assert z1.shape == (100, 4) and np.isfinite(z1).all() and np.allclose(z1, z2)
    assert not model.training


# ------------------------------------------------------------------ hierarchy tools

TREE = {"root": None, "a": "root", "b": "root", "a1": "a", "a2": "a", "b1": "b"}
# Positions chosen so that every child is closer to its parent than to any other cluster; a spanning
# tree over centroids can only recover a hierarchy with that property.
POS = {"root": (0, 0), "a": (2, 2), "b": (2, -2), "a1": (4, 4), "a2": (4.5, 0.5), "b1": (4, -4)}


def _tree_cells(n_per=60, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    names = list(TREE)
    labels = np.repeat(names, n_per)
    z = np.array([POS[n] for n in labels], dtype=float) + noise * rng.normal(size=(len(labels), 2))
    return z, labels


def test_cluster_tree_recovers_a_planted_tree():
    z, labels = _tree_cells()
    tree = cluster_tree(z, labels, root="root")
    assert tree["parents"] == TREE
    assert tree_agreement(tree["parents"], TREE) == {"edge_jaccard": 1.0, "path_spearman": 1.0, "parent_accuracy": 1.0}
    assert cluster_tree(z, labels)["parents"] == TREE                      # root found from the origin
    wrong = dict(TREE, a2="b")
    score = tree_agreement(wrong, TREE)
    assert score["edge_jaccard"] < 1.0 and score["parent_accuracy"] == pytest.approx(5 / 6)
    centroids, names = cluster_centroids(z, labels)
    assert np.allclose(centroids, [POS[n] for n in names], atol=0.1)
    hyp = cluster_tree(0.5 * z, labels, root="root", geometry="hyperbolic")   # runs with geodesic distances
    assert set(hyp["parents"]) == set(TREE) and hyp["parents"]["root"] is None
    with pytest.raises(ValueError):
        cluster_tree(z, labels, root="nope")


def test_graph_pseudotime_follows_the_tree_and_bridges_gaps():
    z, labels = _tree_cells()
    depth = {"root": 0, "a": 1, "b": 1, "a1": 2, "a2": 2, "b1": 2}
    pt = graph_pseudotime(z, labels == "root", k=10)
    from scipy.stats import spearmanr
    assert np.isfinite(pt).all() and pt[labels == "root"].max() == 0.0
    assert spearmanr(pt, [depth[n] for n in labels]).statistic > 0.9
    far = np.vstack([z, z[:30] + 100.0])                                    # a second, disconnected piece
    pt_far = graph_pseudotime(far, np.r_[labels == "root", np.zeros(30, bool)], k=5)
    assert np.isfinite(pt_far).all() and pt_far[-30:].min() > 50
    with pytest.raises(ValueError):
        graph_pseudotime(z, np.zeros(len(z), bool))


# ------------------------------------------------------------------ presets and experiment runner

def test_presets_build_working_models():
    x, y = _batch()
    dims = {"rna": 40, "adt": 10}
    for cfg in (presets.paired_config(dims), presets.clustered_config(dims, n_components=4),
                presets.contrastive_config(dims), presets.hierarchy_config(dims, n_labels=5, label_head="ct"),
                presets.paired_config(dims, likelihoods={"rna": "gaussian"}, latent_dim=8, beta=2.0)):
        out = UniVIMultiModalVAE(cfg)(x, epoch=0, y=y)
        assert torch.isfinite(out["loss"])
    assert presets.paired_config(dims, latent_dim=8).latent_dim == 8
    assert presets.hierarchy_config(dims, n_labels=5).latent_geometry == "product"
    for encoder in ("conv", "vit"):
        cfg = presets.spatial_image_config(30, (3, 16, 16), spatial_sigma=1.0, encoder=encoder, latent_dim=6)
        batch = {"rna": torch.randn(8, 30), "image": torch.rand(8, 768), "spatial": torch.rand(8, 2)}
        out = UniVIMultiModalVAE(cfg, loss_mode="v1")(batch, epoch=0)
        out["loss"].backward()
        assert torch.isfinite(out["loss"]) and "spatial" in out
    with pytest.raises(ValueError):
        presets.spatial_image_config(30, (3, 16, 16), spatial_sigma=1.0, encoder="nope")


def test_compare_models_runs_every_variant_and_seed():
    from univii import make_loader
    x, y = _batch(n=80)
    names = [f"c{i}" for i in range(80)]
    rna, adt = ad.AnnData(x["rna"].numpy()), ad.AnnData(x["adt"].numpy())
    rna.obs_names = adt.obs_names = names
    loader = make_loader({"rna": rna, "adt": adt}, batch_size=40, shuffle=True)
    dims = {"rna": 40, "adt": 10}
    calls = []
    variants = {"euclidean": presets.paired_config(dims, latent_dim=6),
                "product_v2": (presets.paired_config(dims, latent_dim=6, latent_geometry="product"), {"loss_mode": "v2"})}
    runs, models = compare_models(variants, loader, evaluate=lambda m: {"retrieval": _retrieval(m, x)},
                                  seeds=(0, 1), train_cfg=TrainingConfig(n_epochs=5, log_every=10 ** 9),
                                  setup=lambda m: calls.append(m.latent_geometry), keep_models=True)
    assert list(runs["variant"]) == ["euclidean", "euclidean", "product_v2", "product_v2"]
    assert {"seed", "n_parameters", "train_seconds", "final_train_loss", "retrieval"} <= set(runs.columns)
    assert calls == ["euclidean", "euclidean", "product", "product"] and len(models) == 4
    assert models[("product_v2", 0)].loss_mode == "v2" and not models[("euclidean", 1)].training
    table = summarize(runs)
    assert list(table.index) == ["euclidean", "product_v2"] and list(table["n_seeds"]) == [2, 2]
    assert ("retrieval", "mean") in table.columns


# ------------------------------------------------------------------ persistence

def test_reference_bundle_round_trips_the_new_parts(tmp_path):
    from univii import load_reference, save_reference
    torch.manual_seed(0)
    mods = [ModalityConfig("rna", 30, [], [], encoder_type="graph", graph=GraphConfig(k=3, hidden=(16,)),
                           decoder_type="linear", input_dropout=0.2),
            ModalityConfig("image", 768, [], [32], encoder_type="vit",
                           vit=ViTConfig(image_size=(16, 16), patch_size=4, d_model=32, num_heads=2, num_layers=1,
                                         dim_feedforward=64))]
    cfg = UniVIConfig(latent_dim=6, modalities=mods, prior="mog", prior_components=4, contrastive_weight=1.0)
    model = UniVIMultiModalVAE(cfg).eval()
    x = {"rna": torch.randn(8, 30), "image": torch.rand(8, 768), "spatial": torch.rand(8, 2)}
    with torch.no_grad():
        before = model.encode_fused(x)[0]
    save_reference(tmp_path, model, preprocessors={})
    loaded = load_reference(tmp_path)[0].eval()
    assert isinstance(loaded.cfg.modalities[0].graph, GraphConfig) and isinstance(loaded.cfg.modalities[1].vit, ViTConfig)
    assert loaded.cfg.prior == "mog" and loaded.cfg.modalities[0].input_dropout == 0.2
    with torch.no_grad():
        assert torch.allclose(loaded.encode_fused(x)[0], before, atol=1e-6)
    assert torch.equal(loaded.mixture_prior.means, model.mixture_prior.means)
    assert torch.equal(loaded.decoder_loadings("rna"), model.decoder_loadings("rna"))
