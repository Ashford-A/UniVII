"""Tests for the hyperbolic mixture prior, cycle consistency, adapting to a new cohort, and
stain normalisation."""
from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import torch

from univii import (ModalityConfig, TrainingConfig, UniVIConfig, UniVIMultiModalVAE, UniVITrainer, load_reference,
                    make_loader, presets, save_reference)
from univii.atlas import adapt_to_cohort, cohort_mixing, encode_cohort, label_purity
from univii.geometry import tree_distance_matrix
from univii.histology import MacenkoNormalizer, od_to_rgb, rgb_to_od

QUIET = dict(log_every=10 ** 9)


def _mods(adt_likelihood="gaussian"):
    return [ModalityConfig("rna", 40, [32], [32]), ModalityConfig("adt", 10, [16], [16], likelihood=adt_likelihood)]


def _batch(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 5, (n,), generator=g)
    centers = 2.0 * torch.randn(5, 40, generator=g)
    x = {"rna": centers[y] + 0.3 * torch.randn(n, 40, generator=g),
         "adt": centers[y][:, :10] + 0.3 * torch.randn(n, 10, generator=g)}
    return x, {"ct": y}


# ------------------------------------------------------------------ learnable hyperbolic space

def test_one_unit_component_at_the_origin_matches_the_standard_hyperbolic_prior():
    torch.manual_seed(0)
    standard = UniVIMultiModalVAE(UniVIConfig(latent_dim=4, modalities=_mods(), latent_geometry="hyperbolic"))
    mog = UniVIMultiModalVAE(UniVIConfig(latent_dim=4, modalities=_mods(), latent_geometry="hyperbolic",
                                         prior="mog", prior_components=1))
    with torch.no_grad():
        mog.mixture_prior.means.zero_()
    mu = 0.5 * torch.randn(1, 4).expand(6000, 4).contiguous()
    logvar = torch.full((6000, 4), -1.0)
    zeros = torch.zeros_like(mu)
    with torch.no_grad():
        a = standard._kl_gaussian(mu, logvar, zeros, zeros, is_prior=True).mean()
        b = mog._kl_gaussian(mu, logvar, zeros, zeros, is_prior=True).mean()
    assert abs(float(a) - float(b)) < 0.05 * abs(float(a)) + 0.03          # two Monte Carlo estimates of one number


def test_hyperbolic_prototypes_and_curvature_are_learned_and_follow_the_tree():
    torch.manual_seed(0)
    cfg = UniVIConfig(latent_dim=4, modalities=_mods(), latent_geometry="hyperbolic", learn_curvature=True,
                      prior="mog", prior_components=5, prior_label_head="ct", prior_tree_weight=2.0)
    model = UniVIMultiModalVAE(cfg)
    dist, _, _ = tree_distance_matrix({"a": None, "b": "a", "c": "a", "d": "b", "e": "b"})
    model.set_hierarchy(dist, root_index=0)
    x, y = _batch(n=128)
    before_c, before_centres = float(model.curvature), model.prior_centres().clone()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = last = None
    for step in range(250):
        out = model(x, epoch=step, y=y)
        opt.zero_grad(); out["loss"].backward(); opt.step()
        first = out if first is None else first
        last = out
    assert float(model.curvature) != before_c and model.latent.log_curvature.grad is not None
    centres = model.prior_centres()
    assert centres.shape == (5, 4) and not torch.allclose(centres, before_centres)
    assert float(last["prior_tree"]) < 0.5 * float(first["prior_tree"])
    geodesic = model.pairwise_latent_distance(centres).detach()
    iu = torch.triu_indices(5, 5, 1)
    from scipy.stats import spearmanr
    assert spearmanr(geodesic[iu[0], iu[1]].numpy(), dist[iu[0], iu[1]]).statistic > 0.8   # centres laid out like the tree
    with torch.no_grad():
        resp = model.prior_responsibilities(model.eval().encode_fused(x)[0])
    assert torch.allclose(resp.sum(1), torch.ones(128), atol=1e-5)
    assert float((resp.argmax(1) == y["ct"]).float().mean()) > 0.85


# ------------------------------------------------------------------ cycle consistency

def test_cycle_term_mechanics():
    torch.manual_seed(0)
    model = UniVIMultiModalVAE(UniVIConfig(latent_dim=6, modalities=_mods(), cycle_weight=1.0))
    x, _ = _batch()
    out = model(x, epoch=0)
    assert "cycle" in out and torch.isfinite(out["cycle"])
    assert "cycle" in model({"rna": x["rna"]}, epoch=0)                   # runs through the missing modality
    assert "cycle" not in UniVIMultiModalVAE(UniVIConfig(latent_dim=6, modalities=_mods()))(x, epoch=0)

    bn = [m for m in model.encoders["adt"].modules() if isinstance(m, torch.nn.BatchNorm1d)][0]
    mu_dict = model.train().encode_modalities({"rna": x["rna"]})[0]
    stats = bn.running_mean.clone()
    cyc = model._cycle_term(mu_dict)
    assert torch.equal(bn.running_mean, stats) and bn.momentum != 0.0      # predicted inputs leave the statistics alone
    model.zero_grad()
    cyc.mean().backward()
    for part in (model.encoders["adt"], model.decoders["adt"]):
        assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in part.parameters())

    counts = UniVIMultiModalVAE(UniVIConfig(latent_dim=6, modalities=_mods("nb"), cycle_weight=1.0))
    assert "cycle" not in counts({"rna": x["rna"]}, epoch=0)               # count targets are skipped
    with pytest.raises(ValueError):
        UniVIConfig(latent_dim=6, modalities=_mods(), cycle_weight=-1.0).validate()


def test_training_with_the_cycle_term_reduces_it():
    torch.manual_seed(0)
    model = UniVIMultiModalVAE(UniVIConfig(latent_dim=6, modalities=_mods(), cycle_weight=1.0), loss_mode="v2")
    x, _ = _batch()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    values = []
    for step in range(150):
        out = model(x, epoch=step)
        opt.zero_grad(); out["loss"].backward(); opt.step()
        values.append(float(out["cycle"]))
    assert np.mean(values[-10:]) < 0.5 * np.mean(values[:10])


# ------------------------------------------------------------------ adapting to a new cohort

def _adata(x, prefix):
    a = ad.AnnData(np.asarray(x, dtype=np.float32))
    a.obs_names = [f"{prefix}{i}" for i in range(a.n_obs)]
    return a


def test_adapt_to_cohort_aligns_a_new_cohort_and_leaves_the_atlas_untouched(tmp_path):
    rng = np.random.default_rng(0)
    centers, offset = 2.0 * rng.normal(size=(5, 40)), 1.4 * rng.normal(size=40)
    y_ref, y_new = rng.integers(0, 5, 300), rng.integers(0, 5, 200)
    ref = {"rna": _adata(centers[y_ref] + 0.3 * rng.normal(size=(300, 40)), "r")}
    new = {"rna": _adata(centers[y_new] + offset + 0.3 * rng.normal(size=(200, 40)), "q")}
    torch.manual_seed(0)
    cfg = presets.atlas_config({"rna": 40}, n_cohorts=1, latent_dim=6, cohort_mmd_weight=0.0)
    model = UniVIMultiModalVAE(cfg)
    UniVITrainer(model, make_loader(ref, labels={"cohort": np.zeros(300, dtype=np.int64)}, batch_size=50, shuffle=True),
                 train_cfg=TrainingConfig(n_epochs=40, lr=3e-3, **QUIET)).fit()
    model.train()                                                          # the mode must be restored afterwards

    z_ref = encode_cohort(model, ref, cohort=0)
    cohort = np.r_[["ref"] * 300, ["new"] * 200]
    cell_type = np.r_[y_ref, y_new]
    before = cohort_mixing(np.vstack([z_ref, encode_cohort(model, new)]), cohort)
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    flags = {n: p.requires_grad for n, p in model.named_parameters()}

    result = adapt_to_cohort(model, make_loader(new, batch_size=50, shuffle=True), reference_z=z_ref,
                             n_epochs=40, lr=0.05, mmd_weight=20.0)
    code = result["code"]
    assert code == 1 and model.cfg.n_cohorts == 2 and model.cfg.cohort_input_shift is True
    assert result["history"][-1] < result["history"][0] and model.training

    assert np.array_equal(encode_cohort(model, ref, cohort=0), z_ref)      # the atlas did not move
    for name, old in weights.items():
        current = model.state_dict()[name]
        assert torch.equal(current[:old.shape[0]] if name.endswith("embed.weight") else current, old), name
    assert all(p.requires_grad == flags[n] for n, p in model.named_parameters() if n in flags)
    assert float(model.input_shifts["rna"].weight[0].abs().sum()) == 0.0 and float(model.input_shifts["rna"].weight[1].abs().sum()) > 0

    z_new = encode_cohort(model, new, cohort=code)
    z_all = np.vstack([z_ref, z_new])
    after, purity = cohort_mixing(z_all, cohort), label_purity(z_all, cell_type)
    assert before < 0.2 and after > 0.8 and purity > 0.9
    shift = model.input_shifts["rna"].weight[code].detach().numpy()
    assert float(shift @ -offset) / (np.linalg.norm(shift) * np.linalg.norm(offset)) > 0.95   # found the planted shift
    assert model._decode_cohort is None                                    # encode_cohort restores the setting

    save_reference(tmp_path, model, preprocessors={})
    loaded = load_reference(tmp_path)[0]
    assert np.allclose(encode_cohort(loaded, new, cohort=code), z_new, atol=1e-5)

    # a two-modality atlas, and a new cohort that measured only one of them
    two = UniVIMultiModalVAE(presets.atlas_config({"rna": 40, "adt": 10}, n_cohorts=2, latent_dim=6))
    adt_before = two.decoders["adt"].embed.weight.detach().clone()
    out = adapt_to_cohort(two, make_loader(new, batch_size=50), n_epochs=2)
    assert out["code"] == 2 and float(two.input_shifts["rna"].weight[2].abs().sum()) > 0
    assert float(two.input_shifts["adt"].weight.abs().sum()) == 0.0
    assert torch.equal(two.decoders["adt"].embed.weight[:2], adt_before)

    with pytest.raises(ValueError, match="n_cohorts >= 1"):
        adapt_to_cohort(UniVIMultiModalVAE(presets.paired_config({"rna": 40})), [], n_epochs=1)
    with pytest.raises(ValueError, match="outside"):
        model.set_cohort(9)


# ------------------------------------------------------------------ stain normalisation

H_E = np.array([[0.65, 0.07], [0.70, 0.99], [0.29, 0.11]])                # textbook haematoxylin, eosin colours
H_E = H_E / np.linalg.norm(H_E, axis=0)


def _render(conc, stains, scale=1.0):
    """Tiles (n, 3, H, W) from stain amounts (n, H, W, 2)."""
    return np.exp(-(conc * scale) @ stains.T).transpose(0, 3, 1, 2)


def _concentrations(n=12, size=24, seed=0):
    rng = np.random.default_rng(seed)
    conc = rng.exponential(0.6, size=(n, size, size, 2))
    conc[rng.random((n, size, size)) < 0.15] = 0.0                          # some bare glass
    return conc


def test_macenko_recovers_stain_colours_and_removes_a_stain_shift():
    other = H_E + np.array([[0.10, -0.04], [-0.08, 0.02], [0.06, 0.10]])    # a differently stained slide
    other = other / np.linalg.norm(other, axis=0)
    reference = _render(_concentrations(seed=0), H_E)
    stains, max_c = MacenkoNormalizer().estimate(reference)
    assert stains.shape == (3, 2) and np.allclose(np.linalg.norm(stains, axis=0), 1.0)
    assert all(float(stains[:, i] @ H_E[:, i]) > 0.98 for i in range(2))    # haematoxylin first, eosin second
    est_other, _ = MacenkoNormalizer().estimate(_render(_concentrations(seed=1), other, 1.3))
    assert all(float(est_other[:, i] @ other[:, i]) > 0.98 for i in range(2))

    norm = MacenkoNormalizer().fit(reference)
    conc = _concentrations(seed=2)
    wanted, shifted = _render(conc, H_E), _render(conc, other, 1.3)         # same tissue, two stainings
    fixed = norm.transform(shifted)
    assert fixed.shape == shifted.shape and fixed.dtype == np.float32
    err_before, err_after = np.abs(shifted - wanted).mean(), np.abs(fixed - wanted).mean()
    assert err_after < 0.3 * err_before
    assert np.abs(norm.transform(wanted) - wanted).mean() < 0.03            # already-matching tiles barely change


def test_macenko_accepts_tile_anndata_and_rejects_bad_input():
    tiles = _render(_concentrations(n=6, size=16), H_E)
    a = ad.AnnData(tiles.reshape(6, -1).astype(np.float32))
    a.uns["tile_shape"] = [3, 16, 16]
    original = a.X.copy()
    norm = MacenkoNormalizer().fit(a)
    out = norm.transform(a)
    assert isinstance(out, ad.AnnData) and out.shape == a.shape and list(out.uns["tile_shape"]) == [3, 16, 16]
    assert np.array_equal(a.X, original) and float(out.X.min()) >= 0.0 and float(out.X.max()) <= 1.0
    assert np.allclose(od_to_rgb(rgb_to_od(tiles)), np.clip(tiles, 1 / 255, 1), atol=1e-9)
    with pytest.raises(RuntimeError):
        MacenkoNormalizer().transform(tiles)
    with pytest.raises(ValueError, match="too few tissue"):
        MacenkoNormalizer().fit(np.ones((2, 3, 16, 16)))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        MacenkoNormalizer().fit(tiles * 255.0)
    with pytest.raises(ValueError, match="RGB tiles"):
        MacenkoNormalizer().fit(np.zeros((2, 1, 16, 16)))
