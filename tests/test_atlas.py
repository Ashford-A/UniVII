"""Tests for atlas / mosaic / bridge support: the mosaic loader, cohort-conditioned decoders,
the cohort alignment term, and the mixing, purity and label-transfer tools."""
from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pytest
import torch

from univii import (ConvConfig, ModalityConfig, TrainingConfig, UniVIConfig, UniVIMultiModalVAE, UniVITrainer,
                    load_reference, make_loader, presets, save_reference)
from univii.atlas import MosaicLoader, cohort_mixing, label_purity, make_mosaic_loader, transfer_labels
from univii.models.conditioning import CohortConditionedDecoder

QUIET = dict(log_every=10 ** 9)


def _adata(x, prefix):
    a = ad.AnnData(np.asarray(x, dtype=np.float32))
    a.obs_names = [f"{prefix}{i}" for i in range(a.n_obs)]
    return a


def _two_cohorts(n=200, seed=0):
    """The same five cell types measured in two cohorts; the second has a constant technical shift."""
    rng = np.random.default_rng(seed)
    centers = 2.0 * rng.normal(size=(5, 40))
    offset = 1.4 * rng.normal(size=40)
    cohorts, types = {}, {}
    for name, on in (("ref", 0.0), ("query", 1.0)):
        y = rng.integers(0, 5, n)
        cohorts[name] = {"rna": _adata(centers[y] + on * offset + 0.3 * rng.normal(size=(n, 40)), name)}
        types[name] = y
    return cohorts, types


def _embed(model, cohorts):
    model.eval()
    with torch.no_grad():
        return torch.cat([model.encode_fused({"rna": torch.as_tensor(c["rna"].X)})[0] for c in cohorts.values()]).numpy()


# ------------------------------------------------------------------ loader

def test_mosaic_loader_interleaves_and_is_reproducible():
    a, b = [("a", i) for i in range(6)], [("b", i) for i in range(2)]
    loader = MosaicLoader({"big": a, "small": b}, seed=1)
    assert len(loader) == 8 and loader.names == ["big", "small"] and np.allclose(loader.ratios, [0.75, 0.25])
    first, second = list(loader), list(loader)
    assert len(first) == 8 and first != second                               # a new order every epoch
    assert list(MosaicLoader({"big": a, "small": b}, seed=1)) == first       # same seed, same order
    only_a = list(MosaicLoader([a, b], ratios=[1, 0], epoch_length=20))
    assert len(only_a) == 20 and {name for name, _ in only_a} == {"a"}       # restarts a loader that runs out
    counts = np.mean([name == "b" for name, _ in MosaicLoader([a, b], ratios=[1, 3], epoch_length=4000)])
    assert abs(counts - 0.75) < 0.05
    for bad in (dict(ratios=[1]), dict(ratios=[0, 0]), dict(ratios=[-1, 2])):
        with pytest.raises(ValueError):
            MosaicLoader([a, b], **bad)
    with pytest.raises(ValueError):
        MosaicLoader([])


def test_mosaic_of_unimodal_cohorts_and_a_paired_bridge_trains():
    rng = np.random.default_rng(0)
    centers = 2.0 * rng.normal(size=(4, 30))

    def cells(n, prefix, mods):
        y = rng.integers(0, 4, n)
        full = {"rna": centers[y] + 0.3 * rng.normal(size=(n, 30)), "adt": centers[y][:, :8] + 0.3 * rng.normal(size=(n, 8))}
        return {m: _adata(full[m], prefix) for m in mods}, y

    (rna_only, y1), (adt_only, y2), (bridge, y3) = cells(90, "r", ["rna"]), cells(90, "a", ["adt"]), cells(60, "b", ["rna", "adt"])
    loader = make_mosaic_loader({"rna_only": rna_only, "adt_only": adt_only, "bridge": bridge}, batch_size=30,
                                labels={"bridge": {"ct": y3}}, ratios=[1, 1, 2], seed=0)
    assert loader.cohort_codes == {"rna_only": 0, "adt_only": 1, "bridge": 2}
    seen = set()
    for x, y in loader:
        code = int(y["cohort"][0])
        assert (y["cohort"] == code).all()
        seen.add((code, tuple(sorted(x))))
        assert ("ct" in y) == (code == 2)
    assert seen == {(0, ("rna",)), (1, ("adt",)), (2, ("adt", "rna"))}
    cfg = presets.atlas_config({"rna": 30, "adt": 8}, n_cohorts=3, latent_dim=6)
    model = UniVIMultiModalVAE(cfg)
    history = UniVITrainer(model, loader, train_cfg=TrainingConfig(n_epochs=10, lr=3e-3, **QUIET)).fit()
    assert history["train_loss"][-1] < history["train_loss"][0]
    with pytest.raises(ValueError, match="already use the name"):
        make_mosaic_loader({"c": rna_only}, labels={"c": {"cohort": y1}})


# ------------------------------------------------------------------ cohort alignment

def test_conditioned_decoders_with_mmd_mix_cohorts_and_keep_cell_types():
    cohorts, types = _two_cohorts()
    cohort = np.repeat(["ref", "query"], 200)
    cell_type = np.concatenate([types["ref"], types["query"]])
    scores = {}
    for name, extra in (("plain", {}), ("aligned", dict(n_cohorts=2, cohort_mmd_weight=20.0))):
        torch.manual_seed(0)
        cfg = UniVIConfig(latent_dim=6, modalities=[ModalityConfig("rna", 40, [32], [32])], encoder_batchnorm=False, **extra)
        model = UniVIMultiModalVAE(cfg)
        UniVITrainer(model, make_mosaic_loader(cohorts, batch_size=50, seed=0),
                     train_cfg=TrainingConfig(n_epochs=40, lr=3e-3, **QUIET)).fit()
        z = _embed(model, cohorts)
        scores[name] = (cohort_mixing(z, cohort), label_purity(z, cell_type))
    assert scores["plain"][0] < 0.2                                          # the shift separates the cohorts
    assert scores["aligned"][0] > 0.5 and scores["aligned"][1] > 0.9         # mixed, cell types intact


def test_cohort_term_bookkeeping_and_batchnorm_warning():
    with pytest.warns(UserWarning, match="encoder_batchnorm"):
        UniVIMultiModalVAE(UniVIConfig(latent_dim=4, modalities=[ModalityConfig("rna", 40, [16], [16])], cohort_mmd_weight=1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = UniVIMultiModalVAE(UniVIConfig(latent_dim=4, modalities=[ModalityConfig("rna", 40, [16], [16])],
                                               cohort_mmd_weight=1.0, cohort_memory=30, encoder_batchnorm=False))
    x = {"rna": torch.randn(20, 40)}
    zeros, ones = {"cohort": torch.zeros(20, dtype=torch.long)}, {"cohort": torch.ones(20, dtype=torch.long)}
    assert "cohort_mmd" not in model(x, epoch=0, y=zeros)                    # nothing to compare with yet
    out = model(x, epoch=0, y=ones)
    assert "cohort_mmd" in out and torch.isfinite(out["cohort_mmd"])
    out["loss"].backward()
    mixed = {"cohort": torch.tensor([0] * 10 + [1] * 10)}
    assert "cohort_mmd" in model(x, epoch=0, y=mixed)                        # cohorts within one batch
    model(x, epoch=0, y=zeros); model(x, epoch=0, y=zeros)
    assert model._cohort_bank[0].shape[0] == 30 and not model._cohort_bank[0].requires_grad
    assert "cohort_mmd" not in model(x, epoch=0)                             # no cohort labels
    assert "cohort_mmd" not in model.eval()(x, epoch=0, y=ones)              # training only
    model.reset_cohort_memory()
    assert model._cohort_bank == {}
    for bad in (dict(cohort_mmd_weight=-1.0), dict(cohort_memory=0), dict(n_cohorts=-1)):
        with pytest.raises(ValueError):
            UniVIConfig(latent_dim=4, modalities=[ModalityConfig("rna", 40, [16], [16])], **bad).validate()


# ------------------------------------------------------------------ conditioned decoders

def test_cohort_conditioned_decoders():
    torch.manual_seed(0)
    plain = UniVIMultiModalVAE(UniVIConfig(latent_dim=4, modalities=[ModalityConfig("rna", 40, [16], [16])]))
    assert not isinstance(plain.decoders["rna"], CohortConditionedDecoder)   # off by default
    mods = [ModalityConfig("rna", 40, [16], [], decoder_type="linear"),
            ModalityConfig("image", 3 * 8 * 8, [], [], encoder_type="conv", decoder_type="conv",
                           conv=ConvConfig(in_channels=3, spatial_shape=(8, 8), channels=(4, 8), hidden_dim=16))]
    model = UniVIMultiModalVAE(UniVIConfig(latent_dim=4, modalities=mods, n_cohorts=3, cohort_embedding_dim=5))
    assert all(isinstance(d, CohortConditionedDecoder) for d in model.decoders.values())
    assert model.decoder_loadings("rna").shape == (40, 4)
    assert model.decoders["image"].cfg.in_channels == 3                     # attributes of the wrapped decoder
    x = {"rna": torch.randn(6, 40), "image": torch.rand(6, 192)}
    out = model(x, epoch=0, y={"cohort": torch.tensor([0, 1, 2, 0, 1, -1])})
    out["loss"].backward()
    assert torch.isfinite(out["loss"]) and model.decoders["rna"].embed.weight.grad is not None
    assert all(d.codes is None for d in model.decoders.values())             # restored after the pass
    dec, z = model.decoders["rna"].eval(), torch.randn(6, 4)
    with torch.no_grad():
        default = dec(z)
        model.set_decode_cohort(2)
        as_two = dec(z)
        dec.codes = torch.tensor([2, 2, 2, 2, 2, 99])                        # out-of-range code -> cohort 0
        per_row = dec(z)
        model.set_decode_cohort(None)
        assert torch.equal(dec(z), default)
    assert not torch.allclose(default, as_two)
    assert torch.equal(per_row[:5], as_two[:5]) and torch.equal(per_row[5], default[5])
    dec.codes = torch.zeros(3, dtype=torch.long)
    with pytest.raises(ValueError, match="cohort codes for"):
        dec(z)
    with pytest.raises(ValueError, match="outside"):
        model.set_decode_cohort(7); dec(z)


def test_reference_bundle_round_trips_cohort_settings(tmp_path):
    torch.manual_seed(0)
    cfg = presets.atlas_config({"rna": 40, "adt": 10}, n_cohorts=3, latent_dim=6)
    model = UniVIMultiModalVAE(cfg).eval()
    x = {"rna": torch.randn(8, 40), "adt": torch.randn(8, 10)}
    with torch.no_grad():
        before = model.decoders["rna"](model.encode_fused(x)[0])
    save_reference(tmp_path, model, preprocessors={})
    loaded = load_reference(tmp_path)[0].eval()
    assert loaded.cfg.n_cohorts == 3 and loaded.cfg.encoder_batchnorm is False
    with torch.no_grad():
        after = loaded.decoders["rna"](loaded.encode_fused(x)[0])
    assert torch.allclose(before, after, atol=1e-6)


# ------------------------------------------------------------------ atlas scores and label transfer

def test_mixing_purity_and_label_transfer():
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0], [6, 0], [0, 6]])
    y_ref, y_query = rng.integers(0, 3, 300), rng.integers(0, 3, 150)
    z_ref = centers[y_ref] + 0.4 * rng.normal(size=(300, 2))
    z_query = centers[y_query] + 0.4 * rng.normal(size=(150, 2))
    z, cohort = np.vstack([z_ref, z_query]), np.r_[["ref"] * 300, ["query"] * 150]
    assert 0.85 < cohort_mixing(z, cohort) < 1.15                            # same distribution
    assert cohort_mixing(np.vstack([z_ref, z_query + 50.0]), cohort) == 0.0  # disjoint
    assert label_purity(z, np.r_[y_ref, y_query]) > 0.98
    with pytest.raises(ValueError):
        cohort_mixing(z, np.zeros(len(z)))
    names = np.array(["B", "T", "NK"])
    result = transfer_labels(z_ref, names[y_ref], z_query)
    assert list(result.columns) == ["label", "confidence", "novelty"] and len(result) == 150
    assert (result["label"].to_numpy() == names[y_query]).mean() > 0.98
    assert result["confidence"].mean() > 0.95 and 0.7 < result["novelty"].median() < 1.5
    unseen = transfer_labels(z_ref, names[y_ref], np.array([[30.0, 30.0], [3.0, 3.0]]))
    assert unseen["novelty"].iloc[0] > 10 and unseen["novelty"].iloc[1] > 2  # far away, and between clusters
    hyp = transfer_labels(0.3 * z_ref, names[y_ref], 0.3 * z_query, geometry="hyperbolic")
    assert (hyp["label"].to_numpy() == names[y_query]).mean() > 0.95
