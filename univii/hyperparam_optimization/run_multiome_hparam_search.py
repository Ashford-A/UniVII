# univii/hyperparam_optimization/run_multiome_hparam_search.py

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from anndata import AnnData

from univii.config import TrainingConfig
from .common import (
    iter_hparam_configs,
    train_single_config,
    results_to_dataframe,
)


def run_multiome_hparam_search(
    rna_train: AnnData,
    atac_train: AnnData,
    rna_val: AnnData,
    atac_val: AnnData,
    celltype_key: Optional[str] = "cell_type",
    device: str = "cuda",
    layer: str = "counts",  # raw counts for NB / Poisson / ZINB
    X_key: str = "X",
    max_configs: int = 100,
    seed: int = 0,
    base_train_cfg: Optional[TrainingConfig] = None,
):
    """
    Hyperparameter random search for RNA+ATAC multiome.

    Assumes:
    - rna_train / atac_train are paired and same obs_names
    - rna_val / atac_val are paired and same obs_names
    - raw counts stored in .layers[layer] (default 'counts')
    """

    assert rna_train.n_obs == atac_train.n_obs
    assert rna_val.n_obs == atac_val.n_obs

    adata_train = {"rna": rna_train, "atac": atac_train}
    adata_val = {"rna": rna_val, "atac": atac_val}
    modalities = ["rna", "atac"]

    if base_train_cfg is None:
        base_train_cfg = TrainingConfig(
            n_epochs=80,
            batch_size=256,
            lr=1e-3,
            weight_decay=1e-5,
            device=device,
            log_every=5,
            grad_clip=5.0,
            num_workers=0,
            seed=42,
            early_stopping=True,
            patience=15,
            min_delta=0.0,
        )

    # ----- architecture options -----
    rna_arch_options = [
        {"name": "rna_med2", "enc": [512, 256], "dec": [256, 512]},
        {"name": "rna_wide2", "enc": [1024, 512], "dec": [512, 1024]},
        {"name": "rna_wide3", "enc": [1024, 512, 256], "dec": [256, 512, 1024]},
    ]
    atac_arch_options = [
        {"name": "atac_med2", "enc": [512, 256], "dec": [256, 512]},
        {"name": "atac_wide2", "enc": [1024, 512], "dec": [512, 1024]},
        {"name": "atac_wide3", "enc": [2048, 1024, 512], "dec": [512, 1024, 2048]},
    ]

    mod_arch_space = {
        "rna": rna_arch_options,
        "atac": atac_arch_options,
    }

    # ----- likelihood options -----
    likelihood_per_mod = {
        # for raw counts
        "rna": ["nb", "zinb"],
        "atac": ["nb", "poisson", "zinb"],
    }

    # ----- hyperparameter search space -----
    search_space = {
        "latent_dim":       [10, 20, 32, 40, 50, 64, 82, 120, 160, 200],
        "beta":             [0.0, 1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 160.0, 200.0, 300.0],
        "gamma":            [0.0, 1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 160.0, 200.0, 300.0],
        "lr":               [1e-3, 5e-4],
        "weight_decay":     [1e-4, 1e-5],
        "encoder_dropout":  [0.0, 0.1],
        "decoder_batchnorm":[False, True],
        "rna_arch":         rna_arch_options,
        "atac_arch":        atac_arch_options,
        "rna_likelihood":   likelihood_per_mod["rna"],
        "atac_likelihood":  likelihood_per_mod["atac"],
    }

    input_dims = {
        "rna": rna_train.n_vars,
        "atac": atac_train.n_vars,
    }

    results: List[Any] = []
    best_score = float("inf")
    best_model = None
    best_cfg = None

    for cfg_id, hp in enumerate(
        iter_hparam_configs(search_space, max_configs=max_configs, seed=seed),
        start=1,
    ):
        res = train_single_config(
            config_id=cfg_id,
            hparams=hp,
            mod_arch_space=mod_arch_space,
            modalities=modalities,
            input_dims=input_dims,
            likelihood_per_mod={
                "rna": likelihood_per_mod["rna"],
                "atac": likelihood_per_mod["atac"],
            },
            adata_train=adata_train,
            adata_val=adata_val,
            base_train_cfg=base_train_cfg,
            layer=layer,
            X_key=X_key,
            celltype_key=celltype_key,
            device=device,
            multimodal_eval=True,
        )
        results.append(res)

        score = res.metrics["composite_score"]
        if score < best_score:
            best_score = score
            best_model = res  # we'll re-train or save separate; for now keep config
            best_cfg = hp
            print(f"--> New best config (id={cfg_id}) with score={score:.3f}")

    df = results_to_dataframe(results)
    return df, best_model, best_cfg
