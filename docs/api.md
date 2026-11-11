# API reference

Most names are importable from the top-level package (`from univii import ...`); the tables list their home modules.

## Configuration

```{eval-rst}
.. currentmodule:: univii

.. autosummary::
   :toctree: generated
   :nosignatures:

   UniVIConfig
   ModalityConfig
   TrainingConfig
   ClassHeadConfig
   config.TokenizerConfig
   config.TransformerConfig
   refinement.RefinementConfig
```

## Model and training

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   UniVIMultiModalVAE
   trainer.UniVITrainer
   refinement.UniVIRefiner
```

## Data, preprocessing and datasets

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   preprocessing.RNAPreprocessor
   preprocessing.ADTPreprocessor
   preprocessing.ATACPreprocessor
   preprocessing.split_by_label
   workflows.make_loader
   data.MultiModalDataset
   data.align_paired_obs_names
   data.collate_multimodal_xy_recon
   datasets.list_datasets
   datasets.load
   datasets.fetch
   datasets.dataset_info
   datasets.register_dataset
   datasets.export_dataset
   datasets.get_data_dir
   datasets.pbmc_multiome_10k
   datasets.hao_citeseq_pbmc
   datasets.scnmt_gastrulation
   datasets.load_scnmt_gastrulation_genebody_triplet
   datasets.build_univi_inputs_from_scnmt_triplet
```

## Latent geometry and reference anchoring

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   geometry.LatentGeometry
   geometry.split_latent_dims
   geometry.latent_distance
   geometry.pairwise_latent_distance
   geometry.latent_radius
   geometry.to_poincare
   geometry.tree_distance_matrix
   geometry.hierarchy_metrics
   reference.refine_latent
   reference.reconstruction_nll
   reference.deviation_scores
   reference.anchor_on_label
```

## Spatial data, conv encoders and plug-ins

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   config.ConvConfig
   models.conv.ConvGaussianEncoder
   models.conv.ConvDecoder
   models.conv.BackboneGaussianEncoder
   models.registry.register_encoder
   models.registry.register_decoder
   models.registry.available_encoders
   models.registry.available_decoders
   spatial.make_spatial_loader
   spatial.SpatialBlockSampler
   spatial.niche_modality
   spatial.neighborhood_aggregate
   spatial.knn_graph
   spatial.median_neighbor_distance
   spatial.tiles_from_visium
   spatial.extract_tiles
   spatial.unflatten_tiles
   spatial.coords_modality
```

## Embedding, prediction and generation

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   evaluation.encode_adata
   evaluation.encode_fused_adata_pair
   workflows.stack_embeddings
   evaluation.cross_modal_predict
   evaluation.denoise_adata
   evaluation.generate_from_latent
   evaluation.fit_label_latent_gaussians
   evaluation.sample_latent_by_label
   evaluation.encode_moe_gates_from_tensors
   refinement.predict_heads_adata
   perturbation.predict_feature_perturbation
```

## Metrics

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   evaluation.evaluate_alignment
   evaluation.compute_foscttm
   evaluation.compute_match_recall_at_k
   evaluation.compute_modality_mixing
   evaluation.compute_modality_entropy
   evaluation.label_transfer_knn
   evaluation.reconstruction_metrics
   evaluation.evaluate_cross_reconstruction
   evaluation.pearson_corr_per_feature
   evaluation.mse_per_feature
   evaluation.add_lsc17_scores
```

## Saving and loading

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   workflows.save_reference
   workflows.load_reference
   utils.io.save_checkpoint
   utils.io.load_checkpoint
   utils.io.restore_checkpoint
   utils.seed.set_seed
```

## Plotting

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   plotting.set_style
   plotting.umap
   plotting.umap_by_modality
   plotting.plot_confusion_matrix
   plotting.write_gates_to_obs
   plotting.plot_moe_gate_summary
   plotting.compare_raw_vs_pred_umap_features
   plotting.compare_raw_vs_denoised_umap_features
   plotting.plot_reconstruction_error_summary
   plotting.plot_featurewise_reconstruction_scatter
```
