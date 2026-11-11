# The experiment toolkit

Version 0.2.0 adds parts that can be combined freely with everything else: priors, alignment terms, decoders,
image and graph encoders, tools for exploring hierarchy, support for atlases built from several cohorts,
and helpers for running comparisons. Every one is
off by default.

| If you want to | Use | Section |
| --- | --- | --- |
| let the latent form discrete groups, optionally one per class | `UniVIConfig(prior="mog")` | [Mixture prior](#mixture-prior) |
| learn the curvature and the class prototypes of a hyperbolic space | `learn_curvature`, `prior="mog"` | [A learned hyperbolic space](#a-learned-hyperbolic-space) |
| make each cell's modalities retrieve each other | `contrastive_weight` | [Contrastive alignment](#contrastive-alignment) |
| keep predicted modalities consistent with the latent | `cycle_weight` | [Cycle consistency](#cycle-consistency) |
| read which features each latent dimension drives | `decoder_type="linear"` | [Linear decoders](#linear-decoders) |
| make the model robust to missing features | `input_dropout` | [Input masking](#input-masking) |
| encode image tiles with a transformer, with augmentation | `encoder_type="vit"`, `augment` | [Image encoders](#image-encoders) |
| encode each spot together with its neighbours | `encoder_type="graph"` | [Graph encoder](#graph-encoder) |
| turn a latent space into a tree and a pseudotime | `univii.hierarchy` | [Hierarchy tools](#hierarchy-tools) |
| segment cells into runs of states along an ordering | `univii.hmm` | [Hidden Markov model](#hidden-markov-model) |
| combine cohorts that measured different modalities | `univii.atlas`, `n_cohorts`, `cohort_mmd_weight` | [Atlases, mosaics and bridges](#atlases-mosaics-and-bridges) |
| remove staining differences between slides | `univii.histology` | [Stain normalisation](#stain-normalisation) |
| start from a sensible configuration | `univii.presets` | [Presets](#presets) |
| train several configurations and tabulate metrics | `univii.experiments` | [Comparing configurations](#comparing-configurations) |

## Mixture prior

```python
cfg = UniVIConfig(latent_dim=20, modalities=[...], prior="mog", prior_components=12)
model = UniVIMultiModalVAE(cfg)
...
clusters = model.prior_responsibilities(z).argmax(1)        # z from model.encode_fused(...)
```

The prior becomes a learned mixture, so cells can settle around several centres. It covers the whole
latent: Gaussian components in a Euclidean block, and wrapped normals centred at learned points in a
hyperbolic block. `model.prior_centres()` returns the learned centres. Two optional extensions:

- **Components as classes.** With `prior_label_head="celltype"` and `prior_components` equal to the number
  of classes, a cell whose label is known is scored under the component with that index (labels arrive as
  `y={"celltype": ...}`; `-1` means unknown and falls back to the full mixture).
- **Components arranged like a tree.** With `prior_tree_weight > 0` and a hierarchy registered through
  `model.set_hierarchy(dist)`, distances between component means are pulled toward the tree distances.
  The term is reported as `prior_tree`.

The KL term to a mixture is a one-sample estimate, and `kl_hyperbolic_weight` does not apply to it.

## A learned hyperbolic space

Three parts of a hyperbolic latent can be learned, separately or together:

```python
cfg = UniVIConfig(
    latent_dim=8, modalities=[...], latent_geometry="hyperbolic",
    learn_curvature=True,                                    # 1. how strongly the space curves
    prior="mog", prior_components=n_types, prior_label_head="celltype",   # 2. one prototype per class
    prior_tree_weight=2.0,                                   # 3. prototypes arranged like the label tree
)
model = UniVIMultiModalVAE(cfg)
model.set_hierarchy(tree_dist, root_index=root)              # needed for 3
...
float(model.curvature), model.prior_centres()
```

1. **Curvature.** `learn_curvature=True` makes the curvature a parameter. Higher curvature gives more room
   near the boundary for bushy trees. A fixed curvature is easier to compare across runs.
2. **Prototypes.** With the mixture prior, every component is a point in hyperbolic space with its own
   spread, learned with the rest of the model. Tied to labels, each class gets one.
3. **Layout.** `prior_tree_weight` pulls the geodesic distances between prototypes toward the tree
   distances, times `hierarchy_scale`. Leave `hierarchy_scale` fixed (the default) unless you have a
   reason: a learned scale tends to shrink everything toward the origin.

This is separate from `hierarchy_weight`, which pulls the *cells* toward tree distances. They can be used
together. With `latent_geometry="product"`, each prototype has a Euclidean and a hyperbolic part.

## Contrastive alignment

```python
cfg = UniVIConfig(..., contrastive_weight=1.0, contrastive_temperature=0.5)
```

A symmetric InfoNCE term over every pair of modalities present: within a batch, a cell's posterior mean in
one modality should be nearer to its own mean in the other modality than to any other cell's. Similarity is
the negative squared latent distance, so it follows the latent geometry. It adds to the existing alignment
(`gamma`) and is reported as `contrastive`. It needs at least two modalities and batches larger than one;
larger batches give more negatives.

## Cycle consistency

```python
cfg = UniVIConfig(..., cycle_weight=1.0)
```

For every modality in a batch, the model predicts each *other* modality from the latent mean, encodes that
prediction with the other modality's own encoder, and penalises the latent distance back to where it
started. This also runs for modalities the batch does not contain, which is the point for single-modality
cohorts: an RNA-only cell is asked to land in the same place after a round trip through predicted ATAC.
Reported as `cycle`.

It applies to target modalities with a Gaussian-type likelihood, where the decoder's mean is in the same
space as the encoder's input. Count modalities are skipped as targets, because their encoders usually see
transformed data while their decoders predict counts. Predicted inputs do not update the encoders'
batch-norm statistics.

## Linear decoders

```python
ModalityConfig("rna", n_genes, [512, 256], [], likelihood="nb", decoder_type="linear")
...
loadings = model.decoder_loadings("rna")                    # (genes, latent_dim)
```

One linear map from the latent to the likelihood's parameters, for any likelihood. Entry `[g, k]` says how
much latent dimension `k` moves gene `g` on the scale of the likelihood's main parameter. This trades some
reconstruction accuracy for dimensions that can be read directly. Loadings are most interpretable with a
Euclidean latent.

## Input masking

```python
ModalityConfig("atac", n_peaks, [256], [256], input_dropout=0.2)
```

At training time a random 20% of that modality's input features are set to zero before encoding, while the
reconstruction target stays complete. The model learns to fill features in. Nothing is masked at
evaluation.

## Image encoders

```python
from univii import ConvConfig, ViTConfig

vit = ViTConfig(in_channels=3, image_size=(32, 32), patch_size=8, d_model=128, num_layers=4,
                augment=("flip", "rot90"))
image = ModalityConfig("image", 3 * 32 * 32, [], [], encoder_type="vit", vit=vit,
                       decoder_type="conv", conv=ConvConfig(in_channels=3, spatial_shape=(32, 32)))
```

`encoder_type="vit"` cuts the tile into patches and runs a transformer over them. Pair it with a conv
decoder, as above, or an MLP decoder. `augment` is available on both `ViTConfig` and `ConvConfig`: during
training the encoder sees randomly flipped and quarter-turned tiles, while the reconstruction target is the
original. One transform is drawn per batch. Transformers need more data than convolutions to do well on
small tiles; compare both on your data.

## Graph encoder

```python
from univii import GraphConfig
from univii.spatial import make_spatial_loader, encode_spatial

rna = ModalityConfig("rna", n_genes, [], [256], encoder_type="graph", graph=GraphConfig(k=6, hidden=(256, 128)))
loader = make_spatial_loader({"rna": adata}, batch_size=256)          # spatially local batches
...
z = encode_spatial(model, {"rna": adata}, batch_size=256, n_passes=3)  # one latent mean per spot
```

Each spot is encoded from its own features and the mean (or max) of its `k` nearest neighbours, for one
round per entry of `hidden`.

**The neighbours are taken from the batch.** That makes training simple, and it has a consequence: a spot's
embedding depends on which other spots share its batch. Train with `make_spatial_loader` and embed with
`encode_spatial`, which both draw spatially local batches, so that in-batch neighbours are real neighbours.
`encode_spatial` averages over the batches a spot falls into; `n_passes` repeats that with different
batches. In a batch without coordinates the encoder treats each spot as its own neighbour, which is not what
it was trained on.

If you want neighbourhood information without this dependence, `univii.spatial.niche_modality` computes the
neighbourhood average once, over the whole tissue.

## Hierarchy tools

```python
from univii.hierarchy import cluster_tree, graph_pseudotime, tree_agreement

z = model.encode_fused(x_dict, anchored=True)[0]            # after model.set_latent_anchor(stem cells)
geo = dict(geometry=model.latent_geometry, hyperbolic_dim=model.latent.hyperbolic_dim, curvature=model.curvature)

tree = cluster_tree(z, labels, root="HSC", **geo)           # tree["parents"], tree["edge_length"]
pseudotime = graph_pseudotime(z, labels == "HSC", k=15, **geo)
score = tree_agreement(tree["parents"], reference_parents)  # edge_jaccard, path_spearman, parent_accuracy
```

- `cluster_tree` builds a minimum spanning tree over cluster centroids, using geodesic distances. It always
  returns a tree, and it links each cluster to its nearest neighbour: two sibling clusters that are closer to
  each other than to their parent will be joined to each other. Check it against a reference.
- `graph_pseudotime` is the shortest-path distance from the root cells on a k-nearest-neighbour graph. If
  the graph is in several pieces, each is joined by its shortest connecting edge so every cell gets a value.
- `tree_agreement` compares two trees over the same nodes.

## Hidden Markov model

```python
from univii.hmm import segment_by_order

states, probabilities, hmm = segment_by_order(z, order=pseudotime, n_states=4)
```

`segment_by_order` sorts cells by `order`, fits a Gaussian hidden Markov model to the sorted values, and
returns the most probable state and the state probabilities for each cell, in the original cell order. Use
it for latent coordinates along pseudotime, a score across a time course, or a signal along genomic
position. `stickiness` (default 0.95) sets how strongly runs are favoured. State numbers are arbitrary
labels. `univii.hmm.GaussianHMM` exposes the likelihood, posterior and most probable path directly.

Inference is exact. The fit is by gradient ascent from a deterministic start and can settle in a local
optimum; compare a few values of `n_states` and `seed`.

## Atlases, mosaics and bridges

A mosaic is a set of cohorts that measured different modalities: say RNA only, ATAC only, and a smaller
paired cohort that bridges them. The model accepts a batch with any subset of its modalities, so the paired
cohort teaches it how the modalities relate and the single-modality cohorts fill in the space.

```python
from univii import presets, UniVIMultiModalVAE, UniVITrainer, TrainingConfig
from univii.atlas import make_mosaic_loader, cohort_mixing, label_purity, transfer_labels

cohorts = {"rna_only": {"rna": rna_a}, "atac_only": {"atac": atac_b}, "bridge": {"rna": rna_c, "atac": atac_c}}
loader = make_mosaic_loader(cohorts, batch_size=256, ratios=[1, 1, 2],
                            labels={"bridge": {"celltype": codes}})        # labels where you have them

cfg = presets.atlas_config({"rna": 2000, "atac": 100}, n_cohorts=3)
model = UniVIMultiModalVAE(cfg)
UniVITrainer(model, loader, train_cfg=TrainingConfig(n_epochs=200)).fit()
```

Three things work together here.

**The loader.** `make_mosaic_loader` draws each batch from one cohort, chosen at random with probability
`ratios` (by default in proportion to cohort size), and gives every cell an integer cohort code in
`y["cohort"]`. Raise a ratio to oversample a small bridge; lower one to let a noisy cohort trickle in.
Modalities shared between cohorts must use the same features, prepared the same way.

**Cohort-conditioned decoders** (`n_cohorts`). Every decoder also receives a learned embedding of the cell's
cohort. Technical differences between cohorts can then be explained by the embedding, and the latent no
longer has to store them in order to reconstruct the data. Outside training, decoders assume cohort 0;
`model.set_cohort(code)` picks another, which expresses predictions "as if measured in" that cohort.
For a modality only some cohorts measured, pick one of those.

**Cohort alignment** (`cohort_mmd_weight`). The fused latent means of each cohort are pulled toward those of
the other cohorts with a kernel two-sample (MMD) penalty, reported as `cohort_mmd`. Because a batch holds
one cohort, the model keeps the last `cohort_memory` latent means of every cohort and compares against
those. It is applied during training only.

On a simulated pair of cohorts with a technical shift, the two were needed together: alignment without
conditioned decoders left the cohorts separate and raised the reconstruction loss, conditioned decoders
alone left them separate, and both together mixed them while keeping cell types apart. Treat that as a
starting point to verify on your data, not a guarantee.

```{warning}
Turn encoder batch normalisation off (`encoder_batchnorm=False`; `presets.atlas_config` does) whenever
batches hold a single cohort. Batch normalisation centres each batch, which hides cohort differences during
training but not at evaluation: cohorts look aligned while training and separate again afterwards. The same
applies to `ConvConfig(norm="batch")`; use `"group"`. The model warns if `cohort_mmd_weight > 0` is combined
with encoder batch normalisation.
```

Alignment assumes the cohorts contain broadly the same cell populations. If one cohort has a population the
others lack, forcing the distributions together will misplace it; lower the weight and check with the
scores below.

### Adding a cohort to a trained atlas

```python
from univii.atlas import adapt_to_cohort, encode_cohort

z_atlas = np.vstack([encode_cohort(model, data, cohort=code) for data, code in atlas_cohorts])
result = adapt_to_cohort(model, new_loader, reference_z=z_atlas, n_epochs=50)
z_new = encode_cohort(model, new_data, cohort=result["code"])
```

`adapt_to_cohort` freezes everything the model has learned and trains two small things for the new cohort
only: a per-feature shift added to its data, and its row in the decoders' cohort embedding. Existing
cohorts' embeddings are exactly unchanged, so nothing built on the atlas needs redoing.

The shift is a learned correction that moves the new cohort into the feature space the atlas was trained
on; read it from `model.input_shifts[modality].weight[code]`. For Gaussian-type modalities the shifted data
are both the encoder's input and the reconstruction target, so the shift settles where the frozen model
explains the data best. With `reference_z`, an MMD term also pulls the new cells toward the atlas.

Always pass the cohort's code when embedding its cells (`encode_cohort(..., cohort=code)`, or
`model.set_cohort(code)` first); otherwise the shift is not applied. A shift corrects differences that are
roughly constant per feature. It cannot repair a cohort whose features relate to biology differently; for
that, fine-tune the encoders, which does move the atlas.

### Checking the result

```python
z = ...                                                     # fused latent means of all cells
mix = cohort_mixing(z, cohort_names)                        # 0 = separate, about 1 = fully mixed
pure = label_purity(z, cell_types)                          # 1 = neighbours share the cell's label
```

Read the two together. A latent that has collapsed mixes cohorts perfectly and is useless, so high mixing
only counts if purity stays high.

### Transferring labels from a reference

```python
result = transfer_labels(z_reference, reference_labels, z_query, k=15)
unfamiliar = result["novelty"] > 2
```

Each query cell gets the most common label among its nearest reference cells, the share of neighbours that
agree (`confidence`), and a `novelty` score: its distance to those neighbours relative to how far reference
cells sit from each other. A cell can be confidently labelled and still novel, which is the case to look
at, since it means the reference has nothing like it. The threshold of 2 above is an illustration; set it
from held-out reference cells.

All three functions take `geometry`, `hyperbolic_dim` and `curvature` and use geodesic distances.

## Stain normalisation

```python
from univii.histology import MacenkoNormalizer

normalizer = MacenkoNormalizer().fit(tiles_reference_slide)       # tile AnnData or an (n, 3, H, W) array
tiles_b = normalizer.transform(tiles_slide_b)                     # one slide at a time
```

Slides stained or scanned in different batches differ in colour for reasons unrelated to the tissue.
Macenko's method estimates the two stain colours of a slide, splits every pixel into stain amounts, and
re-renders the tiles with the reference slide's colours and intensity. Intensities must be in `[0, 1]`.

Fit and transform one slide (one staining batch) at a time, since the colours are estimated from the pixels
passed in. It assumes two stains, as in H&E, and needs enough tissue pixels; nearly empty tiles are best
left out of the estimate. Normalise before training, so that the same transform is applied to every slide.

## Presets

```python
from univii import presets

cfg = presets.paired_config({"rna": 2000, "adt": 130}, likelihoods={"rna": "nb"})
cfg = presets.clustered_config({"rna": 2000, "atac": 100}, n_components=20)
cfg = presets.contrastive_config({"rna": 2000, "atac": 100})
cfg = presets.hierarchy_config({"rna": 2000}, n_labels=14, label_head="celltype", geometry="product")
cfg = presets.atlas_config({"rna": 2000, "atac": 100}, n_cohorts=3)
cfg = presets.spatial_image_config(n_genes=2000, tile_shape=(3, 32, 32), spatial_sigma=sigma, encoder="vit")
```

Each returns a `UniVIConfig`; any extra keyword overrides the preset. Layer sizes are chosen from the
feature counts and are starting points, not tuned values.

## Comparing configurations

```python
from univii.experiments import compare_models, summarize

variants = {
    "euclidean":  presets.paired_config(dims),
    "product":    presets.paired_config(dims, latent_geometry="product"),
    "product_v2": (presets.paired_config(dims, latent_geometry="product"), {"loss_mode": "v2"}),
}
runs = compare_models(variants, train_loader, evaluate=my_metrics, seeds=(0, 1, 2),
                      train_cfg=TrainingConfig(n_epochs=100))
print(summarize(runs))
```

`evaluate` is your function from a trained model to a dictionary of numbers; compute it on held-out data.
`compare_models` returns one row per variant and seed, with the parameter count, training time and final
training loss alongside your metrics. `setup` runs on each model before training, for steps such as
`model.set_hierarchy(...)`. Several seeds per variant are what make a difference between variants
believable.
