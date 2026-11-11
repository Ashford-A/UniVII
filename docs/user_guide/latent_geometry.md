# Latent geometry: Euclidean, hyperbolic and product spaces

UniVI's latent space is Euclidean by default. Two further options put some or all latent dimensions in
hyperbolic space, which represents tree-like structure (differentiation hierarchies, taxonomies) with less
distortion in few dimensions.

```python
from univii import UniVIConfig, UniVIMultiModalVAE

cfg = UniVIConfig(
    latent_dim=16,
    modalities=[...],
    latent_geometry="product",   # "euclidean" (default) | "hyperbolic" | "product"
    hyperbolic_dim=8,            # product only: the last 8 dimensions are hyperbolic
    curvature=1.0,               # magnitude c; the hyperbolic block has curvature -c
    learn_curvature=False,
)
model = UniVIMultiModalVAE(cfg, loss_mode="v1")
```

With `latent_geometry="euclidean"` the model is numerically identical to earlier releases, and checkpoints
saved by earlier releases load unchanged.

## What the coordinates mean

Latent vectors keep their shape, `(cells, latent_dim)`, in every geometry. The Euclidean columns are ordinary
coordinates. The hyperbolic columns are **tangent coordinates at the origin**: the point they denote is the
exponential map of that vector. Three practical consequences:

- **Radius is exact.** The norm of the hyperbolic block is the geodesic distance from the origin.
- **Euclidean distance between rows is not the geodesic distance.** Use `model.latent_distance`,
  `model.pairwise_latent_distance`, or the functions of the same name in `univii.geometry`. Neighbour graphs,
  UMAPs and the metrics in `univii.evaluation` operate on the stored coordinates with Euclidean distance; on
  a hyperbolic block that is an approximation that is good near the origin and degrades far from it. Pass a
  precomputed distance matrix where the difference matters.
- **Decoders and heads read the stored coordinates**, so nothing downstream changes shape.

## What changes in the objective

| Term | Euclidean block | Hyperbolic block |
| --- | --- | --- |
| Posterior | diagonal Gaussian | wrapped normal (Gaussian in the tangent space at the mean) |
| KL to prior | closed form | one-sample Monte Carlo estimate |
| Cross-modal KL (`loss_mode="v1"`) | closed form | one-sample Monte Carlo estimate |
| Alignment (`loss_mode="v2"`) | squared distance between means | squared geodesic distance between means |
| Mixture-of-experts fusion | precision-weighted means | the same, applied to tangent coordinates at the origin |

The forward pass reports `kl_e` and `kl_h` (the two parts of the prior KL) and `curvature` alongside the
existing outputs. `kl_hyperbolic_weight` rescales the hyperbolic part before `beta` is applied.

Things to expect:

- The hyperbolic KL is noisier than the Euclidean one and can be slightly negative for individual cells.
- Fusion on the hyperbolic block is tangent-space fusion at the origin. It is exact for one expert and an
  approximation for several.
- Hyperbolic computations run in float64 internally. Apple MPS has no float64, so they run in float32 there;
  keep `hyperbolic_max_radius` at or below about 4 on MPS.

## When hyperbolic geometry helps

In simulations with a known tree, hyperbolic latents matched Euclidean ones on a 14-label tree and only pulled
ahead on a tree with 121 labels, and only when cells sat well away from the origin. Left free, the prior pulls
everything toward the origin, where hyperbolic space is nearly flat. Two settings control this:

- `hierarchy_scale` (latent distance per unit of tree distance, used by the hierarchy loss below). A value of
  1 to 2 keeps a tree of depth 4 at radius 4 to 8.
- `curvature`. Larger values make the same coordinates more strongly curved.

Compare against a Euclidean model of the same size on your own data before committing to either.

## Supervising with a label tree

If cells carry labels that sit on a known tree, a weak loss can tie latent distances to tree distances.

```python
from univii.geometry import tree_distance_matrix

parents = {"HSC": None, "MPP": "HSC", "GMP": "MPP", "CLP": "MPP", "Mono": "GMP", "B": "CLP"}
head_labels = ["HSC", "MPP", "GMP", "CLP", "Mono", "B"]           # order of the "celltype" head's classes
dist, names, depth = tree_distance_matrix(parents, order=head_labels)

cfg = UniVIConfig(..., latent_geometry="hyperbolic",
                  class_heads=[ClassHeadConfig(name="celltype", n_classes=len(head_labels))],
                  hierarchy_head="celltype", hierarchy_weight=5.0,
                  hierarchy_root_weight=1.0, hierarchy_scale=1.0)
model = UniVIMultiModalVAE(cfg)
model.set_hierarchy(dist, root_index=names.index("HSC"))
```

Each labelled cell is paired with a random labelled cell in the batch and penalised by
`(latent distance - hierarchy_scale * tree distance) ** 2`. Unlabelled cells (`-1`) are skipped. The term is
reported as `hierarchy` in the forward output and works in every geometry, which makes Euclidean and
hyperbolic models directly comparable. Score the result with `univii.geometry.hierarchy_metrics`.

## Anchoring at a root population

Hyperbolic space has no preferred centre. Anchoring applies the distance-preserving map that moves a chosen
population, such as stem cells, to the origin, so that radius reads as distance from that population.

```python
mu = model.encode_fused(x_dict)[0]
model.set_latent_anchor(mu[is_stem_cell])                 # stored in the checkpoint
anchored = model.encode_fused(x_dict, anchored=True)[0]   # or model.anchor_latent(mu)
radius = model.latent_radius(anchored)
disk = model.to_poincare(anchored)                        # hyperbolic block in the unit ball, for plotting
```

Anchored coordinates are an analysis view. Decoders and heads expect the un-anchored ones.

## Placing new cells on a frozen reference

`univii.reference` scores how well a frozen model explains cells it was not trained on.

```python
from univii.reference import refine_latent, reconstruction_nll, deviation_scores

z = refine_latent(model, x_query, steps=100)              # gradient descent through frozen decoders
nll, per_modality = reconstruction_nll(model, x_query, z)
z_null = refine_latent(model, x_heldout_healthy, steps=100)
nll_null, _ = reconstruction_nll(model, x_heldout_healthy, z_null)
score = deviation_scores(nll, nll_null)                   # robust z-score against the healthy cells
```

Treat the query and the null population the same way (both refined, or both encoder-only); otherwise the
comparison is biased.
