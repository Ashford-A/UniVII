# Spatial data, image tiles and convolutional encoders

UniVI can use spatial information in four ways. They are independent and combine freely.

| Approach | What it adds | What you need |
| --- | --- | --- |
| Niche modality | each spot's surroundings as a second expression-like modality | coordinates |
| Image modality | a tile of the tissue image around each spot, with conv encoder and decoder | a tissue image |
| Embedding modality | features from a pretrained image model, as an ordinary matrix | precomputed features |
| Spatial smoothness prior | nearby spots are pulled together in the latent space | coordinates |

## Niche modality

```python
from univii.spatial import niche_modality

niche = niche_modality(adata, k=6, group_key="slide")     # mean of the 6 nearest spots, per slide
adata_by_mod = {"rna": adata, "niche": niche}             # then configure "niche" like any modality
```

`group_key` keeps neighbour searches within a slide or sample. `kernel="gaussian"` weights neighbours by
distance; `include_self=True` adds the spot itself.

## Image tiles and convolutional encoders

A conv modality is stored as a flat `(cells, C * H * W)` matrix in channel-first order, so loaders, losses
and exports treat it like any other modality. The conv modules reshape internally.

```python
from univii import ConvConfig, ModalityConfig, UniVIConfig, UniVIMultiModalVAE
from univii.spatial import tiles_from_visium, unflatten_tiles

tiles = tiles_from_visium(adata, img_key="hires", out_size=32)      # AnnData, X in [0, 1]
C, H, W = tiles.uns["tile_shape"]

conv = ConvConfig(in_channels=C, spatial_shape=(H, W), channels=(32, 64, 128),
                  output_activation="sigmoid")                      # 3 stages: 32 -> 16 -> 8 -> 4
cfg = UniVIConfig(latent_dim=32, modalities=[
    ModalityConfig("rna", adata.n_vars, [512, 256], [256, 512], likelihood="nb"),
    ModalityConfig("image", tiles.n_vars, [], [], likelihood="gaussian",
                   encoder_type="conv", decoder_type="conv", conv=conv),
])
model = UniVIMultiModalVAE(cfg, loss_mode="v1")

# after training: predict the tile from expression alone
z = model.encode_fused({"rna": x_rna})[0]
pred = unflatten_tiles(model.decode_modalities(z)["image"].detach(), (C, H, W))
```

Rules and options:

- `input_dim` must equal `in_channels * prod(spatial_shape)`, and each spatial size must be divisible by
  `2 ** len(channels)`. Both are checked when the config is validated.
- `encoder_type` and `decoder_type` are independent. A conv encoder with the default MLP decoder is valid.
- The conv decoder supports the `gaussian` (also `normal`, `mse`), `gaussian_diag` and `bernoulli`
  likelihoods.
- `spatial_shape=(L,)` gives 1-D convolutions, for ordered tracks rather than images.
- For tiles from any image array, use `univii.spatial.extract_tiles(image, centers_xy, tile_size)`.

**With `loss_mode="v2"`, pass `recon_normalize_by_dim=False`** (or raise the modality's `recon_weight`).
The v2 objective divides each modality's reconstruction loss by its number of features by default. For an
image with hundreds or thousands of pixels that leaves the reconstruction term far smaller than the KL
term, and the decoder learns only the mean image. `loss_mode="v1"` does not have this problem.

Memory: tiles are held in memory as float32, `n_spots * C * H * W * 4` bytes. 5,000 spots at 3 x 32 x 32
take about 61 MB; at 3 x 224 x 224 they take about 3 GB, at which point precomputed embeddings are the better
route.

## Pretrained image models

Two options, in order of simplicity.

**Precompute features** with your image model and store them as a matrix. That is an ordinary modality with
the default MLP encoder, and needs nothing from this page.

**Wrap the model as an encoder**, frozen or trainable, through the plug-in registry:

```python
from univii.models import BackboneGaussianEncoder, register_encoder

@register_encoder("my_image_model")
def build(uni_cfg, mod_cfg):
    backbone = load_my_pretrained_model()                 # any nn.Module returning features
    return BackboneGaussianEncoder(backbone, feature_dim=1536, latent_dim=uni_cfg.latent_dim,
                                   input_shape=(3, 224, 224), freeze=True)

ModalityConfig("image", 3 * 224 * 224, [], [256], encoder_type="my_image_model")
```

A frozen backbone stays in eval mode and receives no gradients; only the head trains.

## Plugging in your own encoder or decoder

```python
from univii.models import register_encoder, register_decoder

@register_encoder("my_encoder")
def build_encoder(uni_cfg, mod_cfg):
    # forward(x) must return (mu, logvar), each (cells, uni_cfg.latent_dim); x is (cells, input_dim)
    return MyEncoder(mod_cfg.input_dim, uni_cfg.latent_dim, **(mod_cfg.encoder_kwargs or {}))

@register_decoder("my_decoder")
def build_decoder(uni_cfg, mod_cfg, likelihood):
    # forward(z) must return what the built-in decoder for `likelihood` returns
    return MyDecoder(uni_cfg.latent_dim, mod_cfg.input_dim)
```

Register in every process that builds or loads a model using the custom name. Built-in names cannot be
replaced.

## Spatial smoothness prior

```python
from univii.spatial import make_spatial_loader, median_neighbor_distance

sigma = median_neighbor_distance(adata.obsm["spatial"])            # typical spot spacing
cfg = UniVIConfig(..., spatial_weight=1.0, spatial_sigma=sigma, spatial_k=6)
loader = make_spatial_loader({"rna": adata, "image": tiles}, batch_size=256, group_key="slide")
```

Each spot is compared with its `spatial_k` nearest neighbours **within the batch**, and the squared latent
distance is weighted by `exp(-d^2 / (2 * spatial_sigma^2))`. The term is reported as `spatial` in the
forward output and uses the geodesic distance in hyperbolic and product latent spaces.

- `spatial_sigma` is in coordinate units and is required. Pairs much further apart than `spatial_sigma`
  contribute nothing, so an ordinary shuffled loader makes the term nearly inactive.
- `make_spatial_loader` therefore draws spatially local batches by default (`contiguous=True`). Because
  neighbourhoods overlap, a spot can appear in more than one batch per epoch. Use `contiguous=False` or
  `make_loader` for evaluation.
- Batches without coordinates are skipped, so spatial and non-spatial cohorts can share a model.
- The prior smooths the latent space. Too large a weight blurs sharp tissue boundaries; compare against
  `spatial_weight=0` on a held-out slide.
