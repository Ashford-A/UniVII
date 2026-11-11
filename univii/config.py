# univii/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Sequence, Any, Union, Mapping


# =============================================================================
# Transformer + tokenizer config
# =============================================================================

@dataclass
class TransformerConfig:
    d_model: int
    num_heads: int
    num_layers: int
    dim_feedforward: int = 4096
    dropout: float = 0.1
    attn_dropout: float = 0.1
    activation: Literal["relu", "gelu"] = "gelu"
    pooling: Literal["cls", "mean"] = "mean"
    max_tokens: Optional[int] = None

    use_relpos_bias: bool = False
    relpos_num_bins: int = 32
    relpos_max_dist: float = 1e6  # basepairs


@dataclass
class TokenizerConfig:
    mode: Literal["topk_scalar", "topk_channels", "patch", "topk_embed"] = "topk_scalar"

    n_tokens: int = 256
    channels: Sequence[Literal["value", "rank", "dropout"]] = ("value",)

    patch_size: int = 32
    patch_proj_dim: Optional[int] = None

    add_cls_token: bool = False

    n_features: Optional[int] = None
    d_model: Optional[int] = None
    value_mlp_hidden: int = 256

    use_coords: bool = False
    chrom_vocab_size: int = 0
    coord_scale: float = 1e6
    feature_info: Optional[Dict[str, Any]] = None

    # Feature-identity and genomic-coordinate embeddings used by the encoders' tokenizer
    # (top-k modes). Declared as fields since 1.2.0 so that they survive dataclasses.replace
    # (used by the fused encoder) and are written by save_reference / read by load_reference.
    use_feature_embedding: bool = False
    feature_emb_dim: Optional[int] = None
    feature_emb_mode: Literal["add", "concat"] = "add"
    use_coord_embedding: bool = False
    n_chroms: Optional[int] = None
    coord_emb_dim: Optional[int] = None
    coord_mode: Literal["midpoint", "interval"] = "midpoint"
    coord_mlp_hidden: int = 128
    coord_num_frequencies: int = 16
    token_proj_dim: Optional[int] = None


# =============================================================================
# Core UniVI configs
# =============================================================================

LikelihoodName = Literal[
    # existing / common
    "gaussian", "normal", "mse", "gaussian_diag",
    "bernoulli", "poisson",
    "nb", "negative_binomial",
    "zinb", "zero_inflated_negative_binomial",
    "categorical", "cat", "ce", "cross_entropy", "multinomial", "softmax",
    "logistic_normal",
    # likely additions (safe to configure now; model/decoder can ignore until implemented)
    "beta",
    "beta_binomial",
    "zinb_with_libsize",
    "nb_with_libsize",
    "lognormal",
    "gamma",
    "student_t",
    "hurdle_poisson",
    "hurdle_nb",
]


@dataclass
class ConvConfig:
    """Convolutional encoder/decoder settings for grid-shaped modalities.

    The model still receives the modality as a flat ``(cells, input_dim)`` matrix,
    with ``input_dim == in_channels * prod(spatial_shape)`` in channel-first order.
    Conv modules reshape internally, so loaders, losses and exports are unchanged.

    ``spatial_shape`` is ``(H, W)`` for image tiles or ``(L,)`` for 1-D signals
    (ordered tracks). Each entry of ``channels`` is one stage that halves the
    resolution, so every spatial size must be divisible by ``2 ** len(channels)``.
    """
    in_channels: int = 3
    spatial_shape: Sequence[int] = (32, 32)
    channels: Sequence[int] = (32, 64, 128)
    kernel_size: int = 3
    blocks_per_stage: int = 1                  # residual blocks per stage
    norm: Literal["group", "batch", "none"] = "group"
    groups: int = 8
    activation: Literal["relu", "gelu", "silu", "leakyrelu"] = "silu"
    dropout: float = 0.0
    pool: Literal["flatten", "mean"] = "flatten"          # encoder: feature map -> vector
    hidden_dim: int = 256                      # dense layer between the feature map and the latent
    upsample: Literal["nearest", "transpose"] = "nearest"  # decoder upsampling
    output_activation: Literal["none", "sigmoid"] = "none"  # decoder mean; "sigmoid" for data in [0, 1]
    # Train-time augmentation of the encoder input (2-D only): any of "flip", "rot90".
    # The reconstruction target is the un-augmented image.
    augment: Sequence[str] = ()

    @property
    def ndim(self) -> int:
        return len(tuple(self.spatial_shape))

    @property
    def flat_dim(self) -> int:
        n = int(self.in_channels)
        for v in self.spatial_shape:
            n *= int(v)
        return n

    def validate(self, mod_name: str = "?", input_dim: Optional[int] = None) -> None:
        pre = f"Modality {mod_name!r}: conv"
        shape = tuple(int(v) for v in self.spatial_shape)
        if len(shape) not in (1, 2):
            raise ValueError(f"{pre}.spatial_shape must have 1 or 2 entries, got {self.spatial_shape!r}")
        if int(self.in_channels) < 1 or any(v < 1 for v in shape):
            raise ValueError(f"{pre}.in_channels and spatial_shape must be positive")
        chans = [int(c) for c in self.channels]
        if not chans or any(c < 1 for c in chans):
            raise ValueError(f"{pre}.channels must be a non-empty list of positive ints")
        factor = 2 ** len(chans)
        if any(v % factor for v in shape):
            raise ValueError(
                f"{pre}.spatial_shape {shape} must be divisible by {factor} (2 ** number of stages)")
        if int(self.kernel_size) < 1 or int(self.kernel_size) % 2 == 0:
            raise ValueError(f"{pre}.kernel_size must be a positive odd int, got {self.kernel_size}")
        for field_name, allowed in (("norm", ("group", "batch", "none")), ("pool", ("flatten", "mean")),
                                    ("upsample", ("nearest", "transpose")),
                                    ("output_activation", ("none", "sigmoid")),
                                    ("activation", ("relu", "gelu", "silu", "leakyrelu"))):
            if getattr(self, field_name) not in allowed:
                raise ValueError(f"{pre}.{field_name} must be one of {allowed}, got {getattr(self, field_name)!r}")
        if int(self.blocks_per_stage) < 0 or int(self.hidden_dim) < 1 or int(self.groups) < 1:
            raise ValueError(f"{pre}: blocks_per_stage >= 0, hidden_dim >= 1 and groups >= 1 are required")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError(f"{pre}.dropout must be in [0, 1), got {self.dropout}")
        if input_dim is not None and int(input_dim) != self.flat_dim:
            raise ValueError(
                f"Modality {mod_name!r}: input_dim={input_dim} must equal in_channels * prod(spatial_shape) "
                f"= {self.flat_dim}")
        _validate_augment(pre, self.augment, shape)


def _validate_augment(pre: str, augment: Sequence[str], shape: Sequence[int]) -> None:
    bad = [a for a in augment if a not in ("flip", "rot90")]
    if bad:
        raise ValueError(f"{pre}.augment entries must be 'flip' or 'rot90', got {bad}")
    if augment and len(shape) != 2:
        raise ValueError(f"{pre}.augment is only available for 2-D inputs")
    if "rot90" in augment and shape[0] != shape[1]:
        raise ValueError(f"{pre}.augment 'rot90' needs square inputs, got {tuple(shape)}")


@dataclass
class ViTConfig:
    """Vision-transformer encoder settings for image modalities.

    Like conv modalities, the image is stored flat as ``(cells, C * H * W)`` in
    channel-first order. It is cut into non-overlapping ``patch_size`` patches,
    each patch is embedded, and a transformer encoder processes the sequence.
    """
    in_channels: int = 3
    image_size: Sequence[int] = (32, 32)
    patch_size: int = 8
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.0
    pool: Literal["mean", "cls"] = "mean"
    augment: Sequence[str] = ()                # as in ConvConfig

    @property
    def flat_dim(self) -> int:
        return int(self.in_channels) * int(self.image_size[0]) * int(self.image_size[1])

    @property
    def n_patches(self) -> int:
        return (int(self.image_size[0]) // int(self.patch_size)) * (int(self.image_size[1]) // int(self.patch_size))

    def validate(self, mod_name: str = "?", input_dim: Optional[int] = None) -> None:
        pre = f"Modality {mod_name!r}: vit"
        size = tuple(int(v) for v in self.image_size)
        if len(size) != 2 or any(v < 1 for v in size) or int(self.in_channels) < 1:
            raise ValueError(f"{pre}.image_size must be (H, W) with positive entries and in_channels >= 1")
        if int(self.patch_size) < 1 or any(v % int(self.patch_size) for v in size):
            raise ValueError(f"{pre}.image_size {size} must be divisible by patch_size {self.patch_size}")
        if int(self.d_model) % int(self.num_heads):
            raise ValueError(f"{pre}.d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})")
        if int(self.num_layers) < 1 or self.pool not in ("mean", "cls") or not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError(f"{pre}: num_layers >= 1, pool in ('mean', 'cls') and 0 <= dropout < 1 are required")
        if input_dim is not None and int(input_dim) != self.flat_dim:
            raise ValueError(f"Modality {mod_name!r}: input_dim={input_dim} must equal in_channels * H * W "
                             f"= {self.flat_dim}")
        _validate_augment(pre, self.augment, size)


@dataclass
class GraphConfig:
    """Graph (GraphSAGE-style) encoder settings.

    Each cell is encoded from its own features and the mean (or max) of its
    ``k`` nearest neighbours *within the batch*, repeated for ``len(hidden)``
    rounds. Neighbours come from the coordinates the batch carries under
    ``UniVIConfig.spatial_key``; a batch without coordinates falls back to each
    cell being its own neighbour. Use spatially local batches
    (``univii.spatial.make_spatial_loader`` / ``encode_spatial``) at training
    and at inference, so that a cell's in-batch neighbours are its real ones.
    """
    k: int = 6
    hidden: Sequence[int] = (256, 128)
    aggregate: Literal["mean", "max"] = "mean"
    dropout: float = 0.0

    def validate(self, mod_name: str = "?") -> None:
        pre = f"Modality {mod_name!r}: graph"
        if int(self.k) < 1 or not list(self.hidden) or any(int(h) < 1 for h in self.hidden):
            raise ValueError(f"{pre}: k >= 1 and a non-empty list of positive hidden sizes are required")
        if self.aggregate not in ("mean", "max") or not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError(f"{pre}: aggregate in ('mean', 'max') and 0 <= dropout < 1 are required")


@dataclass
class ModalityConfig:
    """
    Configuration for a single modality.

    Backwards compatible with prior versions.

    Added decoder/loss-related optional knobs so the model can support more
    likelihoods without changing the config schema again.
    """
    name: str
    input_dim: int
    encoder_hidden: List[int]
    decoder_hidden: List[int]
    likelihood: str = "gaussian"

    # ---- per-modality reconstruction weighting ----
    recon_weight: float = 1.0

    # ---- Count model settings (NB/ZINB and relatives) ----
    dispersion: str = "gene"  # e.g. "global", "gene", "feature" (model decides support)
    init_log_theta: float = 0.0

    # Optional library-size handling (for count decoders that use offsets)
    use_library_size: bool = False
    library_key: Optional[str] = None          # e.g. obs key if data loader uses obs-derived features
    library_log1p: bool = True                 # whether supplied library is already log1p-transformed
    predict_library: bool = False              # if you later add a library-size head/decoder branch

    # ---- Zero-inflation / hurdle settings ----
    use_zero_inflation: Optional[bool] = None  # None => infer from likelihood
    hurdle: bool = False                       # for hurdle-* likelihoods if implemented

    # ---- Bounded / proportion data (e.g. methylation fractions in [0,1]) ----
    # If you preprocess methylation to fractions, beta-like decoders often want clipping.
    clip_targets_min: Optional[float] = None
    clip_targets_max: Optional[float] = None

    # For beta/binomial-style likelihoods if you add them later
    concentration_init: float = 0.0           # log-concentration init or analogous parameter
    total_count_key: Optional[str] = None     # obs key / side-input key for trials if needed
    successes_are_fraction: bool = False      # True if x is fraction instead of counts

    # ---- Positive continuous data (if you add gamma/lognormal decoders) ----
    positive_eps: float = 1e-8

    # ---- Categorical modality support ----
    ignore_index: int = -1
    input_kind: Literal["matrix", "obs"] = "matrix"
    obs_key: Optional[str] = None

    # ---- Optional masking / weighting hooks (future-proof) ----
    mask_key: Optional[str] = None            # if your dataset returns feature or sample masks
    sample_weight_key: Optional[str] = None   # if you later support weighted recon losses

    # ---- Encoder backend (per-modality only) ----
    # "mlp", "transformer", "conv", or any name added with univii.models.registry.register_encoder
    encoder_type: str = "mlp"
    transformer: Optional[TransformerConfig] = None
    tokenizer: Optional[TokenizerConfig] = None

    # ---- Arbitrary decoder/likelihood kwargs passthrough (future-proof) ----
    # Example:
    # decoder_kwargs={"dispersion":"gene-cell", "min_scale":1e-4}
    decoder_kwargs: Optional[Dict[str, Any]] = None

    # "mlp" (default: chosen by likelihood), "linear" (one linear map from the latent, for any
    # likelihood; read the loadings with model.decoder_loadings), "conv", or a registered name
    decoder_type: str = "mlp"
    # settings for encoder_type="conv" and/or decoder_type="conv"
    conv: Optional[ConvConfig] = None
    # free-form settings passed to registered (custom) encoder builders
    encoder_kwargs: Optional[Dict[str, Any]] = None
    # settings for encoder_type="vit" and encoder_type="graph"
    vit: Optional[ViTConfig] = None
    graph: Optional[GraphConfig] = None
    # Fraction of this modality's input features zeroed at random before encoding, at training time
    # only. The reconstruction target stays complete, so the model learns to fill features in.
    input_dropout: float = 0.0


@dataclass
class ClassHeadConfig:
    """
    Supervised / adversarial head configuration.

    Backwards compatible defaults preserve the old behavior:
    - head_type defaults to categorical (even if n_classes==2)
    """
    name: str
    n_classes: int
    loss_weight: float = 1.0
    ignore_index: int = -1
    from_mu: bool = True
    warmup: int = 0

    adversarial: bool = False
    adv_lambda: float = 1.0

    # NEW (optional): allow passing what the model reads via getattr
    head_type: Optional[Literal["categorical", "binary"]] = None
    hidden_dims: Optional[List[int]] = None
    dropout: Optional[float] = None
    batchnorm: Optional[bool] = None
    activation: Optional[Literal["relu", "gelu", "elu", "leakyrelu", "silu", "tanh"]] = None
    pos_weight: float = 1.0  # binary only
    layernorm: bool = False  # LayerNorm on hidden activations (used for refinement heads)


@dataclass
class UniVIConfig:
    latent_dim: int
    modalities: List[ModalityConfig]

    beta: float = 1.0
    gamma: float = 1.0

    encoder_dropout: float = 0.0
    decoder_dropout: float = 0.0
    encoder_batchnorm: bool = True
    decoder_batchnorm: bool = False

    kl_anneal_start: int = 0
    kl_anneal_end: int = 0
    align_anneal_start: int = 0
    align_anneal_end: int = 0

    class_heads: Optional[List[ClassHeadConfig]] = None
    label_head_name: str = "label"

    fused_encoder_type: Literal["moe", "multimodal_transformer"] = "moe"
    fused_transformer: Optional[TransformerConfig] = None
    fused_modalities: Optional[Sequence[str]] = None
    fused_add_modality_embeddings: bool = True
    fused_require_all_modalities: bool = True

    # ------------------------------------------------------------------
    # MoE gating (learned per-cell modality weights for fusion)
    # ------------------------------------------------------------------
    use_moe_gating: bool = False
    moe_gating_type: Literal["per_modality", "shared"] = "per_modality"
    moe_gating_hidden: Optional[List[int]] = None
    moe_gating_dropout: float = 0.0
    moe_gating_batchnorm: bool = False
    moe_gating_activation: Literal["relu", "gelu", "elu", "leakyrelu", "silu", "tanh"] = "relu"
    moe_gate_eps: float = 1e-6

    # ------------------------------------------------------------------
    # Optional global reconstruction/likelihood defaults (future-proof)
    # These do NOT override per-modality fields unless your model chooses to.
    # ------------------------------------------------------------------
    default_dispersion: Optional[str] = None
    default_use_library_size: Optional[bool] = None
    default_positive_eps: Optional[float] = None

    # ------------------------------------------------------------------
    # Latent geometry (see univii.geometry for the coordinate convention)
    # ------------------------------------------------------------------
    # "euclidean": the original model, unchanged.
    # "hyperbolic": every latent dimension is hyperbolic.
    # "product": the last `hyperbolic_dim` dimensions are hyperbolic, the rest Euclidean.
    latent_geometry: Literal["euclidean", "hyperbolic", "product"] = "euclidean"
    hyperbolic_dim: Optional[int] = None       # product only; None => half of latent_dim (rounded up)
    curvature: float = 1.0                     # magnitude c > 0; the hyperbolic block has curvature -c
    learn_curvature: bool = False
    hyperbolic_max_radius: float = 8.0         # cap on sqrt(c) * ||mu_hyperbolic||
    kl_hyperbolic_weight: float = 1.0          # multiplier on the hyperbolic part of the prior KL

    # Weak supervision from a label tree (inactive until model.set_hierarchy(...) is called
    # and hierarchy_weight > 0). Labels come from `hierarchy_head` in a dict `y`, or from
    # the legacy label tensor when hierarchy_head is None.
    hierarchy_weight: float = 0.0
    hierarchy_root_weight: float = 0.0         # extra pull of the root label toward the origin
    hierarchy_scale: Optional[float] = 1.0     # latent distance per unit tree distance; None => learned
    hierarchy_head: Optional[str] = None

    # ------------------------------------------------------------------
    # Prior over the latent (see univii.priors)
    # ------------------------------------------------------------------
    # "standard": N(0, I) (wrapped normal at the origin in a hyperbolic block), the original model.
    # "mog": a learned mixture of `prior_components` components, which lets the latent form clusters.
    # It covers every block of the latent: in a hyperbolic block each component is a wrapped normal
    # at a learned point, so the prototypes are learned in hyperbolic space. With `prior_label_head`, a cell whose
    # label is known is tied to the component with that index (labels in a dict `y`), so components
    # become classes. With `prior_tree_weight > 0` and a registered hierarchy (model.set_hierarchy),
    # distances between component means are pulled toward the label-tree distances.
    prior: Literal["standard", "mog"] = "standard"
    prior_components: int = 16
    prior_label_head: Optional[str] = None
    prior_tree_weight: float = 0.0

    # Contrastive (InfoNCE) alignment between modalities: each cell's posterior mean in one modality
    # should be closer to its own mean in another modality than to any other cell's in the batch.
    # Uses the latent geometry's distance. Adds to, and does not replace, `gamma`.
    contrastive_weight: float = 0.0
    contrastive_temperature: float = 1.0

    # Cohort alignment for atlases and mosaics (see univii.atlas). With weight > 0, the fused latent
    # means of each cohort are pulled toward those of the other cohorts with a kernel two-sample
    # (MMD) penalty. Cohort codes are read from `y[cohort_head]`. Because a batch usually holds one
    # cohort, the model keeps the last `cohort_memory` latent means of every cohort (detached) and
    # compares against those. Applied at training time only.
    cohort_mmd_weight: float = 0.0
    cohort_head: str = "cohort"
    cohort_memory: int = 1024
    # Cohort-conditioned decoders: with n_cohorts > 0 every decoder also receives a learned
    # embedding of the cell's cohort (code in `y[cohort_head]`), so technical differences between
    # cohorts need not be stored in the latent. Combine with cohort_mmd_weight, which then no longer
    # works against reconstruction. Outside training, decoders use cohort 0 unless
    # model.set_decode_cohort(code) says otherwise.
    n_cohorts: int = 0
    cohort_embedding_dim: int = 8
    # Per-cohort learned shift added to each modality's encoder input. All shifts start at zero and
    # are only trained by univii.atlas.adapt_to_cohort, for a cohort added with model.add_cohort();
    # that function switches this on.
    cohort_input_shift: bool = False

    # Cycle consistency: predict every other modality from a modality's latent mean, encode the
    # prediction with that modality's own encoder, and penalise the distance back to the starting
    # latent. It also runs for modalities missing from the batch, which is what makes it useful for
    # single-modality cohorts. Applied to target modalities with a Gaussian-type likelihood, where
    # the decoder's mean is in the same space as the encoder's input.
    cycle_weight: float = 0.0

    # ------------------------------------------------------------------
    # Spatial smoothness prior (see univii.spatial)
    # ------------------------------------------------------------------
    # Active when spatial_weight > 0 and the batch carries coordinates under `spatial_key`
    # in x_dict (univii.spatial.make_spatial_loader adds them). Each cell is pulled toward its
    # `spatial_k` nearest in-batch neighbours with weight exp(-d^2 / (2 * spatial_sigma^2)),
    # so pairs much further apart than spatial_sigma contribute nothing.
    spatial_weight: float = 0.0
    spatial_key: str = "spatial"
    spatial_sigma: Optional[float] = None      # kernel bandwidth in coordinate units; required when active
    spatial_k: int = 6

    # ------------------------------------------------------------------
    # Arbitrary model kwargs passthrough (future-proof)
    # ------------------------------------------------------------------
    model_kwargs: Optional[Dict[str, Any]] = None

    def validate(self) -> None:
        if int(self.latent_dim) <= 0:
            raise ValueError(f"latent_dim must be > 0, got {self.latent_dim}")

        from .geometry import split_latent_dims
        split_latent_dims(int(self.latent_dim), self.latent_geometry, self.hyperbolic_dim)
        if not float(self.curvature) > 0.0:
            raise ValueError(f"curvature must be > 0, got {self.curvature}")
        if not float(self.hyperbolic_max_radius) > 0.0:
            raise ValueError(f"hyperbolic_max_radius must be > 0, got {self.hyperbolic_max_radius}")
        for _name in ("kl_hyperbolic_weight", "hierarchy_weight", "hierarchy_root_weight"):
            if float(getattr(self, _name)) < 0.0:
                raise ValueError(f"{_name} must be >= 0, got {getattr(self, _name)}")
        if self.hierarchy_scale is not None and not float(self.hierarchy_scale) > 0.0:
            raise ValueError(f"hierarchy_scale must be > 0 or None, got {self.hierarchy_scale}")

        if self.prior not in ("standard", "mog"):
            raise ValueError(f"prior must be 'standard' or 'mog', got {self.prior!r}")
        if self.prior == "mog":
            if int(self.prior_components) < 1:
                raise ValueError(f"prior_components must be >= 1, got {self.prior_components}")
        elif self.prior_label_head is not None or float(self.prior_tree_weight) != 0.0:
            raise ValueError("prior_label_head and prior_tree_weight require prior='mog'")
        if float(self.prior_tree_weight) < 0.0 or float(self.contrastive_weight) < 0.0:
            raise ValueError("prior_tree_weight and contrastive_weight must be >= 0")
        if float(self.cohort_mmd_weight) < 0.0 or int(self.cohort_memory) < 1:
            raise ValueError("cohort_mmd_weight must be >= 0 and cohort_memory >= 1")
        if int(self.n_cohorts) < 0 or int(self.cohort_embedding_dim) < 1:
            raise ValueError("n_cohorts must be >= 0 and cohort_embedding_dim >= 1")
        if bool(self.cohort_input_shift) and int(self.n_cohorts) < 1:
            raise ValueError("cohort_input_shift requires n_cohorts >= 1")
        if float(self.cycle_weight) < 0.0:
            raise ValueError(f"cycle_weight must be >= 0, got {self.cycle_weight}")
        if not float(self.contrastive_temperature) > 0.0:
            raise ValueError(f"contrastive_temperature must be > 0, got {self.contrastive_temperature}")

        if float(self.spatial_weight) < 0.0:
            raise ValueError(f"spatial_weight must be >= 0, got {self.spatial_weight}")
        if float(self.spatial_weight) > 0.0:
            if self.spatial_sigma is None or not float(self.spatial_sigma) > 0.0:
                raise ValueError(
                    "spatial_weight > 0 requires spatial_sigma > 0 (the kernel bandwidth in coordinate units; "
                    "univii.spatial.median_neighbor_distance gives a sensible starting value)")
            if int(self.spatial_k) < 1:
                raise ValueError(f"spatial_k must be >= 1, got {self.spatial_k}")
            if self.spatial_key in {m.name for m in self.modalities}:
                raise ValueError(f"spatial_key {self.spatial_key!r} clashes with a modality name")

        names = [m.name for m in self.modalities]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Duplicate modality names in cfg.modalities: {dupes}")

        mod_by_name: Dict[str, ModalityConfig] = {m.name: m for m in self.modalities}

        for m in self.modalities:
            if int(m.input_dim) <= 0:
                raise ValueError(f"Modality {m.name!r}: input_dim must be > 0, got {m.input_dim}")

            if float(getattr(m, "recon_weight", 1.0)) < 0.0:
                raise ValueError(f"Modality {m.name!r}: recon_weight must be >= 0, got {m.recon_weight}")

            lk = (m.likelihood or "").lower().strip()

            # ---- categorical checks ----
            if lk in ("categorical", "cat", "ce", "cross_entropy", "multinomial", "softmax"):
                if int(m.input_dim) < 2:
                    raise ValueError(f"Categorical modality {m.name!r}: input_dim must be n_classes >= 2.")
                if m.input_kind == "obs" and not m.obs_key:
                    raise ValueError(f"Categorical modality {m.name!r}: input_kind='obs' requires obs_key.")

            # ---- count-like checks ----
            if lk in (
                "nb", "negative_binomial",
                "zinb", "zero_inflated_negative_binomial",
                "nb_with_libsize", "zinb_with_libsize",
                "hurdle_nb", "hurdle_poisson",
            ):
                disp = str(getattr(m, "dispersion", "gene")).lower().strip()
                # keep permissive (model can support subset)
                allowed_disp = {"global", "gene", "feature", "gene_cell", "gene-cell"}
                if disp not in allowed_disp:
                    raise ValueError(
                        f"Modality {m.name!r}: unsupported dispersion={m.dispersion!r}. "
                        f"Expected one of {sorted(allowed_disp)}."
                    )

                if float(getattr(m, "positive_eps", 1e-8)) <= 0:
                    raise ValueError(f"Modality {m.name!r}: positive_eps must be > 0.")

                if bool(getattr(m, "use_library_size", False)) and m.library_key is None:
                    # not always required if library is computed from x on-the-fly, so only warn via ValueError? No:
                    # keep permissive for backward compatibility.
                    pass

            # ---- bounded / proportion checks ----
            if lk in ("beta", "beta_binomial"):
                cmin = getattr(m, "clip_targets_min", None)
                cmax = getattr(m, "clip_targets_max", None)
                if cmin is not None and cmax is not None and float(cmin) >= float(cmax):
                    raise ValueError(
                        f"Modality {m.name!r}: clip_targets_min must be < clip_targets_max "
                        f"(got {cmin} >= {cmax})."
                    )
                if float(getattr(m, "positive_eps", 1e-8)) <= 0:
                    raise ValueError(f"Modality {m.name!r}: positive_eps must be > 0.")

            # ---- positive continuous checks ----
            if lk in ("gamma", "lognormal", "student_t"):
                if float(getattr(m, "positive_eps", 1e-8)) <= 0:
                    raise ValueError(f"Modality {m.name!r}: positive_eps must be > 0.")

            # ---- decoder kwargs sanity ----
            if m.decoder_kwargs is not None and not isinstance(m.decoder_kwargs, dict):
                raise ValueError(f"Modality {m.name!r}: decoder_kwargs must be dict or None.")

            # ---- encoder backend checks ----
            enc_type = (m.encoder_type or "mlp").lower().strip()
            from .models.registry import DECODER_BUILDERS, ENCODER_BUILDERS
            from .models import conv as _conv_builtin  # noqa: F401  (registers the built-in "conv" builders)
            from .models import vision as _vit_builtin, graph as _graph_builtin  # noqa: F401
            if enc_type not in ("mlp", "transformer") and enc_type not in ENCODER_BUILDERS:
                raise ValueError(
                    f"Modality {m.name!r}: unknown encoder_type {m.encoder_type!r}. Built in: 'mlp', "
                    f"'transformer'; registered: {sorted(ENCODER_BUILDERS)}"
                )
            dec_type = (getattr(m, "decoder_type", "mlp") or "mlp").lower().strip()
            if dec_type not in ("mlp", "linear") and dec_type not in DECODER_BUILDERS:
                raise ValueError(
                    f"Modality {m.name!r}: unknown decoder_type {m.decoder_type!r}. Built in: 'mlp', 'linear'; "
                    f"registered: {sorted(DECODER_BUILDERS)}"
                )
            if "conv" in (enc_type, dec_type):
                if m.conv is None:
                    raise ValueError(f"Modality {m.name!r}: encoder_type/decoder_type 'conv' requires conv=ConvConfig(...)")
                m.conv.validate(m.name, int(m.input_dim))
            if enc_type == "vit":
                if m.vit is None:
                    raise ValueError(f"Modality {m.name!r}: encoder_type='vit' requires vit=ViTConfig(...)")
                m.vit.validate(m.name, int(m.input_dim))
            if enc_type == "graph":
                if m.graph is None:
                    raise ValueError(f"Modality {m.name!r}: encoder_type='graph' requires graph=GraphConfig(...)")
                m.graph.validate(m.name)
            if not 0.0 <= float(getattr(m, "input_dropout", 0.0)) < 1.0:
                raise ValueError(f"Modality {m.name!r}: input_dropout must be in [0, 1), got {m.input_dropout}")

            if enc_type == "transformer":
                if m.transformer is None:
                    raise ValueError(f"Modality {m.name!r}: encoder_type='transformer' requires transformer config.")
                if m.tokenizer is None:
                    raise ValueError(f"Modality {m.name!r}: encoder_type='transformer' requires tokenizer config.")
                _validate_tokenizer(m.name, m.tokenizer)

        # -------------------------
        # fused encoder checks
        # -------------------------
        fe = (self.fused_encoder_type or "moe").lower().strip()
        if fe not in ("moe", "multimodal_transformer"):
            raise ValueError(
                f"fused_encoder_type must be 'moe' or 'multimodal_transformer', got {self.fused_encoder_type!r}"
            )

        if fe == "multimodal_transformer":
            if self.fused_transformer is None:
                raise ValueError("fused_encoder_type='multimodal_transformer' requires UniVIConfig.fused_transformer.")

            fused_names = list(self.fused_modalities) if self.fused_modalities is not None else list(mod_by_name.keys())
            if not fused_names:
                raise ValueError("fused_modalities is empty; expected at least one modality name.")

            missing = [n for n in fused_names if n not in mod_by_name]
            if missing:
                raise ValueError(f"fused_modalities contains unknown modalities: {missing}. Known: {list(mod_by_name)}")

            for n in fused_names:
                tok = mod_by_name[n].tokenizer
                if tok is None:
                    raise ValueError(
                        f"Fused multimodal transformer requires ModalityConfig.tokenizer for modality {n!r}."
                    )
                _validate_tokenizer(n, tok)

        # -------------------------
        # MoE gating checks
        # -------------------------
        if bool(self.use_moe_gating):
            gt = (self.moe_gating_type or "per_modality").lower().strip()
            if gt not in ("per_modality", "shared"):
                raise ValueError(
                    f"moe_gating_type must be 'per_modality' or 'shared', got {self.moe_gating_type!r}"
                )

            h = self.moe_gating_hidden
            if h is not None:
                if not isinstance(h, (list, tuple)) or len(h) == 0:
                    raise ValueError("moe_gating_hidden must be a non-empty list[int] if provided.")
                bad = [x for x in h if int(x) <= 0]
                if bad:
                    raise ValueError(f"moe_gating_hidden must contain only positive ints; bad entries: {bad}")

            if float(self.moe_gating_dropout) < 0.0:
                raise ValueError(f"moe_gating_dropout must be >= 0, got {self.moe_gating_dropout}")
            if float(self.moe_gate_eps) < 0.0:
                raise ValueError(f"moe_gate_eps must be >= 0, got {self.moe_gate_eps}")

            act = (self.moe_gating_activation or "relu").lower().strip()
            if act not in ("relu", "gelu", "elu", "leakyrelu", "silu", "tanh"):
                raise ValueError(
                    "moe_gating_activation must be one of "
                    "['relu','gelu','elu','leakyrelu','silu','tanh'], "
                    f"got {self.moe_gating_activation!r}"
                )

        # -------------------------
        # class head checks
        # -------------------------
        if self.class_heads is not None:
            hn = [h.name for h in self.class_heads]
            if len(set(hn)) != len(hn):
                dupes = sorted({n for n in hn if hn.count(n) > 1})
                raise ValueError(f"Duplicate class head names in cfg.class_heads: {dupes}")

            for h in self.class_heads:
                ht = (h.head_type or "categorical")
                if int(h.n_classes) < 2 and ht != "binary":
                    raise ValueError(f"Class head {h.name!r}: n_classes must be >= 2 for categorical heads.")
                if float(h.loss_weight) < 0:
                    raise ValueError(f"Class head {h.name!r}: loss_weight must be >= 0.")
                if int(h.warmup) < 0:
                    raise ValueError(f"Class head {h.name!r}: warmup must be >= 0.")
                if float(getattr(h, "adv_lambda", 1.0)) < 0.0:
                    raise ValueError(f"Class head {h.name!r}: adv_lambda must be >= 0.")
                if float(getattr(h, "pos_weight", 1.0)) < 0.0:
                    raise ValueError(f"Class head {h.name!r}: pos_weight must be >= 0.")

                if h.hidden_dims is not None:
                    if not isinstance(h.hidden_dims, (list, tuple)) or len(h.hidden_dims) == 0:
                        raise ValueError(f"Class head {h.name!r}: hidden_dims must be non-empty if provided.")
                    bad = [x for x in h.hidden_dims if int(x) <= 0]
                    if bad:
                        raise ValueError(f"Class head {h.name!r}: hidden_dims must be positive ints; bad: {bad}")

        # -------------------------
        # annealing checks
        # -------------------------
        for k in ("kl_anneal_start", "kl_anneal_end", "align_anneal_start", "align_anneal_end"):
            v = int(getattr(self, k))
            if v < 0:
                raise ValueError(f"{k} must be >= 0, got {v}")

        # -------------------------
        # optional global defaults checks
        # -------------------------
        if self.default_positive_eps is not None and float(self.default_positive_eps) <= 0:
            raise ValueError("default_positive_eps must be > 0 if provided.")

        if self.model_kwargs is not None and not isinstance(self.model_kwargs, dict):
            raise ValueError("model_kwargs must be dict or None.")


def _validate_tokenizer(mod_name: str, tok: TokenizerConfig) -> None:
    mode = (tok.mode or "").lower().strip()
    if mode not in ("topk_scalar", "topk_channels", "patch", "topk_embed"):
        raise ValueError(
            f"Modality {mod_name!r}: tokenizer.mode must be one of "
            f"['topk_scalar','topk_channels','patch','topk_embed'], got {tok.mode!r}"
        )

    if mode in ("topk_scalar", "topk_channels", "topk_embed"):
        if int(tok.n_tokens) <= 0:
            raise ValueError(f"Modality {mod_name!r}: tokenizer.n_tokens must be > 0 for topk_*")

    if bool(getattr(tok, "use_feature_embedding", False)) and mode in ("topk_scalar", "topk_channels"):
        if tok.n_features is None or int(tok.n_features) <= 0:
            raise ValueError(
                f"Modality {mod_name!r}: tokenizer.n_features must be set (>0) when use_feature_embedding=True"
            )
    if str(getattr(tok, "feature_emb_mode", "add")).lower() not in ("add", "concat"):
        raise ValueError(f"Modality {mod_name!r}: tokenizer.feature_emb_mode must be 'add' or 'concat'")
    if bool(getattr(tok, "use_coord_embedding", False)):
        if mode == "patch":
            raise ValueError(f"Modality {mod_name!r}: use_coord_embedding requires a top-k tokenizer mode")
        n_chroms = getattr(tok, "n_chroms", None) or int(getattr(tok, "chrom_vocab_size", 0) or 0)
        if not n_chroms or int(n_chroms) <= 0:
            raise ValueError(
                f"Modality {mod_name!r}: set tokenizer.n_chroms (>0) when use_coord_embedding=True"
            )
    if str(getattr(tok, "coord_mode", "midpoint")).lower() not in ("midpoint", "interval"):
        raise ValueError(f"Modality {mod_name!r}: tokenizer.coord_mode must be 'midpoint' or 'interval'")
    if int(getattr(tok, "coord_num_frequencies", 16)) <= 0:
        raise ValueError(f"Modality {mod_name!r}: tokenizer.coord_num_frequencies must be > 0")

    if mode == "topk_channels":
        if not tok.channels:
            raise ValueError(f"Modality {mod_name!r}: tokenizer.channels must be non-empty for topk_channels")
        bad = [c for c in tok.channels if c not in ("value", "rank", "dropout")]
        if bad:
            raise ValueError(f"Modality {mod_name!r}: tokenizer.channels has invalid entries: {bad}")

    if mode == "topk_embed":
        if tok.n_features is None or int(tok.n_features) <= 0:
            raise ValueError(f"Modality {mod_name!r}: tokenizer.n_features must be set (>0) for topk_embed")
        if tok.d_model is None or int(tok.d_model) <= 0:
            raise ValueError(f"Modality {mod_name!r}: tokenizer.d_model must be set (>0) for topk_embed")
        if not tok.channels:
            raise ValueError(f"Modality {mod_name!r}: tokenizer.channels must be non-empty for topk_embed")
        bad = [c for c in tok.channels if c not in ("value", "rank", "dropout")]
        if bad:
            raise ValueError(f"Modality {mod_name!r}: tokenizer.channels has invalid entries: {bad}")

        if tok.use_coords:
            if int(tok.chrom_vocab_size) <= 0:
                raise ValueError(
                    f"Modality {mod_name!r}: tokenizer.chrom_vocab_size must be > 0 when use_coords=True"
                )
            if tok.feature_info is not None:
                for k in ("chrom", "start", "end"):
                    if k not in tok.feature_info:
                        raise ValueError(
                            f"Modality {mod_name!r}: tokenizer.feature_info missing key {k!r} (required for coords)"
                        )

    if mode == "patch":
        if int(tok.patch_size) <= 0:
            raise ValueError(f"Modality {mod_name!r}: tokenizer.patch_size must be > 0 for patch")
        if tok.patch_proj_dim is not None and int(tok.patch_proj_dim) <= 0:
            raise ValueError(f"Modality {mod_name!r}: tokenizer.patch_proj_dim must be > 0 if set")


@dataclass
class TrainingConfig:
    n_epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    log_every: int = 10
    grad_clip: Optional[float] = None
    num_workers: int = 0
    seed: int = 0

    early_stopping: bool = False
    patience: int = 20
    min_delta: float = 0.0

    best_epoch_warmup: int = 0

