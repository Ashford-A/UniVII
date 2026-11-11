# User guide

These pages explain the choices behind a UniVI analysis. The [tutorials](../tutorials/index.md) show them in running code.

| Page | Read it when you want to |
| --- | --- |
| [Preparing data](data.md) | get your AnnData objects, features, and splits into shape |
| [How the model works](model.md) | understand encoders, fusion, the training objective, and which latent representation to use |
| [Training and tuning](training.md) | pick hyperparameters, read the training curve, fix common problems |
| [Using a trained model](using_models.md) | embed, predict, denoise, generate, save and reload |
| [Evaluating an integration](evaluation.md) | measure alignment and prediction quality |
| [Advanced features](advanced.md) | heads, refinement, transformers, alternative objectives, scripts |
| [Latent geometry](latent_geometry.md) | use hyperbolic or product latent spaces, tree supervision, anchoring |
| [Spatial data and conv encoders](spatial_and_conv.md) | add niche or image-tile modalities, conv or custom encoders, a spatial prior |
| [FAQ](faq.md) | quick answers |

```{toctree}
:hidden:

data
model
training
using_models
evaluation
advanced
latent_geometry
spatial_and_conv
faq
```
