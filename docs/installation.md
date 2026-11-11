# Installation

UniVII supports Python 3.10 and newer on Linux, macOS, and Windows.

## With pip

UniVII is installed from GitHub until its first PyPI release; it is not on PyPI or conda-forge yet.

```bash
python -m pip install "univii @ git+https://github.com/Ashford-A/UniVII.git"
```

To run the tutorial notebooks locally, include the `tutorials` extra (Jupyter and `scikit-misc`, which Scanpy's Seurat-v3 highly-variable-gene selection needs):

```bash
python -m pip install "univii[tutorials] @ git+https://github.com/Ashford-A/UniVII.git"
```

```{note}
UniVII (`univii`) and the published UniVI (`univi`) are separate packages and can be installed in the same
environment. Datasets are cached separately, in `~/.cache/univii`.
```

### PyTorch and GPUs

UniVII depends on PyTorch. Installing it pulls the default PyTorch wheel for your platform, which on Linux includes CUDA support. For a specific CUDA version, or a CPU-only build, install PyTorch first with the selector at [pytorch.org](https://pytorch.org/get-started/locally/), then install UniVI.

Pass `device="cuda"`, `"mps"` (Apple silicon), or `"cpu"` to `TrainingConfig` and the encoding functions. The tutorials pick automatically:

```python
import torch
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
```

Training runs on a CPU, just more slowly. If an operation is not supported on MPS, fall back to CPU.

## Google Colab

Every tutorial has an **Open in Colab** badge and a first cell that installs UniVI when it detects Colab. Choose *Runtime → Change runtime type → GPU* for faster training.

## From source

```bash
git clone https://github.com/Ashford-A/UniVII.git
cd UniVII
python -m pip install -e ".[tutorials,test]"
python -m pytest -q
```

## Check the installation

```python
import univii
print(univii.__version__)

import univii.datasets as uds
print(uds.list_datasets())
```

Datasets are cached in `~/.cache/univii` by default; set the `UNIVII_DATA_DIR` environment variable to use another location (for example a scratch disk on a cluster).

## Which version to use

| Goal | Install |
| --- | --- |
| New analyses with hyperbolic, spatial or conv features | UniVII, from this repository |
| New analyses with the published feature set | UniVII, or the published package: `pip install univi` |
| Rerun the archived Genome Research notebooks exactly | `pip install univi==0.4.7`, the published release the article was prepared against |

The tutorials on this site use APIs added after 0.4.7 (fitted preprocessors, reference bundles, refinement, dataset downloads). The [reproduction section](reproducibility/index.md) explains how the archived notebooks relate to them.
