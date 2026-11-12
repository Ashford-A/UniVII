"""Execute the tutorial and paper-reproduction notebooks end to end on small synthetic stand-in data.

This checks that every tutorial runs against the current API; it does not
check biological results. Datasets are served from local synthetic files via
``UNIVII_DATASET_REGISTRY`` and training is shortened by overriding the cell
tagged ``parameters`` in each notebook.

Run with:  pytest tests/test_tutorials.py     (skipped unless nbclient is installed)
"""
import os
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (sorted((ROOT / "docs" / "tutorials").glob("*.ipynb"))
             + sorted((ROOT / "docs" / "tutorials" / "experimental").glob("*.ipynb"))
             + sorted((ROOT / "docs" / "reproducibility" / "api").glob("*.ipynb")))

# Small values injected after each notebook's "parameters" cell.
OVERRIDES = """
N_EPOCHS = 3
REFINE_EPOCHS = 3
HEAD_WARMUP_EPOCHS = 3
ENCODER_PATIENCE = 3
BRIDGE_EPOCHS = 3
N_HVG = 300
N_LSI = 21
MAX_CELLS = 1500
N_GENERATED = 200
MAX_TEST_CELLS = 400
MIN_PER_CLASS = 5
HIGHRES_HEAD_EPOCHS = 3
N_PEAKS = 800
N_NULL = 5
RUN_MOTIF_SCAN = False
N_TOKENS_RNA = 32
N_TOKENS_ATAC = 32
MAX_ATTENTION_CELLS = 300
N_PERMUTATIONS = 20
N_CITE_CELLS = 900
N_CITE_TEST = 300
N_SAMPLES = 5
N_NULL_LSI = 3
EXTRA_HOLDOUTS = 1
N_UNTRAINED_CONTROLS = 1
ADV_LAMBDAS = [1.0]
N_TILING_NULL = 15
"""


@pytest.fixture(scope="module")
def synthetic_registry(tmp_path_factory):
    from synthetic_data import write_registry

    root = tmp_path_factory.mktemp("univii-synthetic")
    return write_registry(root), root


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_tutorial_executes(path, synthetic_registry, tmp_path, monkeypatch):
    registry, root = synthetic_registry
    monkeypatch.setenv("UNIVII_DATASET_REGISTRY", str(registry))
    monkeypatch.setenv("UNIVII_DATA_DIR", str(root / "cache"))
    monkeypatch.setenv("MPLBACKEND", "Agg")

    nb = nbformat.read(path, as_version=4)
    idx = next((i for i, c in enumerate(nb.cells) if "parameters" in c.metadata.get("tags", [])), None)
    assert idx is not None, f"{path.name} needs a cell tagged 'parameters'"
    nb.cells.insert(idx + 1, nbformat.v4.new_code_cell(OVERRIDES))

    if path.name == "toolkit.ipynb":
        prepared = _write_toolkit_fixture(tmp_path / "prepared")
        nb.cells.insert(idx + 2, nbformat.v4.new_code_cell(f"""
# Offline AML/geometry smoke test; public atlas/image sections are omitted.
DATA_MODE = 'prepared'
DATA_DIR = Path({str(prepared)!r})
RESULTS = Path({str(tmp_path / 'results')!r})
N_EPOCHS = 2
BATCH_SIZE = 64
SEEDS = (0, 1)
MAX_GRAPH_CELLS = 300
RUN_GEOMETRY_SWEEP = True
RUN_MOSAIC = False
RUN_ANNOTATED_ATLAS = False
RUN_HISTOLOGY = False
import torch, urllib.request
torch.set_num_threads(1)
# Fail rather than silently downloading real data during this smoke test.
def _offline_urlopen(*args, **kwargs):
    raise AssertionError('Unexpected network access in toolkit CI')
urllib.request.urlopen = _offline_urlopen
"""))

    client = nbclient.NotebookClient(nb, timeout=900, kernel_name="python3",
                                     resources={"metadata": {"path": str(tmp_path)}})
    client.execute()
    if path.name == "toolkit.ipynb":
        import pandas as pd
        runs = pd.read_csv(tmp_path / "results" / "geometry_comparison.csv")
        assert set(runs["variant"]) == {"euclidean", "hyperbolic", "product"}
        assert len(runs) == 6  # Three geometries, two seeds each.
        for name in ["latent_anchored.npy", "sampled_graph_pseudotime.csv",
                     "hmm_states.png", "run_manifest.json"]:
            assert (tmp_path / "results" / name).is_file(), name


def _write_toolkit_fixture(folder):
    """Small, explicitly synthetic prepared AML input; never biological evidence.

    Exercise the notebook's prepared-data validation, training, anchoring,
    hierarchy, HMM, figures and all three latent geometries. Public download,
    mosaic/annotated atlas and histology sections are not exercised here.
    """
    import json
    import numpy as np
    import pandas as pd
    import anndata as ad

    folder.mkdir(parents=True)
    rng = np.random.default_rng(42)
    parents = {"HSC": None, "Prog": "HSC", "GMP": "Prog",
               "ProMono": "GMP", "Mono": "ProMono"}
    labels = np.tile(np.repeat(list(parents), 16), 3)
    sample = np.repeat(["ci_train", "ci_null", "ci_query"], 80)
    train = sample == "ci_train"
    null = sample == "ci_null"
    codes = pd.Categorical(labels, categories=list(parents)).codes
    centres = rng.normal(0, 0.25, (5, 16))
    X = (centres[codes] + rng.normal(0, 0.08, (len(labels), 16))).astype("float32")
    obs = pd.DataFrame({"label": labels, "sample": sample, "donor": sample,
                        "cohort": "synthetic_ci", "healthy_donor": train | null},
                       index=[f"ci_cell_{i}" for i in range(len(labels))])
    obs.to_csv(folder / "prepared_obs.csv")
    np.save(folder / "X_anchor.npy", X)
    np.savez_compressed(folder / "split_masks.npz", train=train, null=null)
    a = ad.AnnData(X, obs=obs)
    a.uns["tree_json"] = json.dumps(parents)
    a.write_h5ad(folder / "anchored_input.h5ad")
    return folder
