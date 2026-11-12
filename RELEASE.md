# Releasing UniVII

Tags use the form `vX.Y.Z` (for example `v0.2.1`). The version must match in `pyproject.toml`, `univii/__init__.py`, `CITATION.cff`, `conda.recipe/meta.yaml`, and the top entry of `CHANGELOG.md`.

## Naming, in one place

| Thing | Value |
| --- | --- |
| Repository | `Ashford-A/UniVII` |
| Distribution name (PyPI, conda-forge) | `univii` |
| Python import and command-line entry point | `univii` |
| Published predecessor, not touched by anything here | `univi` (PyPI, conda-forge, `Ashford-A/UniVI`) |

When this file was written (11 November 2026) the name `univii` was unregistered on both PyPI and conda-forge. Neither can be reserved in advance: the first upload claims it. Check again before releasing.

## First push to the repository

```bash
cd UniVII
git init -b main                                   # skip if you cloned the empty repository
git remote add origin https://github.com/Ashford-A/UniVII.git
git add -A
git commit -m "UniVII 0.2.1: UniVI 1.2.1 plus latent geometry, conv encoders and spatial tools"
git push -u origin main
```

The checkout is about 260 MB, mostly the archived notebooks in `notebooks/`; the largest single file is 31 MB, under GitHub's 50 MB warning and 100 MB limit. To keep the repository small, leave `notebooks/` out and link to the UniVI repository for them.

GitHub Actions then run the tests, documentation build and packaging check on every push to `main` and on pull requests (`.github/workflows/docs.yml`). No secrets are needed.

## One-time setup

- **PyPI**: an API token from <https://pypi.org/manage/account/token/>, saved in `~/.pypirc`. Use an account-wide token for the first upload (a project-scoped token cannot exist before the project does), then replace it with one scoped to `univii`.

  ```ini
  [pypi]
  username = __token__
  password = pypi-...
  ```

- **GitHub CLI** (optional, for creating releases from the terminal): `gh auth login`.
- **Zenodo software DOIs** (optional): sign in at <https://zenodo.org> with GitHub, open *GitHub* in your account settings, and switch the `Ashford-A/UniVII` repository on. This must be done *before* the release is created.
- **Read the Docs** (optional): import the repository at <https://readthedocs.org>; `.readthedocs.yaml` is ready. Set the default version to `stable`.

## 1. Optional: publish more datasets

The export code for each paper dataset is in `docs/reproducibility/building_datasets.md`. For files you already have on disk:

```bash
python scripts/prepare_zenodo_release.py prepare hao_citeseq_pbmc \
    --file rna=/path/Hao_RNA_data.h5ad --file adt=/path/Hao_ADT_data.h5ad \
    --labels celltype.l1 celltype.l2 celltype.l3 --outdir zenodo/hao_citeseq_pbmc
python scripts/zenodo_upload.py zenodo/hao_citeseq_pbmc --creator "Ashford, Andrew J."   # then publish the draft on zenodo.org
python scripts/prepare_zenodo_release.py register zenodo/hao_citeseq_pbmc --record-id <RECORD_ID>
python -c "import univii.datasets as u; print(u.list_datasets())"
```

Registry changes ship with the package, so do this before building the release.

## 2. Optional: refresh tutorial outputs

The notebooks store their outputs. After changing a tutorial, execute it on real data (GPU recommended) and review the results:

```bash
python scripts/execute_tutorials.py            # or: python scripts/execute_tutorials.py quickstart
git diff --stat docs/tutorials
```

## 3. Check everything locally

```bash
python -m pip install -e ".[tutorials,test,docs]" build twine
python -m pytest -q                                           # unit tests + tutorial notebooks
python -m sphinx -W --keep-going -b html docs docs/_build/html
rm -rf dist && python -m build --outdir dist
python -m twine check --strict dist/*
python scripts/verify_release.py dist
```

## 4. Commit and push to `main`

```bash
git checkout main
git pull
git add -A
git status                      # review the file list
git commit -m "Release v0.2.1"
git push origin main
```

Wait for the GitHub Actions checks to pass.

## 5. Tag

```bash
git tag -a v0.2.1 -m "Release v0.2.1"
git push origin v0.2.1
```

## 6. Upload to PyPI

Upload the files built and checked in step 3 (rebuild them if anything changed since). The first upload registers the name `univii` to your account.

```bash
python -m twine upload dist/univii-0.2.1.tar.gz dist/univii-0.2.1-py3-none-any.whl
```

Published files cannot be replaced; a mistake needs a new version number. If you want a rehearsal, upload to <https://test.pypi.org> first with `twine upload --repository testpypi dist/*`.

Then switch the install commands in the README, docs and notebook Colab cells from the GitHub URL to the PyPI name, and commit:

```bash
python scripts/switch_install_commands.py --dry-run      # lists the files
python scripts/switch_install_commands.py
```

## 7. Create the GitHub release

```bash
awk '/^## 0.2.1/{f=1; next} /^## /{f=0} f' CHANGELOG.md > /tmp/univii-notes.md
gh release create v0.2.1 dist/univii-0.2.1.tar.gz dist/univii-0.2.1-py3-none-any.whl \
    --title "v0.2.1" --notes-file /tmp/univii-notes.md
```

or use *Releases → Draft a new release* on GitHub with the same tag, title, notes, and files. With the Zenodo integration on, the release is archived and receives a DOI within a few minutes.

## 8. Verify the PyPI package

```bash
python -m venv /tmp/univii-check && source /tmp/univii-check/bin/activate
python -m pip install "univii==0.2.1"
python -c "import univii, univii.datasets as u; print(univii.__version__); print(u.list_datasets())"
deactivate
```

## 9. conda-forge

A new package enters conda-forge through a pull request to <https://github.com/conda-forge/staged-recipes>. It needs the PyPI sdist from step 6.

1. Fork and clone `staged-recipes`, and create a branch.
2. Generate a recipe from the published sdist: `pip install grayskull && grayskull pypi univii`. This writes `univii/meta.yaml` with the version, the sdist URL and its SHA-256.
3. Move it to `recipes/univii/meta.yaml` and compare it with `conda.recipe/meta.yaml` in this repository, which lists the tested `run` requirements, the `noarch: python` build, the entry point, and the import tests. Note the conda names `pytorch` and `python-igraph`. Add yourself under `extra: recipe-maintainers`.
4. Open the pull request and follow the checklist in its template. Reviewers usually respond within days.
5. After it is merged, conda-forge creates `conda-forge/univii-feedstock` and the package appears on the channel.

For later versions the conda-forge bot opens a pull request on the feedstock after each PyPI release. If it does not, `python scripts/update_conda_recipe.py --feedstock /path/to/univii-feedstock --version X.Y.Z` updates the recipe from the published sdist; that script was carried over from UniVI and has not yet been run for `univii`.

Verify:

```bash
conda create -n univii-check -c conda-forge "univii=0.2.1" && conda activate univii-check
python -c "import univii; print(univii.__version__)"
```

## 10. Documentation

If you set up Read the Docs, pushing to `main` rebuilds `latest` and pushing the tag updates `stable`. Then replace the links to the `docs/` folder in `README.md`, `pyproject.toml` (`Documentation`), `CITATION.cff` (`url`) and `.github/ISSUE_TEMPLATE/config.yml` with the site address.

## 11. Afterwards

- If Zenodo minted a software DOI, add its badge to `README.md` and the DOI to `CITATION.cff` (as `doi:`).
- Add PyPI and conda-forge badges to `README.md`.
- On the GitHub repository page, add topics (for example `single-cell`, `multi-omics`, `variational-autoencoder`, `hyperbolic-geometry`, `spatial-transcriptomics`, `pytorch`, `anndata`).
- For the next development cycle, add an `## Unreleased` section at the top of `CHANGELOG.md`.
