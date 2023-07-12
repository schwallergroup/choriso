<p align="center">
    <a href="https://github.com/schwallergroup/choriso-fr/actions/workflows/tests.yml">
        <img alt="Tests" src="https://github.com/schwallergroup/choriso-fr/workflows/Tests/badge.svg" />
    </a>
    <a href="https://doi.org/10.48550/arXiv.2304.05376">
        <img alt="DOI" src="https://zenodo.org/badge/DOI/10.48550/arXiv.2304.05376.svg" />
    </a>
    <a href="https://pypi.org/project/choriso">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/choriso" />
    </a>
    <a href="https://pypi.org/project/choriso">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/choriso" />
    </a>
    <a href="https://github.com/schwallergroup/choriso/blob/main/LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/choriso" />
    </a>
    <a href='https://choriso.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/choriso/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="https://codecov.io/gh/schwallergroup/choriso/branch/main">
        <img src="https://codecov.io/gh/schwallergroup/choriso/branch/main/graph/badge.svg" alt="Codecov status" />
    </a>  
    <a href="https://github.com/cthoyt/cookiecutter-python-package">
        <img alt="Cookiecutter template from @cthoyt" src="https://img.shields.io/badge/Cookiecutter-snekpack-blue" /> 
    </a>
    <a href='https://github.com/psf/black'>
        <img src='https://img.shields.io/badge/code%20style-black-000000.svg' alt='Code style: black' />
    </a>
    <a href="https://github.com/schwallergroup/choriso/blob/main/.github/CODE_OF_CONDUCT.md">
        <img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Contributor Covenant"/>
    </a>
</p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/choriso_logo_dark.png" width='100%'>
  <source media="(prefers-color-scheme: light)" srcset="./assets/choriso_logo_light.png" width='100%'>
  <img alt="Choriso logo" src="/assets/" width="100%">
</picture>

<br>
<br>


**ChORISO** (**Ch**emical **O**rganic **R**eact**I**on **S**miles **O**mnibus) is a benchmarking suite for reaction prediction machine learning models.

We release:

- A highly curated dataset of academic chemical reactions ([download ChORISO and splits](https://drive.switch.ch/index.php/s/VaSVBCiXrmzYzGD))
- A suite of standardized evaluation metrics
- A compilation of models for reaction prediction ([choriso-models](https://github.com/schwallergroup/choriso-models))


It is derived from the [CJHIF dataset](https://ieeexplore.ieee.org/document/9440947/footnotes#footnotes-id-fn7).
This repo provides all the code use for dataset curation, splitting and analysis reported in the paper, as well as the metrics for evaluation of models.

---

## 🚀 Installation

First clone this repo:

```bash
git clone https://github.com/schwallergroup/choriso.git
cd choriso
```

Set up and activate the environment:

```
conda env create -f environment.yml
conda activate choriso
pip install rxnmapper --no-deps
```

## 🔥 Quick start
To download the preprocessed dataset and split it randomly, run the following command:
```
choriso --download_processed \
	--run split \
	--split_mode random
```

---

##  :brain: Advanced usage
Using this repo lets you reproduce the results in the paper using different flags and modes.

### 📥 Download preprocessed dataset:

```
choriso --download_processed \
	--out-dir data/processed/
```

### :gear: Preprocessing:

Get the raw datasets (CJHIF, USPTO) and preprocess:

**NOTE: To run the `clean` step you need to have Leadmine (v3.18.1) and NameRXN (v3.4.0) installed.**

```
choriso --download_raw \
	--uspto \
    	--data-dir=data/raw/ \
	--out-dir data/processed/ \
	--run clean \
	--run atom_map
```

### :mag: Data analysis:

For this step you need to have either downloaded the preprocessed dataset, or running the preprocessing pipeline.

```
choriso --run analysis
```

### :heavy_division_sign: Splitting
In the paper, we describe 3 data splits which can be obtained using the flag `--run split` specifying a `--split_mode`
- random split
```
choriso --run split \
	--split_mode random
```
- split by product
```
choriso --run split \
	--split_mode products
```
- split by Molecular Weight:
  - test on high MW
  - test on low MW

   For example, to create a split by MW, testing on low MW with a threshold of 150 a.m.u., and another split on high MW with threshold of 700 a.m.u. run
```
choriso --run split \
	--split_mode mw \
	--low_mw=150
	--high_mw=700
```

You can optionally augment the SMILES to double the size of the trainig set:
```
choriso --run split \
	--split_mode products \
	--augment
```

---

## 📊 Logging

By default the execution of any step will store all results locally.

Optionally, you can log all results from the preprocessing and analysis to W&B using the `wandb_log` flag at any step.

As an example
```
choriso --run analysis \
	--wandb_log
```
will execute the analysis step and upload all results (plots, metrics) to W&B.

## 👐 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated. See
[CONTRIBUTING.md](https://github.com/schwallergroup/choriso/blob/master/.github/CONTRIBUTING.md) for more information on getting involved.

## 👋 Attribution

### ⚖️ License

The code in this package is licensed under the MIT License.

<!--
### 📖 Citation

Citation goes here!
-->

<!--
### 🎁 Support

This project has been supported by the following organizations (in alphabetical order):

- [Harvard Program in Therapeutic Science - Laboratory of Systems Pharmacology](https://hits.harvard.edu/the-program/laboratory-of-systems-pharmacology/)

-->

<!--
### 💰 Funding

This project has been supported by the following grants:

| Funding Body                                             | Program                                                                                                                       | Grant           |
|----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|-----------------|
| DARPA                                                    | [Automating Scientific Knowledge Extraction (ASKE)](https://www.darpa.mil/program/automating-scientific-knowledge-extraction) | HR00111990009   |
-->

### 🍪 Cookiecutter

This package was created with [@audreyfeldroy](https://github.com/audreyfeldroy)'s
[cookiecutter](https://github.com/cookiecutter/cookiecutter) package using [@cthoyt](https://github.com/cthoyt)'s
[cookiecutter-snekpack](https://github.com/cthoyt/cookiecutter-snekpack) template.

## 🛠️ For Developers

<details>
  <summary>See developer instructions</summary>

The final section of the README is for if you want to get involved by making a code contribution.

### Development Installation

To install in development mode, use the following:

```bash
$ git clone git+https://github.com/schwallergroup/choriso.git
$ cd choriso
$ pip install -e .
```

### 🥼 Testing

After cloning the repository and installing `tox` with `pip install tox`, the unit tests in the `tests/` folder can be
run reproducibly with:

```shell
$ tox
```

Additionally, these tests are automatically re-run with each commit in a [GitHub Action](https://github.com/schwallergroup/choriso/actions?query=workflow%3ATests).

### 📖 Building the Documentation

The documentation can be built locally using the following:

```shell
$ git clone git+https://github.com/schwallergroup/choriso.git
$ cd choriso
$ tox -e docs
$ open docs/build/html/index.html
``` 

The documentation automatically installs the package as well as the `docs`
extra specified in the [`setup.cfg`](setup.cfg). `sphinx` plugins
like `texext` can be added there. Additionally, they need to be added to the
`extensions` list in [`docs/source/conf.py`](docs/source/conf.py).

### 📦 Making a Release

After installing the package in development mode and installing
`tox` with `pip install tox`, the commands for making a new release are contained within the `finish` environment
in `tox.ini`. Run the following from the shell:

```shell
$ tox -e finish
```

This script does the following:

1. Uses [Bump2Version](https://github.com/c4urself/bump2version) to switch the version number in the `setup.cfg`,
   `src/choriso/version.py`, and [`docs/source/conf.py`](docs/source/conf.py) to not have the `-dev` suffix
2. Packages the code in both a tar archive and a wheel using [`build`](https://github.com/pypa/build)
3. Uploads to PyPI using [`twine`](https://github.com/pypa/twine). Be sure to have a `.pypirc` file configured to avoid the need for manual input at this
   step
4. Push to GitHub. You'll need to make a release going with the commit where the version was bumped.
5. Bump the version to the next patch. If you made big changes and want to bump the version by minor, you can
   use `tox -e bumpversion -- minor` after.
</details>
