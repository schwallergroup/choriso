<p align="center">
    <a href="https://github.com/schwallergroup/choriso/actions/workflows/tests.yml">
        <img alt="Tests" src="https://github.com/schwallergroup/choriso/workflows/Tests/badge.svg" />
    </a>
    <a href="https://openreview.net/forum?id=yLydB04RxR">
        <img alt="DOI" src="https://zenodo.org/badge/DOI/10.48550/arXiv.2304.05376.svg" />
    </a>
    <a href='https://choriso.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/choriso/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="https://github.com/cthoyt/cookiecutter-python-package">
        <img alt="Cookiecutter template from @cthoyt" src="https://img.shields.io/badge/Cookiecutter-snekpack-blue" /> 
    </a>
    <a href='https://github.com/psf/black'>
        <img src='https://img.shields.io/badge/code%20style-black-000000.svg' alt='Code style: black' />
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
This repo provides all the code used for dataset curation, splitting and analysis reported in the paper, as well as the metrics for evaluation of models.

---

## 🚀 Installation

First clone this repo:

```bash
git clone https://github.com/schwallergroup/choriso.git
cd choriso
```

Set up and activate the environment:

```bash
conda env create -f environment.yml
conda activate choriso
pip install rxnmapper --no-deps
```

## 🔥 Quick start
To download the preprocessed dataset and split it to obtain the corresponding train, validation and test sets, run the following command:
```bash
choriso --download_processed \
	--run split
```

After executing `some command from choriso-models`, run the analysis of your model's results using:

```bash
analyse --results_folders='path/to/results/folder' 
```

Results will be stored in the same directory as `benchmarking-results`. 

---

<details>
  <summary>Advanced usage</summary>

## 🧠 Advanced usage
Using this repo lets you reproduce the results in the paper using different flags and modes.

### 📥 Download preprocessed dataset:

```bash
choriso --download_processed \
	--out-dir data/processed/
```

### :gear: Preprocessing

Get the raw datasets (CJHIF, USPTO) and preprocess. The `--upsto` command runs the same processing pipeline for the raw USPTO data:

**NOTE: To run the `clean` step you need to have NameRXN (v3.4.0) installed.**

```bash
choriso --download_raw \
	--uspto \
    	--data-dir=data/raw/ \
	--out-dir data/processed/ \
	--run clean \
	--run atom_map
```

### :mag: Stereo check

For this step you need to have either downloaded the preprocessed dataset, or running the preprocessing pipeline. The step checks reactions where there are stereochemistry issues and corrects the dataset.

```
choriso --run analysis
```

### :heavy_division_sign: Splitting
In the paper, we describe a splitting scheme to obtain test splits by product, product molecular weight and random. When doing the splitting, all the testing reactions go to a single test set file, with the `split` column indicating to which split they belong. To run the splitting:

```bash
choriso --run split 
```

By default, reactions with products below 150 a.m.u go to the low MW set and reactions with products above 700 a.m.u go to the high MW set. These values can be modified and adapted to your preferences. For example, to create a split to test on low MW with a threshold of 100 a.m.u., and another split on high MW with threshold of 750 a.m.u. run:

```bash
choriso --run split \
	--low_mw=150
	--high_mw=700
```

You can optionally augment the SMILES to double the size of the training set:
```bash
choriso --run split \
	--augment
```
By default, the splitting will be done on the choriso dataset, which is called `choriso.tsv`. If you want to split a different dataset, you can specify the path to the dataset using the `--split_file_name` option. For example, to split the USPTO dataset, run:
```bash
choriso --run split \
    --split_file_name=uspto.tsv
```
---

## 📊 Logging

By default the execution of any step will store all results locally.

Optionally, you can log all results from the preprocessing to W&B using the `wandb_log` flag at any step.

As an example
```bash
choriso --run clean \
	--wandb_log
```
will execute the analysis step and upload all results (plots, metrics) to W&B.

##  📈 Metrics
You can also use the implemented metrics from the paper to evaluate your own results. We have adapted the evaluation pipeline to the files from the [benchmarking repo](https://github.com/schwallergroup/choriso-models). As an example:
```
analyse --results_folders='OpenNMT_Transformer'
```
This will launch the analysis on all the files of the `OpenNMT_Transformer` folder. The output files should have the same structure as the one included on the benchmarking repo as an example. The program computes the chemistry metrics by default, which require the presence of a template with radius=0 and a template with radius=1 (these columns should be present on the test set file). 

### Flagging individual reactions
You can use the metrics functions to check if a specific reaction is regio or stereoselective. As an example:

```python
from choriso.metrics.selectivity import flag_regio_problem, flag_stereo_problem

regio_rxn = 'BrCc1ccccc1.C1CCOC1.C=CC(O)CO.[H-].[Na+]>>C=CC(O)COCc1ccccc1'
stereo_rxn = 'C=C(NC(C)=O)c1ccc(OC)cc1.ClCCl.[H][H].[Rh+]>>COc1ccc([C@@H](C)NC(C)=O)cc1'

print(flag_regio_problem(regio_rxn))
print(flag_stereo_problem(stereo_rxn))

```
The output will display the flagging labels 

```python
True
True
```
</details>


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
