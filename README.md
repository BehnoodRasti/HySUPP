# HySUPP

An open-source **Hy**per**S**pectral **U**nmixing **P**ython **P**ackage

---

## Introduction

HySUPP is an open-source Python toolbox for hyperspectral unmixing practitioners.

## HySUPP key numbers

* 3 unmixing categories
* 20 unmixing methods
* 4 metrics
* 6 simulated datasets (*soon to be released*)

## License

HySUPP is distributed under MIT license.

## Citing HySUPP

**Coming soon!**

## Installation

### Using `conda`

We recommend using a `conda` virtual Python environment to install HySUPP.

In the following steps we will use `conda` to handle the Python distribution and `pip` to install the required Python packages.
If you do not have `conda`, please install it using [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

```
conda create --name hysupp python=3.10
```

Activate the new `conda` environment to install the Python packages.

```
conda activate hysupp
```

Clone the Github repository.

```
git clone git@github.com:BehnoodRasti/HySUPP.git
```

Change directory and install the required Python packages.

```
cd HySUPP && pip install -r requirements.txt
```

If you encounter any issue when installing `spams`, we recommend reading the Installation section [here](https://pypi.org/project/spams/).


## Getting started

This toolbox uses [MLXP](https://inria-thoth.github.io/mlxp/) to manage multiple experiments built on top of [hydra](https://hydra.cc/).

There are a few required parameters to define in order to run an experiment:
* `mode`: unmixing mode
* `data`: hyperspectral unmixing dataset
* `model`: unmixing model
* `noise.SNR`: input SNR (*optional*)

An example of a corresponding command line is simply:

```shell
python main.py mode=semi data=DC1 model=SUnCNN noise.SNR=30
```

## Data

### Data format

Datasets consist in a dedicated `.mat` file containing the following keys:

* `Y`: original hyperspectral image (dimension `L` x `N`)
* `E`: ground truth endmembers (dimension `L` x `p`)
* `A`: ground truth abundances (dimension `p` x `N`)
* `H`: HSI number of rows
* `W`: HSI number of columns
* `p`: number of endmembers
* `L`: number of channels
* `N`: number of pixels (`N` == `H`*`W`)

For sparse unmixing, a dictionary `D` containing `M` atoms is required.

* `D`: endmembers library (dimension `L` x `M`)
* `M`: number of atoms

We provide a utility script to turn any existing datasets composed of separated files to fit the required format used throughout the toolbox (See `utils/bundle_data.py`).
