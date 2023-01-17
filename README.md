# HySUPP

An Open-Source Hyperspectral Unmixing Python Package

---

## Data

### Data format

Datasets consists in a dedicated `.mat` file containing the following keys:

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

We provide a utility script to turn an existing datasets composed of separated files to fit the required format used throughout the toolbox.


### Simulated


### Real
