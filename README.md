# PARADE: a generative framework for enhanced cell-type specificity in rationally designed mRNAs

Matvei Khoroshkin, Arsenii Zinkevich, Elizaveta Aristova, _et al._, A generative framework for enhanced cell-type specificity in rationally designed mRNAs, biorXiv, 2024; doi: 10.1101/2024.12.31.630783v1

[[`Preprint`](https://www.biorxiv.org/content/10.1101/2024.12.31.630783v1)]


# Overview

mRNA delivery offers new opportunities for disease treatment by directing cells to produce therapeutic proteins. However, designing highly stable mRNAs with programmable cell type-specificity remains a challenge. Here, we present **PARADE** (Prediction And RAtional DEsign of mRNA UTRs), a generative AI framework for the design of untranslated RNA regions (UTRs) with tailored cell type-specific activity.


# System Requirements

## Hardware Requirements

The approach presented in PARADE requires only a regular PC with enough RAM to perform the operations defined by a user. To satisfy the minimal requirements and use the basic functions of PARADE (i.e. sequence activity prediction), only a computer with only about 4 GiB of RAM is necessary. However, if you would like to use PARADE generator, the generation results may depend on the number of sequences tested, therefore we recommend a computer with the following specs for optimal performance:

* RAM: 16+ GiB
* CPU: 8+ threads, 3.3+ GHz/core
* GPU: CUDA v. 11.8+, 8+ GiB RAM

## OS Requirements

The development version was tested on Linux operating system:

OS: Ubuntu 20.04 LTS (GNU/Linux 5.15.0-116-generic x86_64)

The code, however, should be compatible with Windows, Mac, and Linux operating systems.

# Usage

### Setting up dependencies

To get going with PARADE, you should perform the following steps:

1. Install the latest version of `conda` package manager. We recommend using the latest version of [Miniforge](https://github.com/conda-forge/miniforge), however, Miniconda or Anaconda should also work fine.
2. Clone the current repository and `cd` into it.
3. Create a conda environment from YAML: `conda env create -f environment.yml`
4. The created environment can then be used in any convenient way, e.g. in JupyterLab, Visual Studio Code, PyCharm and other IDEs.

### Demo usage

1. Download data from Zenodo (will be published along with the complete MPRA data) and put in into the `data` subdirectory.
2. Normalize the data with the scripts available in `data_preprocessing`.
3. For training PARADE predictor, use Jupyter Notebooks in `predictor/regression_multiple/regression_utr[53].ipynb`.
4. For training and using PARADE generator, use Jupyter Notebooks in `generator` subdirectory.
