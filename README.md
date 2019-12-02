## Introduction
A Keras  implementation of trVAE. trVAE is a deep generative model which learns mapping between multiple different styles (conditions). trVAE can be used for style transfer on images, predicting single-cell perturbations responses and batch removal.
## Getting Started

## Installation

### Installation with pip
To install the latest version from PyPI, simply use the following bash script:
```bash
pip install trvae_pytorch
```
or install the development version via pip: 
```bash
pip install git+https://github.com/theislab/trvae.git
```

or you can first install flit and clone this repository:
```bash
pip install flit
git clone https://github.com/theislab/trVAE
cd trVAE
flit install
```

## Examples

## Reproducing paper results:
In order to reproduce paper results visit [here](https://github.com/Naghipourfar/trVAE_reproducibility).