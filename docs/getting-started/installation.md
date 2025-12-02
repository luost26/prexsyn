
# Installation

## Clone the Repository

```bash
git clone https://github.com/luost26/PrexSyn.git
cd PrexSyn
```

## Pixi (Recommended)

[Pixi](https://pixi.sh/) is highly recommended for managing PrexSyn environments. We used pixi for development. 

Pixi is a conda-based environment manager. It creates environments on a per-project basis, so you don't need to worry about conflicts with your existing conda/mamba installations.

To get started, please install Pixi by following the instructions on the [Pixi documentation](https://pixi.sh/latest/installation/).

Now, you are all set! To activate the environment, simply run the following command in the root directory of the project (it may take a while for the first time):

```bash
pixi shell
```

## Conda/Mamba + PyPI

Create and activate conda (mamba) environment:

```bash
conda create -n prexsyn
conda activate prexsyn
```

Install [PrexSyn Engine](https://github.com/luost26/prexsyn-engine). This package is only available via conda for now. RDKit will be installed as a dependency in this step.

```bash
conda install luost26::prexsyn-engine
```

Setup PrexSyn package. PyTorch and other dependencies will be installed in this step.

```bash
pip install -e .
```
