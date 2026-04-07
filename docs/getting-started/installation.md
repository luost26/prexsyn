
# Installation

## Clone the repository

```bash
git clone https://github.com/luost26/prexsyn.git
cd prexsyn
```

## UV (recommended)

[UV](https://docs.astral.sh/uv/) is a modern and fast Python environment manager. We highly recommend using UV for managing PrexSyn environments.

With UV, You don't need to explicitly create a virtual environment. Simply run any uv command in the PrexSyn directory, and UV will automatically create and manage an isolated environment for you according to the `pyproject.toml` and `uv.lock` files, for example:

```bash
uv run python
```

This will start a Python interpreter with the PrexSyn environment activated. You can try importing PrexSyn to verify the installation:

```python
import prexsyn
import prexsyn_engine
```

## Changing PyTorch CUDA version

By default, UV installs the latest PyTorch version (probably) with the latest CUDA version. Some users may want to use a different CUDA version for compatibility with their hardware or other software.

In such cases, please refer to the UV documentation for instructions on how to change the PyTorch CUDA version in your UV environment by editing the `pyproject.toml` file: [https://docs.astral.sh/uv/guides/integration/pytorch/#using-a-pytorch-index](https://docs.astral.sh/uv/guides/integration/pytorch/#using-a-pytorch-index)
