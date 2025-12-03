# Molecular Sampling

## Prerequisite

Oracle functions used in the benchmarks require additional dependencies (GuacaMol, PyTorch Geometric, etc.). Please follow the instructions below to make sure these dependencies are installed properly.

If you are using Pixi, make sure to activate the `dev` environment:

```bash
pixi shell -e dev
```

If you installed the environment manually using conda/mamba and pypi, make sure you have the `eval` extra dependencies installed ([refer to the installation instructions](../getting-started/installation.md#condamamba-pypi)).

```bash
pip install -e .[eval]
```

## GuacaMol Benchmark

To reproduce the results in Table 2 of [the PrexSyn paper](https://arxiv.org/abs/2512.00384), run the following benchmark script:

```bash
python scripts/benchmarks/optim.py
```

This script runs molecular optimization tasks sequentially and each task is repeated 5 times.
A subset of tasks can be specified using the `-t` or `--task` argument, and the full list of available tasks can be found in the script file.
For example, to run only the Amlodipine MPO and Celecoxib Rediscovery tasks, use:

```bash
python scripts/benchmarks/optim.py -t amlodipine -t celecoxib_rediscovery
```

Results will be saved in the `outputs/benchmarks/optim` directory. For example, the result for Amlodipine MPO can be found at `outputs/benchmarks/optim/amlodipine/log.txt`. The log file should be similar to the following, which corresponds to the results in Table 2:

```
==== Summary ====
Oracle: amlodipine
- Runs: 5
- AUC-Top10: 0.781 ± 0.023
```

## sEH Proxy

```
pixi run -e dev python scripts/benchmarks/optim.py -t sEH_proxy
```
