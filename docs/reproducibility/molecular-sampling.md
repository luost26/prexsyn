# Molecular Sampling

## GuacaMol Benchmark

Install the [GuacaMol package](https://github.com/luost26/PrexSyn/tree/main/third_party) shipped with the PrexSyn repository:

```bash
pip install ./third_party
```

To reproduce the results in Table 2 of [the PrexSyn paper](https://arxiv.org/abs/2512.00384), run the following benchmark script:

```bash
python scripts/benchmarks/optim.py
```

Results will be saved in the `outputs/benchmarks/optim` directory. For example, the result for Amlodipine MPO can be found at `outputs/benchmarks/optim/amlodipine/log.txt`.

The output should be similar to the following, which corresponds to the results in Table 2:

```
==== Summary ====
Oracle: amlodipine
- Runs: 5
- AUC-Top10: 0.781 ± 0.023
```
