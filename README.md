# PrexSyn

[![arXiv](https://img.shields.io/badge/arXiv-2512.00384-b31b1b.svg)](https://arxiv.org/abs/2512.00384)
[![readthedocs](https://app.readthedocs.org/projects/prexsyn/badge/?version=v0.1.0)](https://prexsyn.readthedocs.io)
[![data](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-blue)](https://huggingface.co/datasets/luost26/prexsyn-data/tree/main)

PrexSyn is an efficient, accurate, and programmable framework for exploring synthesizable chemical space.

PrexSyn is based on a decoder-only transformer architecture that autoregressively generates [*postfix notations of
synthesis*](https://arxiv.org/abs/2406.04628) (a molecular representation based on chemical reactions and purchasable building blocks) conditioned on molecular descriptors.


PrexSyn is trained on a billion-scale datastream of postfix notations paired with molecular descriptors using only two GPUs and 32 CPU cores in two days. This is made possible by [PrexSyn Engine](https://github.com/luost26/prexsyn-engine), a real-time, high-throughput C++-based data generation pipeline.


[[Documentation]](https://prexsyn.readthedocs.io)
[[Paper]](https://arxiv.org/abs/2512.00384)
[[PrexSyn Engine]](https://github.com/luost26/prexsyn-engine)
[[Data and Model Weights]](https://huggingface.co/datasets/luost26/prexsyn-data/tree/main)

## Capabilities

| Capability | Input | Output | Performance |
| :---: | :---: | :---: | :---: |
| **Chemical space projection** | ![](docs/imgs/proj-in.png) | ![](docs/imgs/proj-out.png) | ![](docs/imgs/projection-compare.png) |
| **Fingerprint/descriptor based generation** | ![](docs/imgs/fp-in.png) | ![](docs/imgs/proj-out.png) | ![](docs/imgs/projection-compare.png) |
| **Molecular sampling** | ![](docs/imgs/sample-in.png) | ![](docs/imgs/sample-out.png) | ![](docs/imgs/sampling-compare-1.png) |


## Usage

Please refer to the [documentation](https://prexsyn.readthedocs.io) for detailed usage instructions on installation, data setup, reproducibility, and customization.


## Upgrade to PrexSyn v1

We have substantially refactored both the PrexSyn codebase and the [PrexSyn Engine](https://github.com/luost26/prexsyn-engine) to improve usability, performance, and extensibility. Key updates include:

- ✅ **Improved usability:** PrexSyn Engine is now available via PyPI.
- ✅ **Higher performance and stability:** Data generation is now approximately 2× faster than reported in the paper, with improved robustness thanks to a more reliable compilation pipeline.
- ✅ **Greater flexibility:** Chemical space definitions and training workflows are now easier to customize for new use cases.
- ✅ **Cleaner interfaces:** Simplified and more consistent APIs for projection, fingerprint/descriptor-based generation, and sampling.
- 🚧 Migrate molecular sampling benchmarks to the new codebase.

**Note:** Some features described in the original paper (mostly property-based queries) are no longer supported in the current version of PrexSyn. If you need these features, please use the [v0 branch](https://github.com/luost26/prexsyn/tree/dev-v0).


## Citation

```bibtex
@article{luo2025prexsyn,
  title   = {Efficient and Programmable Exploration of Synthesizable Chemical Space},
  author  = {Shitong Luo and Connor W. Coley},
  year    = {2025},
  journal = {arXiv preprint arXiv: 2512.00384}
}
```
