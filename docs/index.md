# PrexSyn


## Introduction

PrexSyn is an efficient, accurate, and programmable framework for synthesizable molecular design.
It is based on a decoder-only transformer architecture that autoregressively generates *postfix notations of
synthesis*[^chemprojector] (a molecular representation based on chemical reactions and purchasable building blocks) conditioned on molecular descriptors.

[^chemprojector]: Projecting Molecules into Synthesizable Chemical Spaces. [https://arxiv.org/abs/2406.04628](https://arxiv.org/abs/2406.04628)

PrexSyn is trained on a billion-scale datastream of postfix notations paired with molecular descriptors using only two GPUs and 32 CPU cores in two days. This is made possible by [PrexSyn Engine](https://github.com/luost26/prexsyn-engine), a real-time, high-throughput C++-based data generation pipeline.


## Capabilities

| Capability | Input | Output |
| :---: | :---: | :---: |
| **Chemical space projection** | ![](imgs/proj-in.png) <br/> Graph / SMILES | ![](imgs/proj-out.png) <br/> |
| **Fingerprint/descriptor based generation** | ![](imgs/fp-in.png) <br/> Fingerprint / descriptor | ![](imgs/proj-out.png) <br/>  |
| **Molecular sampling** | ![](imgs/sample-in.png) <br/> Scoring functions | ![](imgs/sample-out.png) <br/> |

## Performance

| Capability |     Result      |
| :--- | :------------: |
| Record-high accuracy and speed in chemical space projection and fingerprint/descriptor-based generation | ![Performance comparison](imgs/projection-compare.png) |
| Record-high sample efficiency in molecular sampling against scoring functions | ![Molecular Sampling Performance](imgs/sampling-compare-1.png) |


## Resources

### Repositories

- **PrexSyn**: [https://github.com/luost26/prexsyn](https://github.com/luost26/prexsyn)
- **PrexSyn Engine**: The C++ backend that provides a high-throughput training data pipeline and a fast synthesis detokenizer. [https://github.com/luost26/prexsyn-engine](https://github.com/luost26/prexsyn-engine)
- **Data and Weights**: Preprocessed chemical space data and trained model weights hosted on Hugging Face. [https://huggingface.co/datasets/luost26/prexsyn-data/tree/main](https://huggingface.co/datasets/luost26/prexsyn-data/tree/main)

### Papers and Documentation

- **PrexSyn Paper**: Efficient and Programmable Exploration of Synthesizable Chemical Space. [https://arxiv.org/abs/2512.00384](https://arxiv.org/abs/2512.00384)
- **PrexSyn Documentation**: [https://prexsyn.readthedocs.io](https://prexsyn.readthedocs.io)

### Miscellaneous

- **MIT Coley Research Group**: [https://coley.mit.edu/](https://coley.mit.edu/)

## Citation

```bibtex
@article{luo2025prexsyn,
  title   = {Efficient and Programmable Exploration of Synthesizable Chemical Space},
  author  = {Shitong Luo and Connor W. Coley},
  year    = {2025},
  journal = {arXiv preprint arXiv: 2512.00384}
}
```
