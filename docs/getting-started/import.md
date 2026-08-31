# Python API

## Load a released model

```python
import torch

from prexsyn.shortcuts import AllInOneLoader, MoleculeProjector

config_path = "./data/trained_models/enamine2310_rxn115_202511.yml"
device = "cuda" if torch.cuda.is_available() else "cpu"

loader = AllInOneLoader(config_path)
projector = MoleculeProjector(
    model=loader.model().to(device).eval(),
    detokenizer=loader.detokenizer(),
    descriptor="ecfp4",
    num_samples=16,
)
```

`loader.model()`, `loader.chemical_space()`, and `loader.detokenizer()` are cached after their first call. Missing released assets are downloaded from the URLs in the YAML.

## Project one molecule

`one()` accepts a SMILES string, an RDKit `Mol`, or a `prexsyn_engine.chemistry.Molecule`:

```python
result = projector.one("COc1ccc(-c2ccnc(Nc3ccccc3)n2)cc1")

for i, item in enumerate(result.items[:3]):
    print(item.molecule.smiles(), item.similarity)
    print(item.get_tree())

    # Graphviz is required for image rendering.
    img = item.get_image()
    img.save(f"output_{i}.png")
    img.close()
```

Products are sorted by descending Tanimoto similarity. `result.best()` returns the highest-ranked item or `None`; `result.best_similarity()` returns `0.0` when no product was generated. Timing is available as `result.time.model`, `result.time.detok`, and `result.time.total`.

## Project a batch

```python
batch = projector.many(
    [
        "COc1ccc(-c2ccnc(Nc3ccccc3)n2)cc1",
        "COc1ccc(-c2ccnc(Cl)n2)cc1",
    ]
)

for target_result in batch.results:
    best = target_result.best()
    if best is not None:
        print(best.molecule.smiles(), best.similarity)
```

`many()` returns one result per input molecule. Sampling is internally chunked according to `batch_size_limit`, which defaults to 64.

## Generate from fingerprints

`desc()` accepts a two-dimensional NumPy array or PyTorch tensor with shape `(batch_size, descriptor_size)`. Its descriptor must match the name passed to `MoleculeProjector`. For a CUDA model, pass a tensor on the model device.

```python
import torch

from prexsyn_engine.chemistry import Molecule

fingerprint_array = projector.descriptor_function(Molecule.from_smiles("CCO"))[None, :]
fingerprint = torch.from_numpy(fingerprint_array).to(projector.model.device)
result = projector.desc(fingerprint).results[0]
```

The released model supports `ecfp4` and `fcfp4`.
