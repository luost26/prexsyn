# Data and Weights

## Automatic Download

PrexSyn can automatically download the preprocessed chemical space data and trained model weights upon first use.

You can trigger the download by running the following example:

```bash
python scripts/examples/projection.py --smiles "COc1ccc(-c2ccnc(Nc3ccccc3)n2)cc1"
```

The output should look like this:

```
[load_model] Model checkpoint not found locally at data/trained_models/v1_converted.ckpt, trying to download from https://huggingface.co/datasets/luost26/prexsyn-data/resolve/main/trained_models/v1_converted.ckpt...
Downloading: 100%|██████████████████████████| 2.36G/2.36G [00:04<00:00, 545MB/s]
[load_model] Chemical space data not found locally at data/chemical_spaces/enamine_rxn115, trying to download from https://huggingface.co/datasets/luost26/prexsyn-data/resolve/main/chemical_spaces...
Downloading primary_building_blocks: 100%|████| 326M/326M [00:00<00:00, 522MB/s]
Downloading primary_index: 100%|████████████| 36.1M/36.1M [00:00<00:00, 497MB/s]
Downloading reactions: 100%|█████████████████| 374k/374k [00:00<00:00, 85.6MB/s]
Downloading secondary_building_blocks: 100%|█| 1.44G/1.44G [00:02<00:00, 545MB/s
Downloading secondary_index: 100%|██████████| 32.4M/32.4M [00:00<00:00, 402MB/s]
[2025-12-02 10:56:20.399] [prexsyn_engine] [info] Loading building blocks from cache: data/chemical_spaces/enamine_rxn115/primary_building_blocks
[2025-12-02 10:56:23.609] [prexsyn_engine] [info] BuildingBlockList: 223243 building blocks loaded from cache
[2025-12-02 10:56:43.437] [prexsyn_engine] [info] ReactionList: Loading reactions from cache data/chemical_spaces/enamine_rxn115/reactions
[2025-12-02 10:56:43.446] [prexsyn_engine] [info] ReactionList: Loaded 115 reactions
Input: COc1ccc(-c2ccnc(Nc3ccccc3)n2)cc1
Target (Canonical SMILES): COc1ccc(-c2ccnc(Nc3ccccc3)n2)cc1
Results:
- SMILES: COc1ccc(-c2ccnc(Nc3ccccc3)n2)cc1
  Similarity: 1.0000
  Synthesis:
  - Reaction Index: 103
    Possible Products:
    - COc1ccc(-c2ccnc(Nc3ccccc3)n2)cc1
    Reactants:
    - SMILES: COc1ccc(-c2ccnc(Cl)n2)cc1
      Building Block Index: 198209
      ID: EN300-249263
    - SMILES: Nc1ccccc1
      Building Block Index: 95219
      ID: EN300-29997

...
```


## Manual Download

Our preprocessed chemical space data and trained model weights are hosted on Hugging Face: [https://huggingface.co/datasets/luost26/prexsyn-data/tree/main](https://huggingface.co/datasets/luost26/prexsyn-data/tree/main).

You can download them manually and place them in the [`data/`](https://github.com/luost26/PrexSyn/tree/main/data) directory of the project.