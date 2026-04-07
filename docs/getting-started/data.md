# Data and model weights

## Automatic download

PrexSyn will automatically download the preprocessed chemical space data and trained model checkpoints upon first use.

You can trigger the download by running the following example:

```bash
uv run python scripts/examples/projection.py --smiles "COc1ccc(-c2ccnc(Nc3ccccc3)n2)cc1"
```

The output should look like this:

```
[PrexSyn All-in-One Loader]
- Model name: Enamine US Oct 2023 + Rxn115
- Config path: data/trained_models/enamine2310_rxn115_202511.yml
- Checkpoint path: data/trained_models/enamine2310_rxn115_202511.ckpt
- Chemspace path: data/chemical_spaces/enamine2310_rxn115.chemspace
- Description:
  > Enamine Rush Delivery Building Blocks (US) Oct 2023, initially used in ChemProjector model.
  > Rxn115 (115 reactions) template set.
  > Model released in November 2025. This is the model used in the paper.

Downloading chemical space: 100%|███████████| 1.08G/1.08G [00:01<00:00, 896MB/s]
Downloading checkpoint: 100%|███████████████| 2.29G/2.29G [00:02<00:00, 773MB/s]
[2026-04-07 10:11:07.517] [prexsyn] [info] Deserializing chemical space...
[2026-04-07 10:11:07.517] [prexsyn] [info]  - Serialization version: 1
[2026-04-07 10:11:07.517] [prexsyn] [info]  - Sizes: 223243 building blocks, 115 reactions, 349117 intermediates
[2026-04-07 10:11:30.750] [prexsyn] [info]  - Building block library deserialized. Size: 223243
[2026-04-07 10:11:30.759] [prexsyn] [info]  - Reaction library deserialized. Size: 115
[2026-04-07 10:12:06.338] [prexsyn] [info]  - Intermediate library deserialized. Size: 349117
[2026-04-07 10:12:06.448] [prexsyn] [info]  - Reactant-building block mapping deserialized. Matches: 4509428
[2026-04-07 10:12:06.617] [prexsyn] [info]  - Reactant-intermediate mapping deserialized. Matches: 8821093
- Target: COc1ccc(-c2ccnc(Nc3ccccc3)n2)cc1
  Similarity: 1.0
  SMILES: COc1ccc(-c2ccnc(Nc3ccccc3)n2)cc1
  Reaction: RXN_104
  Precursors:
  - R0:
      SMILES: Brc1ccccc1
      BuildingBlock: EN300-19359
    R1:
      SMILES: COc1ccc(-c2ccnc(N)n2)cc1
      BuildingBlock: EN300-186470

...
```


## Manual download (not recommended)

The preprocessed chemical space data and trained model checkpoints are hosted on Hugging Face: [https://huggingface.co/datasets/luost26/prexsyn-data/tree/main](https://huggingface.co/datasets/luost26/prexsyn-data/tree/main). You can download them manually and place them in the [`data/`](https://github.com/luost26/prexsyn/tree/main/data) directory of the project.

This is **not recommended** since the data repository contains old versions of the data and checkpoint files that are no longer used by the current codebase. The automatic download mechanism will ensure that you get the correct and up-to-date files.
