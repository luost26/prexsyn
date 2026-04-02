from pathlib import Path

import click
import torch

from prexsyn.factory import Config, get_model

prefix_mappings = {
    "synthesis_embedder.": "synthesis_embedder.",
    "property_embedders.ecfp4.": "descriptor_embedders.ecfp4.",
    "property_embedders.fcfp4.": "descriptor_embedders.fcfp4.",
    "transformer_layers.": "transformer_layers.",
    "synthesis_output.": "synthesis_output.",
}


@click.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("-i", "--input-path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("-o", "--output-path", type=click.Path(path_type=Path), required=True)
def main(config_path: Path, input_path: Path, output_path: Path):
    config = Config.from_yaml(config_path)

    state_dict_old: dict[str, torch.Tensor] = torch.load(input_path)

    model = get_model(config)
    new_state_dict = model.state_dict()

    for old_key, value in state_dict_old.items():
        new_key = None
        for old_prefix, new_prefix in prefix_mappings.items():
            if old_key.startswith(old_prefix):
                new_key = new_prefix + old_key[len(old_prefix) :]
                break
        if new_key is not None and new_key in new_state_dict:
            old_shape = value.shape
            new_shape = new_state_dict[new_key].shape
            print(f"Mapping {old_key} to {new_key}, shape {old_shape} -> {new_shape}")
            new_state_dict[new_key].copy_(value)
        else:
            print(f"Warning: no mapping found for key {old_key}, skipping")

    print(f"Saving converted checkpoint to {output_path}")
    torch.save(new_state_dict, output_path)


if __name__ == "__main__":
    main()
