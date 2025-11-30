# type: ignore
import pathlib

import hydra
import torch
from omegaconf import DictConfig

from prexsyn.factories.facade import Facade

prefix_mapping = {
    "synthesis_embedder": "model.synthesis_embedder",
    "property_embedders.ecfp4": "model.property_embedders.product_ecfp4_fingerprint",
    "property_embedders.fcfp4": "model.property_embedders.product_fcfp4_fingerprint",
    "property_embedders.rdkit_descriptors": "model.property_embedders.product_rdkit_properties",
    "property_embedders.rdkit_descriptor_upper_bound": "model.property_embedders.product_rdkit_property_upper_bounds",
    "property_embedders.brics": "model.property_embedders.product_fragment_fingerprints",
    "transformer_layers": "model.transformer_layers",
    "synthesis_output": "model.synthesis_output",
}


def crop_tensor(tensor, target_shape):
    """Crop the input tensor to the target shape."""
    slices = tuple(slice(0, size) for size in target_shape)
    return tensor[slices]


@hydra.main(version_base=None, config_path="../configs", config_name="config")  # type: ignore
def main(cfg: DictConfig) -> None:
    from_path = pathlib.Path(cfg["from"])
    to_path = pathlib.Path(cfg["to"])

    facade = Facade(cfg)
    model = facade.build_model()
    tgt_state_dict = model.state_dict()

    ckpt = torch.load(
        from_path,
        map_location="cpu",
        weights_only=False,
    )
    src_state_dict = ckpt["state_dict"]

    for tgt_key in tgt_state_dict.keys():
        for prefix, src_prefix in prefix_mapping.items():
            if tgt_key.startswith(prefix):
                print(f"Mapping {tgt_key} <- {src_prefix}")
                src_key = tgt_key.replace(prefix, src_prefix)
                if src_key in src_state_dict:
                    tgt_state_dict[tgt_key].data.copy_(
                        crop_tensor(src_state_dict[src_key], target_shape=tgt_state_dict[tgt_key].shape)
                    )
                else:
                    raise ValueError(f"Source key {src_key} not found in source state dict.")
                break
        else:
            raise ValueError(f"No matching prefix found for target key {tgt_key}.")

    torch.save(tgt_state_dict, to_path.with_suffix(".ckpt"))
    facade.save_config(to_path.with_suffix(".yaml"))


if __name__ == "__main__":
    main()
