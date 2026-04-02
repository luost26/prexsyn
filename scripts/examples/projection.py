import pathlib

import click
import torch

from prexsyn.factory import get_chemical_space, get_detokenizer, load_model_and_config
from prexsyn.shortcuts.projector import MoleculeProjector
from prexsyn.utils.draw import SynthesisDrawer


@click.command()
@click.option(
    "--model",
    "-m",
    "model_path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default="./data/trained_models/enamine2310_rxn115_202511.ckpt",
)
@click.option("--smiles", type=str, required=True)
@click.option("--draw-output-dir", "-o", type=click.Path(path_type=pathlib.Path), default=None)
@click.option("--top", type=int, default=10)
@click.option("--num-samples", type=int, default=64)
def main(
    model_path: pathlib.Path,
    smiles: str,
    draw_output_dir: pathlib.Path | None,
    top: int,
    num_samples: int,
):
    torch.set_grad_enabled(False)
    if draw_output_dir is not None:
        draw_output_dir.mkdir(parents=True, exist_ok=True)

    model, config = load_model_and_config(model_path)
    cs = get_chemical_space(config.chemical_space)

    projector = MoleculeProjector(
        model=model,
        detokenizer=get_detokenizer(config, chemspace=cs),
        descriptor="ecfp4",
        num_samples=num_samples,
    )

    draw = SynthesisDrawer()

    result = projector(smiles)
    for i, (mol, syn, sim) in enumerate(result.results):
        if i >= top:
            break
        if draw_output_dir is not None:
            img = draw.draw(syn, cs)
            img.save(draw_output_dir / f"synthesis_{i}_sim{sim:.4f}.png")
            img.close()
        print(f"Sample {i}: similarity={sim:.4f}, product={mol.smiles()}")


if __name__ == "__main__":
    main()
