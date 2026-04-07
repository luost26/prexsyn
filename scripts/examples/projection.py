import pathlib

import click
import torch
import yaml

from prexsyn.shortcuts import AllInOneLoader, MoleculeProjector
from prexsyn.utils.draw import SynthesisDraw
from prexsyn.utils.syndag import SynthesisDAG


@click.command()
@click.option(
    "--config",
    "-c",
    "config_path",
    default="./data/trained_models/enamine2310_rxn115_202511.yml",
)
@click.option("--smiles", type=str, required=True)
@click.option("--draw-output-dir", "-o", type=click.Path(path_type=pathlib.Path), default=None)
@click.option("--top", type=int, default=10)
@click.option("--num-samples", type=int, default=64)
@click.option("--device", type=str, default="cuda")
def main(
    config_path: str,
    smiles: str,
    draw_output_dir: pathlib.Path | None,
    top: int,
    num_samples: int,
    device: str,
):
    torch.set_grad_enabled(False)
    if draw_output_dir is not None:
        draw_output_dir.mkdir(parents=True, exist_ok=True)

    loader = AllInOneLoader(config_path)
    model = loader.model().to(device).eval()
    detokenizer = loader.detokenizer()

    projector = MoleculeProjector(
        model=model,
        detokenizer=detokenizer,
        descriptor="ecfp4",
        num_samples=num_samples,
    )

    draw = SynthesisDraw()

    result = projector.one(smiles)
    for i, item in enumerate(result.items):
        if i >= top:
            break

        dag = SynthesisDAG(item.synthesis)
        out_dict: dict[str, object] = {
            "Target": smiles,
            "Similarity": item.similarity,
            **dag.to_dict(item.molecule.smiles()),
        }
        print(yaml.dump([out_dict], sort_keys=False))

        if draw_output_dir is not None:
            img = draw.draw(item.synthesis, highlight_smiles=item.molecule.smiles())
            img.save(draw_output_dir / f"synthesis_{i}_sim{item.similarity:.4f}.png")
            img.close()


if __name__ == "__main__":
    main()
