from collections.abc import Sequence
from pathlib import Path

import click
import numpy as np
import torch
from rdkit.Chem import QED

from prexsyn.shortcuts import AllInOneLoader, MoleculeProjector
from prexsyn.shortcuts.genetic import EvolutionaryTreeDraw, evolve, initialize
from prexsyn_engine.chemistry import Molecule
from prexsyn_engine.chemspace import Synthesis


class ExampleScoringFunction:
    def __call__(self, phenotypes: Sequence[tuple[Synthesis, Molecule]]) -> np.ndarray:
        qed_list = [QED.qed(mol.to_rdkit_mol()) for _, mol in phenotypes]
        return np.array(qed_list)


@click.command()
@click.option(
    "--config",
    "-c",
    "config_path",
    default="./data/trained_models/enamine2310_rxn115_202511.yml",
)
@click.option("--device", type=str, default="cuda")
@click.option("--out-fig", type=click.Path(path_type=Path), default=None)
def main(config_path: str, device: str, out_fig: Path | None):
    torch.set_grad_enabled(False)
    loader = AllInOneLoader(config_path)
    model = loader.model().to(device).eval()
    detokenizer = loader.detokenizer()
    projector = MoleculeProjector(
        model=model,
        detokenizer=detokenizer,
        descriptor="ecfp4",
        num_samples=8,
    )
    fn = ExampleScoringFunction()

    ppl, history = initialize(size=100, projector=projector, fn=fn)
    for i in range(20):
        evolve(ppl, history, projector, fn, k=50, t=0.5)
        print(f"step: {i}, best fitness: {ppl.fitnesses.max():.4f}, avg fitness: {ppl.fitnesses.mean():.4f}")

    if out_fig is not None:
        best_indiv = next(iter(history.individuals.values()))
        for indiv in history.individuals.values():
            if indiv.fitness > best_indiv.fitness:
                best_indiv = indiv

        draw = EvolutionaryTreeDraw()
        img = draw.draw(best_indiv, history)
        img.save(out_fig)


if __name__ == "__main__":
    main()
