import pathlib

import click
import torch
from rdkit import Chem

from prexsyn.applications.analog import generate_analogs
from prexsyn.factories.facade import load_model
from prexsyn.samplers.basic import BasicSampler
from prexsyn_engine.fingerprints import tanimoto_similarity


@click.command()
@click.option(
    "--model",
    "model_path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default="./data/trained_models/v1_converted.ckpt",
)
@click.option("--smiles", type=str, required=True)
def main(
    model_path: pathlib.Path,
    smiles: str,
) -> None:
    torch.set_grad_enabled(False)
    facade, model = load_model(model_path, train=False)
    model = model.to("cuda")

    mol = Chem.MolFromSmiles(smiles)
    canonical_smi = Chem.MolToSmiles(mol, canonical=True)
    print("Canonical SMILES:", canonical_smi)

    sampler = BasicSampler(
        model,
        token_def=facade.tokenization.token_def,
        num_samples=512,
        max_length=16,
    )

    entry = generate_analogs(
        facade=facade,
        model=model,
        sampler=sampler,
        fp_property=facade.property_set["ecfp4"],
        mol=mol,
    )

    results: dict[str, float] = {}
    for i, (synthesis, max_sim) in enumerate(zip(entry["synthesis"], entry["similarity"])):
        if synthesis.stack_size() != 1:
            continue
        for prod in synthesis.top().to_list():
            prod_smi = Chem.MolToSmiles(prod, canonical=True)
            prod_sim = tanimoto_similarity(prod, mol, fp_type="ecfp4")
            results[prod_smi] = prod_sim

    for smi, sim in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{smi}\t{sim:.4f}")


if __name__ == "__main__":
    main()
