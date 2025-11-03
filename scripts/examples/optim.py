import datetime
import logging
import pathlib
import sys

import click
import torch
from rdkit import Chem
from rdkit.Chem import QED

from prexsyn.applications.optim import Optimizer
from prexsyn.applications.optim.step import FingerprintGenetic
from prexsyn.factories.facade import load_model
from prexsyn.properties import PropertySet
from prexsyn.queries import Query
from prexsyn.utils.oracles import CustomOracle


class ExampleOracle(CustomOracle):
    def evaluate(self, mol: Chem.Mol) -> float:
        return float(QED.qed(mol))  # type: ignore[no-untyped-call]

    def evaluate_many(self, mols: list[Chem.Mol]) -> list[float]:
        # Custom batch evaluation logic can be implemented here
        # If not implemented, it will default to calling evaluate() for each molecule
        return super().evaluate_many(mols)


def init_query_lipinski(ps: PropertySet, pn: str = "rdkit_descriptor_upper_bound") -> Query:
    p = ps[pn]
    return (
        p.lt("amw", 500.0)
        & p.lt("CrippenClogP", 5.0)
        & p.lt("lipinskiHBD", 4)
        & p.lt("lipinskiHBA", 9)
        & p.lt("NumRotatableBonds", 9)
        & p.lt("tpsa", 140.0)
    )


@click.command()
@click.option(
    "--model",
    "model_path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default="./data/trained_models/v1_converted.ckpt",
)
@click.option("--out", "output_dir", type=click.Path(path_type=pathlib.Path), default="./outputs/examples/optim")
def main(
    model_path: pathlib.Path,
    output_dir: pathlib.Path,
) -> None:
    torch.set_grad_enabled(False)
    facade, model = load_model(model_path, train=False)
    model = model.to("cuda")
    output_dir = output_dir / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)

    logger = logging.getLogger("optim")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(output_dir / "log.txt"))

    optimizer = Optimizer(
        facade=facade,
        model=model,
        init_query=init_query_lipinski(facade.property_set),
        num_init_samples=1000,
        max_evals=50_000,
        step_strategy=FingerprintGenetic(
            bottleneck_size=50,
            bottleneck_temperature=0.5,
        ),
        oracle_fn=ExampleOracle(),
        constraint_fn=None,
        cond_query=None,
        time_limit=None,
        handle_interrupt=True,
    )
    tracker = optimizer.run()
    df_result = tracker.get_dataframe()
    auc_top10 = tracker.auc_top10(optimizer.max_evals)
    df_result.to_pickle(output_dir / "result.pkl")

    logger.info("==== Summary ====")
    logger.info(f"- AUC-Top10({optimizer.max_evals / 1000}k): {auc_top10:.4f}, Evals: {len(df_result)}")
    logger.info(f"Results saved to: {output_dir / 'result.pkl'}")


if __name__ == "__main__":
    main()
