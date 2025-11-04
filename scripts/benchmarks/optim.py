import heapq
import logging
import pathlib
import sys

import click
import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from prexsyn.applications.optim import Optimizer
from prexsyn.applications.optim.step import FingerprintGenetic, StepStrategy
from prexsyn.factories.facade import Facade, load_model
from prexsyn.models.prexsyn import PrexSyn
from prexsyn.properties import PropertySet
from prexsyn.queries import Query
from prexsyn.utils.oracles import CachedOracle, get_oracle


def query_lipinski(ps: PropertySet, pn: str = "rdkit_descriptor_upper_bound") -> Query:
    p = ps[pn]
    return (
        p.lt("amw", 500.0)
        & p.lt("CrippenClogP", 5.0)
        & p.lt("lipinskiHBD", 4)
        & p.lt("lipinskiHBA", 9)
        & p.lt("NumRotatableBonds", 9)
        & p.lt("tpsa", 140.0)
    )


def query_scaffold_hop_demo1_condition(ps: PropertySet, pn: str = "brics") -> Query:
    deco1 = Chem.MolFromSmiles("[16*]c1ccc2ncsc2c1")
    deco2 = Chem.MolFromSmiles("CCCO[15*]")
    p = ps[pn]
    return p.has(deco1, deco2)


def auc_top10_from_df(df: pd.DataFrame, max_evals: int) -> float:
    scores: list[float] = df["score"].tolist()

    top10: list[float] = []
    moving_top10_avg: list[float] = []
    for score in scores:
        heapq.heappush(top10, score)
        if len(top10) > 10:
            heapq.heappop(top10)
        moving_top10_avg.append(sum(top10) / len(top10) if top10 else 0.0)

    if len(moving_top10_avg) < max_evals:
        moving_top10_avg += [moving_top10_avg[-1]] * (max_evals - len(moving_top10_avg))

    return float(np.mean(moving_top10_avg[:max_evals]))


class Task:
    def __init__(
        self,
        name: str,
        oracle_name: str | None = None,
        constraint_name: str = "null",
        num_runs: int = 5,
        max_evals: int = 10_000,
        num_init_samples: int = 500,
        cond_query: Query | None = None,
        step_strategy: StepStrategy | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.oracle_fn = CachedOracle(get_oracle(oracle_name or name))
        self.constraint_fn = CachedOracle(get_oracle(constraint_name))
        self.num_runs = num_runs
        self.max_evals = max_evals
        self.num_init_samples = num_init_samples
        self.cond_query = cond_query
        self.step_strategy = step_strategy or FingerprintGenetic(
            bottleneck_size=50,
            bottleneck_temperature=0.5,
        )

    def run(self, facade: Facade, model: PrexSyn, out_root: pathlib.Path, time_limit: int | None = None) -> None:
        task_dir = out_root / self.name
        task_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(self.name)
        logger.propagate = False
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.addHandler(logging.FileHandler(task_dir / "log.txt"))

        auc_top10_all = []
        for run_id in range(1, self.num_runs + 1):
            logger.info(f"Running task: {task_dir.name}, run {run_id}/5")
            result_path = task_dir / f"run_{run_id:02d}.df.pkl"
            if result_path.exists():
                logger.info(f"Skipping existing run: {result_path}")
                df_result: pd.DataFrame = pd.read_pickle(result_path)
                auc_top10 = auc_top10_from_df(df_result, self.max_evals)
            else:
                optimizer = Optimizer(
                    facade=facade,
                    model=model,
                    init_query=query_lipinski(facade.property_set),
                    num_init_samples=self.num_init_samples,
                    max_evals=self.max_evals,
                    step_strategy=self.step_strategy,
                    oracle_fn=self.oracle_fn,
                    constraint_fn=self.constraint_fn,
                    cond_query=self.cond_query,
                    time_limit=time_limit,
                )
                tracker = optimizer.run()
                df_result = tracker.get_dataframe()
                auc_top10 = tracker.auc_top10(self.max_evals)
                df_result.to_pickle(result_path)

            auc_top10_all.append(auc_top10)
            logger.info(
                f"Run {run_id}/{self.num_runs}, "
                f"AUC-Top10({self.max_evals / 1000}k): {auc_top10:.4f}, "
                f"Evals: {len(df_result)}"
            )

        logger.info("==== Summary ====")
        logger.info(f"Oracle: {self.name}")
        logger.info(f"- Runs: {len(auc_top10_all)}")
        logger.info(f"- AUC-Top10: {np.mean(auc_top10_all):.3f} ± {np.std(auc_top10_all):.3f}")


@click.command()
@click.option(
    "--model",
    "model_path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default="./data/trained_models/v1_converted.ckpt",
)
@click.option("--out", "output_dir", type=click.Path(path_type=pathlib.Path), default="./outputs/benchmarks/optim")
@click.option("--time-limit", type=int, default=None)
@click.option("selected_tasks", "--task", "-t", multiple=True, default=None)
def main(
    model_path: pathlib.Path,
    output_dir: pathlib.Path,
    time_limit: int | None,
    selected_tasks: list[str] | None,
) -> None:
    torch.set_grad_enabled(False)
    facade, model = load_model(model_path, train=False)
    model = model.to("cuda")
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        Task("amlodipine"),
        Task("fexofenadine"),
        Task("osimertinib"),
        Task("perindopril"),
        Task("ranolazine"),
        Task("sitagliptin"),
        Task("zaleplon"),
        Task("celecoxib_rediscovery"),
        Task("scaffold_hop_demo1_baseline", oracle_name="scaffold_hop_demo1", max_evals=5000),
        Task(
            "scaffold_hop_demo1_conditioned",
            oracle_name="scaffold_hop_demo1",
            cond_query=query_scaffold_hop_demo1_condition(facade.property_set),
            max_evals=5000,
        ),
    ]

    if selected_tasks:
        tasks = [task for task in tasks if task.name in selected_tasks]

    for task in tasks:
        task.run(facade, model, output_dir, time_limit)


if __name__ == "__main__":
    main()
