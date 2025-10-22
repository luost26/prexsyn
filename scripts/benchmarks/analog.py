import pathlib
from collections.abc import Sequence
from typing import Any

import click
import pandas as pd
import torch
from rdkit import Chem
from tqdm.auto import tqdm

from prexsyn.applications.analog.analog import (
    AnalogGenerationDatabase,
    generate_analogs,
)
from prexsyn.factories.facade import Facade
from prexsyn.models.prexsyn import PrexSyn
from prexsyn.samplers.basic import BasicSampler


def _run(
    facade: Facade,
    model: PrexSyn,
    df: pd.DataFrame,
    output_path: pathlib.Path,
    num_samples: int,
    max_length: int,
) -> pd.DataFrame:
    smi_list: Sequence[str] = df["SMILES"].tolist()

    summary: list[dict[str, Any]] = []

    with AnalogGenerationDatabase(output_path) as db:
        for smi in db.keys():
            entry = db.get_without_extra(smi)
            sim = float(entry["similarity"].max())
            summary.append(
                {
                    "smiles": smi,
                    "similarity": sim,
                    "time": entry["time_taken"],
                }
            )

        if len(summary) == len(smi_list):
            print("All SMILES already processed.")
            return pd.DataFrame(summary)

        sampler = BasicSampler(
            model,
            token_def=facade.tokenization.token_def,
            num_samples=num_samples,
            max_length=max_length,
        )
        with tqdm(total=len(smi_list)) as pbar:
            for smi in smi_list:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    print(f"Invalid SMILES: {smi}")
                    continue
                entry = generate_analogs(
                    model=model,
                    sampler=sampler,
                    detokenizer=facade.get_detokenizer(),
                    fp_property=facade.property_set["ecfp4"],
                    mol=mol,
                )
                db[smi] = entry
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "count": len(db),
                        "avg_sim": f"{db.get_average_similarity():.4f}",
                        "recons": f"{db.get_reconstruction_rate():.4f}",
                    }
                )
                sim = float(entry["similarity"].max())
                summary.append(
                    {
                        "smiles": smi,
                        "similarity": sim,
                        "time": entry["time_taken"],
                    }
                )
    return pd.DataFrame(summary)


@click.command()
@click.option(
    "--model",
    "model_path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default="./data/trained_models/v1_converted.ckpt",
)
@click.option("--out", "output_dir", type=click.Path(path_type=pathlib.Path), default="./outputs/benchmarks/analog")
@click.option("--num-runs", type=int, default=5)
@click.option("--device", type=str, default="cuda")
def main(model_path: pathlib.Path, output_dir: pathlib.Path, num_runs: int, device: str) -> None:
    datasets = {
        "Enamine": pd.read_csv("data/benchmarks/enamine_real_1k.txt"),
        "ChEMBL": pd.read_csv("data/benchmarks/chembl_1k.txt"),
    }
    num_samples_grid = [64, 128, 256]
    output_dir.mkdir(parents=True, exist_ok=True)

    facade = Facade.from_file(model_path.with_suffix(".yaml"))
    model = facade.load_model(model_path).eval().to(device)
    torch.set_grad_enabled(False)

    summary_list: list[pd.DataFrame] = []

    try:
        for dataset_name, df in datasets.items():
            for num_samples in num_samples_grid:
                for run_idx in range(num_runs):
                    print(
                        f"Running benchmark: dataset={dataset_name}, "
                        f"num_samples={num_samples}, run_idx={run_idx}({num_runs})"
                    )
                    output_path = output_dir / f"{dataset_name}_{num_samples}_{run_idx}.out"
                    summary_this = _run(
                        facade=facade,
                        model=model,
                        df=df,
                        output_path=output_path,
                        num_samples=num_samples,
                        max_length=16,
                    )
                    summary_this["dataset"] = dataset_name
                    summary_this["num_samples"] = num_samples
                    summary_this["run_idx"] = run_idx
                    summary_list.append(summary_this)
    except KeyboardInterrupt:
        print("Interrupted by user, saving results...")

    summary_df = pd.concat(summary_list, ignore_index=True)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    pd.set_option("display.float_format", "{:.4f}".format)

    summary_df["reconstructed"] = summary_df["similarity"] == 1.0
    recons_df = (
        summary_df.groupby(["dataset", "num_samples", "run_idx"])
        .aggregate(
            similarity=("similarity", "mean"),
            recons=("reconstructed", "mean"),
        )
        .reset_index()
    )
    recons_stat = (
        recons_df.groupby(["dataset", "num_samples"])
        .aggregate(
            similarity_mean=("similarity", "mean"),
            similarity_std=("similarity", "std"),
            recons_mean=("recons", "mean"),
            recons_std=("recons", "std"),
        )
        .reset_index()
    )
    print(recons_stat)

    time_stat = (
        summary_df.groupby(["num_samples"])
        .aggregate(
            time_mean=("time", "mean"),
            time_std=("time", "std"),
        )
        .reset_index()
    )
    print(time_stat)


if __name__ == "__main__":
    main()
