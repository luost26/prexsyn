import pathlib
from collections.abc import Sequence
from typing import Any

import click
import pandas as pd
import torch
from tqdm.auto import tqdm

import prexsyn_engine
from prexsyn.factory import get_detokenizer, load_model_and_config
from prexsyn.shortcuts.projector import MoleculeProjector
from prexsyn.utils.lmdb_dict import LmdbDict


def _run(
    projector: MoleculeProjector,
    df: pd.DataFrame,
    output_path: pathlib.Path,
) -> pd.DataFrame:
    smi_list: Sequence[str] = df["SMILES"].tolist()
    summary: list[dict[str, Any]] = []

    def _average_sim():
        sims = []
        for result in summary:
            sims.append(result["similarity"])
        return sum(sims) / len(sims)

    def _recons_rate():
        recons = 0
        for result in summary:
            if result["similarity"] == 1.0:
                recons += 1
        return recons / len(summary)

    def _average_time():
        times = []
        for result in summary:
            times.append(result["time"])
        return sum(times) / len(times)

    with LmdbDict(output_path) as db:
        pbar = tqdm(total=len(smi_list))
        for i, smi in enumerate(smi_list):
            key_metric = f"{i}_metric"
            if key_metric in db:
                row = db[key_metric]
                if row is None:
                    print(f"Previous error for SMILES: {smi}, skipping.")
                    continue
            else:
                try:
                    result = projector(smi)
                except prexsyn_engine.chemistry.MoleculeError:
                    print(f"Error processing SMILES: {smi}, skipping.")
                    db[key_metric] = None
                    continue

                row = {
                    "smiles": smi,
                    "similarity": result.best_similarity(),
                    "time": result.time,
                }
                db[key_metric] = row

            summary.append(row)
            pbar.update(1)
            pbar.set_postfix(
                {
                    "count": len(summary),
                    "avg_sim": _average_sim(),
                    "recons": f"{_recons_rate() * 100:.2f}%",
                    "avg_time": f"{_average_time():.2f}s",
                }
            )

    return pd.DataFrame(summary)


@click.command()
@click.option(
    "--model",
    "-m",
    "model_path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default="./data/trained_models/enamine2310_rxn115_202511.ckpt",
)
@click.option("--out", "output_dir", type=click.Path(path_type=pathlib.Path), default="./outputs/benchmarks/analog")
@click.option("--num-runs", type=int, default=5)
@click.option("--device", type=str, default="cuda")
def main(model_path: pathlib.Path, output_dir: pathlib.Path, num_runs: int, device: str):
    datasets = {
        "Enamine": pd.read_csv("data/benchmarks/enamine_real_1k.txt"),
        "ChEMBL": pd.read_csv("data/benchmarks/chembl_1k.txt"),
    }
    num_samples_grid = [64, 128, 256]
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    model, config = load_model_and_config(model_path)
    model = model.to(device).eval()

    detokenizer = get_detokenizer(config)

    summary_list: list[pd.DataFrame] = []

    for dataset_name, df in datasets.items():
        for num_samples in num_samples_grid:
            for run_idx in range(num_runs):
                print(
                    f"Running benchmark: dataset={dataset_name}, "
                    f"num_samples={num_samples}, run={run_idx + 1}/({num_runs})"
                )
                output_path = output_dir / f"{dataset_name}_{num_samples}_{run_idx}.out"
                projector = MoleculeProjector(model, detokenizer, "ecfp4", num_samples)
                summary_this = _run(
                    projector=projector,
                    df=df,
                    output_path=output_path,
                )
                summary_this["dataset"] = dataset_name
                summary_this["num_samples"] = num_samples
                summary_this["run_idx"] = run_idx
                summary_list.append(summary_this)

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
