import logging
import pathlib
import pickle
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem.AllChem import Compute2DCoords  # type: ignore[attr-defined]

from prexsyn.factories.facade import Facade, load_model
from prexsyn.factories.property import PropertySet
from prexsyn.models.prexsyn import PrexSyn
from prexsyn.queries import Query
from prexsyn.samplers.query import QuerySampler
from prexsyn.utils.oracles import OracleProtocol, get_oracle
from prexsyn_engine.fingerprints import diversity
from prexsyn_engine.synthesis import Synthesis


def preset_lipinski(ps: PropertySet, pn: str = "rdkit_descriptor_upper_bound") -> Query:
    p = ps[pn]
    return (
        p.lt("amw", 500.0)
        & p.lt("CrippenClogP", 5.0)
        & p.lt("lipinskiHBD", 4)
        & p.lt("lipinskiHBA", 9)
        & p.lt("NumRotatableBonds", 9)
        & p.lt("tpsa", 140.0)
    )


def run_query(
    facade: Facade,
    model: PrexSyn,
    query: Query,
    score_fn: OracleProtocol,
    n_samples: int,
) -> tuple[pd.DataFrame, list[Synthesis]]:
    sampler = QuerySampler(model, facade.tokenization.token_def, num_samples=n_samples)
    samples = sampler.sample(query)
    syns = list(
        facade.get_detokenizer()(
            token_types=samples["token_types"].cpu().numpy(),
            bb_indices=samples["bb_indices"].cpu().numpy(),
            rxn_indices=samples["rxn_indices"].cpu().numpy(),
        )
    )

    df_list: list[dict[str, Any]] = []
    product_set: set[str] = set()
    for i, syn in enumerate(syns):
        if syn.stack_size() != 1:
            continue
        for j, product in enumerate(syn.top().to_list()):
            smi = Chem.MolToSmiles(product)
            if smi in product_set:
                continue
            product_set.add(smi)
            score = score_fn(product)
            df_list.append(
                {
                    "smiles": smi,
                    "product": product,
                    "score": score,
                    "synthesis_idx": i,
                    "product_idx": j,
                }
            )
    df = pd.DataFrame(df_list)
    df.drop_duplicates("smiles", inplace=True)
    df.sort_values("score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df, syns


def save_results(df: pd.DataFrame, syn_list: Sequence[Synthesis], output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path.with_suffix(".syn.pkl"), "wb") as f:
        pickle.dump(syn_list, f)
    df.to_pickle(output_path.with_suffix(".df.pkl"))
    df_top = df.head(500)
    df_top["product"].apply(Compute2DCoords)
    df_top["product"].apply(PandasTools.PrintAsImageString)
    df_top.to_html(output_path.with_suffix(".top500.html"), float_format="%.4f", escape=False)


def top_p_score_diversity(df: pd.DataFrame, p: float) -> tuple[float, float]:
    score_at_p = df["oracle"].quantile(p)
    sub_df = df.loc[df["oracle"] >= score_at_p]
    average_above = sub_df["oracle"].mean()
    diversity_above = diversity(sub_df["product"].tolist(), "ecfp4")
    return float(average_above), float(diversity_above)


def run_task(
    facade: Facade,
    model: PrexSyn,
    query: Query,
    score_fn: OracleProtocol,
    n_samples: int,
    task_dir: pathlib.Path,
    logger: logging.Logger,
    n_runs: int = 5,
) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)

    count_products: list[int] = []
    avg_all: list[float] = []
    avg_best: list[float] = []
    avg_t1: list[float] = []
    avg_t5: list[float] = []
    avg_t10: list[float] = []
    div_t1: list[float] = []
    div_t5: list[float] = []
    div_t10: list[float] = []
    for i in range(n_runs):
        result_path = task_dir / f"run{i}"
        if result_path.with_suffix(".df.pkl").exists():
            logger.info(f"Run {i} already exists, loading results")
            df = pd.read_pickle(result_path.with_suffix(".df.pkl"))
        else:
            df, syn_list = run_query(
                facade=facade,
                model=model,
                query=query,
                score_fn=score_fn,
                n_samples=n_samples,
            )
            save_results(df, syn_list, result_path)

        count_products.append(len(df))
        avg_all.append(float(df["oracle"].mean()))

        avg_best.append(float(df["oracle"].max()))
        s_t1, d_t1 = top_p_score_diversity(df, 0.99)
        s_t5, d_t5 = top_p_score_diversity(df, 0.95)
        s_t10, d_t10 = top_p_score_diversity(df, 0.90)
        avg_t1.append(s_t1)
        avg_t5.append(s_t5)
        avg_t10.append(s_t10)
        div_t1.append(d_t1)
        div_t5.append(d_t5)
        div_t10.append(d_t10)

    logger.info("==== Summary ====")
    logger.info(f"Query: {query}, Scoring Function: {score_fn}")
    logger.info(f"- Runs: {n_runs}")
    logger.info(f"- Count products : {count_products}")
    logger.info(f"- Best: {np.mean(avg_best):.4f} ± {np.std(avg_best):.4f}")
    logger.info(f"- Top 5%  average: {np.mean(avg_t5):.4f} ± {np.std(avg_t5):.4f}")
    logger.info(f"- Top 10% average: {np.mean(avg_t10):.4f} ± {np.std(avg_t10):.4f}")
    logger.info(f"- All     average: {np.mean(avg_all):.4f} ± {np.std(avg_all):.4f}")
    logger.info(f"- Top 5%  diver. : {np.mean(div_t5):.4f} ± {np.std(div_t5):.4f}")
    logger.info(f"- Top 10% diver. : {np.mean(div_t10):.4f} ± {np.std(div_t10):.4f}")


if __name__ == "__main__":
    facade, model = load_model("data/trained_models/v1_converted.ckpt")
    model = model.eval().to("cuda")

    output_dir = pathlib.Path("./outputs/benchmarks/query")

    logger = logging.getLogger("query_benchmarks")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(output_dir / "log.txt"))

    run_task(
        facade=facade,
        model=model,
        query=preset_lipinski(facade.property_set),
        score_fn=get_oracle("lipinski"),
        n_samples=1000,
        task_dir=output_dir / "lipinski",
        logger=logger,
        n_runs=5,
    )
