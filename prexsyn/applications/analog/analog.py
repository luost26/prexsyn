import pathlib
import time
from typing import Any, Self, TypedDict

import numpy as np
from rdkit import Chem

from prexsyn.data.struct import move_to_device
from prexsyn.factories.property.fingerprint import StandardFingerprintProperty
from prexsyn.models.prexsyn import PrexSyn
from prexsyn.samplers.basic import BasicSampler
from prexsyn_engine.detokenizer import Detokenizer
from prexsyn_engine.fingerprints import mol_to_syntheses_tanimoto_similarity
from prexsyn_engine.synthesis import Synthesis

from .db import ResultDatabase


class AnalogGenerationResult(TypedDict):
    synthesis: list[Synthesis]
    # logp: np.ndarray[Any, Any]
    similarity: np.ndarray[Any, Any]
    max_sim_product_idx: np.ndarray[Any, Any]
    time_taken: float


class AnalogGenerationDatabase(ResultDatabase[AnalogGenerationResult]):
    def __init__(self, path: pathlib.Path | str) -> None:
        super().__init__(path, extra_object_fields=["synthesis"])
        self._max_similarities: dict[str, float] = {}

    def __enter__(self) -> Self:
        super().__enter__()
        for key in iter(self):
            data = self.get_without_extra(key)
            self._max_similarities[key] = data["similarity"].max()
        return self

    def __setitem__(self, key: str, data: AnalogGenerationResult) -> None:
        super().__setitem__(key, data)
        self._max_similarities[key] = data["similarity"].max()

    def get_average_similarity(self) -> float:
        return sum(self._max_similarities.values()) / len(self._max_similarities) if self._max_similarities else 0.0

    def get_reconstruction_rate(self) -> float:
        return (
            sum(1 for sim in self._max_similarities.values() if sim == 1.0) / len(self._max_similarities)
            if self._max_similarities
            else 0.0
        )

    def get_time_statistics(self) -> tuple[float, float]:
        times = [self.get_without_extra(key)["time_taken"] for key in iter(self)]
        return (float(np.mean(times)), float(np.std(times))) if times else (0.0, 0.0)


def generate_analogs(
    model: PrexSyn,
    sampler: BasicSampler,
    detokenizer: Detokenizer,
    fp_property: StandardFingerprintProperty,
    mol: Chem.Mol,
    eval_fp_type: str = "ecfp4",
) -> AnalogGenerationResult:
    t_start = time.perf_counter()
    property_repr = {fp_property.name: move_to_device(fp_property.evaluate_mol(mol), model.device)}
    synthesis_repr = sampler.sample(property_repr)
    syn_list = detokenizer(
        token_types=synthesis_repr["token_types"].cpu().numpy(),
        bb_indices=synthesis_repr["bb_indices"].cpu().numpy(),
        rxn_indices=synthesis_repr["rxn_indices"].cpu().numpy(),
    )
    sim_matrix = mol_to_syntheses_tanimoto_similarity(mol, syn_list, fp_type=eval_fp_type)
    sim_list = sim_matrix.max(axis=1)
    max_sim_product_idx = sim_matrix.argmax(axis=1)
    t_end = time.perf_counter()
    return {
        "synthesis": list(syn_list),
        "similarity": sim_list,
        "max_sim_product_idx": max_sim_product_idx,
        "time_taken": t_end - t_start,
    }
