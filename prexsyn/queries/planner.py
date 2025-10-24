from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
from rdkit import Chem

from prexsyn.data.struct import PropertyRepr
from prexsyn_engine.synthesis import Synthesis

from .base import DNF, Query, to_dnf


def _property_repr_size(prop_repr: PropertyRepr) -> int:
    if isinstance(prop_repr, Sequence):
        return sum(_property_repr_size(pr) for pr in prop_repr)

    batch_size = 0
    for _, embedder_params in prop_repr.items():
        for param in embedder_params.values():
            batch_size = max(batch_size, param.shape[0])
    return batch_size


class QueryPlanner:
    def __init__(self, query_or_dnf: DNF) -> None:
        if isinstance(query_or_dnf, Query):
            dnf = to_dnf(query_or_dnf)
        else:
            dnf = query_or_dnf
        self._dnf = dnf

    def get_property_reprs(self) -> list[PropertyRepr]:
        prop_repr_list: list[PropertyRepr] = []
        for conjunction in self._dnf:
            prop_repr_conj: list[Any] = []
            for condition, _ in conjunction:
                prop_repr_this = condition.get_property_repr()
                if _property_repr_size(prop_repr_this) != 1:
                    raise ValueError("Each condition's property representation must have exactly batch size 1.")

                if isinstance(prop_repr_this, Sequence):
                    prop_repr_conj.extend(prop_repr_this)
                else:
                    prop_repr_conj.append(prop_repr_this)

            prop_repr_list.append(prop_repr_conj)
        return prop_repr_list

    def get_weights(self) -> list[torch.Tensor]:
        weight_list: list[torch.Tensor] = []
        for conjunction in self._dnf:
            weights = []
            for condition, is_positive in conjunction:
                weight = condition.weight
                weights.append(weight if is_positive else -weight)
            weight_list.append(torch.tensor(weights, dtype=torch.float32))
        return weight_list

    def get_scorer(self, and_reduction: Literal["sum", "prod"] = "sum") -> Callable[[Synthesis, Chem.Mol], float]:
        if and_reduction not in {"sum", "prod"}:
            raise ValueError(f"Invalid and_reduction: {and_reduction}. Must be 'sum' or 'prod'.")

        def scorer(synthesis: Synthesis, product: Chem.Mol) -> float:
            scores = []
            for conjunction in self._dnf:
                conj_score = 0.0
                for condition, is_positive in conjunction:
                    score = condition.score(synthesis, product)
                    score = 1.0 - score if not is_positive else score
                    score *= condition.weight

                    if and_reduction == "sum":
                        conj_score += score
                    elif and_reduction == "prod":
                        conj_score = conj_score * score

                scores.append(conj_score)
            return max(scores) if scores else 0.0

        return scorer
