from collections.abc import Mapping, Sequence
from typing import TypeAlias, TypedDict, TypeVar, no_type_check

import torch

EmbedderName: TypeAlias = str
EmbedderParams: TypeAlias = Mapping[str, torch.Tensor]


class SynthesisRepr(TypedDict):
    token_types: torch.Tensor
    bb_indices: torch.Tensor
    rxn_indices: torch.Tensor


def concat_synthesis_reprs(*reprs: "SynthesisRepr") -> "SynthesisRepr":
    return SynthesisRepr(
        token_types=torch.cat([r["token_types"] for r in reprs], dim=0),
        bb_indices=torch.cat([r["bb_indices"] for r in reprs], dim=0),
        rxn_indices=torch.cat([r["rxn_indices"] for r in reprs], dim=0),
    )


PropertyRepr: TypeAlias = Sequence[Mapping[EmbedderName, EmbedderParams]] | Mapping[EmbedderName, EmbedderParams]


def get_property_repr_batch_size(prop_repr: PropertyRepr) -> int:
    if isinstance(prop_repr, Sequence):
        return sum(get_property_repr_batch_size(pr) for pr in prop_repr)

    batch_size = 0
    for _, embedder_params in prop_repr.items():
        for param in embedder_params.values():
            batch_size = max(batch_size, param.shape[0])
    return batch_size


class SynthesisTrainingBatch(TypedDict):
    synthesis_repr: SynthesisRepr
    property_repr: PropertyRepr


_T = TypeVar("_T")


@no_type_check
def move_to_device(d: _T, device: torch.device | str) -> _T:
    if isinstance(d, torch.Tensor):
        return d.to(device)
    elif isinstance(d, Mapping):
        return {k: move_to_device(v, device) for k, v in d.items()}
    elif isinstance(d, Sequence):
        return [move_to_device(item, device) for item in d]
    else:
        return d
