from collections.abc import Generator, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from rdkit import RDLogger

from prexsyn.data.struct import EmbedderName, EmbedderParams, SynthesisTrainingBatch
from prexsyn.factories.chemical_space import ChemicalSpace
from prexsyn.factories.property import PropertySet
from prexsyn.factories.tokenization import Tokenization
from prexsyn_engine.chemspace import SynthesisGeneratorOption
from prexsyn_engine.featurizer.synthesis import PostfixNotationFeaturizer
from prexsyn_engine.pipeline import DataPipelineV2


class OnlineSynthesisDataset:
    def __init__(
        self,
        chemical_space: ChemicalSpace,
        property_set: PropertySet,
        tokenization: Tokenization,
        batch_size: int = 128,
        num_threads: int = 16,
        virtual_length: int = 1000,
        base_seed: int = 2025,
    ) -> None:
        super().__init__()
        self.chemical_space = chemical_space
        self.property_set = property_set
        self.tokenization = tokenization

        self.batch_size = batch_size
        self.num_threads = num_threads
        self.virtual_length = virtual_length
        self.base_seed = base_seed

    if TYPE_CHECKING:
        _pipeline: DataPipelineV2

    @property
    def pipeline(self) -> DataPipelineV2:
        RDLogger.DisableLog("rdApp.*")  # type: ignore
        if hasattr(self, "_pipeline"):
            return self._pipeline

        self._featurizer_set, self._property_names = self.property_set.get_featurizer_set()
        self._property_names_and_slices: list[tuple[str, slice]] = []
        num_properties = len(self._property_names)
        if num_properties == 0:
            raise ValueError("At least one property must be defined in the property set.")
        subbatch_size = self.batch_size // num_properties
        for i, prop_name in enumerate(self._property_names):
            start = i * subbatch_size
            end = (i + 1) * subbatch_size if i < num_properties - 1 else self.batch_size
            self._property_names_and_slices.append((prop_name, slice(start, end)))

        self._featurizer_set.add(PostfixNotationFeaturizer(token_def=self.tokenization.token_def))

        self._pipeline = DataPipelineV2(
            num_threads=self.num_threads,
            csd=self.chemical_space.get_csd(),
            gen_option=SynthesisGeneratorOption(),
            featurizer=self._featurizer_set,
            base_seed=self.base_seed,
        )
        self._pipeline.start()
        return self._pipeline

    def __len__(self) -> int:
        return self.virtual_length

    def __getitem__(self, idx: int) -> SynthesisTrainingBatch:
        if idx >= self.virtual_length:
            raise IndexError("Index out of range")

        data = self.pipeline.get(self.batch_size)

        grouped: dict[str, dict[str, np.ndarray[Any, Any]]] = {}
        for key, value in data.items():
            prop_name, param_name = key.split(".")
            if prop_name not in self._property_names:
                continue
            grouped.setdefault(prop_name, {})[param_name] = value

        property_repr: list[Mapping[EmbedderName, EmbedderParams]] = []
        for prop_name, prop_slice in self._property_names_and_slices:
            property_repr.append(
                {
                    prop_name: {
                        param_name: torch.from_numpy(param_value[prop_slice])
                        for param_name, param_value in grouped[prop_name].items()
                    }
                }
            )

        out: SynthesisTrainingBatch = {
            "synthesis_repr": {
                "token_types": torch.from_numpy(data["synthesis.token_types"]),
                "bb_indices": torch.from_numpy(data["synthesis.bb_indices"]),
                "rxn_indices": torch.from_numpy(data["synthesis.rxn_indices"]),
            },
            "property_repr": property_repr,
        }
        return out

    def __iter__(self) -> Generator[SynthesisTrainingBatch, Any, None]:
        for idx in range(len(self)):
            yield self[idx]
