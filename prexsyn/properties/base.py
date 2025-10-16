import abc
from collections.abc import Mapping, Sequence
from typing import Self

import torch
from rdkit import Chem
from torch import nn

from prexsyn.models.embeddings import BasePropertyEmbedder
from prexsyn_engine.featurizer.base import Featurizer, FeaturizerSet
from prexsyn_engine.synthesis import Synthesis


class BasePropertyDef(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def get_featurizer(self) -> Featurizer | None: ...

    @abc.abstractmethod
    def get_embedder(self, model_dim: int) -> BasePropertyEmbedder: ...

    def evaluate_mol(self, mol: Chem.Mol | list[Chem.Mol]) -> Mapping[str, torch.Tensor]:
        raise NotImplementedError(f"Property '{self.name}' does not support molecule evaluation.")

    def evaluate_synthesis(self, synthesis: Synthesis | list[Synthesis]) -> Mapping[str, torch.Tensor]:
        raise NotImplementedError(f"Property '{self.name}' does not support synthesis evaluation.")


class PropertySet:
    def __init__(self, properties: Sequence[BasePropertyDef]) -> None:
        super().__init__()
        self._properties = list(properties)
        self._property_map = {p.name: p for p in self._properties}

    def add(self, prop: BasePropertyDef) -> Self:
        if prop.name in self._property_map:
            raise ValueError(f"Property with name '{prop.name}' already exists in the property set.")
        self._properties.append(prop)
        self._property_map[prop.name] = prop
        return self

    def get_featurizer_set(self) -> tuple[FeaturizerSet, set[str]]:
        fs = FeaturizerSet()
        names: set[str] = set()
        for p in self._properties:
            f = p.get_featurizer()
            if f is not None:
                fs.add(f)
                names.add(p.name)
        return fs, names

    def get_embedders(self, model_dim: int) -> nn.ModuleDict:
        return nn.ModuleDict({p.name: p.get_embedder(model_dim=model_dim) for p in self._properties})
