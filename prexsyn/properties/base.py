import abc
import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from rdkit import Chem
from torch import nn

from prexsyn_engine.featurizer.base import Featurizer
from prexsyn_engine.synthesis import Synthesis


@dataclasses.dataclass(frozen=True)
class PropertyEmbedding:
    embedding: torch.Tensor
    padding_mask: torch.Tensor

    def pad(self, target_length: int) -> "PropertyEmbedding":
        if self.sequence_length >= target_length:
            return self
        pad_length = target_length - self.sequence_length
        padded_embedding = torch.nn.functional.pad(self.embedding, (0, 0, 0, pad_length), value=0.0)
        padded_padding_mask = torch.nn.functional.pad(self.padding_mask, (0, pad_length), value=float("-inf"))
        return PropertyEmbedding(padded_embedding, padded_padding_mask)

    def join(self, *others: "PropertyEmbedding") -> "PropertyEmbedding":
        if not others:
            return self
        return PropertyEmbedding(
            torch.cat([self.embedding] + [o.embedding for o in others], dim=-2),
            torch.cat([self.padding_mask] + [o.padding_mask for o in others], dim=-1),
        )

    @property
    def device(self) -> torch.device:
        return self.embedding.device

    @property
    def batch_size(self) -> int:
        return int(self.embedding.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.embedding.shape[1])

    @property
    def dim(self) -> int:
        return int(self.embedding.shape[2])

    def expand(self, n: int) -> "PropertyEmbedding":
        return PropertyEmbedding(
            embedding=self.embedding.expand(n, -1, -1),
            padding_mask=self.padding_mask.expand(n, -1),
        )

    def repeat(self, n: int) -> "PropertyEmbedding":
        return PropertyEmbedding(
            embedding=self.embedding.repeat(n, 1, 1),
            padding_mask=self.padding_mask.repeat(n, 1),
        )

    @staticmethod
    def create_full_padding_mask(h: torch.Tensor) -> torch.Tensor:
        return torch.zeros(h.shape[:-1], dtype=h.dtype, device=h.device)

    @staticmethod
    def concat_along_seq_dim(*embeddings: "PropertyEmbedding") -> "PropertyEmbedding":
        h = torch.cat([e.embedding for e in embeddings], dim=1)
        m = torch.cat([e.padding_mask for e in embeddings], dim=1)
        return PropertyEmbedding(h, m)

    @staticmethod
    def concat_along_batch_dim(*embeddings: "PropertyEmbedding", max_length: int | None = None) -> "PropertyEmbedding":
        if max_length is None:
            max_length = max(e.sequence_length for e in embeddings)
        padded_embeddings = [e.pad(max_length) for e in embeddings]
        h = torch.cat([e.embedding for e in padded_embeddings], dim=0)
        m = torch.cat([e.padding_mask for e in padded_embeddings], dim=0)
        return PropertyEmbedding(h, m)


class BasePropertyEmbedder(nn.Module, abc.ABC):
    if TYPE_CHECKING:

        def __call__(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> PropertyEmbedding: ...


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
