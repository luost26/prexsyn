import abc
from collections.abc import Mapping

import torch
from rdkit import Chem
from torch import nn

from prexsyn_engine.featurizer.fingerprint import FingerprintFeaturizer
from prexsyn_engine.fingerprints import get_fingerprints

from .base import BasePropertyDef, BasePropertyEmbedder, PropertyEmbedding


class FingerprintEmbedder(BasePropertyEmbedder, nn.Module):
    def __init__(self, fingerprint_dim: int, embedding_dim: int, num_tokens: int) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.mlp = nn.Sequential(
            nn.Linear(fingerprint_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim * num_tokens),
        )

    def forward(self, fingerprint: torch.Tensor) -> PropertyEmbedding:
        h = self.mlp(fingerprint)
        h = h.reshape(*h.shape[:-1], self.num_tokens, -1)
        return PropertyEmbedding(
            embedding=h,
            padding_mask=PropertyEmbedding.create_full_padding_mask(h),
        )


class StandardFingerprintProperty(BasePropertyDef, abc.ABC):
    def __init__(self, name: str, num_embedding_tokens: int = 4) -> None:
        super().__init__()
        self._name = name
        self._num_embedding_tokens = num_embedding_tokens

    @abc.abstractmethod
    @property
    def fp_type(self) -> str: ...

    @abc.abstractmethod
    @property
    def fp_dim(self) -> int: ...

    @property
    def name(self) -> str:
        return self._name

    def get_featurizer(self) -> FingerprintFeaturizer:
        return FingerprintFeaturizer(name=self._name, fp_type=self.fp_type)

    def get_embedder(self, model_dim: int) -> FingerprintEmbedder:
        return FingerprintEmbedder(
            fingerprint_dim=self.fp_dim,
            embedding_dim=model_dim,
            num_tokens=self._num_embedding_tokens,
        )

    def evaluate_mol(self, mol: Chem.Mol | list[Chem.Mol]) -> Mapping[str, torch.Tensor]:
        mol = [mol] if isinstance(mol, Chem.Mol) else mol
        return {f"{self.name}.fingerprint": torch.from_numpy(get_fingerprints(mol, fp_type=self.fp_type))}


class ECFP4(StandardFingerprintProperty):
    def __init__(self, name: str = "ecfp4", num_embedding_tokens: int = 4) -> None:
        super().__init__(name, num_embedding_tokens)

    @property
    def fp_type(self) -> str:
        return "ECFP4"

    @property
    def fp_dim(self) -> int:
        return 2048


class FCFP4(StandardFingerprintProperty):
    def __init__(self, name: str = "fcfp4", num_embedding_tokens: int = 4) -> None:
        super().__init__(name, num_embedding_tokens)

    @property
    def fp_type(self) -> str:
        return "FCFP4"

    @property
    def fp_dim(self) -> int:
        return 2048
