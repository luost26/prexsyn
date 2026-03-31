from collections.abc import Mapping
from pathlib import Path
from dataclasses import dataclass
from typing import cast

import omegaconf


@dataclass
class ChemicalSpaceConfig:
    cache_path: Path
    remote_url: str | None = None
    sdf_path: Path | None = None
    rxn_path: Path | None = None


@dataclass
class DescriptorConfig:
    type: str
    num_embedding_tokens: int


@dataclass
class FeaturizerConfig:
    max_length: int = 16
    pad_token: int | None = None
    end_token: int | None = None
    start_token: int | None = None
    bb_token: int | None = None
    rxn_token: int | None = None


@dataclass
class ModelConfig:
    dim: int
    nhead: int
    dim_feedforward: int
    num_layers: int
    bb_embed_dim: int


@dataclass
class Config:
    chemical_space: ChemicalSpaceConfig
    descriptors: Mapping[str, DescriptorConfig]
    featurizer: FeaturizerConfig
    model: ModelConfig

    @staticmethod
    def from_yaml(path: Path):
        schema = omegaconf.OmegaConf.structured(Config)
        base_conf = omegaconf.OmegaConf.load(path)
        conf = omegaconf.OmegaConf.merge(schema, base_conf)
        return cast(Config, omegaconf.OmegaConf.to_object(conf))
