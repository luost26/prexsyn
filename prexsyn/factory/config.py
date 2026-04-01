from pathlib import Path
from dataclasses import dataclass
from typing import cast

import omegaconf


@dataclass
class ChemicalSpaceConfig:
    cache_path: Path
    remote_url: str | None = None
    bb_path: Path | None = None
    rxn_path: Path | None = None
    building_block_selectivity_cutoff: int | None = None


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
class DataPipelineConfig:
    heavy_atom_limit: int | None = None
    max_building_blocks: int | None = None
    max_outcomes_per_reaction: int | None = None
    selectivity_cutoff: int | None = None


@dataclass
class TrainingConfig:
    batch_size: int
    val_freq: int
    seed: int
    data_pipeline_num_threads: int
    num_val_batches: int
    val_seed: int


@dataclass
class Config:
    chemical_space: ChemicalSpaceConfig
    descriptors: dict[str, DescriptorConfig]
    featurizer: FeaturizerConfig
    model: ModelConfig
    training: TrainingConfig

    @staticmethod
    def from_yaml(path: Path):
        schema = omegaconf.OmegaConf.structured(Config)
        base_conf = omegaconf.OmegaConf.load(path)
        conf = omegaconf.OmegaConf.merge(schema, base_conf)
        return cast(Config, omegaconf.OmegaConf.to_object(conf))
