from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import omegaconf


@dataclass
class ChemicalSpaceConfig:
    cache_path: Path
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
    loss_weights: dict[str, float]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any] | None


@dataclass
class RemoteConfig:
    checkpoint_url: str | None = None
    chemical_space_url: str | None = None


@dataclass
class Note:
    name: str = ""
    description: str = ""


@dataclass
class Config:
    chemical_space: ChemicalSpaceConfig
    descriptors: list[DescriptorConfig]
    featurizer: FeaturizerConfig
    model: ModelConfig
    training: TrainingConfig

    remote: RemoteConfig = field(default_factory=RemoteConfig)
    note: Note = field(default_factory=Note)

    @staticmethod
    def from_yaml(path: Path):
        schema = omegaconf.OmegaConf.structured(Config)
        base_conf = omegaconf.OmegaConf.load(path)
        conf = omegaconf.OmegaConf.merge(schema, base_conf)
        return cast(Config, conf)  # duck typing
