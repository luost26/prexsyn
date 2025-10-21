import pathlib

import torch
from omegaconf import DictConfig, OmegaConf

from prexsyn.models.prexsyn import PrexSyn
from prexsyn_engine.detokenizer import Detokenizer

from .chemical_space import ChemicalSpace
from .property import PropertySet
from .tokenization import Tokenization


class Facade:
    def __init__(self, cfg: DictConfig):
        self._cfg = cfg
        self._chemical_space = ChemicalSpace.from_config(cfg.chemical_space)
        self._property_set = PropertySet.from_config(cfg.property)
        self._tokenization = Tokenization.from_config(cfg.tokenization)

    @classmethod
    def from_file(cls, path: pathlib.Path | str) -> "Facade":
        cfg = OmegaConf.load(path)
        if not isinstance(cfg, DictConfig):
            raise ValueError("Config file does not contain a valid DictConfig.")
        return cls(cfg)

    @property
    def config(self) -> DictConfig:
        return self._cfg

    @property
    def chemical_space(self) -> ChemicalSpace:
        return self._chemical_space

    @property
    def property_set(self) -> PropertySet:
        return self._property_set

    @property
    def tokenization(self) -> Tokenization:
        return self._tokenization

    def get_detokenizer(self) -> Detokenizer:
        csd = self.chemical_space.get_csd()
        return Detokenizer(
            building_blocks=csd.get_primary_building_blocks(),
            reactions=csd.get_reactions(),
            token_def=self.tokenization.token_def,
        )

    def save_config(self, path: pathlib.Path | str) -> None:
        OmegaConf.save(self.config, path)

    def build_model(self) -> PrexSyn:
        return PrexSyn(
            **self.config.model,
            property_embedders=self.property_set.get_embedders(model_dim=self.config.model.dim),
            num_token_types=self.tokenization.token_def.num_token_types,
            max_bb_index=self.chemical_space.count_building_blocks() - 1,
            max_rxn_index=self.chemical_space.count_reactions() - 1,
            pad_token=self.tokenization.token_def.PAD,
            end_token=self.tokenization.token_def.END,
            start_token=self.tokenization.token_def.START,
            bb_token=self.tokenization.token_def.BB,
            rxn_token=self.tokenization.token_def.RXN,
        )

    def load_model(self, path: pathlib.Path | str) -> PrexSyn:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model = self.build_model()
        model.load_state_dict(state_dict)
        return model
