from omegaconf import DictConfig

from .chemical_space import ChemicalSpace
from .property import PropertySet
from .tokenization import Tokenization


class Facade:
    def __init__(self, cfg: DictConfig):
        self._cfg = cfg
        self._chemical_space = ChemicalSpace.from_config(cfg.chemical_space)
        self._property_set = PropertySet.from_config(cfg.property)
        self._tokenization = Tokenization.from_config(cfg.tokenization)

    @property
    def chemical_space(self) -> ChemicalSpace:
        return self._chemical_space

    @property
    def property_set(self) -> PropertySet:
        return self._property_set

    @property
    def tokenization(self) -> Tokenization:
        return self._tokenization
