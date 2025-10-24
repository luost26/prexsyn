import random
from typing import cast

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from prexsyn.data.struct import SynthesisTrainingBatch
from prexsyn.factories.facade import Facade

from .online_dataset import OnlineSynthesisDataset


class SynthesisDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self) -> None:
        facade = Facade(self.cfg)
        val_data_generator = OnlineSynthesisDataset(
            chemical_space=facade.chemical_space,
            property_set=facade.property_set.subset(self.cfg.data.val_properties),
            tokenization=facade.tokenization,
            batch_size=self.cfg.data.batch_size,
            num_threads=1,
            virtual_length=self.cfg.data.num_val_batches,
            base_seed=self.cfg.data.val_base_seed,
        )
        self.val_dataset: list[SynthesisTrainingBatch] = [data for data in val_data_generator]

    def setup(self, stage: str | None = None) -> None:
        if self.trainer is None:
            raise ValueError("Trainer must be set before calling setup().")
        facade = Facade(self.cfg)
        self.train_dataset = OnlineSynthesisDataset(
            chemical_space=facade.chemical_space,
            property_set=facade.property_set,
            tokenization=facade.tokenization,
            batch_size=self.cfg.data.batch_size,
            num_threads=self.cfg.data.num_threads,
            virtual_length=self.cfg.data.virtual_length * self.trainer.world_size,
            base_seed=random.randint(0, 2**32 - 1) + 1024 * self.trainer.global_rank,
        )

    def train_dataloader(self) -> DataLoader[SynthesisTrainingBatch]:
        return DataLoader(
            cast(Dataset[SynthesisTrainingBatch], self.train_dataset),
            batch_size=None,
            num_workers=1,
            collate_fn=lambda x: x,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader[SynthesisTrainingBatch]:
        return DataLoader(
            cast(Dataset[SynthesisTrainingBatch], self.val_dataset),
            batch_size=None,
            num_workers=0,
            collate_fn=lambda x: x,
        )
