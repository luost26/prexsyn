from typing import TYPE_CHECKING, TypedDict, cast

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

import prexsyn_engine
from prexsyn.factory import Config, get_data_pipeline


class SynthesisBatch(TypedDict):
    descriptors: list[tuple[str, torch.Tensor]]
    synthesis: torch.Tensor


class SynthesisDataStream:
    def __init__(
        self,
        config: Config,
        virtual_length: int,
        worker_seeds: list[int],
        descriptor_subset: set[str] | None = None,
    ):
        super().__init__()
        self.config = config
        self.virtual_length = virtual_length
        self.worker_seeds = worker_seeds

        desc_confs = (
            config.descriptors
            if descriptor_subset is None
            else {k: v for k, v in config.descriptors.items() if k in descriptor_subset}
        )
        self.descriptor_slices: list[tuple[str, slice]] = []
        slice_size = config.training.batch_size // len(desc_confs)
        for i, d_name in enumerate(desc_confs.keys()):
            if i == len(desc_confs) - 1:
                self.descriptor_slices.append((d_name, slice(i * slice_size, None)))
            else:
                self.descriptor_slices.append((d_name, slice(i * slice_size, (i + 1) * slice_size)))

    if TYPE_CHECKING:
        _data_pipeline: prexsyn_engine.datapipe.DataPipeline

    @property
    def data_pipeline(self):
        if not hasattr(self, "_data_pipeline"):
            self._data_pipeline = get_data_pipeline(self.config)
            self._data_pipeline.start_workers(self.worker_seeds)
        return self._data_pipeline

    def __getitem__(self, idx: int):
        if idx >= self.virtual_length:
            raise IndexError

        data = self.data_pipeline.get(self.config.training.batch_size)
        batch: SynthesisBatch = {"descriptors": [], "synthesis": torch.from_numpy(data["synthesis"])}
        for d_name, d_slice in self.descriptor_slices:
            batch["descriptors"].append((d_name, torch.from_numpy(data[d_name][d_slice])))

        return batch

    def __iter__(self):
        for idx in range(self.virtual_length):
            yield self[idx]


class SynthesisDataModule(L.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    @property
    def val_cache_path(self):
        return self.config.chemical_space.cache_path.with_suffix(f".val_cache.{self.config.training.val_seed}")

    @property
    def train_seeds(self):
        return [self.config.training.seed * (i + 1) for i in range(16)]

    def prepare_data(self):
        if not self.val_cache_path.exists():
            val_stream = SynthesisDataStream(
                config=self.config,
                virtual_length=self.config.training.num_val_batches,
                worker_seeds=[self.config.training.val_seed],
                descriptor_subset=set(list(self.config.descriptors.keys())[0]),
            )
            self.val_dataset = [batch for batch in val_stream]
            torch.save(self.val_dataset, self.val_cache_path)
        else:
            self.val_dataset = torch.load(self.val_cache_path)

    def setup(self, stage: str | None = None) -> None:
        if self.trainer is None:
            raise ValueError("Trainer must be set before calling setup().")
        self.train_dataset = SynthesisDataStream(
            config=self.config,
            virtual_length=self.config.training.val_freq * self.trainer.world_size,
            worker_seeds=self.train_seeds,
        )

    def train_dataloader(self):
        return DataLoader(
            cast(Dataset, self.train_dataset),
            batch_size=None,
            num_workers=1,
            collate_fn=lambda x: x,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            cast(Dataset, self.val_dataset),
            batch_size=None,
            num_workers=0,
            collate_fn=lambda x: x,
        )
