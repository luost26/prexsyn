from typing import TYPE_CHECKING, TypedDict

import torch

import prexsyn_engine
from prexsyn.factory import Config, get_data_pipeline


class SynthesisBatch(TypedDict):
    descriptors: list[tuple[str, torch.Tensor]]
    synthesis: torch.Tensor


class SynthesisDataStream:
    def __init__(self, config: Config, virtual_length: int, worker_seeds: list[int]):
        super().__init__()
        self.config = config
        self.virtual_length = virtual_length
        self.worker_seeds = worker_seeds

        self.descriptor_slices: list[tuple[str, slice]] = []
        slice_size = config.training.batch_size // len(config.descriptors)
        for i, d_name in enumerate(config.descriptors.keys()):
            if i == len(config.descriptors) - 1:
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
