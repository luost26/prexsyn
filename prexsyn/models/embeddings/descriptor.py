import abc
from typing import TYPE_CHECKING, TypedDict, cast

import torch
from torch import nn

from .base import Embedding


class DescriptorEmbedderConfig(TypedDict):
    descriptor_dim: int
    num_tokens: int


class DescriptorEmbedder(nn.Module, abc.ABC):
    def __init__(self, descriptor_dim: int, embedding_dim: int, num_tokens: int) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.mlp = nn.Sequential(
            nn.Linear(descriptor_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim * num_tokens),
        )

    @property
    def dtype(self):
        return cast(torch.dtype, self.mlp[0].weight.dtype)

    if TYPE_CHECKING:

        def __call__(self, descriptor: torch.Tensor) -> Embedding: ...

    def forward(self, descriptor: torch.Tensor) -> Embedding:
        descriptor = descriptor.to(cast(torch.device, self.mlp[0].weight.device))
        h = self.mlp(descriptor)
        h = h.reshape(*h.shape[:-1], self.num_tokens, -1)
        return Embedding(
            embedding=h,
            padding_mask=Embedding.create_full_padding_mask(h),
        )
