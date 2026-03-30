from typing import TYPE_CHECKING

import torch
from torch import nn

from ..attention.pe import PositionalEncoding
from .base import Embedding


class SynthesisEmbedder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_token_types: int,
        max_bb_index: int,
        max_rxn_index: int,
        bb_embed_dim: int | None,
        pad_token: int,
        bb_token: int,
        rxn_token: int,
        end_token: int,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(num_token_types, dim, padding_idx=pad_token)
        bb_embed_dim = bb_embed_dim or dim
        self.bb_embedding = nn.Sequential(
            nn.Embedding(max_bb_index + 1, bb_embed_dim),
            nn.Identity() if bb_embed_dim == dim else nn.Linear(bb_embed_dim, dim),
        )
        self.rxn_embedding = nn.Embedding(max_rxn_index + 1, dim)
        self.positional_encoding = PositionalEncoding()
        self.pad_token = pad_token
        self.bb_token = bb_token
        self.rxn_token = rxn_token
        self.end_token = end_token

    if TYPE_CHECKING:

        def __call__(
            self,
            token_types: torch.Tensor,
            bb_indices: torch.Tensor,
            rxn_indices: torch.Tensor,
        ) -> Embedding: ...

    def forward(
        self,
        token_types: torch.Tensor,
        bb_indices: torch.Tensor,
        rxn_indices: torch.Tensor,
    ) -> Embedding:
        token_emb = self.token_embedding(token_types)
        bb_emb = self.bb_embedding(bb_indices)
        rxn_emb = self.rxn_embedding(rxn_indices)

        is_bb = (token_types == self.bb_token).unsqueeze(-1).expand_as(token_emb)
        is_rxn = (token_types == self.rxn_token).unsqueeze(-1).expand_as(token_emb)
        h = torch.where(is_bb, bb_emb, torch.where(is_rxn, rxn_emb, token_emb))
        h = self.positional_encoding(h)

        m = torch.full(token_types.size(), fill_value=float("-inf"), dtype=h.dtype, device=h.device)
        m.masked_fill_(token_types != self.pad_token, 0.0)

        return Embedding(h, m)
