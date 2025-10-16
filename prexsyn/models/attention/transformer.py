from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from .mha import MultiheadAttention


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[torch.Tensor], torch.Tensor] = nn.functional.relu,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        cross_attn: bool = False,
        self_attn_rot_emb: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, use_rot_emb=self_attn_rot_emb)
        self.cross_attn = (
            MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, use_rot_emb=False) if cross_attn else None
        )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias) if cross_attn else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout) if cross_attn else nn.Identity()
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation

    if TYPE_CHECKING:

        def __call__(
            self,
            tgt: torch.Tensor,
            memory: torch.Tensor | None = None,
            tgt_mask: torch.Tensor | None = None,
            memory_mask: torch.Tensor | None = None,
            tgt_key_padding_mask: torch.Tensor | None = None,
            memory_key_padding_mask: torch.Tensor | None = None,
        ) -> torch.Tensor: ...

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = tgt
        x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
        if memory is not None:
            x = x + self._xa_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
        x = x + self._ff_block(self.norm3(x))
        return x

    # self-attention block
    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return cast(torch.Tensor, self.dropout1(x))

    # multihead attention block
    def _xa_block(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor | float:
        if self.cross_attn is None:
            return 0.0
        x = self.cross_attn(x, mem, mem, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return cast(torch.Tensor, self.dropout2(x))

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return cast(torch.Tensor, self.dropout3(x))
