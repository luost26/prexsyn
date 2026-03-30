import math
from typing import TYPE_CHECKING

import torch
from torch import nn

from .pe import RotaryEmbedding


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        bias: bool,
        use_rot_emb: bool,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reset_parameters()

        self.rot_emb = RotaryEmbedding(dim=self.head_dim) if use_rot_emb else None

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    if TYPE_CHECKING:

        def __call__(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            key_padding_mask: torch.Tensor | None = None,
            attn_mask: torch.Tensor | None = None,
        ) -> torch.Tensor: ...

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if key_padding_mask is not None and not key_padding_mask.is_floating_point():
            raise ValueError("key_padding_mask must be a floating point tensor")
        if attn_mask is not None and not attn_mask.is_floating_point():
            raise ValueError("attn_mask must be a floating point tensor")

        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)

        q: torch.Tensor = self.q_proj(query)  # q scaling is fused to scaled_dot_product_attention
        k: torch.Tensor = self.k_proj(key)
        v: torch.Tensor = self.v_proj(value)

        # (bsz, heads, tgt_len, head_dim)
        q = q.reshape(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.reshape(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.reshape(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        if self.rot_emb is not None:
            q, k = self.rot_emb(q, k)

        merged_attn_mask = self.merge_mask(attn_mask, key_padding_mask, tgt_len=tgt_len)
        if merged_attn_mask is not None:
            merged_attn_mask = merged_attn_mask[:, None, :, :]
        attn_output: torch.Tensor = nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=merged_attn_mask,
            dropout_p=(self.dropout if self.training else 0.0),
            scale=self.scaling,
        )  # (bsz, heads, tgt_len, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output

    @staticmethod
    def merge_mask(
        attn_mask: torch.Tensor | None, key_padding_mask: torch.Tensor | None, tgt_len: int
    ) -> torch.Tensor | None:
        """
        Args:
            attn_mask:  (bsz, tgt_len, src_len)
            key_padding_mask:  (bsz, src_len)
        """
        if attn_mask is not None and key_padding_mask is not None:
            return attn_mask + key_padding_mask[:, None, :].repeat(1, tgt_len, 1)
        elif attn_mask is not None:
            return attn_mask
        elif key_padding_mask is not None:
            return key_padding_mask[:, None, :].repeat(1, tgt_len, 1)
        else:
            return None
