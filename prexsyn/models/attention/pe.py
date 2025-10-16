import math
from typing import TYPE_CHECKING, cast

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    if TYPE_CHECKING:

        def __call__(self, x: torch.Tensor, position: torch.Tensor | None = None) -> torch.Tensor: ...

    def forward(self, x: torch.Tensor, position: torch.Tensor | None = None) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        batch_size, seqlen, d_model = x.size()
        if position is None:
            position = torch.arange(seqlen, device=x.device)[None, :].expand(batch_size, seqlen)
        position = position.unsqueeze(-1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=x.device, dtype=x.dtype) * (-math.log(10000.0) / d_model)
        )
        pe_sin = torch.sin(position * div_term)
        pe_cos = torch.cos(position * div_term)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)
        x = x + pe
        return cast(torch.Tensor, self.dropout(x))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[:, : x.shape[-2], :]
    sin = sin[:, : x.shape[-2], :]

    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.
    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration
    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox
    .. warning: Please note that this embedding is not registered on purpose, as it is transformative
        (it does not create the embedding dimension) and will likely be picked up (imported) on a ad-hoc basis
    """

    def __init__(self, dim: int):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.inv_freq = nn.Parameter(inv_freq, requires_grad=False)

        self._seq_len_cached: int = 0
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None

    def _update_cos_sin_tables(self, x: torch.Tensor, seq_dimension: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len > self._seq_len_cached:
            print(f"Updating cos and sin tables for rotary embeddings (seq_len={seq_len}, old={self._seq_len_cached})")
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, :, :]
            self._sin_cached = emb.sin()[None, :, :]

        if self._cos_cached is None or self._sin_cached is None:
            raise RuntimeError("Cosine and sine tables are not initialized.")

        return self._cos_cached, self._sin_cached

    if TYPE_CHECKING:

        def __call__(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

        head_dim = q.shape[-1]
        seqlen = q.shape[-2]
        q_shape, k_shape = q.shape, k.shape
        q = q.reshape(-1, seqlen, head_dim)
        k = k.reshape(-1, seqlen, head_dim)

        q, k = (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )
        return q.view(q_shape), k.view(k_shape)
