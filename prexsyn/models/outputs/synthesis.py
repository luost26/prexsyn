import dataclasses

import torch
from torch import nn


@dataclasses.dataclass(frozen=True)
class Prediction:
    token_types: torch.Tensor  # (batch, seq_len, num_token_types)
    bb_indices: torch.Tensor
    rxn_indices: torch.Tensor

    _bb_token: int
    _rxn_token: int

    def sample(
        self,
        t_types: float = 1.0,
        t_bb: float = 1.0,
        t_rxn: float = 1.0,
        n_bb: int | None = None,
        n_rxn: int | None = None,
    ) -> dict[str, torch.Tensor]:
        def _sample(logits: torch.Tensor, t: float, n: int | None = None) -> torch.Tensor:
            shape = logits.shape[:-1]
            logits = logits.reshape(-1, logits.shape[-1])
            if n is not None:
                logits = logits[:, :n]
            out = torch.multinomial(torch.softmax(logits / t, dim=-1), num_samples=1)
            return out.reshape(shape)

        return {
            "token_types": _sample(self.token_types, t_types),
            "bb_indices": _sample(self.bb_indices, t_bb, n_bb),
            "rxn_indices": _sample(self.rxn_indices, t_rxn, n_rxn),
        }

    def topk(self, k: int, n_bb: int | None = None, n_rxn: int | None = None) -> dict[str, torch.Tensor]:
        n_types = self.token_types.shape[-1]
        n_bb = n_bb or self.bb_indices.shape[-1]
        n_rxn = n_rxn or self.rxn_indices.shape[-1]

        logp_types = torch.log_softmax(self.token_types, dim=-1)
        logp_bb = torch.log_softmax(self.bb_indices[..., :n_bb], dim=-1) + logp_types[..., self._bb_token][..., None]
        logp_rxn = (
            torch.log_softmax(self.rxn_indices[..., :n_rxn], dim=-1) + logp_types[..., self._rxn_token][..., None]
        )

        logp_types[..., self._bb_token] = float("-inf")
        logp_types[..., self._rxn_token] = float("-inf")

        logp_all = torch.cat([logp_types, logp_bb, logp_rxn], dim=-1)
        topk_logp, topk_indices = torch.topk(logp_all, k=k, dim=-1, largest=True)

        is_bb = torch.logical_and(topk_indices >= n_types, topk_indices < n_types + n_bb)
        is_rxn = torch.logical_and(topk_indices >= n_types + n_bb, topk_indices < n_types + n_bb + n_rxn)
        is_other = topk_indices < n_types

        token_types = torch.where(is_other, topk_indices, torch.where(is_bb, self._bb_token, self._rxn_token))
        bb_indices = torch.where(is_bb, topk_indices - n_types, 0)
        rxn_indices = torch.where(is_rxn, topk_indices - (n_types + n_bb), 0)
        return {
            "token_types": token_types,
            "bb_indices": bb_indices,
            "rxn_indices": rxn_indices,
            "logp": topk_logp,
        }

    def composite(self, coef: torch.Tensor) -> "Prediction":
        n_props = coef.size(0)
        n_samples = self.token_types.size(0) // n_props
        seqlen = self.token_types.size(1)
        coef = coef.reshape(1, -1, 1, 1)
        return Prediction(
            token_types=(self.token_types.reshape(n_samples, n_props, seqlen, -1) * coef).sum(dim=1),
            bb_indices=(self.bb_indices.reshape(n_samples, n_props, seqlen, -1) * coef).sum(dim=1),
            rxn_indices=(self.rxn_indices.reshape(n_samples, n_props, seqlen, -1) * coef).sum(dim=1),
            _bb_token=self._bb_token,
            _rxn_token=self._rxn_token,
        )

    def add(self, other: "Prediction", coef: float) -> "Prediction":
        return Prediction(
            token_types=self.token_types + other.token_types * coef,
            bb_indices=self.bb_indices + other.bb_indices * coef,
            rxn_indices=self.rxn_indices + other.rxn_indices * coef,
            _bb_token=self._bb_token,
            _rxn_token=self._rxn_token,
        )


class SynthesisOutput(nn.Module):
    def __init__(
        self,
        dim: int,
        num_token_types: int,
        max_bb_indices: int,
        max_rxn_indices: int,
        bb_embed_dim: int | None,
        pad_token: int,
        bb_token: int,
        rxn_token: int,
        end_token: int,
    ) -> None:
        super().__init__()
        self.token_head = nn.Linear(dim, num_token_types)
        bb_embed_dim = bb_embed_dim or dim
        self.bb_head = nn.Sequential(
            nn.Identity() if bb_embed_dim == dim else nn.Linear(dim, bb_embed_dim),
            nn.Linear(bb_embed_dim, max_bb_indices + 1),
        )
        self.rxn_head = nn.Linear(dim, max_rxn_indices + 1)
        self.pad_token = pad_token
        self.bb_token = bb_token
        self.rxn_token = rxn_token
        self.end_token = end_token

    def get_loss(
        self,
        h: torch.Tensor,
        token_types: torch.Tensor,
        bb_indices: torch.Tensor,
        rxn_indices: torch.Tensor,
        data_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        ar_shape = list(token_types.shape[:-1]) + [token_types.shape[-1] - 1]
        if data_weights is None:
            data_weights = torch.ones(ar_shape, dtype=h.dtype, device=h.device).reshape(-1)
        else:
            data_weights = data_weights.unsqueeze(-1).expand(ar_shape).reshape(-1)

        h = h[..., :-1, :].reshape(-1, h.shape[-1])
        token_types = token_types[..., 1:].reshape(-1)
        bb_indices = bb_indices[..., 1:].reshape(-1)
        rxn_indices = rxn_indices[..., 1:].reshape(-1)

        loss_dict: dict[str, torch.Tensor] = {}

        token_types_m = token_types != self.pad_token
        loss_dict["token_types"] = (
            nn.functional.cross_entropy(
                input=self.token_head(h[token_types_m]), target=token_types[token_types_m], reduction="none"
            )
            * data_weights[token_types_m]
        ).mean()

        bb_indices_m = token_types == self.bb_token
        loss_dict["bb_indices"] = (
            nn.functional.cross_entropy(
                input=self.bb_head(h[bb_indices_m]),
                target=bb_indices[bb_indices_m],
            )
            * data_weights[bb_indices_m]
        ).mean()

        rxn_indices_m = token_types == self.rxn_token
        loss_dict["rxn_indices"] = (
            nn.functional.cross_entropy(
                input=self.rxn_head(h[rxn_indices_m]),
                target=rxn_indices[rxn_indices_m],
            )
            * data_weights[rxn_indices_m]
        ).mean()

        return loss_dict

    def predict(self, h: torch.Tensor) -> Prediction:
        return Prediction(
            token_types=self.token_head(h),
            bb_indices=self.bb_head(h),
            rxn_indices=self.rxn_head(h),
            _bb_token=self.bb_token,
            _rxn_token=self.rxn_token,
        )
