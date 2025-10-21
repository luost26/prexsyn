import dataclasses
from typing import cast

import torch
from torch import nn


@dataclasses.dataclass(frozen=True)
class Prediction:
    type_logits: torch.Tensor  # (batch, seq_len, num_token_types)
    bb_logits: torch.Tensor
    rxn_logits: torch.Tensor

    bb_token: int
    rxn_token: int

    def flatten(self) -> "FlattenedPrediction":
        type_logp = nn.functional.log_softmax(self.type_logits, dim=-1)
        type_bb_logp = type_logp[..., self.bb_token].unsqueeze(-1)
        type_rxn_logp = type_logp[..., self.rxn_token].unsqueeze(-1)
        bb_logp = nn.functional.log_softmax(self.bb_logits, dim=-1) + type_bb_logp
        rxn_logp = nn.functional.log_softmax(self.rxn_logits, dim=-1) + type_rxn_logp

        type_logp[..., self.bb_token] = float("-inf")
        type_logp[..., self.rxn_token] = float("-inf")

        logp = torch.cat([bb_logp, rxn_logp, type_logp], dim=-1)
        return FlattenedPrediction(
            logp,
            num_building_blocks=self.bb_logits.shape[-1],
            num_reactions=self.rxn_logits.shape[-1],
            bb_token=self.bb_token,
            rxn_token=self.rxn_token,
        )


@dataclasses.dataclass(frozen=True)
class FlattenedPrediction:
    logp: torch.Tensor

    num_building_blocks: int
    num_reactions: int
    bb_token: int
    rxn_token: int

    def unflatten_tokens(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        n_bb_rxn = self.num_building_blocks + self.num_reactions
        is_bb = torch.where(tokens < self.num_building_blocks, True, False)
        is_rxn = torch.where((tokens >= self.num_building_blocks) & (tokens < n_bb_rxn), True, False)

        token_types = torch.where(is_bb, self.bb_token, torch.where(is_rxn, self.rxn_token, tokens - n_bb_rxn))
        bb_indices = torch.where(is_bb, tokens, 0)
        rxn_indices = torch.where(is_rxn, tokens - self.num_building_blocks, 0)
        return {
            "token_types": token_types,
            "bb_indices": bb_indices,
            "rxn_indices": rxn_indices,
        }


class SynthesisOutput(nn.Module):
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
        self.token_head = nn.Linear(dim, num_token_types)
        bb_embed_dim = bb_embed_dim or dim
        self.bb_head = nn.Sequential(
            nn.Identity() if bb_embed_dim == dim else nn.Linear(dim, bb_embed_dim),
            nn.Linear(bb_embed_dim, max_bb_index + 1),
        )
        self.rxn_head = nn.Linear(dim, max_rxn_index + 1)
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
            type_logits=self.token_head(h),
            bb_logits=self.bb_head(h),
            rxn_logits=self.rxn_head(h),
            bb_token=self.bb_token,
            rxn_token=self.rxn_token,
        )

    def predict_type(self, h: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.token_head(h))

    def predict_bb(self, h: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.bb_head(h))

    def predict_rxn(self, h: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.rxn_head(h))
