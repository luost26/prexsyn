import dataclasses

import torch
from torch import nn


@dataclasses.dataclass(frozen=True)
class Prediction:
    token_types: torch.Tensor  # (batch, seq_len, num_token_types)
    bb_indices: torch.Tensor
    rxn_indices: torch.Tensor


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
        )
