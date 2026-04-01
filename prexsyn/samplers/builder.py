from typing import Self

import torch

from prexsyn_engine.detokenizer import MultiThreadedDetokenizer


class SynthesisTensorBuilder:
    def __init__(
        self,
        batch_size: int,
        device: torch.device | str,
        bb_token: int,
        rxn_token: int,
        pad_token: int,
        start_token: int,
        end_token: int,
    ) -> None:
        super().__init__()
        self.token_types = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        self.bb_indices = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        self.rxn_indices = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        self.ended = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self.bb_token = bb_token
        self.rxn_token = rxn_token
        self.pad_token = pad_token
        self.end_token = end_token

    def append(self, token_types: torch.Tensor, bb_indices: torch.Tensor, rxn_indices: torch.Tensor) -> Self:
        token_types = torch.where(self.ended[:, None], self.pad_token, token_types)
        bb_indices = torch.where(self.ended[:, None] | (token_types != self.bb_token), 0, bb_indices)
        rxn_indices = torch.where(self.ended[:, None] | (token_types != self.rxn_token), 0, rxn_indices)

        self.token_types = torch.cat([self.token_types, token_types], dim=-1)
        self.bb_indices = torch.cat([self.bb_indices, bb_indices], dim=-1)
        self.rxn_indices = torch.cat([self.rxn_indices, rxn_indices], dim=-1)
        self.ended = torch.logical_or(self.ended, (token_types == self.end_token).any(dim=-1))
        return self

    def get(self, fix_length: int | None = None):
        out = {
            "token_types": self.token_types,
            "bb_indices": self.bb_indices,
            "rxn_indices": self.rxn_indices,
        }
        if fix_length is not None:
            if fix_length < out["token_types"].size(1):
                out = {k: v[:, :fix_length] for k, v in out.items()}
            elif fix_length > out["token_types"].size(1):
                pad_size = fix_length - out["token_types"].size(1)
                out = {k: torch.nn.functional.pad(v, (0, pad_size), value=0) for k, v in out.items()}
        return out

    def get_tensor(self, fix_length: int | None = None):
        d = self.get(fix_length)
        return torch.stack([d["token_types"], d["bb_indices"], d["rxn_indices"]], dim=-1)

    def detokenize(self, detokenizer: MultiThreadedDetokenizer):
        return detokenizer(self.get_tensor().cpu().numpy())
