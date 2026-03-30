from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from .attention import TransformerLayer, generate_square_subsequent_mask
from .embeddings import Embedding, SynthesisEmbedder, DescriptorEmbedder, DescriptorEmbedderConfig
from .outputs.synthesis import Prediction, SynthesisOutput


class PrexSyn(nn.Module):
    def __init__(
        self,
        dim: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        bb_embed_dim: int,
        descriptor_configs: Mapping[str, DescriptorEmbedderConfig],
        num_token_types: int,
        max_bb_index: int,
        max_rxn_index: int,
        pad_token: int,
        end_token: int,
        start_token: int,
        bb_token: int,
        rxn_token: int,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.synthesis_embedder = SynthesisEmbedder(
            dim=dim,
            num_token_types=num_token_types,
            max_bb_index=max_bb_index,
            max_rxn_index=max_rxn_index,
            bb_embed_dim=bb_embed_dim,
            pad_token=pad_token,
            bb_token=bb_token,
            rxn_token=rxn_token,
            end_token=end_token,
        )
        self.descriptor_embedders = self.create_descriptor_embedders(descriptor_configs, dim)
        self.transformer_layers = cast(
            Sequence[TransformerLayer],
            nn.ModuleList(
                TransformerLayer(
                    d_model=dim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                )
                for _ in range(num_layers)
            ),
        )
        self.synthesis_output = SynthesisOutput(
            dim=dim,
            num_token_types=num_token_types,
            max_bb_index=max_bb_index,
            max_rxn_index=max_rxn_index,
            bb_embed_dim=bb_embed_dim,
            pad_token=pad_token,
            bb_token=bb_token,
            rxn_token=rxn_token,
            end_token=end_token,
        )

    @staticmethod
    def create_descriptor_embedders(
        descriptor_configs: Mapping[str, DescriptorEmbedderConfig],
        embedding_dim: int,
    ) -> Mapping[str, DescriptorEmbedder]:
        mods = nn.ModuleDict()
        for name, config in descriptor_configs.items():
            mods[name] = DescriptorEmbedder(
                descriptor_dim=config["descriptor_dim"],
                embedding_dim=embedding_dim,
                num_tokens=config["num_tokens"],
            )
        return cast(Mapping[str, DescriptorEmbedder], mods)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def embed_descriptor(self, name: str, descriptor: torch.Tensor) -> Embedding:
        return self.descriptor_embedders[name](descriptor)

    def embed_synthesis(
        self, token_types: torch.Tensor, bb_indices: torch.Tensor, rxn_indices: torch.Tensor
    ) -> Embedding:
        return self.synthesis_embedder(token_types=token_types, bb_indices=bb_indices, rxn_indices=rxn_indices)

    def encode(self, e_descriptor: Embedding, e_synthesis: Embedding) -> torch.Tensor:
        e = Embedding.concat_along_seq_dim(e_descriptor, e_synthesis)

        attn_mask = generate_square_subsequent_mask(
            e_descriptor.sequence_length + e_synthesis.sequence_length,
            device=e_synthesis.padding_mask.device,
            dtype=e_synthesis.padding_mask.dtype,
        )
        attn_mask[..., : e_descriptor.sequence_length] = 0.0

        h = e.embedding
        padding_mask = e.padding_mask
        for layer in self.transformer_layers:
            h = layer(h, tgt_mask=attn_mask, tgt_key_padding_mask=padding_mask)
        h_syn = h[..., -e_synthesis.sequence_length :, :]
        return h_syn

    def predict(self, h_syn: torch.Tensor) -> Prediction:
        return self.synthesis_output.predict(h_syn)

    def predict_token_type(self, h_syn: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.synthesis_output.token_head(h_syn))

    def predict_building_block(self, h_syn: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.synthesis_output.bb_head(h_syn))

    def predict_reaction(self, h_syn: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.synthesis_output.rxn_head(h_syn))

    if TYPE_CHECKING:

        def __call__(
            self,
            descriptor_name: str,
            descriptor: torch.Tensor,
            token_types: torch.Tensor,
            bb_indices: torch.Tensor,
            rxn_indices: torch.Tensor,
            data_weights: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]: ...

    def forward(
        self,
        descriptor_name: str,
        descriptor: torch.Tensor,
        token_types: torch.Tensor,
        bb_indices: torch.Tensor,
        rxn_indices: torch.Tensor,
        data_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        e_descriptor = self.embed_descriptor(descriptor_name, descriptor)
        e_synthesis = self.embed_synthesis(token_types, bb_indices, rxn_indices)
        h_syn = self.encode(e_descriptor, e_synthesis)

        loss_dict = self.synthesis_output.get_loss(
            h_syn,
            token_types=token_types,
            bb_indices=bb_indices,
            rxn_indices=rxn_indices,
            data_weights=data_weights,
        )

        return loss_dict
