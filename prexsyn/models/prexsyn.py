from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import torch
from torch import nn

from prexsyn.data.struct import EmbedderName, PropertyRepr, SynthesisRepr

from .attention import TransformerLayer
from .embeddings import Embedding, SynthesisEmbedder
from .outputs.synthesis import SynthesisOutput


class PrexSyn(nn.Module):
    def __init__(
        self,
        dim: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        property_embedders: nn.ModuleDict,
        synthesis_embedder_config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.dim = dim
        self.synthesis_embedder = SynthesisEmbedder(dim=dim, **synthesis_embedder_config)
        self.property_embedders = cast(Mapping[EmbedderName, Callable[..., Embedding]], property_embedders)
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
        self.synthesis_output = SynthesisOutput(dim=dim, **synthesis_embedder_config)

    def embed_properties(self, property_repr: PropertyRepr) -> Embedding:
        prop_embs: list[Embedding] = []
        for prop_dict in property_repr:
            prop_embs.append(
                Embedding.concat_along_seq_dim(
                    *[self.property_embedders[name](**params) for name, params in prop_dict.items()]
                )
            )
        e_property = Embedding.concat_along_batch_dim(*prop_embs)
        return e_property

    def embed_synthesis(self, synthesis_repr: SynthesisRepr) -> Embedding:
        return self.synthesis_embedder(**synthesis_repr)

    def encode(self, e_property: Embedding, e_synthesis: Embedding) -> torch.Tensor:
        e = Embedding.concat_along_seq_dim(e_property, e_synthesis)

        attn_mask = nn.Transformer.generate_square_subsequent_mask(
            e_property.sequence_length + e_synthesis.sequence_length,
            device=e_synthesis.padding_mask.device,
            dtype=e_synthesis.padding_mask.dtype,
        )
        attn_mask[..., : e_property.sequence_length] = 0.0

        h = e.embedding
        padding_mask = e.padding_mask
        for layer in self.transformer_layers:
            h = layer(h, tgt_mask=attn_mask, tgt_key_padding_mask=padding_mask)
        h_syn = h[..., -e_synthesis.sequence_length :, :]
        return h_syn

    def forward(
        self,
        property_repr: PropertyRepr,
        synthesis_repr: SynthesisRepr,
        data_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        e_property = self.embed_properties(property_repr)
        e_synthesis = self.embed_synthesis(synthesis_repr)
        h_syn = self.encode(e_property, e_synthesis)

        loss_dict = self.synthesis_output.get_loss(
            h_syn,
            token_types=synthesis_repr["token_types"],
            bb_indices=synthesis_repr["bb_indices"],
            rxn_indices=synthesis_repr["rxn_indices"],
            data_weights=data_weights,
        )

        return loss_dict
