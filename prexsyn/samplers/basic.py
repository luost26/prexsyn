import torch

from prexsyn.models.embeddings import Embedding
from prexsyn.models.prexsyn import PrexSyn

from .builder import SynthesisTensorBuilder


class BasicSampler:
    def __init__(
        self,
        model: PrexSyn,
        num_samples: int,
        batch_size_limit: int,
        t_types: float = 1.0,
        t_bb: float = 1.0,
        t_rxn: float = 1.0,
        max_length: int = 16,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        self.batch_size_limit = batch_size_limit
        self.t_types = t_types
        self.t_bb = t_bb
        self.t_rxn = t_rxn
        self.max_length = max_length

        if self.batch_size_limit <= 0:
            raise ValueError("batch_size_limit must be positive.")

    def _create_builder(self, batch_size: int):
        return SynthesisTensorBuilder(
            batch_size=batch_size,
            device=self.model.device,
            bb_token=self.model.bb_token,
            rxn_token=self.model.rxn_token,
            pad_token=self.model.pad_token,
            start_token=self.model.start_token,
            end_token=self.model.end_token,
        )

    def _predict_and_sample(self, h_next: torch.Tensor, ended: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_shape = h_next.shape[:-1]
        h_next = h_next.flatten(0, -2)
        ended = ended.flatten()

        type_logits = self.model.predict_token_type(h_next) / self.t_types
        next_type = torch.multinomial(torch.softmax(type_logits, dim=-1), num_samples=1).squeeze(-1)
        next_type = torch.where(ended, self.model.pad_token, next_type)

        next_bb = torch.zeros_like(next_type)
        bb_pos = (next_type == self.model.bb_token).nonzero(as_tuple=True)[0]
        if bb_pos.numel() > 0:
            bb_logits = self.model.predict_building_block(h_next[bb_pos]) / self.t_bb
            next_bb_subset = torch.multinomial(torch.softmax(bb_logits, dim=-1), num_samples=1).squeeze(-1)
            next_bb[bb_pos] = next_bb_subset

        next_rxn = torch.zeros_like(next_type)
        rxn_pos = (next_type == self.model.rxn_token).nonzero(as_tuple=True)[0]
        if rxn_pos.numel() > 0:
            rxn_logits = self.model.predict_reaction(h_next[rxn_pos]) / self.t_rxn
            next_rxn_subset = torch.multinomial(torch.softmax(rxn_logits, dim=-1), num_samples=1).squeeze(-1)
            next_rxn[rxn_pos] = next_rxn_subset

        return {
            "token_types": next_type.reshape(batch_shape),
            "bb_indices": next_bb.reshape(batch_shape),
            "rxn_indices": next_rxn.reshape(batch_shape),
        }

    def _chunk_descriptor(self, descriptor: tuple[str, torch.Tensor]):
        name, values = descriptor
        if values.size(0) <= self.batch_size_limit:
            yield descriptor
            return

        for start in range(0, values.size(0), self.batch_size_limit):
            yield (name, values[start : start + self.batch_size_limit])

    @staticmethod
    def _concat_builders(builders: list[SynthesisTensorBuilder]) -> SynthesisTensorBuilder:
        if not builders:
            raise ValueError("builders must not be empty.")
        if len(builders) == 1:
            return builders[0]

        max_seq_len = max(builder.token_types.size(1) for builder in builders)
        first = builders[0]
        out = SynthesisTensorBuilder(
            batch_size=sum(builder.token_types.size(0) for builder in builders),
            device=first.token_types.device,
            bb_token=first.bb_token,
            rxn_token=first.rxn_token,
            pad_token=first.pad_token,
            start_token=int(first.token_types[0, 0].item()),
            end_token=first.end_token,
        )

        padded = [builder.get(fix_length=max_seq_len) for builder in builders]
        out.token_types = torch.cat([item["token_types"] for item in padded], dim=0)
        out.bb_indices = torch.cat([item["bb_indices"] for item in padded], dim=0)
        out.rxn_indices = torch.cat([item["rxn_indices"] for item in padded], dim=0)
        out.ended = torch.cat([builder.ended for builder in builders], dim=0)
        return out

    @torch.no_grad()
    def _sample_with_property(self, e_property: Embedding) -> SynthesisTensorBuilder:
        # e_property is an Embedding object from model.embed_descriptors.
        batch_size = e_property.batch_size
        builder = self._create_builder(batch_size)
        for _ in range(self.max_length):
            e_synthesis = self.model.embed_synthesis(**builder.get())
            h_syn = self.model.encode(e_property, e_synthesis)

            next = self._predict_and_sample(h_syn[..., -1:, :], builder.ended)
            builder.append(**next)

            if builder.ended.all():
                break
        return builder

    @torch.no_grad()
    def _sample_once(self, descriptors: list[tuple[str, torch.Tensor]]) -> SynthesisTensorBuilder:
        e_property = self.model.embed_descriptors(descriptors)
        return self._sample_with_property(e_property)

    @torch.no_grad()
    def sample(self, descriptors: list[tuple[str, torch.Tensor]] | tuple[str, torch.Tensor]):
        if isinstance(descriptors, tuple):
            descriptors = [descriptors]

        chunked_descriptors: list[tuple[str, torch.Tensor]] = []
        for descriptor in descriptors:
            chunked_descriptors.extend(self._chunk_descriptor(descriptor))

        builders: list[SynthesisTensorBuilder] = []
        for _ in range(self.num_samples):
            for descriptor_chunk in chunked_descriptors:
                builders.append(self._sample_once([descriptor_chunk]))

        return self._concat_builders(builders)
