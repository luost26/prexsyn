import torch

from prexsyn.data.struct import (
    PropertyRepr,
    SynthesisRepr,
    concat_synthesis_reprs,
    get_property_repr_batch_size,
    move_to_device,
)
from prexsyn.models.embeddings import Embedding
from prexsyn.models.outputs.synthesis import Prediction
from prexsyn.models.prexsyn import PrexSyn
from prexsyn.queries import Query, QueryPlanner
from prexsyn_engine.featurizer.synthesis import PostfixNotationTokenDef

from .builder import SynthesisReprBuilder


def _composite(pred: Prediction, weight: torch.Tensor) -> Prediction:
    n_conds = weight.size(0)
    n_samples = pred.type_logits.size(0) // n_conds
    seqlen = pred.type_logits.size(1)
    weight = weight.reshape(1, -1, 1, 1)

    return Prediction(
        type_logits=(pred.type_logits.reshape(n_samples, n_conds, seqlen, -1) * weight).sum(dim=1),
        bb_logits=(pred.bb_logits.reshape(n_samples, n_conds, seqlen, -1) * weight).sum(dim=1),
        rxn_logits=(pred.rxn_logits.reshape(n_samples, n_conds, seqlen, -1) * weight).sum(dim=1),
        bb_token=pred.bb_token,
        rxn_token=pred.rxn_token,
    )


def _add(pred: Prediction, other: Prediction, coef: float) -> Prediction:
    return Prediction(
        type_logits=pred.type_logits + coef * other.type_logits,
        bb_logits=pred.bb_logits + coef * other.bb_logits,
        rxn_logits=pred.rxn_logits + coef * other.rxn_logits,
        bb_token=pred.bb_token,
        rxn_token=pred.rxn_token,
    )


def _sample_from_pred(
    pred: Prediction, ended: torch.Tensor, token_def: PostfixNotationTokenDef
) -> dict[str, torch.Tensor]:
    batch_shape = pred.type_logits.shape[:-1]
    ended = ended.flatten()

    type_logits = pred.type_logits.flatten(0, -2)
    next_type = torch.multinomial(torch.softmax(type_logits, dim=-1), num_samples=1).squeeze(-1)
    next_type = torch.where(ended, token_def.PAD, next_type)

    next_bb = torch.zeros_like(next_type)
    bb_pos = (next_type == token_def.BB).nonzero(as_tuple=True)[0]
    if bb_pos.numel() > 0:
        bb_logits = pred.bb_logits.flatten(0, -2)[bb_pos]
        next_bb_subset = torch.multinomial(torch.softmax(bb_logits, dim=-1), num_samples=1).squeeze(-1)
        next_bb[bb_pos] = next_bb_subset

    next_rxn = torch.zeros_like(next_type)
    rxn_pos = (next_type == token_def.RXN).nonzero(as_tuple=True)[0]
    if rxn_pos.numel() > 0:
        rxn_logits = pred.rxn_logits.flatten(0, -2)[rxn_pos]
        next_rxn_subset = torch.multinomial(torch.softmax(rxn_logits, dim=-1), num_samples=1).squeeze(-1)
        next_rxn[rxn_pos] = next_rxn_subset

    return {
        "token_types": next_type.reshape(batch_shape),
        "bb_indices": next_bb.reshape(batch_shape),
        "rxn_indices": next_rxn.reshape(batch_shape),
    }


class QuerySampler:
    def __init__(
        self,
        model: PrexSyn,
        token_def: PostfixNotationTokenDef,
        num_samples: int,
        max_length: int = 16,
        model_batch_size: int = 4096,
    ) -> None:
        super().__init__()
        self.model = model
        self.token_def = token_def
        self.num_samples = num_samples
        self.max_length = max_length
        self.model_batch_size = model_batch_size

    def _create_builder(self, batch_size: int) -> SynthesisReprBuilder:
        return SynthesisReprBuilder(
            batch_size=batch_size,
            device=self.model.device,
            bb_token=self.token_def.BB,
            rxn_token=self.token_def.RXN,
            pad_token=self.token_def.PAD,
            start_token=self.token_def.START,
            end_token=self.token_def.END,
        )

    def _sample_conjunctive(self, e_property: Embedding, weight: torch.Tensor, num_samples: int) -> SynthesisRepr:
        num_conditions = e_property.batch_size
        builder = self._create_builder(num_samples)
        e_property = e_property.repeat(num_samples)

        for _ in range(self.max_length):
            e_synthesis = self.model.embed_synthesis(builder.get()).repeat_interleave(num_conditions)
            h_syn = self.model.encode(e_property, e_synthesis)

            next_pred = _composite(self.model.predict(h_syn[..., -1:, :]), weight)
            next = _sample_from_pred(next_pred, builder.ended, self.token_def)
            builder.append(**next)

            if builder.ended.all():
                break

        return builder.get(self.max_length)

    @torch.no_grad()
    def sample(self, query: Query) -> SynthesisRepr:
        planner = QueryPlanner(query)
        property_repr_list = move_to_device(planner.get_property_reprs(), self.model.device)
        weight_list = move_to_device(planner.get_weights(), self.model.device)
        e_property_list = [self.model.embed_properties(prop_repr) for prop_repr in property_repr_list]

        sample_list: list[SynthesisRepr] = []
        for e_property, weight in zip(e_property_list, weight_list):
            n_conds = weight.size(0)
            chunk_size = self.model_batch_size // n_conds
            for i in range(0, self.num_samples, chunk_size):
                samples = self._sample_conjunctive(
                    e_property,
                    weight,
                    num_samples=min(chunk_size, self.num_samples - i),
                )
                sample_list.append(samples)

        return concat_synthesis_reprs(*sample_list)


class QueryConditionedSampler:
    def __init__(
        self,
        model: PrexSyn,
        token_def: PostfixNotationTokenDef,
        max_length: int = 16,
        model_batch_size: int = 4096,
    ) -> None:
        super().__init__()
        self.model = model
        self.token_def = token_def
        self.max_length = max_length
        self.model_batch_size = model_batch_size

    def _create_builder(self, batch_size: int) -> SynthesisReprBuilder:
        return SynthesisReprBuilder(
            batch_size=batch_size,
            device=self.model.device,
            bb_token=self.token_def.BB,
            rxn_token=self.token_def.RXN,
            pad_token=self.token_def.PAD,
            start_token=self.token_def.START,
            end_token=self.token_def.END,
        )

    def _sample_conjunctive(
        self,
        e_batched_property: Embedding,
        e_query_property: Embedding,
        weight: torch.Tensor,
        batched_property_weight: float,
    ) -> SynthesisRepr:
        num_samples = e_batched_property.batch_size
        num_conditions = e_query_property.batch_size
        builder = self._create_builder(num_samples)
        e_batched_property = e_batched_property.repeat_interleave(num_conditions)
        e_query_property = e_query_property.repeat(num_samples)

        for _ in range(self.max_length):
            e_synthesis = self.model.embed_synthesis(builder.get()).repeat_interleave(num_conditions)
            h_syn_q = self.model.encode(e_query_property, e_synthesis)
            next_pred_q = _composite(self.model.predict(h_syn_q[..., -1:, :]), weight)  # (num_samples, vocab_size)

            h_syn_b = self.model.encode(e_batched_property, e_synthesis)
            next_pred_b = self.model.predict(h_syn_b[..., -1:, :])  # (num_samples, vocab_size)

            next_pred = _add(next_pred_q, next_pred_b, coef=batched_property_weight)

            next = _sample_from_pred(next_pred, builder.ended, self.token_def)
            builder.append(**next)

            if builder.ended.all():
                break

        return builder.get(self.max_length)

    @torch.no_grad()
    def sample(
        self, batched_property_repr: PropertyRepr, query: Query, batched_property_weight: float
    ) -> SynthesisRepr:
        planner = QueryPlanner(query)
        query_property_repr_list = move_to_device(planner.get_property_reprs(), self.model.device)
        weight_list = move_to_device(planner.get_weights(), self.model.device)
        e_query_property_list = [self.model.embed_properties(prop_repr) for prop_repr in query_property_repr_list]
        e_batched_property = self.model.embed_properties(batched_property_repr)

        num_samples = get_property_repr_batch_size(batched_property_repr)

        sample_list: list[SynthesisRepr] = []
        for e_query_property, weight in zip(e_query_property_list, weight_list):
            n_conds = weight.size(0)
            chunk_size = self.model_batch_size // n_conds
            for i in range(0, num_samples, chunk_size):
                samples = self._sample_conjunctive(
                    e_batched_property[i : i + chunk_size],
                    e_query_property,
                    weight,
                    batched_property_weight,
                )
                sample_list.append(samples)

        return concat_synthesis_reprs(*sample_list)
