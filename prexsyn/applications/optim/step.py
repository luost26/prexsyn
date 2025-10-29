import abc
from typing import Any

import numpy as np
import torch
from rdkit import Chem

from prexsyn.factories.facade import Facade
from prexsyn.models.prexsyn import PrexSyn
from prexsyn.queries import Query
from prexsyn.samplers.basic import BasicSampler
from prexsyn.samplers.query import QueryConditionedSampler
from prexsyn.utils.oracles import OracleProtocol
from prexsyn_engine.fingerprints import fp_func
from prexsyn_engine.synthesis import Synthesis

from .state import DeltaState, State


class StepStrategy(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        state: State,
        facade: Facade,
        model: PrexSyn,
        oracle_fn: OracleProtocol,
        constraint_fn: OracleProtocol,
        cond_query: Query | None = None,
    ) -> tuple[State, DeltaState]: ...


class FingerprintGenetic(StepStrategy):
    def __init__(
        self,
        bottleneck_size: int,
        bottleneck_temperature: float = 0.5,
        fingerprint_weight: float = 0.75,
        num_syns_per_query: int = 8,
        fp_type: str = "ecfp4",
        flip_bit_ratio: float = 0.0,
    ) -> None:
        self.bottleneck_size = bottleneck_size
        self.bottleneck_temperature = bottleneck_temperature
        self.fingerprint_weight = fingerprint_weight
        self.num_syns_per_query = num_syns_per_query
        self.fp_type = fp_type
        self.flip_bit_ratio = flip_bit_ratio

    @staticmethod
    def random_flip_bits(fp: torch.Tensor, flip_ratio: float) -> torch.Tensor:
        """
        Approximate L1 norm preserving random bit flip.
        Args:
            coords: (N, 2048) tensor of ECFP4 fingerprints
            flip_prob: probability of flipping bits that are 1
        """
        fp_dtype = fp.dtype
        fp = fp != 0.0
        num_1s = fp.sum(dim=1, keepdim=True)
        num_0s = fp.size(1) - num_1s

        num_flips = (num_1s * flip_ratio).long().clamp(min=1)
        prob_0_to_1 = (num_flips / num_0s).clamp(0.0, 1.0).expand_as(fp)
        prob_1_to_0 = (num_flips / num_1s).clamp(0.0, 1.0).expand_as(fp)

        bit_probs = torch.where(fp, 1.0 - prob_1_to_0, prob_0_to_1)
        fp = torch.bernoulli(bit_probs).to(fp_dtype)
        return fp

    def update_fp_bits_ga(self, coords: torch.Tensor) -> torch.Tensor:
        parent_1_index = torch.randint(0, coords.size(0), (coords.size(0),), device=coords.device)
        parent_2_index = torch.randint(0, coords.size(0), (coords.size(0),), device=coords.device)
        parent_1 = coords[parent_1_index]
        parent_2 = coords[parent_2_index]
        bit_prob = (0.5 * (parent_1 + parent_2)).clamp(0.0, 1.0)
        coords = torch.bernoulli(bit_prob).to(coords.dtype)
        if self.flip_bit_ratio > 0:
            coords = FingerprintGenetic.random_flip_bits(coords, flip_ratio=self.flip_bit_ratio)
        return coords

    @staticmethod
    def tanimoto_np(fp1: np.ndarray[Any, Any], fp2: np.ndarray[Any, Any]) -> float:
        intersection = np.sum(np.minimum(fp1, fp2))
        union = np.sum(np.maximum(fp1, fp2))
        return intersection / union if union > 0 else 0.0

    def get_delta_state(
        self,
        facade: Facade,
        model: PrexSyn,
        coords: torch.Tensor,
        ages: torch.Tensor,
        oracle_fn: OracleProtocol,
        constraint_fn: OracleProtocol,
        cond_query: Query | None = None,
    ) -> DeltaState:
        prop_repr = {self.fp_type: {"fingerprint": coords.repeat_interleave(self.num_syns_per_query, 0)}}
        if cond_query is None:
            samples = BasicSampler(model, token_def=facade.get_token_def(), num_samples=1).sample(prop_repr)
        else:
            samples = QueryConditionedSampler(model, facade.get_token_def()).sample(
                prop_repr,
                cond_query,
                batched_property_weight=self.fingerprint_weight,
            )
        detok = facade.get_detokenizer()(**samples)

        mol_to_be_scored: list[Chem.Mol] = []
        mol_to_syn_index: list[int] = []
        mol_to_fp: list[np.ndarray[Any, Any]] = []
        for syn_index in range(coords.shape[0]):
            products: list[Chem.Mol] = []
            for syn in detok[syn_index * self.num_syns_per_query : (syn_index + 1) * self.num_syns_per_query]:
                if syn.stack_size() != 1:
                    continue
                products.extend(syn.top().to_list())

            best_sim = 0.0
            best_mol: Chem.Mol | None = None
            best_fp: np.ndarray[Any, Any] | None = None
            for prod in products:
                prod_fp = fp_func(prod, self.fp_type)
                ref_fp = coords[syn_index].cpu().numpy()
                sim = self.tanimoto_np(prod_fp, ref_fp)
                if sim > best_sim:
                    best_sim = sim
                    best_mol = prod
                    best_fp = prod_fp
            if best_mol is None or best_fp is None:
                continue
            mol_to_be_scored.append(best_mol)
            mol_to_syn_index.append(syn_index)
            mol_to_fp.append(best_fp)

        mol_to_score = oracle_fn(mol_to_be_scored)
        mol_to_constraint_score = constraint_fn(mol_to_be_scored)

        new_scores_list: list[float] = [0.0] * coords.size(0)
        new_constraint_scores_list: list[float] = [0.0] * coords.size(0)
        new_coords_np = np.zeros(list(coords.size()), dtype=np.float32)
        new_syn_list: list[Synthesis | None] = [None] * coords.size(0)
        new_prod_list: list[Chem.Mol | None] = [None] * coords.size(0)
        for mol, score, cstr_score, fp, syn_idx in zip(
            mol_to_be_scored, mol_to_score, mol_to_constraint_score, mol_to_fp, mol_to_syn_index
        ):
            new_scores_list[syn_idx] = score
            new_constraint_scores_list[syn_idx] = cstr_score
            new_coords_np[syn_idx, :] = fp
            new_syn_list[syn_idx] = detok[syn_idx]
            new_prod_list[syn_idx] = mol

        return DeltaState(
            coords=torch.tensor(new_coords_np, dtype=coords.dtype, device=coords.device),
            scores=torch.tensor(new_scores_list, dtype=coords.dtype, device=coords.device),
            constraint_scores=torch.tensor(new_constraint_scores_list, dtype=coords.dtype, device=coords.device),
            ages=ages + 1,
            syntheses=new_syn_list,
            products=new_prod_list,
        )

    def __call__(
        self,
        state: State,
        facade: Facade,
        model: PrexSyn,
        oracle_fn: OracleProtocol,
        constraint_fn: OracleProtocol,
        cond_query: Query | None = None,
    ) -> tuple[State, DeltaState]:
        state = state.samplek(self.bottleneck_size, temperature=self.bottleneck_temperature)

        coords = self.update_fp_bits_ga(state.coords)
        delta_state = self.get_delta_state(
            facade=facade,
            model=model,
            coords=coords,
            ages=state.ages,
            oracle_fn=oracle_fn,
            constraint_fn=constraint_fn,
            cond_query=cond_query,
        )

        state_next = state.concat(delta_state).deduplicate().topk(state.coords.size(0))

        return state_next, delta_state
