import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import overload

import numpy as np
import rdkit.Chem
import torch

from prexsyn.factory import get_descriptor_constructor
from prexsyn.models import PrexSyn
from prexsyn.samplers.basic import BasicSampler
from prexsyn.utils.draw import SynthesisDraw
from prexsyn.utils.metrics import tanimoto_similarity
from prexsyn.utils.syndag import SynthesisDAG
from prexsyn_engine.chemistry import Molecule
from prexsyn_engine.chemspace import Synthesis
from prexsyn_engine.detokenizer import MultiThreadedDetokenizer


@dataclass(frozen=True)
class _Time:
    model: float
    detok: float

    @property
    def total(self) -> float:
        return self.model + self.detok

    def __truediv__(self, divisor: float | int):
        return _Time(model=self.model / divisor, detok=self.detok / divisor)


@dataclass(frozen=True)
class _ResultItem:
    molecule: Molecule
    synthesis: Synthesis
    similarity: float

    def __iter__(self):
        return iter((self.molecule, self.synthesis, self.similarity))

    def get_tree(self):
        return SynthesisDAG(self.synthesis).to_dict(self.molecule.smiles())

    def get_image(self):
        return SynthesisDraw().draw(self.synthesis, highlight_smiles=self.molecule.smiles())


@dataclass
class _Result:
    items: list[_ResultItem]
    time: _Time

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]

    def best(self):
        if not self.items:
            return None
        best_item = max(self.items, key=lambda item: item.similarity)
        return best_item

    def best_similarity(self):
        best = self.best()
        if best is None:
            return 0.0
        return best.similarity


@dataclass
class _BatchedResult:
    results: list[_Result]
    time: _Time

    def __len__(self):
        return len(self.results)

    @overload
    def __getitem__(self, index: int) -> _Result: ...

    @overload
    def __getitem__(self, index: tuple[int, int]) -> _ResultItem: ...

    def __getitem__(self, index: int | tuple[int, int]):
        if isinstance(index, int):
            return self.results[index]
        elif isinstance(index, tuple) and len(index) == 2:
            batch_idx, item_idx = index
            return self.results[batch_idx][item_idx]
        else:
            raise IndexError("Invalid index type or shape")


class MoleculeProjector:
    def __init__(
        self,
        model: PrexSyn,
        detokenizer: MultiThreadedDetokenizer,
        descriptor: str,
        num_samples: int,
    ):
        super().__init__()
        self.model = model
        self.detokenizer = detokenizer
        self.descriptor_name = descriptor
        self.descriptor_function = get_descriptor_constructor(descriptor)()
        self.num_samples = num_samples
        self.sampler = BasicSampler(model, num_samples)

    def _convert_molecule(self, mol: Molecule | str | rdkit.Chem.Mol) -> Molecule:
        if isinstance(mol, Molecule):
            return mol
        elif isinstance(mol, str):
            return Molecule.from_smiles(mol)
        elif isinstance(mol, rdkit.Chem.Mol):
            return Molecule.from_rdkit_mol(mol)
        raise ValueError(f"Unsupported molecule type: {type(mol)}")

    def _convert_molecules(self, mols: Sequence[Molecule | str | rdkit.Chem.Mol]) -> list[Molecule]:
        return [self._convert_molecule(mol) for mol in mols]

    def _compute_descriptors(self, mols: Sequence[Molecule]):
        return self.descriptor_function(mols)

    def _aggregate_results(
        self,
        syn_out: list[Synthesis],
        in_desc: np.ndarray,
        time_stat: _Time,
    ):
        results_2d = [syn_out[i : i + self.num_samples] for i in range(0, len(syn_out), self.num_samples)]
        batched_items: list[_Result] = []
        for batch_idx, syns in enumerate(results_2d):
            mol_syn_index: list[int] = []
            prods_all: list[Molecule] = []
            for syn_i, syn in enumerate(syns):
                prods = syn.products()
                mol_syn_index.extend([syn_i] * len(prods))
                prods_all.extend(prods)

            prod_descs = self.descriptor_function(prods_all)  # (num_prods, desc_dim)
            sims = tanimoto_similarity(in_desc[batch_idx][None, :], prod_descs)  # (num_prods, )

            order = sims.argsort()[::-1]

            items = [_ResultItem(prods_all[i], syns[mol_syn_index[i]], float(sims[i])) for i in order]
            batched_items.append(_Result(items=items, time=time_stat / len(results_2d)))

        return _BatchedResult(results=batched_items, time=time_stat)

    def one(self, mol: Molecule | str | rdkit.Chem.Mol) -> _Result:
        return self.many([mol]).results[0]

    def many(self, mols: Sequence[Molecule | str | rdkit.Chem.Mol]) -> _BatchedResult:
        t_0 = time.perf_counter()
        mols_converted = self._convert_molecules(mols)
        in_desc = self._compute_descriptors(mols_converted)  # (n, desc_dim)
        in_desc_torch = torch.from_numpy(in_desc).to(dtype=self.model.dtype, device=self.model.device)
        samples = self.sampler.sample((self.descriptor_name, in_desc_torch))

        t_1 = time.perf_counter()
        results = samples.detokenize(self.detokenizer)

        t_2 = time.perf_counter()

        return self._aggregate_results(results, in_desc, _Time(model=t_1 - t_0, detok=t_2 - t_1))

    def desc(self, in_descs: torch.Tensor | np.ndarray) -> _BatchedResult:
        if isinstance(in_descs, torch.Tensor):
            in_descs_np = in_descs.cpu().numpy()
            in_descs_torch = in_descs
        else:
            in_descs_np = in_descs
            in_descs_torch = torch.from_numpy(in_descs)

        if in_descs_np.ndim != 2:
            raise ValueError(f"Expected shape (bsz, desc_dim), got {in_descs_np.shape}")

        t_0 = time.perf_counter()
        samples = self.sampler.sample((self.descriptor_name, in_descs_torch))

        t_1 = time.perf_counter()
        results = samples.detokenize(self.detokenizer)

        t_2 = time.perf_counter()

        return self._aggregate_results(results, in_descs_np, _Time(model=t_1 - t_0, detok=t_2 - t_1))
