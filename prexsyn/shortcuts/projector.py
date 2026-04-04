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
from prexsyn.utils.metrics import tanimoto_similarity
from prexsyn_engine.chemistry import Molecule
from prexsyn_engine.chemspace import Synthesis
from prexsyn_engine.detokenizer import MultiThreadedDetokenizer


@dataclass(frozen=True)
class _ResultItem:
    molecule: Molecule
    synthesis: Synthesis
    similarity: float

    def __iter__(self):
        return iter((self.molecule, self.synthesis, self.similarity))


@dataclass
class _Result:
    items: list[_ResultItem]
    time: float

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]

    def best(self):
        best_item = max(self.items, key=lambda item: item.similarity)
        return best_item

    def best_similarity(self):
        if not self.items:
            return 0.0
        return self.best().similarity


@dataclass
class _BatchedResult:
    results: list[_Result]
    time: float

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
        total_time: float,
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
            batched_items.append(_Result(items=items, time=total_time / len(results_2d)))

        return _BatchedResult(results=batched_items, time=total_time)

    def one(self, mol: Molecule | str | rdkit.Chem.Mol) -> _Result:
        return self.many([mol]).results[0]

    def many(self, mols: Sequence[Molecule | str | rdkit.Chem.Mol]) -> _BatchedResult:
        t_start = time.perf_counter()

        mols_converted = self._convert_molecules(mols)
        in_desc = self._compute_descriptors(mols_converted)  # (n, desc_dim)
        in_desc_torch = torch.from_numpy(in_desc).to(dtype=self.model.dtype, device=self.model.device)
        results = self.sampler.sample((self.descriptor_name, in_desc_torch)).detokenize(self.detokenizer)

        t_end = time.perf_counter()
        total_time = t_end - t_start

        return self._aggregate_results(results, in_desc, total_time)

    def desc(self, in_descs: torch.Tensor | np.ndarray) -> _BatchedResult:
        if isinstance(in_descs, torch.Tensor):
            in_descs_np = in_descs.cpu().numpy()
            in_descs_torch = in_descs
        else:
            in_descs_np = in_descs
            in_descs_torch = torch.from_numpy(in_descs)

        if in_descs_np.ndim != 2:
            raise ValueError(f"Expected shape (bsz, desc_dim), got {in_descs_np.shape}")

        t_start = time.perf_counter()
        results = self.sampler.sample((self.descriptor_name, in_descs_torch)).detokenize(self.detokenizer)

        t_end = time.perf_counter()
        total_time = t_end - t_start

        return self._aggregate_results(results, in_descs_np, total_time)
