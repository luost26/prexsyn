import time
from dataclasses import dataclass

import rdkit.Chem
import torch

from prexsyn.factory import get_descriptor_constructor
from prexsyn.models import PrexSyn
from prexsyn.samplers.basic import BasicSampler
from prexsyn.utils.metrics import tanimoto_similarity
from prexsyn_engine.chemistry import Molecule
from prexsyn_engine.chemspace import Synthesis
from prexsyn_engine.detokenizer import MultiThreadedDetokenizer


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

    def _compute_descriptor(self, mol: Molecule):
        return self.descriptor_function(mol)[None, :]

    @dataclass
    class Result:
        results: list[tuple[Molecule, Synthesis, float]]
        time: float

    def __call__(self, mol: Molecule | str | rdkit.Chem.Mol):
        t_start = time.perf_counter()

        in_desc = self._compute_descriptor(self._convert_molecule(mol))
        in_desc_torch = torch.from_numpy(in_desc).to(dtype=self.model.dtype, device=self.model.device)
        results = self.sampler.sample((self.descriptor_name, in_desc_torch)).detokenize(self.detokenizer)

        t_end = time.perf_counter()

        mol_syn_index: list[int] = []
        mols: list[Molecule] = []
        for i, syn in enumerate(results):
            prods = syn.products()
            mol_syn_index.extend([i] * len(prods))
            mols.extend(prods)

        prod_descs = self.descriptor_function(mols)
        sims = tanimoto_similarity(in_desc, prod_descs)  # (num_prods, )

        order = sims.argsort()[::-1]

        return self.Result(
            results=[(mols[i], results[mol_syn_index[i]], float(sims[i])) for i in order],
            time=t_end - t_start,
        )
