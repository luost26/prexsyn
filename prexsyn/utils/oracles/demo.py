from typing import cast

import numpy as np
from rdkit import Chem

from ._registry import OracleProtocol, register
from .common import oracle_function_wrapper


@register
def scaffold_hop_demo1() -> OracleProtocol:
    from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
    from rdkit.DataStructs import TanimotoSimilarity

    ref_mol = Chem.MolFromSmiles("CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C")
    ref_ph_fp = Generate.Gen2DFingerprint(ref_mol, Gobbi_Pharm2D.factory)  # type: ignore[no-untyped-call]
    deco1 = Chem.MolFromSmiles("c1ccc2ncsc2c1")
    deco2 = Chem.MolFromSmiles("CCCO")
    core = Chem.MolFromSmiles("[#7]-c1ncnc2cc(-[#8])ccc12")
    core_bitset = set(cast(list[int], np.array(Chem.LayeredFingerprint(core)).nonzero()[0].tolist()))

    def _score_fn(mol: Chem.Mol) -> float:
        mol_bitset = set(cast(list[int], np.array(Chem.LayeredFingerprint(mol)).nonzero()[0].tolist()))

        contains_deco1 = 1.0 if mol.HasSubstructMatch(deco1) else 0.0
        contains_deco2 = 1.0 if mol.HasSubstructMatch(deco2) else 0.0

        not_contains_core = 1.0 - len(mol_bitset.intersection(core_bitset)) / len(core_bitset)

        mol_ph_fp = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)  # type: ignore[no-untyped-call]
        sim_to_ref: float = TanimotoSimilarity(mol_ph_fp, ref_ph_fp)

        return (contains_deco1 + contains_deco2 + not_contains_core + sim_to_ref) / 4

    return oracle_function_wrapper(_score_fn)
