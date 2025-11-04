from collections.abc import Callable
from typing import overload

import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski

from ._registry import OracleProtocol, register


def oracle_function_wrapper(fn: Callable[[Chem.Mol], float]) -> OracleProtocol:
    @overload
    def wrapper(mol: Chem.Mol) -> float: ...
    @overload
    def wrapper(mol: list[Chem.Mol]) -> list[float]: ...

    def wrapper(mol: Chem.Mol | list[Chem.Mol]) -> list[float] | float:
        if isinstance(mol, list):
            return [float(fn(m)) for m in mol]
        else:
            return float(fn(mol))

    return wrapper


@register
def qed() -> OracleProtocol:
    return oracle_function_wrapper(lambda m: QED.qed(m))  # type: ignore[no-untyped-call]


def _soft_less_than(x: float, target: float, inv_lambda: float) -> float:
    diff = x - target
    if diff < 0:
        diff = 0
    return float(np.exp(-inv_lambda * diff))


def _lipinski(mol: Chem.Mol) -> list[float]:
    rule_1 = _soft_less_than(Descriptors.ExactMolWt(mol), 500 - 10, 10)  # type: ignore[attr-defined]
    rule_2_don = _soft_less_than(Lipinski.NumHDonors(mol), 5 - 1, 1)  # type: ignore[attr-defined]
    rule_2_acc = _soft_less_than(Lipinski.NumHAcceptors(mol), 10 - 1, 1)  # type: ignore[attr-defined]
    rule_3 = _soft_less_than(Descriptors.TPSA(mol), 140 - 10, 10)  # type: ignore[attr-defined]
    rule_4 = _soft_less_than(Crippen.MolLogP(mol), 5 - 0.5, 1)  # type: ignore[attr-defined]
    rule_5 = _soft_less_than(Chem.rdMolDescriptors.CalcNumRotatableBonds(mol), 10 - 1, 1)
    return [rule_1, rule_2_don, rule_2_acc, rule_3, rule_4, rule_5]


@register
def lipinski() -> OracleProtocol:
    def _fn(mol: Chem.Mol) -> float:
        scores = _lipinski(mol)
        return float(np.mean(scores))

    return oracle_function_wrapper(_fn)


@register
def lipinski_product() -> OracleProtocol:
    def _fn(mol: Chem.Mol) -> float:
        scores = _lipinski(mol)
        prod = 1.0
        for s in scores:
            prod *= s
        return prod

    return oracle_function_wrapper(_fn)
