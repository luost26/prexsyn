from typing import overload

from rdkit import Chem

from ._registry import OracleProtocol


class CachedOracle:
    def __init__(self, oracle: OracleProtocol) -> None:
        super().__init__()
        self._oracle = oracle
        self._cache: dict[str, float] = {}

    def _get_id(self, mol: Chem.Mol) -> str:
        return Chem.MolToSmiles(mol, canonical=True)

    def _cached_call_multiple(self, mols: list[Chem.Mol]) -> list[float]:
        scores: list[float] = [0.0] * len(mols)

        cache_miss_indices: list[int] = []
        cache_miss_mols: list[Chem.Mol] = []
        cache_miss_ids: list[str] = []
        for i, mol in enumerate(mols):
            mol_id = self._get_id(mol)
            if mol_id in self._cache:
                scores[i] = self._cache[mol_id]
            else:
                cache_miss_indices.append(i)
                cache_miss_mols.append(mol)
                cache_miss_ids.append(mol_id)

        cache_miss_scores = self._oracle(cache_miss_mols) if cache_miss_mols else []
        for idx, score, mol_id in zip(cache_miss_indices, cache_miss_scores, cache_miss_ids):
            self._cache[mol_id] = score
            scores[idx] = score

        return scores

    def _cached_call_single(self, mol: Chem.Mol) -> float:
        mol_id = self._get_id(mol)
        if mol_id not in self._cache:
            score = self._oracle(mol)
            self._cache[mol_id] = score
        return self._cache[mol_id]

    @overload
    def __call__(self, mol: Chem.Mol) -> float: ...
    @overload
    def __call__(self, mol: list[Chem.Mol]) -> list[float]: ...

    def __call__(self, mol: list[Chem.Mol] | Chem.Mol) -> list[float] | float:
        if isinstance(mol, list):
            return self._cached_call_multiple(mol)
        else:
            return self._cached_call_single(mol)
