from typing import overload

import numpy as np
import torch


def _tanimoto_similarity_torch(x1: torch.Tensor, x2: torch.Tensor):
    intersection = torch.min(x1, x2).sum(dim=-1)
    union = torch.max(x1, x2).sum(dim=-1)
    return intersection / (union + 1e-8)


def _tanimoto_similarity_numpy(x1: np.ndarray, x2: np.ndarray):
    intersection = np.minimum(x1, x2).sum(axis=-1)
    union = np.maximum(x1, x2).sum(axis=-1)
    return intersection / (union + 1e-8)


@overload
def tanimoto_similarity(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor: ...


@overload
def tanimoto_similarity(x1: np.ndarray, x2: np.ndarray) -> np.ndarray: ...


def tanimoto_similarity(x1: torch.Tensor | np.ndarray, x2: torch.Tensor | np.ndarray):
    if isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
        return _tanimoto_similarity_torch(x1, x2)
    elif isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray):
        return _tanimoto_similarity_numpy(x1, x2)
    else:
        raise ValueError("Input types must be both torch.Tensor or both np.ndarray.")
