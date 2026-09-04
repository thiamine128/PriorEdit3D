from __future__ import annotations

import torch


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    
    return tensor.mean(dim=list(range(1, tensor.ndim)))
