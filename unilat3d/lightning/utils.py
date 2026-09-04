from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Iterator, List

import torch
import numpy as np

from unilat3d.utils import dist_utils

def denormalize_latents(latents: torch.Tensor, unilat_normalization: dict) -> torch.Tensor:
    device = latents.device
    mean = torch.tensor(unilat_normalization["mean"], device=device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(unilat_normalization["std"], device=device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
    if mean.shape[1] == latents.shape[1] and std.shape[1] == latents.shape[1]:
        latents = latents * std + mean
    return latents

def save_latents(x_out, path):
    if x_out.ndim == 5 and x_out.shape[0] == 1:
        x_out = x_out[0]
                                                                                 
    out_pack = {"mean": x_out.detach().float().cpu().numpy()}
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out_pack)

def dict_flatten(d: Dict[str, Any], *, sep: str = "/") -> Dict[str, Any]:
    
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            sub = dict_flatten(v, sep=sep)
            for sk, sv in sub.items():
                out[f"{k}{sep}{sk}"] = sv
        else:
            out[k] = v
    return out


def cycle(dataloader: Iterable[Any]) -> Iterator[Any]:
    
    while True:
        for x in dataloader:
            yield x


def split_batch(batch: Dict[str, Any], batch_split: int) -> List[Dict[str, Any]]:
    
    if batch_split <= 1:
        return [batch]
    any_tensor = next(v for v in batch.values() if isinstance(v, torch.Tensor))
    bsz = any_tensor.shape[0]
    assert bsz % batch_split == 0, "batch size per gpu must be divisible by batch_split"
    out = []
    for i in range(batch_split):
        s = i * bsz // batch_split
        e = (i + 1) * bsz // batch_split
        mb = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                mb[k] = v[s:e]
            else:
                mb[k] = v
        out.append(mb)
    return out


def mean_terms(terms_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    
    if not terms_list:
        return {}
    keys = set().union(*[d.keys() for d in terms_list if isinstance(d, dict)])
    out: Dict[str, Any] = {}
    for k in keys:
        vals = [d[k] for d in terms_list if k in d]
        if all(isinstance(v, torch.Tensor) for v in vals):
            out[k] = torch.stack(vals).mean()
        elif all(isinstance(v, (float, int)) for v in vals):
            out[k] = float(np.mean(vals))
        else:
            if all(isinstance(v, dict) for v in vals):
                                                                                                 
                merged: Dict[str, List[Any]] = {}
                for vv in vals:
                    for kk, vvv in vv.items():
                        merged.setdefault(kk, []).append(vvv)

                reduced: Dict[str, Any] = {}
                for kk, vv_list in merged.items():
                    if all(isinstance(x, torch.Tensor) for x in vv_list):
                        reduced[kk] = torch.stack([x.detach() for x in vv_list]).mean()
                    elif all(isinstance(x, (float, int)) for x in vv_list):
                        reduced[kk] = float(np.mean(vv_list))
                    else:
                                                                                   
                        nums: List[float] = []
                        for x in vv_list:
                            if isinstance(x, torch.Tensor):
                                nums.append(float(x.detach().float().mean().cpu().item()))
                            else:
                                nums.append(float(x))
                        reduced[kk] = float(np.mean(nums)) if len(nums) > 0 else 0.0

                out[k] = reduced
            else:
                out[k] = vals[0]
    if "loss" not in out and any("loss" in d for d in terms_list):
        out["loss"] = terms_list[-1]["loss"]
    return out


def torch_load_dist(path: str, *, map_location) -> Any:
    
    data = dist_utils.read_file_dist(path)
    try:
        return torch.load(data, map_location=map_location, weights_only=False)
    except TypeError:
                                          
        return torch.load(data, map_location=map_location)

