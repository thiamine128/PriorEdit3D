from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import torch
from torch.utils.data import Sampler
import torch.distributed as dist


def recursive_to_device(obj: Any, device: torch.device, non_blocking: bool = False) -> Any:
    
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: recursive_to_device(v, device, non_blocking=non_blocking) for k, v in obj.items()}
    if isinstance(obj, list):
        return [recursive_to_device(v, device, non_blocking=non_blocking) for v in obj]
    if isinstance(obj, tuple):
        return tuple(recursive_to_device(v, device, non_blocking=non_blocking) for v in obj)
    return obj


def cycle(dataloader: Iterable[Any]) -> Iterator[Any]:
    
    while True:
        for x in dataloader:
            yield x


class ResumableSampler(Sampler[int]):
    

    def __init__(
        self,
        data_source,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = int(seed)
        self.epoch = 0
        self.pos = 0                                             

    def __len__(self) -> int:
        return len(self.data_source)

    def _get_perm(self) -> List[int]:
        n = len(self.data_source)
        if not self.shuffle:
            return list(range(n))
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        return torch.randperm(n, generator=g).tolist()

    def __iter__(self) -> Iterator[int]:
        perm = self._get_perm()
                             
        for idx in perm[self.pos :]:
            self.pos += 1
            yield idx
                        
        self.epoch += 1
        self.pos = 0

    def state_dict(self) -> Dict[str, int]:
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "pos": self.pos,
            "shuffle": int(self.shuffle),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.seed = int(state.get("seed", self.seed))
        self.epoch = int(state.get("epoch", 0))
        self.pos = int(state.get("pos", 0))
        self.shuffle = bool(state.get("shuffle", int(self.shuffle)))


class ResumableDistributedSampler(Sampler[int]):
    

    def __init__(
        self,
        data_source,
        *,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(data_source)
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.data_source = data_source
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.pos = 0

        self._recompute_sizes()

    def _recompute_sizes(self) -> None:
        
        n = len(self.data_source)
        if self.drop_last and n % self.num_replicas != 0:
            self.num_samples = math.ceil((n - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(n / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __len__(self) -> int:
        return self.num_samples

    def _padded_indices_for_epoch(self) -> List[int]:
        
        n = len(self.data_source)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(n))
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size
        return indices

    def _rank_indices(self) -> List[int]:
        indices = self._padded_indices_for_epoch()
        return indices[self.rank : self.total_size : self.num_replicas]

    def __iter__(self) -> Iterator[int]:
        rank_indices = self._rank_indices()
        for idx in rank_indices[self.pos :]:
            self.pos += 1
            yield idx
        self.epoch += 1
        self.pos = 0

    def state_dict(self) -> Dict[str, int]:
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "pos": self.pos,
            "shuffle": int(self.shuffle),
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "drop_last": int(self.drop_last),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.seed = int(state.get("seed", self.seed))
        self.epoch = int(state.get("epoch", 0))
        self.pos = int(state.get("pos", 0))
        self.shuffle = bool(state.get("shuffle", int(self.shuffle)))
                                                         
                                                                        
                                                           
        if "drop_last" in state:
            self.drop_last = bool(state["drop_last"])
        self._recompute_sizes()
                                                                        
        if self.pos > self.num_samples:
            self.pos = self.num_samples



