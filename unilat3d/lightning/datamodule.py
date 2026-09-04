from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from unilat3d.utils.data_utils import ResumableSampler, ResumableDistributedSampler
import pytorch_lightning as pl
             


class UniLatDataModule(pl.LightningDataModule):
    

    def __init__(
        self,
        dataset,
        *,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = True,
        shuffle: bool = True,
        seed: int = 0,
        exact_resume: bool = False,
        use_distributed_sampler: bool = False,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.exact_resume = bool(exact_resume)
        self.use_distributed_sampler = bool(use_distributed_sampler)
        self._train_sampler = None
        self._pending_sampler_state = None

    def train_dataloader(self) -> DataLoader:
        try:
            n = len(self.dataset)
        except Exception:
            n = None
        if n == 0:
            raise ValueError(
                "Training dataset is empty (len(dataset)=0). "
                "Please check your `data_dir` and dataset config/filters."
            )
        
        is_dist = (
            self.use_distributed_sampler
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )

        if is_dist:
                                                                                         
                                                                              
            self._train_sampler = ResumableDistributedSampler(
                self.dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=self.shuffle,
                seed=self.seed,
                drop_last=False,
            )
        else:
            self._train_sampler = ResumableSampler(self.dataset, shuffle=self.shuffle, seed=self.seed)

        if self._pending_sampler_state is not None:
            self._train_sampler.load_state_dict(self._pending_sampler_state)
            self._pending_sampler_state = None

        effective_num_workers = 0 if self.exact_resume else self.num_workers
        if self.exact_resume and effective_num_workers != self.num_workers:
            print(
                f"[Data] exact_resume=True, forcing num_workers=0 (was {self.num_workers}) "
                "to avoid worker prefetch drift.",
                flush=True,
            )
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self._train_sampler,
            shuffle=False,
            num_workers=effective_num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=(effective_num_workers > 0),
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None,
        )

    def get_train_sampler_state(self):
        if self._train_sampler is None:
            return None
        return self._train_sampler.state_dict()

    def set_train_sampler_state(self, state):
        self._pending_sampler_state = state
        if self._train_sampler is not None and state is not None:
            self._train_sampler.load_state_dict(state)

    def state_dict(self):
        return {"train_sampler_state": self.get_train_sampler_state()}

    def load_state_dict(self, state_dict):
        self.set_train_sampler_state(state_dict.get("train_sampler_state"))

