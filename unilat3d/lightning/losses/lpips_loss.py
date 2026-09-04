from __future__ import annotations

from typing import List, Optional

import torch
import lpips
from easydict import EasyDict as edict


class LPIPSLoss:
    

    def __init__(
        self,
        weight: float = 1.0,
        net: str = "vgg",
        reduction: str = "mean",
    ) -> None:
        
        self.weight = float(weight)
        self.net = str(net)
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported reduction: {reduction} (expected 'mean'|'sum'|'none')")
        self.reduction = reduction

                                                                                
        self._lpips = None

    def _get_lpips(self, device: torch.device) -> torch.nn.Module:
        if self._lpips is None:
            self._lpips = lpips.LPIPS(net=self.net)
            self._lpips.eval()
            for p in self._lpips.parameters():
                p.requires_grad_(False)

                                          
        self._lpips = self._lpips.to(device)
        return self._lpips

    def __call__(
        self,
        img_before: torch.Tensor,
        img_after: torch.Tensor,
        img_ref: torch.Tensor,
        edit_inst: List[str],
        **kwargs,
    ) -> (torch.Tensor, edict):
                                                     
        img_ref = img_ref.clamp(0, 1)
        img_after = img_after.clamp(0, 1)
        x = img_ref * 2.0 - 1.0
        y = img_after * 2.0 - 1.0

        lpips_model = self._get_lpips(device=img_after.device)

                                                                         
        x32 = x.float()
        y32 = y.float()
        with torch.autocast(device_type="cuda", enabled=False):
            d = lpips_model(x32, y32)                     

                                            
        if d.ndim > 1:
            d = d.view(d.shape[0], -1).mean(dim=1)

        if self.reduction == "mean":
            out = d.mean()
        elif self.reduction == "sum":
            out = d.sum()
        else:          
            out = d

        loss_lpips = out.to(img_after.dtype) * self.weight
        return loss_lpips, edict(loss_lpips=loss_lpips.detach())

