from __future__ import annotations

from typing import List, Literal, Optional
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from easydict import EasyDict as edict


class PixelwiseLoss:
    
    
    def __init__(
        self,
        weight: float = 1.0,
        loss_type: Literal["l1", "l2", "mse"] = "l2",
        reduction: str = "mean",
        alpha_as_mask: bool = False,
    ):
        
        self.weight = weight
        self.loss_type = loss_type
        self.reduction = reduction
        self.alpha_as_mask = alpha_as_mask
        self.supervise_alpha = True
    
    def __call__(
        self,
        img_before: torch.Tensor,
        img_after: torch.Tensor,
        img_ref: torch.Tensor,
        edit_inst: List[str],
        img_ref_alpha: Optional[torch.Tensor] = None,
        img_after_alpha: Optional[torch.Tensor] = None,
        save_dir: Optional[str] = None,
        global_step: Optional[int] = None,
        rank: Optional[int] = None,
        **kwargs
    ) -> (torch.Tensor, edict):
        
                                                                        
                                              
        img_ref = img_ref.clamp(0, 1)
        img_after = img_after.clamp(0, 1)
        
        loss_type = self.loss_type.lower()
        
        if img_ref_alpha is not None and img_after_alpha is not None:
            ref_alpha = img_ref_alpha
            if ref_alpha.ndim == 4 and ref_alpha.shape[1] == 4:
                ref_alpha = ref_alpha[:, 3:4]                                   
            elif ref_alpha.ndim == 3:
                ref_alpha = ref_alpha.unsqueeze(1)
            
            after_alpha = img_after_alpha
            if after_alpha.ndim == 3:
                after_alpha = after_alpha.unsqueeze(1)
            
            if ref_alpha.shape[-2:] != img_ref.shape[-2:]:
                ref_alpha = F.interpolate(
                    ref_alpha.float(),
                    size=img_ref.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
            if after_alpha.shape[-2:] != img_after.shape[-2:]:
                after_alpha = F.interpolate(
                    after_alpha.float(),
                    size=img_after.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
            
            ref_alpha = ref_alpha.clamp(0, 1).to(device=img_ref.device, dtype=img_ref.dtype)
            after_alpha = after_alpha.clamp(0, 1).to(device=img_after.device, dtype=img_after.dtype)
            
            img_ref_rgba = torch.cat([img_ref, ref_alpha], dim=1)
            img_after_rgba = torch.cat([img_after, after_alpha], dim=1)

            img_ref_loss = img_ref_rgba if self.supervise_alpha else img_ref
            img_after_loss = img_after_rgba if self.supervise_alpha else img_after
            
            if loss_type == "l1":
                per = (img_ref_loss - img_after_loss).abs()
            elif loss_type in ("l2", "mse"):
                per = (img_ref_loss - img_after_loss).pow(2)
            else:
                raise ValueError(f"Unsupported loss_type={self.loss_type!r}. Use 'l1' or 'l2'/'mse'.")
        else:
            if loss_type == "l1":
                per = (img_ref - img_after).abs()
            elif loss_type in ("l2", "mse"):
                per = (img_ref - img_after).pow(2)
            else:
                raise ValueError(f"Unsupported loss_type={self.loss_type!r}. Use 'l1' or 'l2'/'mse'.")
        
                                                                
        m = None
        if self.alpha_as_mask:
            if img_ref_alpha is not None:
                                                                                       
                alpha = img_ref_alpha
                if alpha.ndim == 4 and alpha.shape[1] == 4:
                    alpha = alpha[:, 3:4]                                   
                elif alpha.ndim == 3:
                    alpha = alpha.unsqueeze(1)
                                                                  
                if alpha.shape[-2:] != per.shape[-2:]:
                    alpha = F.interpolate(alpha.float(), size=per.shape[-2:], mode="bilinear", align_corners=False)
                alpha = alpha.clamp(0, 1).to(device=per.device, dtype=per.dtype)
                m = alpha.detach() if alpha.requires_grad else alpha
        
        if m is not None:
                                                           
            m = m.expand(-1, per.shape[1], -1, -1)
            per = per * m

            if self.reduction == "none":
                out = per
            elif self.reduction == "sum":
                out = per.sum()
            elif self.reduction == "mean":
                denom = m.sum().clamp_min(1e-6)
                out = per.sum() / denom
            else:
                raise ValueError(f"Unsupported reduction={self.reduction!r}. Use 'mean', 'sum', or 'none'.")
        else:
                                               
            if self.reduction == "none":
                out = per
            elif self.reduction == "sum":
                out = per.sum()
            elif self.reduction == "mean":
                out = per.mean()
            else:
                raise ValueError(f"Unsupported reduction={self.reduction!r}. Use 'mean', 'sum', or 'none'.")

        if save_dir is not None and global_step is not None:
            try:
                os.makedirs(save_dir, exist_ok=True)
                
                compare_save_path = os.path.join(save_dir, f"{rank}_{global_step}_compare.png")
                after_img_cmp = img_after[0].detach().cpu().clamp(0, 1).float()
                after_img_cmp_np = (after_img_cmp.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                intent_img_np = (img_ref[0].detach().cpu().clamp(0, 1).float().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                cmp_img = np.concatenate([intent_img_np, after_img_cmp_np], axis=1)
                Image.fromarray(cmp_img).save(compare_save_path)
                
                                                                                               
                if img_after_alpha is not None and img_ref_alpha is not None:
                    ref_alpha = img_ref_alpha[0]
                    if ref_alpha.ndim == 3 and ref_alpha.shape[0] == 4:
                        ref_alpha = ref_alpha[3:4]                           
                    elif ref_alpha.ndim == 2:
                        ref_alpha = ref_alpha.unsqueeze(0)
                    ref_alpha_np = (ref_alpha.detach().cpu().clamp(0, 1).float().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    if ref_alpha_np.shape[2] == 1:
                        ref_alpha_np = ref_alpha_np.squeeze(2)
                    
                    after_alpha_np = (img_after_alpha[0].detach().cpu().clamp(0, 1).float().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    if after_alpha_np.shape[2] == 1:
                        after_alpha_np = after_alpha_np.squeeze(2)
                    
                    if ref_alpha_np.shape != after_alpha_np.shape:
                        ref_alpha_pil = Image.fromarray(ref_alpha_np, mode='L')
                        target_size = after_alpha_np.shape[:2] if after_alpha_np.ndim == 2 else (after_alpha_np.shape[1], after_alpha_np.shape[0])
                        ref_alpha_pil = ref_alpha_pil.resize(target_size[::-1], Image.Resampling.BILINEAR)
                        ref_alpha_np = np.array(ref_alpha_pil)
                    
                    alpha_compare_path = os.path.join(save_dir, f"{rank}_{global_step}_alpha_compare.png")
                    alpha_cmp_img = np.concatenate([ref_alpha_np, after_alpha_np], axis=1)
                    Image.fromarray(alpha_cmp_img, mode='L').save(alpha_compare_path)
            except Exception as e:
                                                     
                import warnings
                warnings.warn(f"Failed to save comparison/alpha images: {e}")

        loss_pixelwise = out * self.weight
        return loss_pixelwise, edict(loss_pixelwise=loss_pixelwise.detach())
