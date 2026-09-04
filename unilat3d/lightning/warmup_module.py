from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

import pytorch_lightning as pl

from unilat3d.utils import dist_utils


class WarmupLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        dataset,
        *,
        output_dir: str,
        max_steps: int,
        optimizer: Dict[str, Any],
        t_schedule: dict = None,
        sigma_min: float = 1e-5,
        image_cond_model: str = "dinov3_vith16plus",
        unilat_normalization: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "dataset", "optimizer"])
        self.model = model
        self.dataset = dataset
        self.output_dir = output_dir
        self.max_steps = int(max_steps)
        self.optimizer_config = optimizer
        self.t_schedule = t_schedule or {"name": "logitNormal", "args": {"mean": 0.0, "std": 1.0}}
        self.sigma_min = float(sigma_min)
        self.unilat_normalization = unilat_normalization
        self.image_cond_model_name = image_cond_model
        self.image_cond_model = None
        self.image_resolution = 518

    def _apply_unilat_norm_for_training(self, latents: torch.Tensor) -> torch.Tensor:
        if getattr(self.dataset, "normalization", None) is not None:
            return latents
        norm = getattr(self, "unilat_normalization", None)
        if not isinstance(norm, dict) or not norm:
            return latents
        try:
            if latents.ndim == 4:
                latents = latents.unsqueeze(0)
            mean = torch.tensor(norm["mean"], device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
            std = torch.tensor(norm["std"], device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
            if latents.ndim == 5 and mean.shape[1] == latents.shape[1] and std.shape[1] == latents.shape[1]:
                return (latents - mean) / std
            return latents
        except Exception:
            return latents

    
    def _init_image_cond_model(self) -> None:
        with dist_utils.local_master_first():
            if "dinov3" in self.image_cond_model_name:
                dino_github = "external/dinov3"
                dino_weight = "external/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
                self.image_resolution = 592
                model = torch.hub.load(dino_github, self.image_cond_model_name, source="local", weights=dino_weight, pretrained=True)
            else:
                self.image_resolution = 518
                model = torch.hub.load("facebookresearch/dinov2", self.image_cond_model_name, pretrained=True)
        model.eval().to(self.device)
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.image_cond_model = {"model": model, "transform": transform}

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        assert isinstance(image, torch.Tensor)
        assert image.ndim == 4, "Image tensor should be batched (B,C,H,W)"
        if image.shape[-1] != self.image_resolution or image.shape[-2] != self.image_resolution:
            image = F.interpolate(image, size=(self.image_resolution, self.image_resolution), mode="bilinear", align_corners=False)
        if self.image_cond_model is None:
            self._init_image_cond_model()
        image = self.image_cond_model["transform"](image).to(self.device)
        features = self.image_cond_model["model"](image, is_training=True)["x_prenorm"]
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens

    def get_cond(self, cond_img: Optional[torch.Tensor], **kwargs) -> Optional[torch.Tensor]:
        if cond_img is None:
            return None
        return self.encode_image(cond_img)

    def get_inference_cond(self, *, cond: torch.Tensor, **kwargs) -> Dict[str, Any]:
        cond = self.encode_image(cond)
        return {"cond": cond, **kwargs}

    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_0)
        t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        x_t = (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise
        return x_t

    def get_v(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (1 - self.sigma_min) * noise - x_0


    def sample_t(self, batch_size: int) -> torch.Tensor:
        if self.t_schedule["name"] == "uniform":
            return torch.rand(batch_size)
        if self.t_schedule["name"] == "logitNormal":
            mean = self.t_schedule["args"]["mean"]
            std = self.t_schedule["args"]["std"]
            return torch.sigmoid(torch.randn(batch_size) * std + mean)
        raise ValueError(f"Unknown t_schedule: {self.t_schedule['name']}")

    
    def configure_optimizers(self):
        opt_cfg = self.optimizer_config
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if hasattr(torch.optim, opt_cfg["name"]):
            return getattr(torch.optim, opt_cfg["name"])(trainable, **opt_cfg.get("args", {}))
        raise ValueError(f"Unknown optimizer: {opt_cfg['name']}")

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        x_0 = batch["x_0"]
        y = batch["y"]
        cond = batch["cond"]
        x_0 = self._apply_unilat_norm_for_training(x_0)
        y = self._apply_unilat_norm_for_training(y)
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device)
        x_t = self.diffuse(x_0, t, noise=noise)
        cond_feat = self.get_cond(cond)

        pred = self.model(x_t, t * 1000, cond_feat, y=y)
        target = self.get_v(x_0, noise, t)

        loss = F.mse_loss(pred, target)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss