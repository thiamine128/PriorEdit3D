from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import inspect

import os

import imageio
import torch
import deepspeed
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from easydict import EasyDict as edict
from torchvision import transforms
from PIL import Image
import numpy as np

import pytorch_lightning as pl

from unilat3d import models as unilat_models

from unilat3d.utils import dist_utils, render_utils
from unilat3d.lightning.render_utils import render, render_dbg
from unilat3d.representations.gaussian.gaussian_model import Gaussian


from . import unpaired_loss



class UnpairedLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_g: nn.Module,
        model_aux: nn.Module,
        output_dir: str,
        max_steps: int,
        optimizer: Dict[str, Any],
        losses: Optional[List[Dict[str, Any]]] = None,
        image_cond_model: str = "dinov3_vith16plus",
        vlm_model: str = "",
        gs_decoder: str = "",
        gen_model: str = "",
        unilat_normalization: Optional[dict] = None,
        sigma_min: float = 1e-5,
        qwen_w: float = 1.0,
        qwen_prob_loss: bool = False,
        aux_steps: int = 100,
        recon_every: int = 0,
        sample_steps: int = 50,
        dmd_enabled: bool = False,
        dmd_w: float = 1.0,
        denoising_step_list: List[float] = [],
        timestep_shift: float = 5.0,
        guidance_scale: float = 7.0,
        image_resolution: int = 512,
        random_bg: bool = False,
        cam: dict = None,
        consistency_cam: dict = None,
        vlm_cam: dict = None,
        warmup_steps: int = 3000,
        **kwargs,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["model_g", "model_aux"])
        self.model_g = model_g
        self.model_aux = model_aux
        self.output_dir = output_dir
        self.max_steps = int(max_steps)

        self.sigma_min = float(sigma_min)
        self.qwen_w = float(qwen_w)
        self.qwen_prob_loss = bool(qwen_prob_loss)
        self.dmd_w = float(dmd_w)
        self.sample_steps = int(sample_steps)
        self.image_cond_model_name = image_cond_model
        self.image_cond_model = None
        self.image_resolution = int(image_resolution)
        self.random_bg = bool(random_bg)
        self.unilat_normalization = unilat_normalization
        self.vlm_model_name = vlm_model
        self.gs_decoder_name = gs_decoder
        self.gen_model_name = gen_model
        self.aux_steps = int(aux_steps)
        self.recon_every = int(recon_every)
        self.gs_decoder = None
        self.gen_model = None
        self.optimizer_config = optimizer
        self.cam = cam
        self.vlm_cam = vlm_cam
        self.consistency_cam = consistency_cam
        self.losses_config = losses
        self.losses = []
        self.model_g_train_params = [p for p in self.model_g.parameters() if p.requires_grad]
        self.model_aux_train_params = [p for p in self.model_aux.parameters() if p.requires_grad]

        self.denoising_step_list = denoising_step_list
        self.dmd_enabled = dmd_enabled
        self.timestep_shift = timestep_shift
        self.guidance_scale = guidance_scale
        self.save_per_rank = True
        self.verbose = False

        self.warmup_steps = warmup_steps

        print(f"DMD enabled: {self.dmd_enabled}", flush=True)

    def _register_loss_submodules(
        self,
        loss_instance: Any,
        idx: int,
        loss_name: str,
    ) -> None:
        
        prefix = f"loss_{idx}"
        if isinstance(loss_instance, nn.Module):
            self.add_module(prefix, loss_instance)
            return
                                                         
        for attr_name in vars(loss_instance):
            if attr_name.startswith("__"):
                continue
            try:
                obj = getattr(loss_instance, attr_name)
            except AttributeError:
                continue
            if isinstance(obj, nn.Module):
                                                                   
                sub_name = f"{prefix}_{attr_name}"
                self.add_module(sub_name, obj)
    
    def _init_image_cond_model(self) -> None:
        with dist_utils.local_master_first():
            dino_github = "external/dinov3"
            dino_weight = "external/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
            self.image_resolution = 592
            model = torch.hub.load(dino_github, self.image_cond_model_name, source="local", weights=dino_weight, pretrained=True)
            model.eval().to(self.device)
        transform = transforms.Compose(
            [
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.image_cond_model = {"model": model, "transform": transform}

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        assert isinstance(image, torch.Tensor), "Lightning training expects tensor images in batch"
        assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        if image.shape[-1] != self.image_resolution or image.shape[-2] != self.image_resolution:
            image = F.interpolate(image, size=(self.image_resolution, self.image_resolution), mode="bilinear", align_corners=False)
        image = self.image_cond_model["transform"](image).to(self.device)
        
        with torch.autocast("cuda", dtype=self.dtype):
            features = self.image_cond_model["model"](image, is_training=True)["x_prenorm"]
        patchtokens = F.layer_norm(features, features.shape[-1:]).to(self.dtype)
        return patchtokens

    def get_cond(self, cond_img: Optional[torch.Tensor], **kwargs) -> Optional[torch.Tensor]:
        if cond_img is None:
            return None
        return self.encode_image(cond_img)
    
        
            
    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in (None, "fit"):
            return
        self._init_image_cond_model()
        self.gs_decoder = unilat_models.from_pretrained(self.gs_decoder_name)
        self.gs_decoder.eval().requires_grad_(False)
        self.gen_model = unilat_models.from_pretrained(self.gen_model_name)
        self.gen_model.eval().requires_grad_(False)
        with dist_utils.master_first():
            for idx, loss_cfg in enumerate(self.losses_config):
                name = loss_cfg["name"]
                args = loss_cfg.get("args", {})
                
                if not hasattr(unpaired_loss, name):
                    available = [attr for attr in dir(unpaired_loss) if isinstance(getattr(unpaired_loss, attr), type) and attr.endswith('Loss')]
                    raise ValueError(f"Unknown loss: {name}. Available losses: {available}")
                
                loss_cls = getattr(unpaired_loss, name)
                loss_instance = loss_cls(**args)
                                                                                                  
                                                                                 
                self._register_loss_submodules(loss_instance, idx, name)
                self.losses.append(loss_instance)

    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_0)
        t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        return (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise
    
    
    def generate_and_sync_list(self, num_denoising_steps, device) -> List[int]:
        
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
                                                                               
                                                                     
            warmup_steps = self.warmup_steps

            use_bias = (
                warmup_steps > 0
                and self.global_step < warmup_steps
            )

            if use_bias:
                                                           
                progress = float(self.global_step) / float(warmup_steps)
                                                                            
                                                                       
                alpha_max = 4.0
                alpha = alpha_max * (1.0 - progress)

                t_list = torch.tensor(
                    self.denoising_step_list[:num_denoising_steps],
                    device=device,
                    dtype=torch.float32,
                ).clamp(min=1e-6)
                weights = t_list.pow(-alpha)
                probs = weights / weights.sum()
                indices = torch.multinomial(probs, 1)
            else:
                                                           
                indices = torch.randint(
                    0,
                    num_denoising_steps,
                    (1,),
                    device=device,
                )
        else:
            indices = torch.empty(1, dtype=torch.int64, device=device)

        dist.broadcast(indices, src=0)
        return indices.tolist()
    
    def update_requires_grad(self, model_g: bool, model_aux: bool) -> None:
        if model_g:
            self.model_g.train()
        else:
            self.model_g.eval()
        if model_aux:
            self.model_aux.train()
        else:
            self.model_aux.eval()
        for param in self.model_g_train_params:
            param.requires_grad = model_g
        for param in self.model_aux_train_params:
            param.requires_grad = model_aux
        
    def save_sample(self, edited_gs: list[Gaussian], before_gs: list[Gaussian]):
        save_dir = os.path.join(self.output_dir, "edited")
        os.makedirs(save_dir, exist_ok=True)
        concat_save_path = os.path.join(save_dir, f"{self.global_rank}_{self.global_step}.png")
        video_save_path = os.path.join(save_dir, f"{self.global_rank}_{self.global_step}_video.mp4")
        
        before_img = render_dbg([before_gs[0]], self.image_resolution)
        edited_img = render_dbg([edited_gs[0]], self.image_resolution)
        
        concat_img = np.concatenate([before_img, edited_img], axis=1)
        Image.fromarray(concat_img).save(concat_save_path)
        
                                
        edited_video = render_utils.render_video(edited_gs[0], bg_color=(1, 1, 1), num_frames=300, resolution=512, verbose=False)['color']
        before_video = render_utils.render_video(before_gs[0], bg_color=(1, 1, 1), num_frames=300, resolution=512, verbose=False)['color']
        
        num_frames = min(len(edited_video), len(before_video))
        edited_video = edited_video[:num_frames]
        before_video = before_video[:num_frames]
        
        concat_video = []
        for before_frame, edited_frame in zip(before_video, edited_video):
            concat_frame = np.concatenate([before_frame, edited_frame], axis=1)
            concat_video.append(concat_frame)
        
                                 
        imageio.mimsave(video_save_path, concat_video, fps=60)

    def edit_model_training_step(self, batch: Dict[str, Any], batch_idx: int):
        self.update_requires_grad(True, False)
        rank = dist.get_rank() if dist.is_initialized() else 0
        y = batch["y"]
        cond = batch["cond"]
        inst = batch["inst"]
        name = batch["name"]
        vlm_view_img = batch["vlm_view_img"]
        vlm_back_img = batch["vlm_back_img"]
        device = y.device
        print(f"rank {rank} Processing sample: {name}", flush=True)
        self.print_memory_info(rank, "step_start", device)
        
        cond_feat = self.get_cond(cond)
        self.print_memory_info(rank, "after_get_cond", device)

        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(num_denoising_steps, y.device)

        noise = torch.randn_like(y)

        print(f"rank {rank} exit_flags: {exit_flags}", flush=True)

        for index, t in enumerate(self.denoising_step_list):
            exit_flag = (index == exit_flags[0])
            t_tensor = torch.tensor([1000 * t] * y.shape[0], device=y.device, dtype=y.dtype)
            if not exit_flag:
                with torch.no_grad():
                    v_theta = self.model_g(noise, t_tensor, cond_feat, y)
                    next_t = self.denoising_step_list[index + 1]
                    noise = noise - (t - next_t) * v_theta
            else:
                v_theta = self.model_g(noise, t_tensor, cond_feat, y)
                noise = noise - t * v_theta
                break

        x0_theta = noise
        self.print_memory_info(rank, "after_denoising", device)

        torch.cuda.empty_cache()

        loss = torch.zeros((), device=y.device, dtype=y.dtype)
        before_gs = self.gs_decoder(y)
        self.print_memory_info(rank, "after_gs_decoder_before", device)
        edited_gs = self.gs_decoder(x0_theta)
        self.print_memory_info(rank, "after_gs_decoder_edited", device)

        torch.cuda.empty_cache()        

        after_render_res = render(edited_gs, self.cam)
        self.print_memory_info(rank, "after_render_edited", device)
        before_render_res = render(before_gs, self.cam)
        self.print_memory_info(rank, "after_render_before", device)
        after_rgb = after_render_res["color"][:, 0]
        before_rgb = before_render_res["color"][:, 0]
        after_alpha = after_render_res["alpha"][:, 0] if after_render_res["alpha"] is not None else None
        after_depth = after_render_res["depth"][:, 0] if after_render_res["depth"] is not None else None
        intent_img = cond.to(y.device).clamp(0, 1)
        vlm_view_img = vlm_view_img.to(y.device).clamp(0, 1)
        vlm_back_img = vlm_back_img.to(y.device).clamp(0, 1)
        if intent_img.shape[-2:] != after_rgb.shape[-2:]:
            intent_img = F.interpolate(
                intent_img,
                size=after_rgb.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
        
        img_ref_rgba = batch.get("ref_img", None)
        img_ref_alpha = None
        if img_ref_rgba.shape[1] == 4:
            img_ref_alpha = img_ref_rgba[:, 3:4].to(y.device).clamp(0, 1)
        
        should_save = False
        save_dir_for_loss = None
        if rank == 0 or self.save_per_rank:
            edit_steps = self.global_step // (1 + self.aux_steps)
            if edit_steps % self.sample_steps == 0:
                should_save = True
                save_dir_for_loss = os.path.join(self.output_dir, "edited")
        
        if should_save:
            self.save_sample(edited_gs, before_gs)
            self.print_memory_info(rank, "after_save_sample", device)
        
        with torch.amp.autocast(dtype=y.dtype, device_type="cuda"):
            for loss_instance in self.losses:
                loss_kwargs = {
                    "before_gs": before_gs,
                    "after_gs": edited_gs,
                    "vlm_cam": self.vlm_cam,
                    "img_before": before_rgb.to(y.dtype),
                    "img_after": after_rgb.to(y.dtype),
                    "img_ref": intent_img.to(y.dtype),
                    "edit_inst": inst,
                    "img_ref_alpha": img_ref_alpha,
                    "img_after_alpha": after_alpha,
                    "render_depth": after_depth,
                    "rank": rank,
                    "consistency_cam": self.consistency_cam,
                    "vlm_view_img": vlm_view_img,
                    "vlm_back_img": vlm_back_img,
                }
                
                if should_save:
                    loss_kwargs["save_dir"] = save_dir_for_loss
                    loss_kwargs["global_step"] = self.global_step
                
                loss_out, log_dict = loss_instance(**loss_kwargs)
                self.print_memory_info(rank, f"after_loss_{loss_instance.__class__.__name__}", device)
                for key, value in log_dict.items():
                    self.log(f"loss_{loss_instance.__class__.__name__}_{key}", value.detach(), on_step=True, logger=True)
                loss_comp = loss_out
                if loss_comp.ndim > 0:
                    loss_comp = loss_comp.mean()
                loss = loss + loss_comp
        
        self.print_memory_info(rank, "after_all_losses", device)
        print(f"rank {rank} Process done", flush=True)
        
        
        
        if self.dmd_enabled:
            dmd_kwargs = {"y": y, "x0_theta": x0_theta, "cond_feat": cond_feat}
            if should_save:
                dmd_kwargs["save_dir"] = save_dir_for_loss
                dmd_kwargs["global_step"] = self.global_step
            dmd_loss = self.dmd_w * self.dmd_loss(**dmd_kwargs)
            self.print_memory_info(rank, "after_dmd_loss", device)
            loss = loss + dmd_loss
            self.log("loss_dmd", dmd_loss.detach(), on_step=True, logger=True)
        
        self.print_memory_info(rank, "step_end", device)


        return loss

    def get_w_t(self, t, type="constant"):
        if type == "constant":
            return torch.ones_like(t)
        elif type == "sigmoid":
                                
            return t * (1 - t) * 4 
        elif type == "log_normal":
                                         
            return torch.exp(-((t.log() - 0.5)**2) / 0.5)

    def dmd_loss(
        self,
        y: torch.Tensor,
        x0_theta: torch.Tensor,
        cond_feat: torch.Tensor,
        save_dir: Optional[str] = None,
        global_step: Optional[int] = None,
    ):
        B = y.shape[0]
        device = y.device
        dtype = y.dtype

        t_u = self.sample_t_uniform(B, device, 0.02, 0.98, same_across_batch=False)

        if self.timestep_shift > 1:
            s = float(self.timestep_shift)
            t_u = (s * t_u) / (1.0 + (s - 1.0) * t_u)
        t_u = t_u.clamp(0.02, 0.98)
        t_int = (t_u * 1000.0).round().long()
        t_tensor = t_int.to(dtype=dtype)
        eps_u = torch.randn_like(y)
        t_view = t_u.view(B, *([1] * (y.ndim - 1)))
        x_theta_t = (1.0 - t_view) * x0_theta + t_view * eps_u
        x_theta_t = x_theta_t.to(dtype)
        with torch.no_grad():

                           
            v_real = self.gen_model(x_theta_t, t_tensor, cond_feat)
            v_real_uncond = self.gen_model(x_theta_t, t_tensor, torch.zeros_like(cond_feat))

                                 
            v_gen = self.model_aux(x_theta_t, t_tensor, cond_feat, y)
            v_real = v_real_uncond + self.guidance_scale * (v_real - v_real_uncond)

                          
            x0_real_hat = x_theta_t - t_view * v_real
                                             
            x0_fake_hat = x_theta_t - t_view * v_gen

                                                                                
            rank = dist.get_rank() if dist.is_initialized() else 0
            if save_dir is not None and global_step is not None and (rank == 0 or self.save_per_rank):
                os.makedirs(save_dir, exist_ok=True)

                try:
                    gen_gs = self.gs_decoder(x0_real_hat[:1].to(dtype=dtype))
                    gen_video = render_utils.render_video(
                        gen_gs[0], bg_color=(1, 1, 1), num_frames=300,
                        resolution=self.image_resolution, verbose=False
                    )['color']
                    video_path = os.path.join(save_dir, f"{rank}_{global_step}_dmd_gen_model.mp4")
                    imageio.mimsave(video_path, gen_video, fps=60)
                except Exception as e:
                    if self.verbose:
                        print(f"[rank {rank}] dmd_loss save gen_model video skip: {e}", flush=True)

            p_real = (x0_theta - x0_real_hat)
            p_fake = (x0_theta - x0_fake_hat)
            denom = p_real.abs().mean(dim=list(range(1, p_real.ndim)), keepdim=True).clamp_min(1e-6)
            grad = (p_real - p_fake) / denom
            grad = torch.nan_to_num(grad)
        loss_dmd = 0.5 * F.mse_loss(
            x0_theta.float(),
            (x0_theta - grad).detach().float(),
            reduction="mean"
        )
        return loss_dmd

    def sample_t_uniform(self, batch_size: int, device, t_min: float = 0.02, t_max: float = 0.98,
                     same_across_batch: bool = False) -> torch.Tensor:
        
        if same_across_batch:
            t = torch.empty(1, device=device).uniform_(t_min, t_max).repeat(batch_size)
        else:
            t = torch.empty(batch_size, device=device).uniform_(t_min, t_max)
        return t
        
        
    
    def edit_model_recon_training_step(self, batch: Dict[str, Any], batch_idx: int):
        
                               
        self.update_requires_grad(True, False)

        rank = dist.get_rank() if dist.is_initialized() else 0
        y = batch["y"]
        ori_img = batch.get("ori_img", None)
        if ori_img is None:
            raise ValueError("Batch must contain 'ori_img' for edit_model_recon_training_step.")

        device = y.device
                                                       
        cond_feat = self.get_cond(ori_img.to(device))

        B = y.shape[0]
        dtype = y.dtype

                              
        t_u = self.sample_t_uniform(B, device, 0.02, 0.98, same_across_batch=False)

        if getattr(self, "timestep_shift", 1.0) > 1:
            s = float(self.timestep_shift)
            t_u = (s * t_u) / (1.0 + (s - 1.0) * t_u)
        t_u = t_u.clamp(0.02, 0.98)
        t_view = t_u.view(B, *([1] * (y.ndim - 1))).to(dtype=dtype)

                                                         
        eps = torch.randn_like(y)
        x_t = (1 - t_view) * y + t_view * eps

                            
        v_target = eps - y

                                                     
        t_tensor = (t_u * 1000.0).to(dtype=dtype)
        v_pred = self.model_g(x_t, t_tensor, cond_feat, y)
        loss_recon = F.mse_loss(v_pred, v_target)
        return loss_recon

    def aux_model_training_step(self, batch: Dict[str, Any], batch_idx: int):
        y = batch["y"]
        cond = batch["cond"]
        self.update_requires_grad(False, True)

        cond_feat = self.get_cond(cond)

        with torch.no_grad():
            x = torch.randn_like(y)

            for i, t in enumerate(self.denoising_step_list):
                t_tensor = torch.full((y.shape[0],), 1000.0 * float(t), device=y.device, dtype=y.dtype)

                next_t = 0.0 if i == (len(self.denoising_step_list) - 1) else float(self.denoising_step_list[i + 1])
                dt = float(t) - next_t

                v = self.model_g(x, t_tensor, cond_feat, y)
                x = x - dt * v

            x0_fake = x


        B = x0_fake.shape[0]
        device, dtype = x0_fake.device, x0_fake.dtype

                              
        t_u = self.sample_t_uniform(B, device, 0.02, 0.98, same_across_batch=False)

        if getattr(self, "timestep_shift", 1.0) > 1:
            s = float(self.timestep_shift)
            t_u = (s * t_u) / (1.0 + (s - 1.0) * t_u)
        t_u = t_u.clamp(0.02, 0.98)
        t_view = t_u.view(B, *([1]*(x0_fake.ndim-1))).to(dtype=dtype)

                                      
        eps = torch.randn_like(x0_fake)
        x_t = (1 - t_view) * x0_fake + t_view * eps

                            
        v_target = eps - x0_fake

                           
        t_tensor = (t_u * 1000.0).to(dtype=dtype)                      
        v_pred = self.model_aux(x_t, t_tensor, cond_feat, y)
        loss_aux = F.mse_loss(v_pred, v_target)
        return loss_aux
    
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        sampler_state = checkpoint.get("train_sampler_state", None)
        if sampler_state is not None and self.trainer is not None and self.trainer.datamodule is not None:
            setter = getattr(self.trainer.datamodule, "set_train_sampler_state", None)
            if callable(setter):
                setter(sampler_state)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.trainer is None or self.trainer.datamodule is None:
            return
        getter = getattr(self.trainer.datamodule, "get_train_sampler_state", None)
        if callable(getter):
            sampler_state = getter()
            if sampler_state is not None:
                checkpoint["train_sampler_state"] = sampler_state

    def configure_optimizers(self):
        model_g_params = list(self.model_g.parameters())
        model_aux_params = list(self.model_aux.parameters())
        if hasattr(torch.optim, self.optimizer_config["name"]):
            return getattr(torch.optim, self.optimizer_config["name"])(model_g_params + model_aux_params, **self.optimizer_config.get("args", {}))
        elif hasattr(deepspeed.ops.adam, self.optimizer_config["name"]):
            return getattr(deepspeed.ops.adam, self.optimizer_config["name"])(model_g_params + model_aux_params, **self.optimizer_config.get("args", {}))
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_config['name']}")
    
    def on_fit_start(self) -> None:
        if not self.trainer.is_global_zero:
            return
        print(
            f"[ResumeCheck] current_epoch={self.current_epoch}, global_step={self.global_step}",
            flush=True,
        )
        optimizers = getattr(self.trainer, "optimizers", None) or []
        if len(optimizers) == 0:
            print("[Optimizer] No optimizer found on trainer.", flush=True)
            return
        for opt_idx, optimizer in enumerate(optimizers):
            print(f"[Optimizer {opt_idx}] class={optimizer.__class__.__name__}", flush=True)
            defaults = getattr(optimizer, "defaults", {})
            if defaults:
                print(f"[Optimizer {opt_idx}] defaults={defaults}", flush=True)
            for group_idx, group in enumerate(optimizer.param_groups):
                group_hparams = {k: v for k, v in group.items() if k != "params"}
                group_param_count = sum(p.numel() for p in group["params"])
                print(
                    f"[Optimizer {opt_idx}] param_group={group_idx}, "
                    f"num_tensors={len(group['params'])}, num_params={group_param_count}, "
                    f"hparams={group_hparams}",
                    flush=True,
                )

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        torch.cuda.empty_cache()
        rank = dist.get_rank() if dist.is_initialized() else 0
        if self.dmd_enabled:
            cycle = 1 + self.aux_steps
            pos = int(self.global_step % cycle) if cycle > 0 else 0

            if pos == 0:
                
                edit_steps = self.global_step // (1 + self.aux_steps)
                if self.recon_every > 0 and edit_steps % self.recon_every == self.recon_every - 1:
                    if rank == 0:
                        print(f"Reconstruction step", flush=True)
                    loss = self.edit_model_recon_training_step(batch, batch_idx)
                    self.log("loss_recon", loss.detach(), on_step=True, logger=True)
                else:
                    if rank == 0:
                        print(f"Main edit step", flush=True)
                    loss = self.edit_model_training_step(batch, batch_idx)
                    self.log("loss_edit", loss.detach(), on_step=True, logger=True)
            else:
                                             
                if rank == 0:
                    print(f"Auxiliary distillation step", flush=True)
                loss = self.aux_model_training_step(batch, batch_idx)
                self.log("loss_aux", loss.detach(), on_step=True, logger=True)
        else:
            loss = self.edit_model_training_step(batch, batch_idx)
            self.log("loss_edit", loss.detach(), on_step=True, logger=True)
        return loss
    
    def print_memory_info(self, rank: int, step_name: str, device: torch.device = None):
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated(device) / 1024**3      
            reserved = torch.cuda.memory_reserved(device) / 1024**3      
            max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3      
            max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3      
            
            if self.verbose:
                print(f"[rank {rank}] Memory at {step_name}: "
                    f"allocated={allocated:.2f}GB, reserved={reserved:.2f}GB, "
                    f"max_allocated={max_allocated:.2f}GB, max_reserved={max_reserved:.2f}GB", flush=True)
            
                              
            torch.cuda.reset_peak_memory_stats(device)