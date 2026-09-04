import os
from typing import Optional
import torch
import torch.nn as nn
import math
from easydict import EasyDict as edict
import utils3d
from unilat3d.renderers import GaussianRenderer
from unilat3d.representations import Gaussian
from PIL import Image
import numpy as np
from unilat3d.lightning.utils import denormalize_latents

def render(gs: list[Gaussian], cam: dict) -> edict:
    render_near = 0.8
    render_far = 1.6
    render_ssaa = 1
    render_num_views = 1
    render_radius = cam['radius']
    render_fov = cam['fov']
    num_views = 1
    cam_dtype = torch.float32

    device = gs[0].device

    renderer = GaussianRenderer(
        {
            "resolution": cam['resolution'],
            "near": render_near,
            "far": render_far,
            "ssaa": render_ssaa,
            "bg_color": cam['bg_color'] if 'bg_color' in cam else (0, 0, 0)
        }
    )

    num_views = render_num_views
    radius = render_radius
    fov = torch.tensor(render_fov, device=device, dtype=cam_dtype)

    yaws = torch.full((num_views,), cam['yaw'], device=device, dtype=cam_dtype)
                                                                                
    pitchs = torch.full((num_views,), cam['pitch'], device=device, dtype=cam_dtype)

    extrinsics = []
    intrinsics = []
    target = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=cam_dtype)
    up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=cam_dtype)
    for yaw, pitch in zip(yaws, pitchs):
        orig = (
            torch.stack(
                [
                    torch.sin(yaw) * torch.cos(pitch),
                    torch.cos(yaw) * torch.cos(pitch),
                    torch.sin(pitch),
                ],
                dim=0,
            )
            * radius
        )
        extr = utils3d.torch.extrinsics_look_at(orig, target, up)
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        extrinsics.append(extr)
        intrinsics.append(intr)
    extrinsics = torch.stack(extrinsics, dim=0).to(dtype=cam_dtype)
    intrinsics = torch.stack(intrinsics, dim=0).to(dtype=cam_dtype)

    colors = []
    alphas = []
    depths = []
    with torch.autocast("cuda", enabled=False):
        for g in gs:
            view_imgs = []
            view_alphas = []
            view_depths = []
            for v in range(num_views):
                out = renderer.render(g, extrinsics[v], intrinsics[v])
                view_imgs.append(out["color"].float())
                if out["alpha"] is not None:
                    view_alphas.append(out["alpha"].float())
                if out["depth"] is not None:
                    view_depths.append(out["depth"].float())
            colors.append(torch.stack(view_imgs, dim=0))
            if len(view_alphas) > 0:
                alphas.append(torch.stack(view_alphas, dim=0))
            if len(view_depths) > 0:
                depths.append(torch.stack(view_depths, dim=0))
        color = torch.stack(colors, dim=0).float()
        if len(alphas) > 0:
            alpha = torch.stack(alphas, dim=0).float()
        else:
            alpha = None
        if len(depths) > 0:
            depth = torch.stack(depths, dim=0).float()
        else:
            depth = None
    return edict({"color": color, "alpha": alpha, "depth": depth})


@torch.no_grad()
def render_dbg(gs: list[Gaussian], render_resolution: int = 512) -> np.ndarray:
    render_near = 0.8
    render_far = 1.6
    render_ssaa = 1
    render_radius = 2.0
    render_fov_deg = 40.0
    
    device = gs[0].device
    
                    
    num_views = 4
                                       
    yaws = torch.tensor([0.0, math.pi/2, math.pi, 3*math.pi/2], device=device, dtype=torch.float32)
    pitchs = torch.zeros(num_views, device=device, dtype=torch.float32)
    
    cam_dtype = torch.float32

    renderer = GaussianRenderer(
        {
            "resolution": render_resolution,
            "near": render_near,
            "far": render_far,
            "ssaa": render_ssaa,
            "bg_color": (0, 0, 0),
        }
    )

    radius = render_radius
    fov = torch.deg2rad(torch.tensor(render_fov_deg, device=device, dtype=cam_dtype))

    extrinsics = []
    intrinsics = []
    target = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=cam_dtype)
    up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=cam_dtype)
    for yaw, pitch in zip(yaws, pitchs):
        orig = (
            torch.stack(
                [
                    torch.sin(yaw) * torch.cos(pitch),
                    torch.cos(yaw) * torch.cos(pitch),
                    torch.sin(pitch),
                ],
                dim=0,
            )
            * radius
        )
        extr = utils3d.torch.extrinsics_look_at(orig, target, up)
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        extrinsics.append(extr)
        intrinsics.append(intr)
    extrinsics = torch.stack(extrinsics, dim=0).to(dtype=cam_dtype)
    intrinsics = torch.stack(intrinsics, dim=0).to(dtype=cam_dtype)

    colors = []
    with torch.autocast("cuda", enabled=False):
        for g in gs:
            view_imgs = []
            for v in range(num_views):
                out = renderer.render(g, extrinsics[v], intrinsics[v])
                view_imgs.append(out["color"].float())
            colors.append(torch.stack(view_imgs, dim=0))
        color = torch.stack(colors, dim=0).float()

                        
    color_np = color.detach().cpu().numpy()
    view_images = []
    for v in range(num_views):
        img = color_np[0, v]
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))             
        view_images.append(img)
    
                          
                
                  
    h, w = view_images[0].shape[:2]
    merged_img = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    merged_img[0:h, 0:w] = view_images[0]     
    merged_img[0:h, w:2*w] = view_images[1]     
    merged_img[h:2*h, 0:w] = view_images[2]     
    merged_img[h:2*h, w:2*w] = view_images[3]     
    
    return merged_img