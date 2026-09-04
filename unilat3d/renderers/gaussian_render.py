 
                           
                                                           
                      
 
                                                                        
                                         
 
                                                  
 

from token import OP
from typing import Optional
import torch
import math
from easydict import EasyDict as edict
import numpy as np
from ..representations.gaussian import Gaussian
from .sh_utils import eval_sh
import torch.nn.functional as F
from easydict import EasyDict as edict


def intrinsics_to_projection(
        intrinsics: torch.Tensor,
        near: float,
        far: float,
    ) -> torch.Tensor:
    
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = - 2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.
    return ret


def render(viewpoint_camera, pc : Gaussian, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    
                 
    if 'GaussianRasterizer' not in globals():
        from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
    
                                                                                                        
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
                                        
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    kernel_size = pipe.kernel_size
    subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2), dtype=torch.float32, device="cuda")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
                                  
                                          
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

                                                                                             
                                           
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

                                                                                                    
                                                                                              
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    alpha = None
    depth = None
                                                                            
    rendered_image, radii, depth, alpha  = rasterizer(
                                         
        means3D = means3D.to(torch.float32),
        means2D = means2D.to(torch.float32),
        shs = shs.to(torch.float32),
        colors_precomp = colors_precomp,
        opacities = opacity.to(torch.float32),
        scales = scales.to(torch.float32),
        rotations = rotations.to(torch.float32),
        cov3D_precomp = cov3D_precomp.to(torch.float32) if cov3D_precomp is not None else None
    )

                                                                                     
                                                                              
    return edict({"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "alpha": alpha,
            "depth": depth,
            })


class GaussianRenderer:
    

    def __init__(self, rendering_options={}) -> None:
        self.pipe = edict({
            "kernel_size": 0.1,
            "convert_SHs_python": False,
            "compute_cov3D_python": False,
            "scale_modifier": 1.0,
            "debug": False
        })
        self.rendering_options = edict({
            "resolution": None,
            "near": None,
            "far": None,
            "ssaa": 1,
            "bg_color": 'random',
        })
        self.rendering_options.update(rendering_options)
        self.bg_color = None
    
    def render(
            self,
            gausssian: Gaussian,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            colors_overwrite: torch.Tensor = None
        ) -> edict:
        
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]
        
        if self.rendering_options["bg_color"] == 'random':
            self.bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
            if np.random.rand() < 0.5:
                self.bg_color += 1
        else:
            self.bg_color = torch.tensor(self.rendering_options["bg_color"], dtype=torch.float32, device="cuda")

        view = extrinsics
        perspective = intrinsics_to_projection(intrinsics, near, far)
        camera = torch.inverse(view)[:3, 3]
        focalx = intrinsics[0, 0]
        focaly = intrinsics[1, 1]
        fovx = 2 * torch.atan(0.5 / focalx)
        fovy = 2 * torch.atan(0.5 / focaly)
            
        camera_dict = edict({
            "image_height": resolution * ssaa,
            "image_width": resolution * ssaa,
            "FoVx": fovx,
            "FoVy": fovy,
            "znear": near,
            "zfar": far,
            "world_view_transform": view.T.contiguous(),
            "projection_matrix": perspective.T.contiguous(),
            "full_proj_transform": (perspective @ view).T.contiguous(),
            "camera_center": camera
        })

                
        render_ret = render(camera_dict, gausssian, self.pipe, self.bg_color, override_color=colors_overwrite, scaling_modifier=self.pipe.scale_modifier)

        if ssaa > 1:
            render_ret.render = F.interpolate(render_ret.render[None], size=(resolution, resolution), mode='bilinear', align_corners=False, antialias=True).squeeze()
            
        ret = edict({
            'color': render_ret['render'],
            'alpha': render_ret['alpha'],
            'depth': render_ret['depth']
        })
        return ret