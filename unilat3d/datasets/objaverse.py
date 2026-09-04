import ast
import math
from typing import *
from abc import abstractmethod
import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

def _composite_to_bg_rgb(img: Image.Image, bg_color: Tuple[float, float, float] = (1, 1, 1)) -> Image.Image:
    r, g, b = bg_color
    r_i = int(max(0.0, min(1.0, float(r))) * 255)
    g_i = int(max(0.0, min(1.0, float(g))) * 255)
    b_i = int(max(0.0, min(1.0, float(b))) * 255)

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img_rgba = img.convert("RGBA")
        bg = Image.new("RGBA", img_rgba.size, (r_i, g_i, b_i, 255))
        img_rgba = Image.alpha_composite(bg, img_rgba)
        return img_rgba.convert("RGB")

    return img.convert("RGB")

class ObjaverseUniLat(torch.utils.data.Dataset):
    def __init__(self,
        roots: str,
        *,
        normalization: Optional[dict] = None,
        image_cond_model: str = "dinov3_vith16plus",
        device: str = "cuda",
        bg_color: Tuple[float, float, float] = (0, 0, 0),
    ):
        super().__init__()
        self.roots = roots.split(',')
        self.normalization = normalization
        self.value_range = (0, 1)
        self.num_workers = 0
        self.device = torch.device(device)
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1)

        root = self.roots[0] if self.roots else roots
        edit_prompts_csv_path = os.path.join(root, "generated_instructions_cleaned.csv")
        prompts_csv = pd.read_csv(edit_prompts_csv_path)
        prompts = {}
        skipped = 0
        for index, row in prompts_csv.iterrows():
            uid = row['uid']
            try:
                prompts[uid] = ast.literal_eval(row['instructions'])
            except:
                skipped += 1
        
        self.cam = {
            'radius': 2.0,
            'fov': 0.8,
            'yaw': -math.pi,
            'pitch': 0,
            'resolution': 512,
            'bg_color': (0, 0, 0),
        }

        self.consistency_cam = {
            'radius': 2.0,
            'fov': 0.8,
            'yaw': 0,
            'pitch': 0,
            'resolution': 512,
            'bg_color': (0, 0, 0),
        }

        self.vlm_cam = {
            'radius': 2.0,
            'fov': 0.8,
            'yaw': math.pi + math.pi / 4,
            'pitch': math.pi / 6,
            'resolution': 512,
            'bg_color': (0, 0, 0),
        }
        
        
        self.metadata = []
        for uid in prompts.keys():
            ori_path = os.path.join(root, f'unilat_latents/dinov2_vitl14_reg_encoder/{uid}.npz')
            for idx, edit_inst in enumerate(prompts[uid]):
                img_path = os.path.join(root, f'objaverse_render_edited_rgba/{uid}_{idx}.png')
                ori_img_path = os.path.join(root, f'objaverse_render/{uid}.png')
                vlm_view_path = os.path.join(root, f'objaverse_render_vlm/{uid}.png')
                vlm_back_path = os.path.join(root, f'objaverse_render_vlm_back/{uid}.png')
                instruction = edit_inst
                
                if os.path.exists(ori_path) and os.path.exists(img_path) and os.path.exists(ori_img_path) and os.path.exists(vlm_view_path) and os.path.exists(vlm_back_path):
                    self.metadata.append({
                        'ori_latents': ori_path,
                        'image': img_path,
                        'instruction': instruction,
                        'name': f'{uid}_{idx}',
                        'ori_img': ori_img_path,
                        'vlm_view_img': vlm_view_path,
                        'vlm_back_img': vlm_back_path
                    })

        print(f"Loaded {len(self.metadata)} samples from {len(prompts.keys())} UIDs")
        print(f"Skipped {skipped} samples due to invalid instructions")
        if len(self.metadata) == 0:
            raise ValueError(f"No valid data found in any of the provided roots: {self.roots}")
        self.image_cond_model = image_cond_model

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        entry = self.metadata[idx]

        ori_latents_path = entry['ori_latents']

        ori_latents = np.load(ori_latents_path)

        z_src = torch.tensor(ori_latents['mean']).float()
        if z_src.ndim == 5 and z_src.shape[0] == 1:
            z_src = z_src[0]
        if z_src.ndim != 4:
            raise ValueError(f"Unexpected latent shape in {ori_latents_path}: {tuple(z_src.shape)} (expected (C,R,R,R) or (1,C,R,R,R))")
        
        if self.normalization is not None:
            z_src = (z_src - self.mean) / self.std

        image_path = entry['image']
        ref_rgba_pil = Image.open(image_path).convert("RGBA")
        ref_rgba = np.array(ref_rgba_pil).astype(np.float32) / 255.0
        ref_rgba = torch.from_numpy(ref_rgba).permute(2, 0, 1).contiguous()                  
        alpha_mask = (ref_rgba[3:4] >= 0.1).to(ref_rgba.dtype)
        cond_rgb = (ref_rgba[:3] * alpha_mask).contiguous()

        ori_img_path = entry['ori_img']
        ori_img = Image.open(ori_img_path).convert("RGB")
        ori_img = np.array(ori_img).astype(np.float32) / 255.0
        ori_img = torch.from_numpy(ori_img).permute(2, 0, 1).contiguous()                  

        vlm_view_img_path = entry['vlm_view_img']
        vlm_view_img = Image.open(vlm_view_img_path).convert("RGB")
        vlm_view_img = np.array(vlm_view_img).astype(np.float32) / 255.0
        vlm_view_img = torch.from_numpy(vlm_view_img).permute(2, 0, 1).contiguous()                  

        vlm_back_img_path = entry['vlm_back_img']
        vlm_back_img = Image.open(vlm_back_img_path).convert("RGB")
        vlm_back_img = np.array(vlm_back_img).astype(np.float32) / 255.0
        vlm_back_img = torch.from_numpy(vlm_back_img).permute(2, 0, 1).contiguous()                  

        return {
            'x_0': z_src,
            'y': z_src,
            'cond': cond_rgb,
            'ref_img': ref_rgba,
            'inst': entry['instruction'],
            'name': entry['name'],
            'ori_img': ori_img,
            'vlm_view_img': vlm_view_img,
            'vlm_back_img': vlm_back_img
        }