from __future__ import annotations

from typing import List, Optional
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from easydict import EasyDict as edict
from transformers import AutoModelForImageTextToText, AutoProcessor

from unilat3d.representations.gaussian.gaussian_model import Gaussian
from unilat3d.lightning.render_utils import render

from ..vlm_utils import calc_vlm_reward


class VLMLoss:
    
    
    def __init__(
        self,
        vlm_model_name: str,
        prob_loss: bool = False,
        weight: float = 1.0,
    ):
        
        self.vlm_model_name = vlm_model_name

        self.weight = weight
        
        self.processor = AutoProcessor.from_pretrained(
            self.vlm_model_name, 
            trust_remote_code=True
        )
        
                                      
        model_kwargs = {
            "trust_remote_code": True,
        }
        self.vlm_model = AutoModelForImageTextToText.from_pretrained(
            self.vlm_model_name,
            **model_kwargs
        ).to(device='cuda')
        self.vlm_model.config.use_cache = False
        self.vlm_model.eval().requires_grad_(False)
    
    def __call__(
        self,
        before_gs: List[Gaussian],
        after_gs: List[Gaussian],
        img_ref: torch.Tensor,
        edit_inst: List[str],
        vlm_cam: dict,
        consistency_cam: dict,
        save_dir: Optional[str] = None,
        global_step: Optional[int] = None,
        rank: Optional[int] = None,
        vlm_view_img: Optional[torch.Tensor] = None,
        vlm_back_img: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> (torch.Tensor, edict):
        
        rr = int(img_ref.shape[-1])
        
        if img_ref.shape[-2:] != (rr, rr):
            img_ref = F.interpolate(
                img_ref, 
                size=(rr, rr), 
                mode="bilinear", 
                align_corners=False
            )
        after_render_res = render(after_gs, vlm_cam)
        after_render_consistency = render(after_gs, consistency_cam)
        after_rgb = after_render_res["color"][:, 0]
        img_after = after_rgb
        img_after_consistency = after_render_consistency["color"][:, 0]
        img_ref = img_ref.clamp(0, 1)

                                                                                   

                                
                                                                 
                                                              
                                                                                              

                                                          

           
                                                                                                                                               

                                                     
     
        prompt_2 = f"""You are evaluating visual quality of a 3D edited object.

You are given:
- Image 1: Rendering of the original 3D model (before editing).
- Image 2: Rendering of the edited 3D model (after editing).

Question:
Is the object edited keeping the original model's style and reserving it's identity?
IGNORE the changes in the second image because of the editing instruction {edit_inst}.

Answer strictly with "Yes" or "No". Do not explain."""

        prompt_3 = f"""You are evaluating visual quality of a 3D edited object.

You are given:
- Image 1: Rendering of the original 3D model (before editing).
- Image 2: Rendering of the edited 3D model (after editing).

Question:
Does edited object successfully apply the editing instruction {edit_inst}?

Answer strictly with "Yes" or "No". Do not explain."""

                                  
                                      
                                       
                                       
                                                    
                                  
           
                                  
                                      
                                       
                                       
                                                         
                                  
           
        torch.cuda.empty_cache()
        vlm_out_2 = calc_vlm_reward(
            processor=self.processor,
            vlm_model=self.vlm_model,
            imgs=[vlm_view_img, img_after],
            promt_text=prompt_2,
        )
        torch.cuda.empty_cache()
        vlm_out_3 = calc_vlm_reward(
            processor=self.processor,
            vlm_model=self.vlm_model,
            imgs=[vlm_view_img, img_after],
            promt_text=prompt_3,
        )

                                                                                

        vlm_out = vlm_out_2.loss_vlm + vlm_out_3.loss_vlm
                                      
        loss_vlm = vlm_out * self.weight

        if save_dir is not None and global_step is not None:
            os.makedirs(save_dir, exist_ok=True)
            
            questions = [
                                              
                ("q2", vlm_out_2, prompt_2),
                ("q3", vlm_out_3, prompt_3),
            ]
            
            for q_name, vlm_out_q, prompt_q in questions:
                if vlm_out_q.concat_imgs:
                    concat_img = vlm_out_q.concat_imgs[0]
                    concat_img_np = concat_img.detach().cpu().permute(1, 2, 0).numpy()
                    if concat_img_np.dtype != np.uint8:
                        concat_img_np = (concat_img_np * 255).astype(np.uint8)
                    img_save_path = os.path.join(save_dir, f"{rank}_{global_step}_vlm_{q_name}_concat.png")
                    Image.fromarray(concat_img_np).save(img_save_path)
                
                if vlm_out_q.gen_texts:
                    text_save_path = os.path.join(save_dir, f"{rank}_{global_step}_vlm_{q_name}_gen_text.txt")
                    with open(text_save_path, 'w', encoding='utf-8') as f:
                        f.write(f"Question: {q_name}\n")
                        f.write(f"Prompt: {prompt_q}\n")
                        f.write(f"Generated Text: {vlm_out_q.gen_texts[0]}\n")

        extra_log = edict(
            total_loss=loss_vlm.detach(),
                                                  
            q2_loss=vlm_out_2.loss_vlm.detach(),
            q3_loss=vlm_out_3.loss_vlm.detach(),
        )
        return loss_vlm, extra_log
