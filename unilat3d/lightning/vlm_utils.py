from __future__ import annotations

from typing import List, Optional, Tuple

import math

import torch
import torch.nn.functional as F
from easydict import EasyDict as edict


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int = 28,
    min_pixels: int = 3136,
    max_pixels: int = 12845056,
) -> Tuple[int, int]:
    
    if height < factor or width < factor:
        factor = min(height, width, factor) if min(height, width, factor) > 0 else 1
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar == 0:
        h_bar = factor
    if w_bar == 0:
        w_bar = factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(max(1, height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
        if h_bar == 0:
            h_bar = factor
        if w_bar == 0:
            w_bar = factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / max(height * width, 1))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
        if h_bar == 0:
            h_bar = factor
        if w_bar == 0:
            w_bar = factor
    h_bar = int(max(factor, h_bar))
    w_bar = int(max(factor, w_bar))
    h_bar = round(h_bar / factor) * factor
    w_bar = round(w_bar / factor) * factor
    return int(h_bar), int(w_bar)


def prepare_initial_image_tensors(processor, image_tensor: torch.Tensor) -> torch.Tensor:
    
    img_processor = processor.image_processor
    patch_size = img_processor.patch_size
    merge_size = img_processor.merge_size

    b, c, h, w = image_tensor.shape
    target_h, target_w = smart_resize(
        h,
        w,
        factor=patch_size * merge_size,
        min_pixels=img_processor.min_pixels or img_processor.size["shortest_edge"],
        max_pixels=img_processor.max_pixels or img_processor.size["longest_edge"],
    )
    x = F.interpolate(image_tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)

    mean = torch.tensor(img_processor.image_mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(img_processor.image_std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def qwen_yesno_reward(
    *,
    processor,
    vlm_model,
    images_bchw: torch.Tensor,
    prompt_text: str,
    concat_images: bool = False,
) -> edict:
    
    vlm_device = vlm_model.device
    img = images_bchw.to(vlm_device).clamp(0, 1)

    panels = []
    target_h, target_w = img[0].shape[-2], img[0].shape[-1]
    for k in range(img.shape[0]):
        panel = img[k]
        if panel.shape[-2:] != (target_h, target_w):
            panel = F.interpolate(
                panel.unsqueeze(0),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        panels.append(panel)

                                                                           
    concat_img = torch.cat(panels, dim=-1)

    image_placeholders = [{"type": "image"}] * img.shape[0]
    content_list = image_placeholders + [{"type": "text", "text": prompt_text}]
    messages = [{"role": "user", "content": content_list}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt", do_rescale=False)
    model_inputs = {k: v.to(vlm_device) for k, v in inputs.items()}

    img_norm = prepare_initial_image_tensors(processor, img)
    processed_image = processor.image_processor.process_precomputed_tensors(img_norm)

    final_inputs = {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "pixel_values": processed_image["pixel_values"].to(vlm_device),
        "image_grid_thw": processed_image["image_grid_thw"].to(vlm_device),
    }

    outputs = vlm_model(**final_inputs)
    logits = outputs.logits
    last_token_logits = logits[0, -1, :]
    yes_token_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = processor.tokenizer.encode("No", add_special_tokens=False)[0]

    logit_yes = last_token_logits[yes_token_id]
    logit_no = last_token_logits[no_token_id]
    loss = -F.logsigmoid(logit_yes - logit_no)

    with torch.no_grad():
        gen_ids = vlm_model.generate(
            **final_inputs,
            max_new_tokens=64,
            do_sample=False,                     
        )

    gen_text = processor.batch_decode(
        gen_ids[:, final_inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0]
    return edict(loss=loss, logit_yes=logit_yes, logit_no=logit_no, gen_text=gen_text, concat_img=concat_img)


def calc_vlm_reward(
    *,
    processor,
    vlm_model,
    imgs: List[torch.Tensor],
    promt_text: str,
) -> edict:
    

    losses: List[torch.Tensor] = []
    logits_yes: List[torch.Tensor] = []
    logits_no: List[torch.Tensor] = []


    b = imgs[0].shape[0]
    gen_texts: List[str] = []
    concat_imgs: List[torch.Tensor] = []
    for i in range(b):
        images_bchw = torch.cat([img[i : i + 1] for img in imgs], dim=0)
        reward = qwen_yesno_reward(
            processor=processor,
            vlm_model=vlm_model,
            images_bchw=images_bchw,
            prompt_text=promt_text,
        )
        losses.append(reward.loss)
        logits_yes.append(reward.logit_yes)
        logits_no.append(reward.logit_no)
        gen_texts.append(reward.gen_text)
        concat_imgs.append(reward.concat_img)
    loss = torch.stack(losses).mean()
    logits_yes = torch.stack(logits_yes).mean()
    logits_no = torch.stack(logits_no).mean()

    return edict(
        loss_vlm=loss,
        logits_yes=logits_yes,
        logits_no=logits_no,
        gen_texts=gen_texts,
        concat_imgs=concat_imgs,
    )