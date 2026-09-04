# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor class for Qwen2-VL."""

# Necessary Imports
import math
from typing import Dict, List, Optional, Union

import numpy as np
import torch # For the new method


from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import (
    convert_to_rgb,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_flat_list_of_images,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...video_utils import VideoInput, make_batched_videos
from ...utils import TensorType, logging

logger = logging.get_logger(__name__)

def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class Qwen2VLImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Qwen2-VL image processor that dynamically resizes images based on the original images.
    Includes standard preprocessing and a method to process pre-computed tensors directly.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions during standard preprocess.
        size (`Dict[str, int]`, *optional*, defaults to `{"shortest_edge": 3136, "longest_edge": 12845056}`):
            Size parameters used during standard resize logic.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
            Resampling filter for standard resize.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor` during standard preprocess.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor for standard rescale.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image during standard preprocess.
        image_mean (`float` or `List[float]`, *optional*, defaults to `OPENAI_CLIP_MEAN`):
            Mean for standard normalization.
        image_std (`float` or `List[float]`, *optional*, defaults to `OPENAI_CLIP_STD`):
            Standard deviation for standard normalization.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB during standard preprocess.
        min_pixels (`int`, *optional*, defaults to `3136`):
            Used in standard resize logic (`smart_resize`).
        max_pixels (`int`, *optional*, defaults to `12845056`):
            Used in standard resize logic (`smart_resize`).
        patch_size (`int`, *optional*, defaults to 14):
            The spatial patch size of the vision encoder. Used in all patching logic.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The temporal patch size of the vision encoder. Used in all patching logic.
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to llm encoder. Used in all patching logic.
    """

    model_input_names = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        print('The differential image processor!')
        # Use provided defaults or fall back to typical values if None
        if size is None:
            size = {"shortest_edge": 3136, "longest_edge": 12845056} # Default values matching user printout
        if image_mean is None:
            image_mean = OPENAI_CLIP_MEAN
        if image_std is None:
            image_std = OPENAI_CLIP_STD

        if "shortest_edge" not in size or "longest_edge" not in size:
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

        # Override size with min_pixels and max_pixels if they are provided (for backward compatibility)
        if min_pixels is not None: size["shortest_edge"] = min_pixels
        if max_pixels is not None: size["longest_edge"] = max_pixels
        # Use effective min/max from size dict
        self.min_pixels = size["shortest_edge"]
        self.max_pixels = size["longest_edge"]
        self.size = size

        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.do_convert_rgb = do_convert_rgb

        # Ensure mean and std are lists for normalization function compatibility
        self.image_mean = self.image_mean if isinstance(self.image_mean, list) else [self.image_mean] * 3
        self.image_std = self.image_std if isinstance(self.image_std, list) else [self.image_std] * 3

    # Standard preprocessing using PIL/NumPy (called by preprocess)
    def _preprocess(
        self,
        images: Union[ImageInput, VideoInput], # Expects list of image frames
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        do_convert_rgb: bool = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        # Assign defaults from self if specific arguments are None
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        patch_size = patch_size if patch_size is not None else self.patch_size
        temporal_patch_size = temporal_patch_size if temporal_patch_size is not None else self.temporal_patch_size
        merge_size = merge_size if merge_size is not None else self.merge_size
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        # Process list of images/frames
        images = make_list_of_images(images) # Ensure it's a list

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
             logger.warning_once(
                 "It looks like you are trying to rescale already rescaled images. If the input"
                 " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
             )
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])

        # Apply transformations to each frame
        processed_images = []
        # Get original size from first frame for potential resize calc
        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width # Initialize

        for image in images:
            # Get current image size if needed (though usually frames are same size)
            current_height, current_width = get_image_size(image, channel_dim=input_data_format)
            current_resized_h, current_resized_w = current_height, current_width

            if do_resize:
                 # Use smart_resize based on the first frame's dimensions or current frame?
                 # Original code seems to imply using the initial H/W. Let's stick to that.
                 # Re-calculate only once if needed
                if processed_images == []: # Only calculate resize dims once based on first image
                    resized_height, resized_width = smart_resize(
                        height,
                        width,
                        factor=patch_size * merge_size,
                        min_pixels=size["shortest_edge"], # Use effective min_pixels
                        max_pixels=size["longest_edge"], # Use effective max_pixels
                    )
                # Apply the calculated resize dimensions
                image = resize(
                    image, size=(resized_height, resized_width), resample=resample, input_data_format=input_data_format
                )
                current_resized_h, current_resized_w = resized_height, resized_width
            else:
                # If not resizing, use the current image's dimensions for grid calculation later
                 current_resized_h, current_resized_w = current_height, current_width


            if do_rescale:
                # Use the instance's rescale method if available, otherwise basic scaling
                if hasattr(self, "rescale"):
                    image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
                else: # Basic fallback
                    image = image.astype(np.float32) * rescale_factor

            if do_normalize:
                 # Use the instance's normalize method if available
                if hasattr(self, "normalize"):
                    image = self.normalize(
                        image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
                    )
                else: # Basic fallback
                     mean = np.array(image_mean).reshape((3, 1, 1) if input_data_format == ChannelDimension.FIRST else (1, 1, 3))
                     std = np.array(image_std).reshape((3, 1, 1) if input_data_format == ChannelDimension.FIRST else (1, 1, 3))
                     image = (image - mean) / std


            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            processed_images.append(image)

        # --- Patching logic using NumPy ---
        patches = np.array(processed_images) # Stack frames: [T, C, H, W] or [T, H, W, C]
        if data_format == ChannelDimension.LAST: # Ensure channel first for patching
            patches = patches.transpose(0, 3, 1, 2) # Now [T, C, H, W]

        # Temporal padding
        num_frames = patches.shape[0]
        padded_num_frames = num_frames
        if num_frames % temporal_patch_size != 0:
            num_padding = temporal_patch_size - (num_frames % temporal_patch_size)
            last_frame = patches[-1][np.newaxis]
            padding = np.repeat(last_frame, num_padding, axis=0)
            patches = np.concatenate([patches, padding], axis=0)
            padded_num_frames += num_padding

        # Get dimensions after potential resize and padding
        final_C, final_H, final_W = patches.shape[1], patches.shape[2], patches.shape[3]

        # Check dimension compatibility before patching
        if final_H % patch_size != 0 or final_W % patch_size != 0:
             raise ValueError(f"Final image H ({final_H}) or W ({final_W}) not divisible by patch_size ({patch_size})")
        grid_h_calc, grid_w_calc = final_H // patch_size, final_W // patch_size
        if grid_h_calc % merge_size != 0 or grid_w_calc % merge_size != 0:
             raise ValueError(f"Final grid H ({grid_h_calc}) or W ({grid_w_calc}) not divisible by merge_size ({merge_size})")


        grid_t = padded_num_frames // temporal_patch_size
        grid_h, grid_w = grid_h_calc, grid_w_calc # Use calculated grid dims
        ghm, gwm = grid_h // merge_size, grid_w // merge_size
        p, m, tp = patch_size, merge_size, temporal_patch_size
        channel = final_C

        # Reshape, transpose, flatten using NumPy
        try:
            patches = patches.reshape(grid_t, tp, channel, ghm, m, p, gwm, m, p)
            patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
            # Shape: [gt, ghm, gwm, m, m, C, tp, p, p]
            flatten_patches = patches.reshape(
                grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
            )
            # Shape: [N_patches, Patch_dim]
        except ValueError as e:
             logger.error(f"Error during numpy reshape/transpose. Check dimensions.")
             logger.error(f"Padded shape: {patches.shape}")
             logger.error(f"grid_t={grid_t}, grid_h={grid_h}, grid_w={grid_w}, ghm={ghm}, gwm={gwm}")
             logger.error(f"p={p}, m={m}, tp={tp}, C={channel}")
             raise e

        return flatten_patches, (grid_t, grid_h, grid_w)

    # Standard public entry point (calls _preprocess)
    def preprocess(
        self,
        images: ImageInput,
        videos: VideoInput = None,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        """
        Preprocesses images and/or videos using standard library logic (PIL, NumPy).

        Args:
            images (`ImageInput`, *optional*): Images to preprocess.
            videos (`VideoInput`, *optional*): Videos to preprocess.
            # ... other arguments as in __init__ ...
            return_tensors (`str` or `TensorType`, *optional*): Tensor type for output.
            data_format (`ChannelDimension` or `str`, *optional*): Output channel format.
            input_data_format (`ChannelDimension` or `str`, *optional*): Input channel format hint.

        Returns:
            BatchFeature: Dictionary containing processed pixel values and grid info.
        """
        # Determine effective parameters, falling back to instance defaults if args are None
        effective_size = size if size is not None else self.size
        # Handle min/max overrides for effective_size dictionary
        if min_pixels is not None: effective_size["shortest_edge"] = min_pixels
        if max_pixels is not None: effective_size["longest_edge"] = max_pixels

        effective_do_resize = do_resize if do_resize is not None else self.do_resize
        effective_resample = resample if resample is not None else self.resample
        effective_do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        effective_rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        effective_do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        effective_image_mean = image_mean if image_mean is not None else self.image_mean
        effective_image_std = image_std if image_std is not None else self.image_std
        effective_patch_size = patch_size if patch_size is not None else self.patch_size
        effective_temporal_patch_size = temporal_patch_size if temporal_patch_size is not None else self.temporal_patch_size
        effective_merge_size = merge_size if merge_size is not None else self.merge_size
        effective_do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        # Input validation and preparation
        if images is not None:
            images = make_flat_list_of_images(images) # List[Image]
        if videos is not None:
            videos = make_batched_videos(videos) # List[List[Image]] (list of videos, each video is list of frames)

        if images is not None and not valid_images(images):
             raise ValueError("Invalid image type provided to preprocess.")
        # Add similar check for videos if necessary

        validate_preprocess_arguments(
            rescale_factor=effective_rescale_factor,
            do_normalize=effective_do_normalize,
            image_mean=effective_image_mean,
            image_std=effective_image_std,
            do_resize=effective_do_resize,
            size=effective_size,
            resample=effective_resample,
        )

        data = {}
        # --- Processing images ---
        if images is not None:
            pixel_values_list, vision_grid_thws_list = [], []
            # Call _preprocess for each image (as a single-frame video)
            for image in images:
                patches, image_grid_thw = self._preprocess(
                    [image], # Pass as a list containing one frame
                    do_resize=effective_do_resize, size=effective_size, resample=effective_resample,
                    do_rescale=effective_do_rescale, rescale_factor=effective_rescale_factor,
                    do_normalize=effective_do_normalize, image_mean=effective_image_mean, image_std=effective_image_std,
                    patch_size=effective_patch_size, temporal_patch_size=effective_temporal_patch_size,
                    merge_size=effective_merge_size, data_format=data_format, # Pass data_format
                    do_convert_rgb=effective_do_convert_rgb, input_data_format=input_data_format,
                )
                pixel_values_list.append(patches)
                vision_grid_thws_list.append(image_grid_thw)

            # Concatenate patches from all images
            pixel_values = np.concatenate(pixel_values_list, axis=0)
            vision_grid_thws = np.array(vision_grid_thws_list) # Shape [num_images, 3]
            data["pixel_values"] = pixel_values
            data["image_grid_thw"] = vision_grid_thws

        # --- Processing videos ---
        if videos is not None:
            pixel_values_list, vision_grid_thws_list = [], []
            # Call _preprocess for each video (list of frames)
            for video_frames in videos:
                patches, video_grid_thw = self._preprocess(
                    video_frames, # Pass the list of frames for one video
                    do_resize=effective_do_resize, size=effective_size, resample=effective_resample,
                    do_rescale=effective_do_rescale, rescale_factor=effective_rescale_factor,
                    do_normalize=effective_do_normalize, image_mean=effective_image_mean, image_std=effective_image_std,
                    patch_size=effective_patch_size, temporal_patch_size=effective_temporal_patch_size,
                    merge_size=effective_merge_size, data_format=data_format, # Pass data_format
                    do_convert_rgb=effective_do_convert_rgb, input_data_format=input_data_format,
                )
                pixel_values_list.append(patches)
                vision_grid_thws_list.append(video_grid_thw)

            # Concatenate patches from all videos
            pixel_values_videos = np.concatenate(pixel_values_list, axis=0)
            video_grid_thws = np.array(vision_grid_thws_list) # Shape [num_videos, 3]
            data["pixel_values_videos"] = pixel_values_videos
            data["video_grid_thw"] = video_grid_thws

        return BatchFeature(data=data, tensor_type=return_tensors)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ MODIFIED METHOD FOR PRECOMPUTED TENSORS WITH GRADIENTS +++
    # +++ Mimics _preprocess temporal logic (T=1)              +++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def process_precomputed_tensors(
            self,
            image_tensors: torch.Tensor,  # Expect shape [B, C, H, W]
            return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Processes pre-computed image tensors (e.g., [B, C, H, W])
        directly into flattened patches using PyTorch, preserving gradients.

        **MODIFIED LOGIC:** This version mimics the temporal handling of the
        `_preprocess` method when processing single images (even in a batch),
        resulting in image_grid_thw with T=1. It simulates temporal padding
        internally for patch dimension calculation.

        **ASSUMPTION:** Input tensors are already correctly sized (H/W divisible
        by patch_size; resulting grid H/W divisible by merge_size), scaled,
        and normalized.

        Args:
            image_tensors (`torch.Tensor`): Batch of image tensors [B, C, H, W].
            return_tensors (`str` or `TensorType`, *optional*): Output tensor type.

        Returns:
            BatchFeature: Dict with "pixel_values", "image_grid_thw" (T=1).
        """
        if not isinstance(image_tensors, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(image_tensors)}")
        if image_tensors.dim() != 4:
            raise ValueError(f"Input tensor must be 4D [B, C, H, W], got {image_tensors.dim()}D")

        device = image_tensors.device

        # Get parameters from self
        B, C, H, W = image_tensors.shape
        p = self.patch_size
        tp = self.temporal_patch_size  # Default temporal patch size (e.g., 2)
        m = self.merge_size

        # --- Dimension Compatibility Checks ---
        if H % p != 0 or W % p != 0:
            raise ValueError(
                f"Input tensor H ({H}) and W ({W}) must be divisible by patch_size ({p})"
            )
        grid_h_calc = H // p
        grid_w_calc = W // p
        if grid_h_calc % m != 0 or grid_w_calc % m != 0:
            raise ValueError(
                f"Grid H ({grid_h_calc}) / W ({grid_w_calc}) must be divisible by merge_size ({m})"
            )

        # --- Calculate Grid Dimensions (Force T=1 logic) ---
        # Mimic padding 1 frame to tp frames for grid_t calculation
        num_frames_per_item = 1
        padded_num_frames_per_item = num_frames_per_item
        if num_frames_per_item % tp != 0:
            num_padding = tp - (num_frames_per_item % tp)
            padded_num_frames_per_item += num_padding  # Should become equal to tp

        target_grid_t = padded_num_frames_per_item // tp  # Should always be 1
        grid_h = grid_h_calc
        grid_w = grid_w_calc
        ghm = grid_h // m
        gwm = grid_w // m

        # --- Patching, Transposing, Flattening (Adapting for T=1 logic) ---
        # To mimic the result of padding 1 frame to tp frames and then patching,
        # we can conceptually repeat the input tensor temporally `tp` times just
        # before the final flattening step integrates the temporal dimension.
        # The standard patching logic already handles the spatial aspect.
        # Let's adapt the original patching sequence carefully.

        try:
            # 1. Initial spatial patch extraction and rearrangement common to both logics:
            # Reshape: [B, C, H, W] -> [B, C, ghm, m, p, gwm, m, p]
            # Note: Using H = ghm * m * p and W = gwm * m * p
            patches_spatial = image_tensors.view(B, C, ghm, m, p, gwm, m, p)

            # Permute to group spatial patches: [B, ghm, gwm, m, m, C, p, p]
            patches_spatial = patches_spatial.permute(0, 2, 5, 3, 6, 1, 4, 7)

            # Reshape to [B * gh * gw, C, p, p] where gh = ghm*m, gw = gwm*m
            # This gives N = B * gh * gw individual spatial patches
            spatial_patches_flat = patches_spatial.contiguous().view(B * grid_h * grid_w, C, p, p)

            # 2. Simulate temporal dimension within the flattened patch dimension:
            # The original _preprocess pads 1 frame to tp frames, then flattens C*tp*p*p.
            # We achieve the same by repeating the C*p*p part `tp` times.
            # Reshape spatial_patches_flat to [N, C*p*p]
            spatial_patches_final_dim = spatial_patches_flat.view(B * grid_h * grid_w, C * p * p)

            # Repeat this C*p*p block `tp` times along a new dimension and flatten
            # Shape becomes [N, tp, C*p*p] -> [N, tp * C*p*p]
            # Note: Correct final dimension is C * tp * p * p
            # Need to reshape C*p*p correctly first.
            # Original flatten was: C * tp * p * p
            # Our spatial_patches_flat is [N, C, p, p]

            # Reshape to [N, C*p*p]
            flatten_spatial = spatial_patches_flat.view(B * grid_h * grid_w, C * p * p)

            # Repeat along a new dimension: [N, 1, C*p*p] -> [N, tp, C*p*p]
            repeated_temporal = flatten_spatial.unsqueeze(1).repeat(1, tp, 1)

            # Flatten to the final target dimension: [N, tp * C*p*p]
            # which is equivalent to [N, C * tp * p * p] if C=3, p=14, tp=2 -> 3*2*14*14 = 1176
            # Let's verify the order: original permute brings C before tp [m, m, C, tp, p, p]
            # So the final view should be C * tp * p * p.

            # Reshape spatial_patches_flat [N, C, p, p] to [N, C, p*p]
            spatial_patches_flat_pp = spatial_patches_flat.view(B * grid_h * grid_w, C, p * p)
            # Repeat temporally: [N, C, 1, p*p] -> [N, C, tp, p*p]
            repeated_temporal_correct = spatial_patches_flat_pp.unsqueeze(2).repeat(1, 1, tp, 1)
            # Flatten C * tp * (p*p): [N, C * tp * p*p]
            flatten_patches = repeated_temporal_correct.contiguous().view(B * grid_h * grid_w, C * tp * p * p)

        except RuntimeError as e:
            logger.error(f"Error during MODIFIED torch view/permute/view. Check dimensions:", exc_info=True)
            logger.error(f"Input shape: {image_tensors.shape}")
            logger.error(f"Target Grid (t,h,w): ({target_grid_t},{grid_h},{grid_w}), Merged Grid (h,w): ({ghm},{gwm})")
            logger.error(f"Params (p,tp,m,C): ({p},{tp},{m},{C})")
            raise e

        # --- Prepare Output (using target_grid_t = 1) ---
        single_grid = torch.tensor([target_grid_t, grid_h, grid_w], dtype=torch.long, device=device)
        # Repeat for each item in the original batch B
        image_grid_thw_tensor = single_grid.unsqueeze(0).repeat(B, 1)  # Shape [B, 3]

        data = {"pixel_values": flatten_patches, "image_grid_thw": image_grid_thw_tensor}

        return BatchFeature(data=data, tensor_type=return_tensors)

# Required for module discovery if installed properly
__all__ = ["Qwen2VLImageProcessor"]