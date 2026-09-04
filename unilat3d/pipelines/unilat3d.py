from typing import *
from contextlib import contextmanager, nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import rembg
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp


class UniLatImageTo3DPipeline(Pipeline):
    
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        unilat_sampler: samplers.Sampler = None,
        image_cond_model: str = None,
        unilat_normalization: dict = None
    ):
        if models is None:
            return
        super().__init__(models)
        self.unilat_sampler = unilat_sampler
        self.unilat_sampler_params = {}

        self.unilat_normalization = unilat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "UniLatImageTo3DPipeline":
        
        pipeline = super(UniLatImageTo3DPipeline, UniLatImageTo3DPipeline).from_pretrained(path)
        new_pipeline = UniLatImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.unilat_sampler = getattr(samplers, args['unilat_sampler']['name'])(**args['unilat_sampler']['args'])
        new_pipeline.unilat_sampler_params = args['unilat_sampler']['params']


        new_pipeline.unilat_normalization = args['unilat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        
        
        if "dinov3" in name:
            self.dino_github = "external/dinov3"
            self.dino_weight = "external/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
            self.image_resolution = 592
            dino_model = torch.hub.load(self.dino_github, name, source = "local", weights = self.dino_weight, pretrained=True)
        elif "dinov2" in name:
            self.dino_github = "dinov2"
            self.image_resolution = 518
            dino_model = torch.hub.load(self.dino_github, name, source="local", pretrained=True)
        dino_model.eval()
        self.models['image_cond_model'] = dino_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        
                                                                             
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)                
        output = output.resize((self.image_resolution, self.image_resolution), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.image_resolution, self.image_resolution), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.image_cond_model_transform(image).to(self.device)
                                                                                                             
        img_model = self.models['image_cond_model']
        try:
            img_dtype = next(img_model.parameters()).dtype
        except StopIteration:
            img_dtype = image.dtype
        if self.device.type == "cuda" and image.dtype != img_dtype:
            image = image.to(dtype=img_dtype)
        features = img_model(image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        
                                 
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        
        flow_dtype = getattr(flow_model, "dtype", None)
        if not isinstance(flow_dtype, torch.dtype):
            try:
                flow_dtype = next(flow_model.parameters()).dtype
            except StopIteration:
                flow_dtype = torch.float32
        noise = torch.randn(
            num_samples, flow_model.in_channels, reso, reso, reso, device=self.device, dtype=flow_dtype
        )
        sampler_params = {**self.unilat_sampler_params, **sampler_params}
        z_s = self.unilat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        )
        z_sample = z_s.samples
        pred_x_t = z_s["pred_x_t"]
        


        return z_sample, pred_x_t

    def decode_unilat(
        self,
        unilat: torch.Tensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['unilat_decoder_mesh'].decode_mesh(self.models['unilat_decoder_gs'].compute_slats(unilat))
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['unilat_decoder_gs'](unilat)

        return ret


    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        unilat_sampler_params: dict = {},
        precision: str = "bf16",
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        sample: bool = False
    ) -> dict:
        
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        if device_type == "cuda" and precision == "bf16":
            amp_ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        elif device_type == "cuda" and precision == "fp16":
            amp_ctx = torch.autocast(device_type=device_type, dtype=torch.float16)
        else:
            amp_ctx = nullcontext()

        with amp_ctx:
            if preprocess_image:
                image = self.preprocess_image(image)
            cond = self.get_cond([image])
            torch.manual_seed(seed)
            unilat, pred_x_t = self.sample_sparse_structure(cond, num_samples, unilat_sampler_params)
            std = torch.tensor(self.unilat_normalization['std'], device=unilat.device, dtype=unilat.dtype)[None]
            mean = torch.tensor(self.unilat_normalization['mean'], device=unilat.device, dtype=unilat.dtype)[None]
            unilat = unilat * std.view(*std.shape,1,1,1).repeat(1,1,16,16,16) + mean.view(*mean.shape,1,1,1).repeat(1,1,16,16,16)
            if sample:
                return unilat
            return self.decode_unilat(unilat, formats)

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m")

            cond_indices = (np.arange(num_steps) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
        
        elif mode =='multidiffusion':
            from .samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f'_old_inference_model')

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        unilat_sampler_params: dict = {},
        precision: str = "bf16",
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ) -> dict:
        
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        if device_type == "cuda" and precision == "bf16":
            amp_ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        elif device_type == "cuda" and precision == "fp16":
            amp_ctx = torch.autocast(device_type=device_type, dtype=torch.float16)
        else:
            amp_ctx = nullcontext()

        with amp_ctx:
            if preprocess_image:
                images = [self.preprocess_image(image) for image in images]
            cond = self.get_cond(images)
            cond['neg_cond'] = cond['neg_cond'][:1]
            torch.manual_seed(seed)
            ss_steps = {**self.unilat_sampler_params, **unilat_sampler_params}.get('steps')
            with self.inject_sampler_multi_image('unilat_sampler', len(images), ss_steps, mode=mode):
                unilat, _ = self.sample_sparse_structure(cond, num_samples, unilat_sampler_params)
                std = torch.tensor(self.unilat_normalization['std'], device=unilat.device, dtype=unilat.dtype)[None]
                mean = torch.tensor(self.unilat_normalization['mean'], device=unilat.device, dtype=unilat.dtype)[None]
            unilat = unilat * std.view(*std.shape,1,1,1).repeat(1,1,16,16,16) + mean.view(*mean.shape,1,1,1).repeat(1,1,16,16,16)

            return self.decode_unilat(unilat, formats)