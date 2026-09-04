from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import convert_module_to_bf16, convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import AbsolutePositionEmbedder, ModulatedTransformerCrossBlock
from ..modules.spatial import patchify, unpatchify


class TimestepEmbedder(nn.Module):
    
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        
                                                                               
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(next(self.parameters()).dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class SparseStructureFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        pe_mode: Literal["ape", "rope"] = "ape",
        precision: str = "fp16",
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        use_segment_tags: bool = True,
        tag_timesteps: Tuple[float, float] = (0.0, 1000.0),
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        if precision == "fp16":
            self.dtype = torch.float16
        elif precision == "fp32":
            self.dtype = torch.float32
        elif precision == "bf16":
            self.dtype = torch.bfloat16
        else:
            raise ValueError(f"Invalid precision: {precision}")
        self.use_segment_tags = use_segment_tags
        self.tag_timesteps = tag_timesteps

        self.t_embedder = TimestepEmbedder(model_channels)
                                                                                                
        if self.use_segment_tags:
            assert len(tag_timesteps) == 2, "Currently only support two segments (target, src)"
            self.register_buffer("tag_time_tensor", torch.tensor(tag_timesteps, dtype=torch.float32))
            if share_mod:
                self.tag_linear = nn.Linear(model_channels, 6 * model_channels, bias=True)
            else:
                self.tag_linear = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(model_channels, 6 * model_channels, bias=True)
                )
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [resolution // patch_size] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)
            self.register_buffer("pos_emb", pos_emb)

        self.input_layer = nn.Linear(in_channels * patch_size**3, model_channels)
            
        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = nn.Linear(model_channels, out_channels * patch_size**3)

        self.initialize_weights()
        if precision == "fp16":
            self.convert_to_fp16()
        elif precision == "fp32":
            self.convert_to_fp32()
        elif precision == "bf16":
            self.convert_to_bf16()

    @property
    def device(self) -> torch.device:
        
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        
        self.blocks.apply(convert_module_to_f32)

    def convert_to_bf16(self) -> None:
        
        self.blocks.apply(convert_module_to_bf16)

    def initialize_weights(self) -> None:
                                        
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

                                            
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

                                                         
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

                               
        if self.use_segment_tags:
            if isinstance(self.tag_linear, nn.Linear):
                nn.init.constant_(self.tag_linear.weight, 0)
                nn.init.constant_(self.tag_linear.bias, 0)
            else:
                nn.init.constant_(self.tag_linear[-1].weight, 0)
                nn.init.constant_(self.tag_linear[-1].bias, 0)

                                 
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3],\
                f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"

        if y is not None:
            assert [*y.shape] == [y.shape[0], self.in_channels, *[self.resolution] * 3],\
                f"Input shape mismatch, got {y.shape}, expected {[y.shape[0], self.in_channels, *[self.resolution] * 3]}"

        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()

                             
        if y is not None:
            tgt_len = h.shape[1]
            h_y = patchify(y, self.patch_size)
            h_y = h_y.view(*h_y.shape[:2], -1).permute(0, 2, 1).contiguous()
            h = torch.cat([h, h_y], dim=1)
        else:
            tgt_len = None
        
        h = self.input_layer(h)
       
        if y is not None:
            if not hasattr(self, 'pos_emb'):
                raise RuntimeError('positional embedding not initialized')
            pos_emb = torch.cat([self.pos_emb, self.pos_emb], dim=0)
        else:
            pos_emb = self.pos_emb
        h = h + pos_emb[None]
        
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond = cond.type(self.dtype)

                                        
        tag_mod_tuple: Optional[Tuple[torch.Tensor, ...]] = None
        if self.use_segment_tags and y is not None and tgt_len is not None:
            tag_times = self.tag_time_tensor.to(t.device)       
            tag_embs = self.t_embedder(tag_times)          
            tag_params = self.tag_linear(tag_embs)           
            tag_params = tag_params.view(2, 6, self.model_channels)           

            B, T, C = h.shape
            device = h.device
            seg_shift_msa = torch.zeros(B, T, C, dtype=h.dtype, device=device)
            seg_scale_msa = torch.zeros_like(seg_shift_msa)
            seg_gate_msa = torch.zeros_like(seg_shift_msa)
            seg_shift_mlp = torch.zeros_like(seg_shift_msa)
            seg_scale_mlp = torch.zeros_like(seg_shift_msa)
            seg_gate_mlp = torch.zeros_like(seg_shift_msa)

            (t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp) = tag_params[0]
            (s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp) = tag_params[1]
            seg_shift_msa[:, :tgt_len] = t_shift_msa
            seg_scale_msa[:, :tgt_len] = t_scale_msa
            seg_gate_msa[:, :tgt_len] = t_gate_msa
            seg_shift_mlp[:, :tgt_len] = t_shift_mlp
            seg_scale_mlp[:, :tgt_len] = t_scale_mlp
            seg_gate_mlp[:, :tgt_len] = t_gate_mlp

            seg_shift_msa[:, tgt_len:] = s_shift_msa
            seg_scale_msa[:, tgt_len:] = s_scale_msa
            seg_gate_msa[:, tgt_len:] = s_gate_msa
            seg_shift_mlp[:, tgt_len:] = s_shift_mlp
            seg_scale_mlp[:, tgt_len:] = s_scale_mlp
            seg_gate_mlp[:, tgt_len:] = s_gate_mlp
            tag_mod_tuple = (seg_shift_msa, seg_scale_msa, seg_gate_msa, seg_shift_mlp, seg_scale_mlp, seg_gate_mlp)

        
        for block in self.blocks:
            h = block(h, t_emb, cond, tag_mod_tuple)
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)

        if y is not None:
            h = h[:, :h_y.shape[1]]

        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
        h = unpatchify(h, self.patch_size).contiguous()

        return h
