from typing import *
import torch
import torch.nn as nn
from ..attention import MultiHeadAttention
from ..norm import LayerNorm32
from .blocks import FeedForwardNet


class ModulatedTransformerBlock(nn.Module):
    
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, tag_mod_tuple: Optional[Tuple[torch.Tensor, ...]] = None) -> torch.Tensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)

                                              
        shift_msa_t = shift_msa.unsqueeze(1)
        scale_msa_t = scale_msa.unsqueeze(1)
        gate_msa_t = gate_msa.unsqueeze(1)
        shift_mlp_t = shift_mlp.unsqueeze(1)
        scale_mlp_t = scale_mlp.unsqueeze(1)
        gate_mlp_t = gate_mlp.unsqueeze(1)

                                                                      
        if tag_mod_tuple is not None:
            (seg_shift_msa, seg_scale_msa, seg_gate_msa, seg_shift_mlp, seg_scale_mlp, seg_gate_mlp) = tag_mod_tuple
            shift_msa_t = shift_msa_t + seg_shift_msa
            scale_msa_t = scale_msa_t + seg_scale_msa
            gate_msa_t = gate_msa_t + seg_gate_msa
            shift_mlp_t = shift_mlp_t + seg_shift_mlp
            scale_mlp_t = scale_mlp_t + seg_scale_mlp
            gate_mlp_t = gate_mlp_t + seg_gate_mlp

        h = self.norm1(x)
        h = h * (1 + scale_msa_t) + shift_msa_t
        h = self.attn(h)
        h = h * gate_msa_t
        x = x + h
        h = self.norm2(x)
        h = h * (1 + scale_mlp_t) + shift_mlp_t
        h = self.mlp(h)
        h = h * gate_mlp_t
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, tag_mod_tuple: Optional[Tuple[torch.Tensor, ...]] = None) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, tag_mod_tuple, use_reentrant=False)
        else:
            return self._forward(x, mod, tag_mod_tuple)


class ModulatedTransformerCrossBlock(nn.Module):
    
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(
        self,
        x: torch.Tensor,
        mod: torch.Tensor,
        context: torch.Tensor,
        tag_mod_tuple: Optional[Tuple[torch.Tensor, ...]] = None,
    ):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)

                                              
        shift_msa_t = shift_msa.unsqueeze(1)
        scale_msa_t = scale_msa.unsqueeze(1)
        gate_msa_t = gate_msa.unsqueeze(1)
        shift_mlp_t = shift_mlp.unsqueeze(1)
        scale_mlp_t = scale_mlp.unsqueeze(1)
        gate_mlp_t = gate_mlp.unsqueeze(1)

                                                                      
        if tag_mod_tuple is not None:
            (seg_shift_msa, seg_scale_msa, seg_gate_msa, seg_shift_mlp, seg_scale_mlp, seg_gate_mlp) = tag_mod_tuple
            shift_msa_t = shift_msa_t + seg_shift_msa
            scale_msa_t = scale_msa_t + seg_scale_msa
            gate_msa_t = gate_msa_t + seg_gate_msa
            shift_mlp_t = shift_mlp_t + seg_shift_mlp
            scale_mlp_t = scale_mlp_t + seg_scale_mlp
            gate_mlp_t = gate_mlp_t + seg_gate_mlp

        h = self.norm1(x)
        h = h * (1 + scale_msa_t) + shift_msa_t
        h = self.self_attn(h)
        h = h * gate_msa_t
        x = x + h
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h
        h = self.norm3(x)
        h = h * (1 + scale_mlp_t) + shift_mlp_t
        h = self.mlp(h)
        h = h * gate_mlp_t
        x = x + h
        return x

    def forward(
        self,
        x: torch.Tensor,
        mod: torch.Tensor,
        context: torch.Tensor,
        tag_mod_tuple: Optional[Tuple[torch.Tensor, ...]] = None,
    ):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, tag_mod_tuple, use_reentrant=False)
        else:
            return self._forward(x, mod, context, tag_mod_tuple)
