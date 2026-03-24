"""Module 3.1: Thermal Gas-Aware Attention (TGAA) Block.

Replaces standard MiT self-attention inside SegFormer-B2 encoder at all 4 stages.

Core mechanism:
  1. Standard MiT efficient attention (reduce K,V spatially)
  2. Gas Intensity Attention Weight: sigmoid(linear(thermal_intensity))
     -> high-emission patches get higher attention weight
  3. Spatial Dispersion Gate: sigmoid(linear(global_avg_pool(x)))
     -> suppresses near-zero background tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TGAABlock(nn.Module):
    """Thermal Gas-Aware Attention block.

    Drop-in replacement for the standard Efficient Self-Attention in MiT.

    Args:
        dim: input/output channel dimension
        num_heads: number of attention heads
        sr_ratio: spatial reduction ratio for K,V (matches MiT stage config)
        qkv_bias: whether to use bias in Q/K/V projections
        attn_drop: dropout on attention weights
        proj_drop: dropout on output projection
    """

    def __init__(self, dim: int, num_heads: int = 1, sr_ratio: int = 8,
                 qkv_bias: bool = True, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)

        # TGAA Addition 1: Gas Intensity Attention Weight
        self.intensity_proj = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, num_heads),
            nn.Sigmoid(),
        )

        # TGAA Addition 2: Spatial Dispersion Gate
        self.dispersion_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, H: int, W: int,
                thermal_intensity: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) patch tokens
            H, W: spatial dimensions of the token grid
            thermal_intensity: (B, N, 1) per-patch gas intensity values.
                If None, intensity weighting is skipped (fallback to standard attention).

        Returns:
            (B, N, C) output tokens
        """
        B, N, C = x.shape

        # Q from full resolution
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # K, V with spatial reduction (standard MiT efficiency)
        if self.sr_ratio > 1:
            x_sr = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_sr = self.sr(x_sr).reshape(B, C, -1).permute(0, 2, 1)
            x_sr = self.sr_norm(x_sr)
        else:
            x_sr = x

        kv = self.kv(x_sr).reshape(B, -1, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N_sr)

        # TGAA Addition 1: Gas Intensity Attention Weight
        if thermal_intensity is not None:
            # intensity_weight: (B, num_heads, N, 1) -- scales each query's scores
            intensity_weight = self.intensity_proj(thermal_intensity)  # (B, N, num_heads)
            intensity_weight = intensity_weight.permute(0, 2, 1).unsqueeze(-1)  # (B, heads, N, 1)
            attn = attn * intensity_weight

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # TGAA Addition 2: Spatial Dispersion Gate
        gap = x.mean(dim=1, keepdim=True)  # (B, 1, C)
        gate = self.dispersion_gate(gap)  # (B, 1, C)
        out = out * gate

        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class TGAATransformerBlock(nn.Module):
    """Full transformer block using TGAA attention + MLP (Mix-FFN).

    Matches the structure of a single MiT encoder block.
    """

    def __init__(self, dim: int, num_heads: int, sr_ratio: int,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TGAABlock(
            dim=dim, num_heads=num_heads, sr_ratio=sr_ratio,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)

        # Mix-FFN (depthwise conv inside FFN, same as MiT)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            DWConvFFN(mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, H: int, W: int,
                thermal_intensity: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W, thermal_intensity)
        x = x + self.mlp(self.norm2(x))
        return x


class DWConvFFN(nn.Module):
    """Depthwise convolution inside FFN (Mix-FFN from SegFormer)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C) but we need spatial dims
        # Store original shape and infer H, W from N
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        if H * W != N:
            # Non-square -- try to find factors
            for h in range(int(N ** 0.5), 0, -1):
                if N % h == 0:
                    H = h
                    W = N // h
                    break
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x
