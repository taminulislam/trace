"""Module 3.2: TRACE Segmentation Branch.

Modified SegFormer-B2 with TGAA blocks replacing standard MiT self-attention
at all 4 encoder stages.

Base: SegFormer-B2 pretrained ImageNet weights (from transformers/timm)
Input: 3-channel thermal overlay (resize to 224x224 or 320x256)
Auxiliary input: binary mask concatenated to Stage-4 features before decode head
Output: (1) per-pixel gas segmentation logits, (2) Stage-4 feature map (for ATF)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.tgaa import TGAATransformerBlock


# ---------- Patch Embedding (same as MiT) ----------

class OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding used in MiT encoder stages."""

    def __init__(self, patch_size: int = 7, stride: int = 4,
                 in_channels: int = 3, embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.norm(x)
        return x, H, W


# ---------- TGAA Encoder Stage ----------

class TGAAEncoderStage(nn.Module):
    """One stage of the TGAA-modified MiT encoder.

    Each stage: PatchEmbed -> N x TGAATransformerBlock -> LayerNorm
    """

    def __init__(self, in_channels: int, embed_dim: int, num_heads: int,
                 depth: int, sr_ratio: int, patch_size: int = 3,
                 stride: int = 2, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(
            patch_size=patch_size, stride=stride,
            in_channels=in_channels, embed_dim=embed_dim,
        )
        self.blocks = nn.ModuleList([
            TGAATransformerBlock(
                dim=embed_dim, num_heads=num_heads, sr_ratio=sr_ratio,
                mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor,
                thermal_intensity: torch.Tensor = None) -> tuple:
        """
        Args:
            x: (B, C_in, H_in, W_in)
            thermal_intensity: (B, 1, H_in, W_in) per-pixel intensity map

        Returns:
            features: (B, C, H, W)
            H, W: spatial dims
        """
        x, H, W = self.patch_embed(x)

        # Compute per-patch thermal intensity if provided
        patch_intensity = None
        if thermal_intensity is not None:
            # Average pool intensity to match patch grid
            patch_intensity = F.adaptive_avg_pool2d(thermal_intensity, (H, W))
            patch_intensity = patch_intensity.flatten(2).transpose(1, 2)  # (B, N, 1)

        for block in self.blocks:
            x = block(x, H, W, patch_intensity)

        x = self.norm(x)
        x = x.reshape(-1, H, W, x.shape[-1]).permute(0, 3, 1, 2)  # (B, C, H, W)
        return x, H, W


# ---------- Lightweight MLP Decode Head (SegFormer style) ----------

class SegFormerDecodeHead(nn.Module):
    """All-MLP decode head from SegFormer.

    Takes multi-scale features from 4 encoder stages, projects each to
    a common dimension, upsamples to the same resolution, concatenates,
    and applies a final linear classifier.
    """

    def __init__(self, in_channels_list: list, embed_dim: int = 256,
                 num_classes: int = 1, aux_channels: int = 0):
        super().__init__()
        self.linear_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, 1),
                nn.BatchNorm2d(embed_dim),
            )
            for c in in_channels_list
        ])
        total_in = embed_dim * len(in_channels_list) + aux_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(total_in, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, features: list, target_size: tuple,
                aux_features: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            features: list of (B, C_i, H_i, W_i) from 4 stages
            target_size: (H, W) for output resolution
            aux_features: optional (B, aux_C, H, W) concatenated before fusion

        Returns:
            (B, num_classes, H, W) logits
        """
        projected = []
        for feat, proj in zip(features, self.linear_projs):
            p = proj(feat)
            p = F.interpolate(p, size=target_size, mode="bilinear", align_corners=False)
            projected.append(p)

        x = torch.cat(projected, dim=1)

        if aux_features is not None:
            aux = F.interpolate(aux_features, size=target_size,
                                mode="bilinear", align_corners=False)
            x = torch.cat([x, aux], dim=1)

        x = self.fuse(x)
        return self.classifier(x)


# ---------- TRACE ----------

class TRACE(nn.Module):
    """TRACE: Modified SegFormer-B2 with TGAA attention.

    SegFormer-B2 configuration:
      Stage 1: C=64,  depth=3, heads=1, sr=8
      Stage 2: C=128, depth=4, heads=2, sr=4
      Stage 3: C=320, depth=6, heads=5, sr=2
      Stage 4: C=512, depth=3, heads=8, sr=1

    Outputs:
      - seg_logits: (B, 1, H, W) per-pixel gas segmentation
      - stage4_features: (B, 512, H/32, W/32) for ATF fusion
    """

    # SegFormer-B2 config
    STAGE_CONFIGS = [
        {"embed_dim": 64,  "depth": 3, "num_heads": 1, "sr_ratio": 8, "patch_size": 7, "stride": 4},
        {"embed_dim": 128, "depth": 4, "num_heads": 2, "sr_ratio": 4, "patch_size": 3, "stride": 2},
        {"embed_dim": 320, "depth": 6, "num_heads": 5, "sr_ratio": 2, "patch_size": 3, "stride": 2},
        {"embed_dim": 512, "depth": 3, "num_heads": 8, "sr_ratio": 1, "patch_size": 3, "stride": 2},
    ]

    def __init__(self, in_channels: int = 3, num_seg_classes: int = 1,
                 decode_dim: int = 256, use_aux_mask: bool = True):
        super().__init__()
        self.use_aux_mask = use_aux_mask

        # Build 4 encoder stages
        self.stages = nn.ModuleList()
        prev_channels = in_channels
        embed_dims = []
        for cfg in self.STAGE_CONFIGS:
            stage = TGAAEncoderStage(
                in_channels=prev_channels,
                embed_dim=cfg["embed_dim"],
                num_heads=cfg["num_heads"],
                depth=cfg["depth"],
                sr_ratio=cfg["sr_ratio"],
                patch_size=cfg["patch_size"],
                stride=cfg["stride"],
            )
            self.stages.append(stage)
            prev_channels = cfg["embed_dim"]
            embed_dims.append(cfg["embed_dim"])

        # Auxiliary mask encoder (lightweight: just a conv to 1 channel)
        aux_ch = 0
        if use_aux_mask:
            aux_ch = 16
            self.mask_encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, aux_ch, 3, padding=1),
                nn.BatchNorm2d(aux_ch),
                nn.ReLU(inplace=True),
            )

        # Decode head
        self.decode_head = SegFormerDecodeHead(
            in_channels_list=embed_dims,
            embed_dim=decode_dim,
            num_classes=num_seg_classes,
            aux_channels=aux_ch,
        )

    def forward(self, overlay: torch.Tensor,
                thermal_intensity: torch.Tensor = None,
                binary_mask: torch.Tensor = None) -> dict:
        """
        Args:
            overlay: (B, 3, H, W) thermal overlay image
            thermal_intensity: (B, 1, H, W) per-pixel gas intensity
            binary_mask: (B, 1, H, W) binary gas mask (for aux input)

        Returns:
            dict with:
                "seg_logits": (B, 1, H, W) segmentation logits
                "stage4_features": (B, 512, H/32, W/32) features for ATF
                "all_features": list of 4 stage features
        """
        B, _, H, W = overlay.shape
        features = []
        x = overlay

        for stage in self.stages:
            x, h, w = stage(x, thermal_intensity)
            features.append(x)
            # thermal_intensity is downsampled inside each stage automatically

        # Auxiliary mask features
        aux = None
        if self.use_aux_mask and binary_mask is not None:
            aux = self.mask_encoder(binary_mask)

        # Segmentation output
        seg_logits = self.decode_head(features, (H, W), aux)

        return {
            "seg_logits": seg_logits,
            "stage4_features": features[-1],
            "all_features": features,
        }

    def load_segformer_pretrained(self, state_dict: dict):
        """Load pretrained SegFormer-B2 weights, skipping TGAA-specific layers.

        The TGAA additions (intensity_proj, dispersion_gate) are randomly
        initialized; everything else is loaded from the pretrained model.
        """
        own_state = self.state_dict()
        loaded = 0
        skipped = 0
        for name, param in state_dict.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1
            else:
                skipped += 1
        print(f"Loaded {loaded} params, skipped {skipped} (TGAA-specific or mismatched)")
