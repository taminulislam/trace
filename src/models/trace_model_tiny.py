"""TRACETiny: TGAA encoder with MiT-B0 configuration.

Same TGAA blocks as TRACE but with smaller dimensions:
  Stage 1: C=32,  depth=2, heads=1, sr=8
  Stage 2: C=64,  depth=2, heads=2, sr=4
  Stage 3: C=160, depth=2, heads=5, sr=2
  Stage 4: C=256, depth=2, heads=8, sr=1

~8-9M params vs 27.7M for full TRACE.
stage4_channels=256 (used by ATF and fusion — set STAGE4_CHANNELS=256 in env).
"""

import torch
import torch.nn as nn

from src.models.trace_model import TGAAEncoderStage, SegFormerDecodeHead


class TRACETiny(nn.Module):
    """TRACETiny: TGAA-SegFormer with MiT-B0 dimensions."""

    STAGE_CONFIGS = [
        {"embed_dim": 32,  "depth": 2, "num_heads": 1, "sr_ratio": 8, "patch_size": 7, "stride": 4},
        {"embed_dim": 64,  "depth": 2, "num_heads": 2, "sr_ratio": 4, "patch_size": 3, "stride": 2},
        {"embed_dim": 160, "depth": 2, "num_heads": 5, "sr_ratio": 2, "patch_size": 3, "stride": 2},
        {"embed_dim": 256, "depth": 2, "num_heads": 8, "sr_ratio": 1, "patch_size": 3, "stride": 2},
    ]

    def __init__(self, in_channels: int = 3, num_seg_classes: int = 1,
                 decode_dim: int = 256, use_aux_mask: bool = True):
        super().__init__()
        self.use_aux_mask = use_aux_mask

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

        aux_ch = 0
        if use_aux_mask:
            aux_ch = 16
            self.mask_encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.Conv2d(16, aux_ch, 3, padding=1), nn.BatchNorm2d(aux_ch), nn.ReLU(inplace=True),
            )

        self.decode_head = SegFormerDecodeHead(
            in_channels_list=embed_dims,
            embed_dim=decode_dim,
            num_classes=num_seg_classes,
            aux_channels=aux_ch,
        )

    def forward(self, overlay, thermal_intensity=None, binary_mask=None):
        B, _, H, W = overlay.shape
        features = []
        x = overlay
        for stage in self.stages:
            x, h, w = stage(x, thermal_intensity)
            features.append(x)

        aux = None
        if self.use_aux_mask and binary_mask is not None:
            aux = self.mask_encoder(binary_mask)

        seg_logits = self.decode_head(features, (H, W), aux)
        return {
            "seg_logits": seg_logits,
            "stage4_features": features[-1],   # (B, 256, H/32, W/32)
            "all_features": features,
        }
