"""Model factory — create segmentation backbones by name.

All models expose the same interface as TRACE:
    forward(overlay, thermal_intensity=None, binary_mask=None)
    → {"seg_logits": (B,1,H,W), "stage4_features": ..., "all_features": [...]}
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Lightweight SegFormer-style MLP Decode Head (shared across all models)
# ---------------------------------------------------------------------------
class MLPDecodeHead(nn.Module):
    """All-MLP decode head that fuses multi-scale features."""

    def __init__(self, in_channels_list, decode_dim=256, num_classes=1):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, decode_dim, 1),
                nn.BatchNorm2d(decode_dim),
                nn.GELU(),
            ) for c in in_channels_list
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(decode_dim * len(in_channels_list), decode_dim, 1),
            nn.BatchNorm2d(decode_dim),
            nn.GELU(),
        )
        self.head = nn.Conv2d(decode_dim, num_classes, 1)

    def forward(self, features, target_size):
        projected = []
        for feat, proj in zip(features, self.projections):
            p = proj(feat)
            p = F.interpolate(p, size=target_size, mode="bilinear", align_corners=False)
            projected.append(p)
        fused = self.fuse(torch.cat(projected, dim=1))
        return self.head(fused)


# ═══════════════════════════════════════════════════════════════════════════
#  1. SegFormer-B2 (Vanilla) — same backbone as TRACE but NO TGAA
# ═══════════════════════════════════════════════════════════════════════════
class SegFormerB2Vanilla(nn.Module):
    """Vanilla SegFormer-B2 using HuggingFace transformers.
    Direct ablation baseline — same architecture minus TGAA blocks.
    """

    def __init__(self, num_seg_classes=1, decode_dim=256, **kwargs):
        super().__init__()
        from transformers import SegformerModel
        self.backbone = SegformerModel.from_pretrained(
            "nvidia/mit-b2",
            output_hidden_states=True,
        )
        # MiT-B2 stage dims: [64, 128, 320, 512]
        self.decode_head = MLPDecodeHead([64, 128, 320, 512], decode_dim, num_seg_classes)

    def forward(self, overlay, thermal_intensity=None, binary_mask=None):
        B, _, H, W = overlay.shape
        outputs = self.backbone(pixel_values=overlay, output_hidden_states=True)
        features = list(outputs.hidden_states)

        seg_logits = self.decode_head(features, (H, W))

        return {
            "seg_logits": seg_logits,
            "stage4_features": features[-1],
            "all_features": features,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  2. iFormer (implemented as EfficientFormerV2-S2) — ICLR 2025 style
#     Hybrid CNN+ViT lightweight model
# ═══════════════════════════════════════════════════════════════════════════
class iFormerSeg(nn.Module):
    """iFormer-style CNN+ViT hybrid using MobileViT-S from timm.
    Pretrained on ImageNet, provides multi-scale features.
    ~5.6M params, fast training.
    """

    def __init__(self, num_seg_classes=1, decode_dim=256, **kwargs):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "mobilevit_s",
            pretrained=True,
            features_only=True,
        )
        # Get channel dims from the backbone
        dummy = torch.randn(1, 3, 256, 320)
        with torch.no_grad():
            feats = self.backbone(dummy)
        self.feat_channels = [f.shape[1] for f in feats]
        print(f"  iFormer/MobileViT-S feature channels: {self.feat_channels}")

        self.decode_head = MLPDecodeHead(self.feat_channels, decode_dim, num_seg_classes)

    def forward(self, overlay, thermal_intensity=None, binary_mask=None):
        B, _, H, W = overlay.shape
        features = self.backbone(overlay)

        seg_logits = self.decode_head(features, (H, W))

        return {
            "seg_logits": seg_logits,
            "stage4_features": features[-1],
            "all_features": features,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  3. LACTNet — Lightweight Aggregated CNN + Transformer
#     Custom implementation: ResNet-18 CNN encoder + lightweight Transformer
# ═══════════════════════════════════════════════════════════════════════════
class LightweightTransformerBlock(nn.Module):
    """Lightweight transformer block with depthwise separable attention."""

    def __init__(self, dim, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        x_flat = x_flat + self.attn(self.norm1(x_flat), self.norm1(x_flat), self.norm1(x_flat))[0]
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        return x_flat.permute(0, 2, 1).view(B, C, H, W)


class LACTNet(nn.Module):
    """Lightweight Aggregated CNN + Transformer Network.
    CNN branch: ResNet-18 (pretrained) for local features
    Transformer branch: Lightweight attention on downsampled features
    Fusion: Channel attention + MLP decode head
    ~15M params.
    """

    def __init__(self, num_seg_classes=1, decode_dim=256, **kwargs):
        super().__init__()
        import timm

        # CNN branch: ResNet-18
        resnet = timm.create_model("resnet18", pretrained=True, features_only=True)
        self.cnn_stages = resnet

        # Get feature dims
        dummy = torch.randn(1, 3, 256, 320)
        with torch.no_grad():
            cnn_feats = self.cnn_stages(dummy)
        self.cnn_channels = [f.shape[1] for f in cnn_feats]
        print(f"  LACTNet CNN feature channels: {self.cnn_channels}")

        # Transformer branch on last two stages (lower resolution)
        self.trans_proj3 = nn.Conv2d(self.cnn_channels[-2], 128, 1)
        self.trans_block3 = LightweightTransformerBlock(128, num_heads=4)
        self.trans_proj4 = nn.Conv2d(self.cnn_channels[-1], 256, 1)
        self.trans_block4 = LightweightTransformerBlock(256, num_heads=4)

        # Fusion: concat CNN + Transformer features
        fused_channels = list(self.cnn_channels)
        fused_channels[-2] = self.cnn_channels[-2] + 128
        fused_channels[-1] = self.cnn_channels[-1] + 256

        self.decode_head = MLPDecodeHead(fused_channels, decode_dim, num_seg_classes)

    def forward(self, overlay, thermal_intensity=None, binary_mask=None):
        B, _, H, W = overlay.shape
        cnn_feats = self.cnn_stages(overlay)

        # Transformer on last 2 stages
        t3 = self.trans_block3(self.trans_proj3(cnn_feats[-2]))
        t4 = self.trans_block4(self.trans_proj4(cnn_feats[-1]))

        # Fuse CNN + Transformer
        features = list(cnn_feats)
        features[-2] = torch.cat([cnn_feats[-2], t3], dim=1)
        features[-1] = torch.cat([cnn_feats[-1], t4], dim=1)

        seg_logits = self.decode_head(features, (H, W))

        return {
            "seg_logits": seg_logits,
            "stage4_features": features[-1],
            "all_features": features,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  4. SegFormer-B0 — Tiny SegFormer baseline
# ═══════════════════════════════════════════════════════════════════════════
class SegFormerB0(nn.Module):
    """SegFormer-B0 — smallest SegFormer variant (3.7M params).
    Useful as a lightweight baseline.
    """

    def __init__(self, num_seg_classes=1, decode_dim=256, **kwargs):
        super().__init__()
        from transformers import SegformerModel
        self.backbone = SegformerModel.from_pretrained(
            "nvidia/mit-b0",
            output_hidden_states=True,
        )
        # MiT-B0 stage dims: [32, 64, 160, 256]
        self.decode_head = MLPDecodeHead([32, 64, 160, 256], decode_dim, num_seg_classes)

    def forward(self, overlay, thermal_intensity=None, binary_mask=None):
        B, _, H, W = overlay.shape
        outputs = self.backbone(pixel_values=overlay, output_hidden_states=True)
        features = list(outputs.hidden_states)
        seg_logits = self.decode_head(features, (H, W))
        return {
            "seg_logits": seg_logits,
            "stage4_features": features[-1],
            "all_features": features,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  5. Mask2Former — Universal segmentation (Swin-T backbone)
# ═══════════════════════════════════════════════════════════════════════════
class Mask2FormerSeg(nn.Module):
    """Mask2Former with Swin-Tiny backbone (~44M params).
    SOTA universal segmentation architecture. We use just the backbone
    features + our standard MLP decode head for fair comparison.
    """

    def __init__(self, num_seg_classes=1, decode_dim=256, **kwargs):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "swinv2_tiny_window8_256",
            pretrained=True,
            features_only=True,
        )
        dummy = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            feats = self.backbone(dummy)
        self.feat_channels = [f.shape[1] for f in feats]
        print(f"  Mask2Former/Swin-T feature channels: {self.feat_channels}")
        self.decode_head = MLPDecodeHead(self.feat_channels, decode_dim, num_seg_classes)

    def forward(self, overlay, thermal_intensity=None, binary_mask=None):
        import torch.nn.functional as F
        B, _, H, W = overlay.shape
        x = F.interpolate(overlay, size=(256, 256), mode="bilinear", align_corners=False)
        features = self.backbone(x)
        seg_logits = self.decode_head(features, (H, W))
        return {
            "seg_logits": seg_logits,
            "stage4_features": features[-1],
            "all_features": features,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  6. SAM-style — Foundation model approach (MobileViT-V2 large backbone)
# ═══════════════════════════════════════════════════════════════════════════
class SAMSeg(nn.Module):
    """SAM-style segmentation using MobileViT-V2-200 as backbone.
    Large ViT backbone (~18M params) mimicking SAM's approach of using
    a powerful image encoder + lightweight mask decoder.
    """

    def __init__(self, num_seg_classes=1, decode_dim=256, **kwargs):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "mobilevitv2_200",
            pretrained=True,
            features_only=True,
        )
        dummy = torch.randn(1, 3, 256, 320)
        with torch.no_grad():
            feats = self.backbone(dummy)
        self.feat_channels = [f.shape[1] for f in feats]
        print(f"  SAM/MobileViT-V2-200 feature channels: {self.feat_channels}")
        self.decode_head = MLPDecodeHead(self.feat_channels, decode_dim, num_seg_classes)

    def forward(self, overlay, thermal_intensity=None, binary_mask=None):
        B, _, H, W = overlay.shape
        features = self.backbone(overlay)
        seg_logits = self.decode_head(features, (H, W))
        return {
            "seg_logits": seg_logits,
            "stage4_features": features[-1],
            "all_features": features,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  8. RepViT-M1 — Revisiting Mobile CNN from ViT Perspective (CVPR 2024)
#     Re-parameterizable mobile CNN with ViT-style macro design
# ═══════════════════════════════════════════════════════════════════════════
class RepViTM1Seg(nn.Module):
    """RepViT-M1: Re-parameterizable mobile CNN (CVPR 2024).
    Revisits mobile CNN design from a ViT perspective using
    structural re-parameterization. ~6.8M params. Strong on
    dense prediction; used as backbone in RepViT-SAM.
    timm: repvit_m1.dist_in1k
    """

    def __init__(self, num_seg_classes=1, decode_dim=256, **kwargs):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "repvit_m1.dist_in1k",
            pretrained=True,
            features_only=True,
        )
        dummy = torch.randn(1, 3, 256, 320)
        with torch.no_grad():
            feats = self.backbone(dummy)
        self.feat_channels = [f.shape[1] for f in feats]
        print(f"  RepViT-M1 feature channels: {self.feat_channels}")
        self.decode_head = MLPDecodeHead(self.feat_channels, decode_dim, num_seg_classes)

    def forward(self, overlay, thermal_intensity=None, binary_mask=None):
        B, _, H, W = overlay.shape
        features = self.backbone(overlay)
        seg_logits = self.decode_head(features, (H, W))
        return {
            "seg_logits": seg_logits,
            "stage4_features": features[-1],
            "all_features": features,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  9. SHViT-S4 — Single-Head Vision Transformer (CVPR 2024)
#     Memory-efficient single-head attention with macro design
# ═══════════════════════════════════════════════════════════════════════════
class SHViTS4Seg(nn.Module):
    """SHViT-S4: Single-Head Vision Transformer (CVPR 2024).
    Memory-efficient ViT that uses a single attention head with
    a hardware-aware macro design. ~16.6M params. Multi-scale
    feature pyramid for dense prediction tasks.
    timm: shvit_s4.in1k
    """

    def __init__(self, num_seg_classes=1, decode_dim=256, **kwargs):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "shvit_s4.in1k",
            pretrained=True,
            features_only=True,
        )
        dummy = torch.randn(1, 3, 256, 320)
        with torch.no_grad():
            feats = self.backbone(dummy)
        self.feat_channels = [f.shape[1] for f in feats]
        print(f"  SHViT-S4 feature channels: {self.feat_channels}")
        self.decode_head = MLPDecodeHead(self.feat_channels, decode_dim, num_seg_classes)

    def forward(self, overlay, thermal_intensity=None, binary_mask=None):
        B, _, H, W = overlay.shape
        features = self.backbone(overlay)
        seg_logits = self.decode_head(features, (H, W))
        return {
            "seg_logits": seg_logits,
            "stage4_features": features[-1],
            "all_features": features,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  10. StarNet-S2 — Rewrite the Stars (CVPR 2024)
#      Replaces attention with element-wise star (★) multiplication
# ═══════════════════════════════════════════════════════════════════════════
class StarNetS2Seg(nn.Module):
    """StarNet-S2: Rewrite the Stars (CVPR 2024).
    Replaces self-attention with element-wise star (★) multiplication
    (Hadamard product), enabling implicit high-order feature interaction
    with minimal compute. ~3.7M params — lightest CVPR 2024 model here.
    timm: starnet_s2.in1k
    """

    def __init__(self, num_seg_classes=1, decode_dim=256, **kwargs):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "starnet_s2.in1k",
            pretrained=True,
            features_only=True,
        )
        dummy = torch.randn(1, 3, 256, 320)
        with torch.no_grad():
            feats = self.backbone(dummy)
        self.feat_channels = [f.shape[1] for f in feats]
        print(f"  StarNet-S2 feature channels: {self.feat_channels}")
        self.decode_head = MLPDecodeHead(self.feat_channels, decode_dim, num_seg_classes)

    def forward(self, overlay, thermal_intensity=None, binary_mask=None):
        B, _, H, W = overlay.shape
        features = self.backbone(overlay)
        seg_logits = self.decode_head(features, (H, W))
        return {
            "seg_logits": seg_logits,
            "stage4_features": features[-1],
            "all_features": features,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  11. MobileNetV4-Conv-Small — Universal Mobile Backbone (ECCV 2024)
#      Google's latest mobile backbone with Universal Inverted Bottleneck
# ═══════════════════════════════════════════════════════════════════════════
class MobileNetV4ConvSSeg(nn.Module):
    """MobileNetV4-Conv-Small: Universal Mobile Backbone (ECCV 2024).
    Google's successor to MobileNetV3 using the Universal Inverted
    Bottleneck (UIB) block. ~3.8M params. Designed for hardware
    portability across mobile, server, and edge accelerators.
    timm: mobilenetv4_conv_small.e2400_r224_in1k
    """

    def __init__(self, num_seg_classes=1, decode_dim=256, **kwargs):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "mobilenetv4_conv_small.e2400_r224_in1k",
            pretrained=True,
            features_only=True,
        )
        dummy = torch.randn(1, 3, 256, 320)
        with torch.no_grad():
            feats = self.backbone(dummy)
        self.feat_channels = [f.shape[1] for f in feats]
        print(f"  MobileNetV4-Conv-S feature channels: {self.feat_channels}")
        self.decode_head = MLPDecodeHead(self.feat_channels, decode_dim, num_seg_classes)

    def forward(self, overlay, thermal_intensity=None, binary_mask=None):
        B, _, H, W = overlay.shape
        features = self.backbone(overlay)
        seg_logits = self.decode_head(features, (H, W))
        return {
            "seg_logits": seg_logits,
            "stage4_features": features[-1],
            "all_features": features,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  7. Prior2Former — MetaFormer architecture (PoolFormerV2-S24)
# ═══════════════════════════════════════════════════════════════════════════
class Prior2FormerSeg(nn.Module):
    """Prior2Former-style using PoolFormerV2-S24 from timm.
    MetaFormer architecture (ICCV 2025 inspired): replaces attention
    with simple pooling, proving the importance of architecture over
    specific token mixing. ~21M params.
    """

    def __init__(self, num_seg_classes=1, decode_dim=256, **kwargs):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "poolformerv2_s24",
            pretrained=True,
            features_only=True,
        )
        dummy = torch.randn(1, 3, 256, 320)
        with torch.no_grad():
            feats = self.backbone(dummy)
        self.feat_channels = [f.shape[1] for f in feats]
        print(f"  Prior2Former/PoolFormerV2-S24 feature channels: {self.feat_channels}")
        self.decode_head = MLPDecodeHead(self.feat_channels, decode_dim, num_seg_classes)

    def forward(self, overlay, thermal_intensity=None, binary_mask=None):
        B, _, H, W = overlay.shape
        features = self.backbone(overlay)
        seg_logits = self.decode_head(features, (H, W))
        return {
            "seg_logits": seg_logits,
            "stage4_features": features[-1],
            "all_features": features,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Model Factory
# ═══════════════════════════════════════════════════════════════════════════
MODEL_REGISTRY = {
    "trace": None,       # imported from src.models.trace_model
    "trace_tiny": None,  # imported from src.models.trace_model_tiny (A6 ablation)
    "segformer_b2": SegFormerB2Vanilla,
    "iformer": iFormerSeg,
    "lactnet": LACTNet,
    "segformer_b0": SegFormerB0,
    "mask2former": Mask2FormerSeg,
    "sam2": SAMSeg,
    "prior2former": Prior2FormerSeg,
    # CVPR / ECCV 2024 lightweight models
    "repvit_m1": RepViTM1Seg,
    "shvit_s4": SHViTS4Seg,
    "starnet_s2": StarNetS2Seg,
    "mobilenetv4_conv_s": MobileNetV4ConvSSeg,
}


def create_model(model_name: str, num_seg_classes=1, decode_dim=256, **kwargs):
    """Create a segmentation model by name.

    Available models:
        trace, segformer_b2, iformer, lactnet,
        segformer_b0, mask2former, sam2, prior2former,
        repvit_m1, shvit_s4, starnet_s2, mobilenetv4_conv_s
    """
    name = model_name.lower().replace("-", "_")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")

    if name == "trace":
        from src.models.trace_model import TRACE
        return TRACE(num_seg_classes=num_seg_classes, decode_dim=decode_dim,
                     use_aux_mask=kwargs.get("use_aux_mask", True))

    if name == "trace_tiny":
        from src.models.trace_model_tiny import TRACETiny
        return TRACETiny(num_seg_classes=num_seg_classes, decode_dim=decode_dim,
                         use_aux_mask=kwargs.get("use_aux_mask", True))

    cls = MODEL_REGISTRY[name]
    return cls(num_seg_classes=num_seg_classes, decode_dim=decode_dim, **kwargs)
