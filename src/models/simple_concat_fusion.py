"""Simple Concat Fusion — ablation replacement for ATF (Ablation A2).

Replaces the cross-attention ATF module with naive concatenation + linear projection.
Uses the same three stream encoders as ATF for a fair comparison:
  Stream A: MaskEncoder (binary gas mask -> D-dim)
  Stream B: overlay_proj (Stage-4 features -> D-dim)
  Stream C: LightweightCNNEncoder (background overlay -> D-dim)

Fusion: cat(A, B, C) -> Linear(3*D, D) -> LayerNorm
No cross-attention, no learnable confidence scalars.
"""

import torch
import torch.nn as nn

from src.models.atf import MaskEncoder, LightweightCNNEncoder


class SimpleConcatFusion(nn.Module):
    """Naive concat fusion — ablation baseline for ATF (A2).

    Same stream encoders as ATF but replaces cross-attention with concat+linear.
    Exposes the same forward() interface as AsymmetricThermalFusion.
    """

    def __init__(self, feature_dim: int = 256, stage4_channels: int = 512):
        super().__init__()
        self.feature_dim = feature_dim

        self.mask_encoder = MaskEncoder(in_channels=1, out_dim=feature_dim)
        self.overlay_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(stage4_channels, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        self.bg_encoder = LightweightCNNEncoder(in_channels=3, out_dim=feature_dim)

        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, binary_mask, stage4_features, background_overlay):
        """Same interface as AsymmetricThermalFusion.forward()."""
        stream_A = self.mask_encoder(binary_mask)
        stream_B = self.overlay_proj(stage4_features)
        stream_C = self.bg_encoder(background_overlay)
        fused = self.fusion(torch.cat([stream_A, stream_B, stream_C], dim=1))
        return {"fused": fused, "confidence_scores": [1.0, 1.0, 1.0]}
