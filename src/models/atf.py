"""Module 3.3: Asymmetric Thermal Fusion (ATF) Module.

Three input streams, all projected to D=256:
  Stream A (Mask features): binary mask -> lightweight SegFormer-B0 encoder -> features
  Stream B (Overlay features): Stage-4 features from TRACE segmentation branch
  Stream C (Background features): overlay with gas region zeroed out -> simple CNN encoder

ATF Mechanism:
  Stream A = Primary Query (dominant semantic signal)
  Stream B = Key/Value with learnable confidence (conf_B)
  Stream C = Residual correction with learnable confidence (conf_C)
  Cross-attention: single head over D=256 vectors
  Output: fused feature vector + confidence scores for ablation analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightCNNEncoder(nn.Module):
    """Simple CNN encoder for background features (Stream C).

    4 conv blocks with stride-2 downsampling -> global average pool -> project to D.
    """

    def __init__(self, in_channels: int = 3, out_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        return self.proj(x)


class MaskEncoder(nn.Module):
    """Lightweight encoder for binary mask features (Stream A).

    Uses a small CNN (similar to SegFormer-B0 complexity) to extract
    spatial features from the binary gas mask, then global-pools to a vector.
    """

    def __init__(self, in_channels: int = 1, out_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, stride=4, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        return self.proj(x)


class AsymmetricThermalFusion(nn.Module):
    """Asymmetric Thermal Fusion (ATF) module.

    Fuses three data streams with asymmetric roles:
      Stream A (mask): primary query
      Stream B (overlay/segmentation features): key/value with learnable confidence
      Stream C (background): residual correction with learnable confidence

    Args:
        feature_dim: dimension D for all projected features (default 256)
        stage4_channels: number of channels in TRACE Stage-4 features
    """

    def __init__(self, feature_dim: int = 256, stage4_channels: int = 512):
        super().__init__()
        self.feature_dim = feature_dim

        # Stream A: mask encoder
        self.mask_encoder = MaskEncoder(in_channels=1, out_dim=feature_dim)

        # Stream B: project Stage-4 features to D
        self.overlay_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(stage4_channels, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        # Stream C: background CNN encoder
        self.bg_encoder = LightweightCNNEncoder(in_channels=3, out_dim=feature_dim)

        # Cross-attention projections (single head)
        self.W_q = nn.Linear(feature_dim, feature_dim)
        self.W_k = nn.Linear(feature_dim, feature_dim)
        self.W_v = nn.Linear(feature_dim, feature_dim)

        # Learnable confidence scalars
        self.conf_B_logit = nn.Parameter(torch.tensor(0.0))
        self.conf_C_logit = nn.Parameter(torch.tensor(0.0))

        # Residual projection for Stream C
        self.W_r = nn.Linear(feature_dim, feature_dim)

        # Output LayerNorm
        self.out_norm = nn.LayerNorm(feature_dim)

    def forward(self, binary_mask: torch.Tensor,
                stage4_features: torch.Tensor,
                background_overlay: torch.Tensor) -> dict:
        """
        Args:
            binary_mask: (B, 1, H, W) binary gas mask
            stage4_features: (B, 512, H/32, W/32) from TRACE
            background_overlay: (B, 3, H, W) overlay with gas region zeroed

        Returns:
            dict with:
                "fused": (B, D) fused feature vector
                "confidence_scores": [1.0, conf_B, conf_C] for logging
        """
        # Encode streams
        stream_A = self.mask_encoder(binary_mask)        # (B, D)
        stream_B = self.overlay_proj(stage4_features)    # (B, D)
        stream_C = self.bg_encoder(background_overlay)   # (B, D)

        # Learnable confidences
        conf_B = torch.sigmoid(self.conf_B_logit)
        conf_C = torch.sigmoid(self.conf_C_logit)

        # Cross-attention: A queries B
        Q = self.W_q(stream_A).unsqueeze(1)              # (B, 1, D)
        K = (self.W_k(stream_B) * conf_B).unsqueeze(1)   # (B, 1, D)
        V = (self.W_v(stream_B) * conf_B).unsqueeze(1)   # (B, 1, D)

        scale = self.feature_dim ** 0.5
        attn = torch.softmax((Q @ K.transpose(-2, -1)) / scale, dim=-1)
        fused = (attn @ V).squeeze(1)                    # (B, D)

        # Stream C residual correction
        fused = fused + self.W_r(stream_C) * conf_C

        fused = self.out_norm(fused)

        return {
            "fused": fused,
            "confidence_scores": [1.0, conf_B.item(), conf_C.item()],
        }
