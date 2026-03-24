"""Module 3.4: Temporal Encoder.

Base: VideoMAE-Small pretrained on Kinetics-400
Input: 16-frame clip of overlay images (T=16, C=3, H=224, W=224)
Output: temporal embedding projected to D=256 via linear adapter
Adapter: Linear(384->256) + LayerNorm
"""

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """VideoMAE-Small wrapper with projection adapter.

    Uses HuggingFace transformers VideoMAEModel as the backbone.
    The backbone is loaded from 'MCG-NJU/videomae-small' pretrained
    on Kinetics-400.

    The [CLS] token embedding (384-dim for VideoMAE-Small) is projected
    to the fusion dimension D=256 via a linear adapter.

    Args:
        output_dim: projection output dimension (default 256)
        pretrained_name: HuggingFace model name
        freeze_backbone: if True, freeze VideoMAE weights (for Stage 2a)
    """

    def __init__(self, output_dim: int = 256,
                 pretrained_name: str = "MCG-NJU/videomae-small-finetuned-kinetics",
                 freeze_backbone: bool = False):
        super().__init__()
        from transformers import VideoMAEModel, VideoMAEConfig

        self.backbone = VideoMAEModel.from_pretrained(pretrained_name)
        hidden_size = self.backbone.config.hidden_size  # 384 for Small

        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Classification head (used in Stage 2 training)
        self.classifier = nn.Linear(output_dim, 3)

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, pixel_values: torch.Tensor,
                return_cls_logits: bool = False) -> dict:
        """
        Args:
            pixel_values: (B, T, C, H, W) clip of overlay images
                          T=16, C=3, H=224, W=224
            return_cls_logits: if True, also return classification logits

        Returns:
            dict with:
                "temporal_embedding": (B, D) temporal feature vector
                "cls_logits": (B, 3) classification logits (if requested)
        """
        outputs = self.backbone(pixel_values=pixel_values)
        # VideoMAE outputs last_hidden_state: (B, num_patches+1, hidden_size)
        # Use the [CLS] token (index 0) or mean-pool all tokens
        cls_token = outputs.last_hidden_state[:, 0]  # (B, 384)
        temporal_emb = self.adapter(cls_token)  # (B, 256)

        result = {"temporal_embedding": temporal_emb}

        if return_cls_logits:
            result["cls_logits"] = self.classifier(temporal_emb)

        return result


class TemporalEncoderStandalone(nn.Module):
    """Standalone temporal encoder without HuggingFace dependency.

    Uses a standard ViT-Small architecture for video understanding.
    Useful for environments where transformers library is not available,
    or for testing with smaller models.

    Args:
        num_frames: number of frames per clip (default 16)
        img_size: spatial resolution per frame (default 224)
        patch_size: spatial patch size (default 16)
        tubelet_size: temporal patch size (default 2)
        embed_dim: transformer embedding dimension (default 384)
        depth: number of transformer layers (default 12)
        num_heads: attention heads (default 6)
        output_dim: final projection dimension (default 256)
    """

    def __init__(self, num_frames: int = 16, img_size: int = 224,
                 patch_size: int = 16, tubelet_size: int = 2,
                 embed_dim: int = 384, depth: int = 12,
                 num_heads: int = 6, output_dim: int = 256):
        super().__init__()
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim

        num_spatial_patches = (img_size // patch_size) ** 2
        num_temporal_patches = num_frames // tubelet_size
        num_patches = num_spatial_patches * num_temporal_patches

        # 3D patch embedding: (B, C, T, H, W) -> (B, num_patches, embed_dim)
        self.patch_embed = nn.Conv3d(
            3, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.classifier = nn.Linear(output_dim, 3)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, pixel_values: torch.Tensor,
                return_cls_logits: bool = False) -> dict:
        """
        Args:
            pixel_values: (B, T, C, H, W) clip

        Returns:
            dict with temporal_embedding and optionally cls_logits
        """
        B = pixel_values.shape[0]
        # Rearrange to (B, C, T, H, W) for Conv3d
        x = pixel_values.permute(0, 2, 1, 3, 4)
        x = self.patch_embed(x)  # (B, embed_dim, T', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        x = self.encoder(x)
        x = self.norm(x)

        cls_out = x[:, 0]
        temporal_emb = self.adapter(cls_out)

        result = {"temporal_embedding": temporal_emb}
        if return_cls_logits:
            result["cls_logits"] = self.classifier(temporal_emb)
        return result
