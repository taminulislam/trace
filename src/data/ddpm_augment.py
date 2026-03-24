"""Step 2.3: Conditional DDPM for synthetic mask augmentation.

Architecture: small U-Net (3 down + 3 up blocks, 64 base channels, ~4M params)
Conditioning: class embedding (3 classes -> learned 64-dim embedding) injected via AdaGN
Training data: all binary masks, T=100 diffusion steps, resolution 64x48 (upscale after)
Use: generate additional synthetic masks for underrepresented short sequences.
These masks serve ONLY as augmentation for segmentation pretraining (Stage 1).

Training is deferred to Stage 6 of Phase 4. This module defines the model and
utilities; the actual training script is in src/train/.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 3
CLASS_EMBED_DIM = 64
BASE_CHANNELS = 64
DIFFUSION_STEPS = 100
TRAIN_RESOLUTION = (48, 64)  # H, W -- masks are 240x320, downscaled 5x


# ---------- Diffusion schedule ----------

def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4,
                         beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def get_diffusion_params(timesteps: int) -> dict:
    betas = linear_beta_schedule(timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance,
    }


# ---------- Adaptive Group Normalization (AdaGN) ----------

class AdaGN(nn.Module):
    """Adaptive Group Normalization conditioned on class + timestep embeddings."""

    def __init__(self, num_channels: int, embed_dim: int, num_groups: int = 8):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.proj = nn.Linear(embed_dim, num_channels * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        scale_shift = self.proj(emb).unsqueeze(-1).unsqueeze(-1)
        scale, shift = scale_shift.chunk(2, dim=1)
        return self.gn(x) * (1 + scale) + shift


# ---------- Sinusoidal timestep embedding ----------

class SinusoidalPositionEmbedding(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ---------- ResBlock with AdaGN ----------

class ResBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, embed_dim: int):
        super().__init__()
        self.norm1 = AdaGN(in_ch, embed_dim)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = AdaGN(out_ch, embed_dim)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x, emb))
        h = self.conv1(h)
        h = self.act(self.norm2(h, emb))
        h = self.conv2(h)
        return h + self.skip(x)


# ---------- Down / Up blocks ----------

class DownBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, embed_dim: int):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, embed_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        h = self.res(x, emb)
        return self.down(h), h  # return skip connection


class UpBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, embed_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res = ResBlock(in_ch * 2, out_ch, embed_dim)  # concat: up(in_ch) + skip(in_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor,
                emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.res(x, emb)


# ---------- Conditional U-Net ----------

class ConditionalUNet(nn.Module):
    """Small U-Net for conditional binary mask generation.

    3 down + 3 up blocks, 64 base channels, ~4M params.
    Conditioning: class embedding + sinusoidal timestep embedding, both
    projected to a shared embed_dim and summed, then injected via AdaGN.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = BASE_CHANNELS,
                 num_classes: int = NUM_CLASSES, embed_dim: int = 128):
        super().__init__()
        bc = base_channels
        self.embed_dim = embed_dim

        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.class_embed = nn.Embedding(num_classes, embed_dim)

        self.input_conv = nn.Conv2d(in_channels, bc, 3, padding=1)

        self.down1 = DownBlock(bc, bc, embed_dim)
        self.down2 = DownBlock(bc, bc * 2, embed_dim)
        self.down3 = DownBlock(bc * 2, bc * 4, embed_dim)

        self.mid = ResBlock(bc * 4, bc * 4, embed_dim)

        self.up3 = UpBlock(bc * 4, bc * 2, embed_dim)
        self.up2 = UpBlock(bc * 2, bc, embed_dim)
        self.up1 = UpBlock(bc, bc, embed_dim)

        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, bc),
            nn.SiLU(),
            nn.Conv2d(bc, in_channels, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                class_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) noisy mask
            t: (B,) integer timestep
            class_labels: (B,) integer class ids

        Returns:
            (B, 1, H, W) predicted noise
        """
        emb = self.time_embed(t) + self.class_embed(class_labels)

        h = self.input_conv(x)

        h1, skip1 = self.down1(h, emb)
        h2, skip2 = self.down2(h1, emb)
        h3, skip3 = self.down3(h2, emb)

        h3 = self.mid(h3, emb)

        h = self.up3(h3, skip3, emb)
        h = self.up2(h, skip2, emb)
        h = self.up1(h, skip1, emb)

        return self.output_conv(h)


# ---------- Forward diffusion ----------

def q_sample(x_0: torch.Tensor, t: torch.Tensor,
             diffusion_params: dict, noise: torch.Tensor = None) -> tuple:
    if noise is None:
        noise = torch.randn_like(x_0)
    device = x_0.device
    sqrt_alpha = diffusion_params["sqrt_alphas_cumprod"].to(device)[t][:, None, None, None]
    sqrt_one_minus_alpha = diffusion_params["sqrt_one_minus_alphas_cumprod"].to(device)[t][:, None, None, None]
    x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    return x_t, noise


# ---------- Sampling (reverse diffusion) ----------

@torch.no_grad()
def p_sample_loop(model: ConditionalUNet, shape: tuple,
                  class_labels: torch.Tensor, diffusion_params: dict,
                  device: torch.device) -> torch.Tensor:
    b = shape[0]
    x = torch.randn(shape, device=device)
    betas = diffusion_params["betas"].to(device)
    alphas = diffusion_params["alphas"].to(device)
    alphas_cumprod = diffusion_params["alphas_cumprod"].to(device)
    posterior_variance = diffusion_params["posterior_variance"].to(device)

    for i in reversed(range(DIFFUSION_STEPS)):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        pred_noise = model(x, t, class_labels)
        alpha_t = alphas[i]
        alpha_bar_t = alphas_cumprod[i]
        beta_t = betas[i]

        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
        )
        if i > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(posterior_variance[i]) * noise
        else:
            x = mean

    return x


def generate_masks(model: ConditionalUNet, n_samples: int,
                   class_id: int, device: torch.device,
                   resolution: tuple = TRAIN_RESOLUTION) -> torch.Tensor:
    """Generate synthetic binary masks.

    Args:
        model: trained ConditionalUNet
        n_samples: number of masks to generate
        class_id: conditioning class (0=HF, 1=Control, 2=LF)
        device: torch device
        resolution: (H, W) of generated masks

    Returns:
        (n_samples, 1, H, W) tensor with values in {0, 1}
    """
    model.eval()
    diffusion_params = get_diffusion_params(DIFFUSION_STEPS)
    class_labels = torch.full((n_samples,), class_id, device=device, dtype=torch.long)
    shape = (n_samples, 1, resolution[0], resolution[1])
    samples = p_sample_loop(model, shape, class_labels, diffusion_params, device)
    return (samples > 0).float()
