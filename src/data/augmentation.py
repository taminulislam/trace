"""Step 2.1: Strong physics-aware augmentation pipeline (CVPR-grade).

Class: ThermalGasAugment
Applies consistently to all three streams (image, mask, overlay) with the
same random state per sample call.

Supported transforms:
  Spatial:
    - Horizontal flip (p=0.5) — vertical flip NOT valid for upward-rising gas
    - Random affine: scale [0.85–1.15], translate ±10%, shear ±5°
    - Random crop & resize (p=0.3)
  Photometric (applied to gas region):
    - Gas intensity scaling [0.7, 1.4]
    - Per-channel gas colour jitter
  Photometric (applied to background region):
    - Stronger brightness/contrast jitter
    - Background Gaussian blur (p=0.2)
  Mask:
    - Boundary noise (erosion-based ring) with morphological dilation (p=0.3)
    - Cutout / mask dropout (p=0.15, zero-out random patch inside gas region)
  Noise:
    - Additive Gaussian noise on overlay (sigma 0–10 on uint8 scale)
  NO rotation >5°, NO vertical flip, NO elastic distortion (physics constraints)
"""

import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None


class ThermalGasAugment:
    """Strong physics-aware augmentation for thermal gas emission data.

    All transforms are applied consistently across the three data streams
    (image, mask, overlay) using a shared random state per sample.
    """

    def __init__(
        self,
        p_hflip: float = 0.5,
        p_affine: float = 0.5,
        p_crop: float = 0.3,
        p_blur: float = 0.2,
        p_cutout: float = 0.15,
        p_morph_dilate: float = 0.3,
        intensity_range: tuple = (0.7, 1.4),
        colour_jitter_range: float = 0.15,
        brightness_range: tuple = (-30, 30),
        contrast_range: tuple = (0.8, 1.2),
        noise_sigma_range: tuple = (0, 10),
        boundary_noise_sigma: float = 0.04,
        erosion_px: int = 3,
        cutout_frac: float = 0.15,
    ):
        self.p_hflip = p_hflip
        self.p_affine = p_affine
        self.p_crop = p_crop
        self.p_blur = p_blur
        self.p_cutout = p_cutout
        self.p_morph_dilate = p_morph_dilate
        self.intensity_range = intensity_range
        self.colour_jitter_range = colour_jitter_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_sigma_range = noise_sigma_range
        self.boundary_noise_sigma = boundary_noise_sigma
        self.erosion_px = erosion_px
        self.cutout_frac = cutout_frac

    def __call__(self, image: np.ndarray, mask: np.ndarray,
                 overlay: np.ndarray, seed: int = None) -> tuple:
        """Apply augmentations.

        Args:
            image: (H, W, 3) uint8
            mask: (H, W) uint8 binary mask (0 or 255)
            overlay: (H, W, 3) uint8 RGB overlay
            seed: optional random seed

        Returns:
            (augmented_image, augmented_mask, augmented_overlay)
        """
        rng = np.random.RandomState(seed)

        image = image.copy()
        mask = mask.copy()
        overlay = overlay.copy()
        binary = (mask > 127).astype(np.uint8)

        # --- Spatial transforms (applied identically to all streams) ---

        # Horizontal flip
        if rng.random() < self.p_hflip:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
            overlay = np.flip(overlay, axis=1).copy()
            binary = np.flip(binary, axis=1).copy()

        # Random affine (scale + translate + shear)
        if rng.random() < self.p_affine:
            image, mask, overlay, binary = self._random_affine(
                image, mask, overlay, binary, rng)

        # Random crop & resize
        if rng.random() < self.p_crop:
            image, mask, overlay, binary = self._random_crop(
                image, mask, overlay, binary, rng)

        # --- Photometric transforms (gas region) ---

        # Gas intensity scaling
        overlay = self._scale_gas_intensity(overlay, binary, rng)

        # Per-channel gas colour jitter
        overlay = self._gas_colour_jitter(overlay, binary, rng)

        # --- Photometric transforms (background region) ---
        image = self._adjust_background(image, binary, rng)
        overlay = self._adjust_background(overlay, binary, rng)

        # Background Gaussian blur
        if rng.random() < self.p_blur:
            image = self._blur_background(image, binary, rng)
            overlay = self._blur_background(overlay, binary, rng)

        # --- Mask augmentation ---
        mask = self._add_boundary_noise(mask, binary, rng)

        # Morphological dilation of mask
        if rng.random() < self.p_morph_dilate:
            kernel_sz = rng.choice([3, 5])
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (kernel_sz, kernel_sz))
            mask = cv2.dilate(mask, kernel, iterations=1)

        # Cutout (zero-out random patch in gas region)
        if rng.random() < self.p_cutout:
            overlay, mask = self._cutout_gas(overlay, mask, binary, rng)

        # --- Global noise ---
        sigma = rng.uniform(*self.noise_sigma_range)
        if sigma > 0.5:
            noise = rng.normal(0, sigma, overlay.shape).astype(np.float32)
            overlay = np.clip(overlay.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            noise_img = rng.normal(0, sigma, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise_img, 0, 255).astype(np.uint8)

        return image, mask, overlay

    # ---- Internal helpers ----

    def _random_affine(self, image, mask, overlay, binary, rng):
        H, W = image.shape[:2]
        scale = rng.uniform(0.85, 1.15)
        tx = rng.uniform(-0.1, 0.1) * W
        ty = rng.uniform(-0.1, 0.1) * H
        shear = rng.uniform(-5, 5) * np.pi / 180

        cx, cy = W / 2, H / 2
        M = np.float32([
            [scale * np.cos(shear), -np.sin(shear), tx + cx * (1 - scale)],
            [np.sin(shear), scale * np.cos(shear), ty + cy * (1 - scale)],
        ])

        image = cv2.warpAffine(image, M, (W, H), borderMode=cv2.BORDER_REFLECT_101)
        overlay = cv2.warpAffine(overlay, M, (W, H), borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        binary = (mask > 127).astype(np.uint8)
        return image, mask, overlay, binary

    def _random_crop(self, image, mask, overlay, binary, rng):
        H, W = image.shape[:2]
        crop_frac = rng.uniform(0.75, 0.95)
        cH, cW = int(H * crop_frac), int(W * crop_frac)
        y0 = rng.randint(0, H - cH + 1)
        x0 = rng.randint(0, W - cW + 1)

        image = cv2.resize(image[y0:y0+cH, x0:x0+cW], (W, H))
        overlay = cv2.resize(overlay[y0:y0+cH, x0:x0+cW], (W, H))
        mask = cv2.resize(mask[y0:y0+cH, x0:x0+cW], (W, H),
                          interpolation=cv2.INTER_NEAREST)
        binary = (mask > 127).astype(np.uint8)
        return image, mask, overlay, binary

    def _scale_gas_intensity(self, overlay, binary, rng):
        factor = rng.uniform(*self.intensity_range)
        gas_mask_3c = np.stack([binary] * 3, axis=-1).astype(bool)
        ovl = overlay.astype(np.float32)
        ovl[gas_mask_3c] *= factor
        return np.clip(ovl, 0, 255).astype(np.uint8)

    def _gas_colour_jitter(self, overlay, binary, rng):
        """Per-channel intensity jitter in the gas region only."""
        gas_mask = binary.astype(bool)
        if gas_mask.sum() == 0:
            return overlay
        ovl = overlay.astype(np.float32)
        for c in range(3):
            shift = rng.uniform(-self.colour_jitter_range, self.colour_jitter_range) * 255
            ovl[:, :, c][gas_mask] += shift
        return np.clip(ovl, 0, 255).astype(np.uint8)

    def _add_boundary_noise(self, mask, binary, rng):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * self.erosion_px + 1, 2 * self.erosion_px + 1),
        )
        eroded = cv2.erode(binary, kernel, iterations=1)
        boundary_ring = binary - eroded
        if boundary_ring.sum() == 0:
            return mask
        noise = rng.normal(0, self.boundary_noise_sigma, mask.shape).astype(np.float32)
        m = mask.astype(np.float32) / 255.0
        m += noise * boundary_ring.astype(np.float32)
        return ((m > 0.5).astype(np.uint8)) * 255

    def _adjust_background(self, img, binary, rng):
        brightness = rng.uniform(*self.brightness_range)
        contrast = rng.uniform(*self.contrast_range)
        bg_mask = (binary == 0)
        if img.ndim == 3:
            bg = np.stack([bg_mask] * img.shape[2], axis=-1)
        else:
            bg = bg_mask
        f = img.astype(np.float32)
        f[bg] = f[bg] * contrast + brightness
        return np.clip(f, 0, 255).astype(np.uint8)

    def _blur_background(self, img, binary, rng):
        ksize = rng.choice([3, 5, 7])
        blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
        bg_mask = (binary == 0)
        if img.ndim == 3:
            bg = np.stack([bg_mask] * 3, axis=-1)
        else:
            bg = bg_mask
        out = img.copy()
        out[bg] = blurred[bg]
        return out

    def _cutout_gas(self, overlay, mask, binary, rng):
        """Zero-out a random rectangular patch within the gas region."""
        H, W = overlay.shape[:2]
        ph = int(H * self.cutout_frac)
        pw = int(W * self.cutout_frac)
        if ph < 2 or pw < 2:
            return overlay, mask

        ys, xs = np.where(binary > 0)
        if len(ys) == 0:
            return overlay, mask

        ci = rng.randint(len(ys))
        cy, cx = ys[ci], xs[ci]
        y1, y2 = max(0, cy - ph // 2), min(H, cy + ph // 2)
        x1, x2 = max(0, cx - pw // 2), min(W, cx + pw // 2)

        overlay[y1:y2, x1:x2] = 0
        mask[y1:y2, x1:x2] = 0
        return overlay, mask


if torch is not None:
    class ThermalGasAugmentTorch:
        """Wrapper that accepts and returns torch tensors."""

        def __init__(self, **kwargs):
            self.augmentor = ThermalGasAugment(**kwargs)

        def __call__(self, image: torch.Tensor, mask: torch.Tensor,
                     overlay: torch.Tensor, seed: int = None) -> tuple:
            img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            mask_np = (mask.squeeze(0).numpy() * 255).astype(np.uint8)
            ovl_np = (overlay.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            img_aug, mask_aug, ovl_aug = self.augmentor(img_np, mask_np, ovl_np, seed)

            img_t = torch.from_numpy(img_aug).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask_aug).unsqueeze(0).float() / 255.0
            ovl_t = torch.from_numpy(ovl_aug).permute(2, 0, 1).float() / 255.0

            return img_t, mask_t, ovl_t
