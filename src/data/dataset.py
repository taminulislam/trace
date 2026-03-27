"""PyTorch Dataset classes for TRACE.

Provides:
  - ThermalFrameDataset: single-frame dataset for segmentation training (Stage 1)
  - ThermalClipDataset: temporal clip dataset for temporal/fusion/e2e training (Stage 2-4)
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from src.data.augmentation import ThermalGasAugment
except ImportError:
    ThermalGasAugment = None

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))


class ThermalFrameDataset(Dataset):
    """Single-frame dataset for segmentation training.

    Loads overlay (input), mask (target), and optionally image.
    Computes thermal_intensity as the per-pixel average of the overlay
    within the gas mask region.

    Args:
        annotations_csv: path to annotations.csv
        split: "train", "val", or "test"
        img_size: (H, W) resize target for segmentation
        augment: whether to apply ThermalGasAugment
        include_excluded: if False, filter out excluded frames
    """

    def __init__(self, annotations_csv: str = None, split: str = "train",
                 img_size: tuple = (256, 320), augment: bool = False,
                 include_excluded: bool = False):
        if annotations_csv is None:
            annotations_csv = str(PROJECT_ROOT / "annotations" / "annotations.csv")

        df = pd.read_csv(annotations_csv)
        df = df[df["split"] == split]
        if not include_excluded:
            df = df[~df["excluded"]]
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.augment = augment and ThermalGasAugment is not None
        if self.augment:
            self.augmentor = ThermalGasAugment()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        overlay = cv2.imread(str(PROJECT_ROOT / row["overlay_path"]))
        mask = cv2.imread(str(PROJECT_ROOT / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(str(PROJECT_ROOT / row["image_path"]))

        # Handle missing files gracefully
        if overlay is None:
            overlay = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        if mask is None:
            mask = np.zeros((self.img_size[0], self.img_size[1]), dtype=np.uint8)
        if image is None:
            image = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)

        overlay = cv2.resize(overlay, (self.img_size[1], self.img_size[0]))
        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]),
                          interpolation=cv2.INTER_NEAREST)
        image = cv2.resize(image, (self.img_size[1], self.img_size[0]))

        if self.augment:
            image, mask, overlay = self.augmentor(image, mask, overlay)

        # Binarize mask
        mask = ((mask > 127).astype(np.float32))

        # Compute thermal intensity: grayscale overlay normalized to [0, 1]
        thermal_intensity = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # Convert to tensors (C, H, W)
        overlay_t = torch.from_numpy(overlay).permute(2, 0, 1).float() / 255.0
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        intensity_t = torch.from_numpy(thermal_intensity).unsqueeze(0).float()

        return {
            "overlay": overlay_t,           # (3, H, W)
            "image": image_t,               # (3, H, W)
            "mask": mask_t,                 # (1, H, W)
            "thermal_intensity": intensity_t,  # (1, H, W)
            "class_id": int(row["class_id"]),
            "frame_id": row["frame_id"],
            "seq_id": str(row["seq_id"]),
        }


class ThermalClipDataset(Dataset):
    """Temporal clip dataset for video-level training.

    Loads clips of T=16 frames from clips.csv.
    Each clip returns overlay frames resized to 224x224 for VideoMAE,
    plus per-frame masks and features.

    Args:
        clips_csv: path to clips.csv
        split: "train", "val", or "test"
        clip_img_size: (H, W) for VideoMAE input
        seg_img_size: (H, W) for segmentation input
    """

    def __init__(self, clips_csv: str = None, split: str = "train",
                 clip_img_size: tuple = (224, 224),
                 seg_img_size: tuple = (256, 320)):
        if clips_csv is None:
            clips_csv = str(PROJECT_ROOT / "annotations" / "clips.csv")

        df = pd.read_csv(clips_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.clip_img_size = clip_img_size
        self.seg_img_size = seg_img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        overlay_paths = row["overlay_paths"].split("|")
        mask_paths = row["mask_paths"].split("|")
        image_paths = row["image_paths"].split("|")

        clip_overlays = []  # (T, 3, 224, 224) for VideoMAE
        seg_overlays = []   # (T, 3, H, W) for segmentation
        masks = []
        intensities = []

        for ovl_p, msk_p, img_p in zip(overlay_paths, mask_paths, image_paths):
            overlay = cv2.imread(str(PROJECT_ROOT / ovl_p))
            mask = cv2.imread(str(PROJECT_ROOT / msk_p), cv2.IMREAD_GRAYSCALE)

            # VideoMAE input
            ovl_clip = cv2.resize(overlay, (self.clip_img_size[1], self.clip_img_size[0]))
            clip_overlays.append(
                torch.from_numpy(ovl_clip).permute(2, 0, 1).float() / 255.0
            )

            # Segmentation input
            ovl_seg = cv2.resize(overlay, (self.seg_img_size[1], self.seg_img_size[0]))
            seg_overlays.append(
                torch.from_numpy(ovl_seg).permute(2, 0, 1).float() / 255.0
            )

            msk = cv2.resize(mask, (self.seg_img_size[1], self.seg_img_size[0]),
                             interpolation=cv2.INTER_NEAREST)
            masks.append(torch.from_numpy((msk > 127).astype(np.float32)).unsqueeze(0))

            intensity = cv2.cvtColor(ovl_seg, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            intensities.append(torch.from_numpy(intensity).unsqueeze(0))

        return {
            "clip_overlays": torch.stack(clip_overlays),    # (T, 3, 224, 224)
            "seg_overlays": torch.stack(seg_overlays),      # (T, 3, H, W)
            "masks": torch.stack(masks),                    # (T, 1, H, W)
            "intensities": torch.stack(intensities),        # (T, 1, H, W)
            "class_id": int(row["class_id"]),
            "clip_id": row["clip_id"],
            "seq_id": str(row["seq_id"]),
        }
