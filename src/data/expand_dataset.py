"""Offline dataset augmentation — physically expand training data on disk.

Creates N augmented copies of every training frame (overlay, mask, image)
and appends them to annotations.csv. This gives the model more diversity
than on-the-fly augmentation alone.

Usage:
    python src/data/expand_dataset.py --copies 5
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.augmentation import ThermalGasAugment

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def augment_and_save(row, copy_id, augmentor, project_root):
    """Load one sample, augment it, save to disk, return new row."""
    overlay = cv2.imread(str(project_root / row["overlay_path"]))
    mask = cv2.imread(str(project_root / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(str(project_root / row["image_path"]))

    if overlay is None or mask is None or image is None:
        return None

    # Apply augmentation
    image_aug, mask_aug, overlay_aug = augmentor(image, mask, overlay)

    # Generate new filenames
    seq_id = row["seq_id"]
    frame_base = Path(row["image_path"]).stem
    aug_suffix = f"_aug{copy_id:02d}"

    img_dir = project_root / f"dataset/SEQ_{seq_id:04d}/images"
    mask_dir = project_root / f"dataset/SEQ_{seq_id:04d}/masks"
    ovl_dir = project_root / f"dataset/SEQ_{seq_id:04d}/overlays"

    new_img_name = f"{frame_base}{aug_suffix}.png"
    new_mask_name = f"{frame_base}{aug_suffix}.png"
    new_ovl_name = f"{frame_base}{aug_suffix}.png"

    cv2.imwrite(str(img_dir / new_img_name), image_aug)
    cv2.imwrite(str(mask_dir / new_mask_name), mask_aug)
    cv2.imwrite(str(ovl_dir / new_ovl_name), overlay_aug)

    # Create new row
    new_row = row.copy()
    new_row["frame_id"] = f"{frame_base}{aug_suffix}"
    new_row["image_path"] = f"dataset/SEQ_{seq_id:04d}/images/{new_img_name}"
    new_row["mask_path"] = f"dataset/SEQ_{seq_id:04d}/masks/{new_mask_name}"
    new_row["overlay_path"] = f"dataset/SEQ_{seq_id:04d}/overlays/{new_ovl_name}"
    new_row["is_interpolated"] = False

    # Recompute gas stats from augmented mask
    binary = mask_aug > 127
    total_pixels = binary.size
    gas_pixels = binary.sum()
    new_row["gas_area_pct"] = float(gas_pixels / total_pixels * 100)
    if gas_pixels > 0:
        ys, xs = np.where(binary)
        new_row["gas_centroid_x"] = float(xs.mean())
        new_row["gas_centroid_y"] = float(ys.mean())
        new_row["gas_dispersion"] = float(np.sqrt(xs.var() + ys.var()))
    else:
        new_row["gas_centroid_x"] = 0.0
        new_row["gas_centroid_y"] = 0.0
        new_row["gas_dispersion"] = 0.0

    return new_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=5,
                        help="Number of augmented copies per training frame")
    parser.add_argument("--annotations", type=str,
                        default=str(PROJECT_ROOT / "annotations" / "annotations.csv"))
    args = parser.parse_args()

    # Backup original
    ann_path = Path(args.annotations)
    backup = ann_path.with_suffix(".csv.backup")
    if not backup.exists():
        import shutil
        shutil.copy(ann_path, backup)
        print(f"📦 Backed up original to: {backup}")

    df = pd.read_csv(ann_path)
    train_df = df[df["split"] == "train"].copy()

    print(f"Original dataset: {len(df)} total, {len(train_df)} train")
    print(f"Generating {args.copies} augmented copies per train frame...")
    print(f"Expected new frames: {len(train_df) * args.copies}")

    augmentor = ThermalGasAugment()
    new_rows = []
    failed = 0

    for copy_id in range(1, args.copies + 1):
        print(f"\n  Copy {copy_id}/{args.copies}...")
        for i, (idx, row) in enumerate(train_df.iterrows()):
            new_row = augment_and_save(row, copy_id, augmentor, PROJECT_ROOT)
            if new_row is not None:
                new_rows.append(new_row)
            else:
                failed += 1

            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(train_df)} frames processed")

    print(f"\n✅ Generated {len(new_rows)} augmented frames ({failed} failed)")

    # Append to dataframe
    aug_df = pd.DataFrame(new_rows)
    expanded_df = pd.concat([df, aug_df], ignore_index=True)

    # Save
    expanded_df.to_csv(ann_path, index=False)
    print(f"\n📄 Updated annotations: {ann_path}")
    print(f"   Original: {len(df)} frames")
    print(f"   Added:    {len(new_rows)} augmented frames")
    print(f"   Total:    {len(expanded_df)} frames")
    print(f"   Train:    {len(expanded_df[expanded_df['split'] == 'train'])} frames")
    print(f"   Val:      {len(expanded_df[expanded_df['split'] == 'val'])} (unchanged)")
    print(f"   Test:     {len(expanded_df[expanded_df['split'] == 'test'])} (unchanged)")


if __name__ == "__main__":
    main()
