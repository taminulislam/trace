"""Step 1.3: Extract per-frame gas features from masks and overlays.

For each frame computes:
  From mask: gas_area_pct, gas_centroid_x, gas_centroid_y, gas_dispersion, gas_connected_components
  From overlay: gas_intensity_mean, gas_intensity_max

Output: annotations/frame_features.csv
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", _ROOT / "dataset"))
ANNOTATIONS_DIR = Path(os.environ.get("ANNOTATIONS_DIR", _ROOT / "annotations"))


def extract_frame_features(mask_path: Path, overlay_path: Path) -> dict:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)

    if mask is None or overlay is None:
        return None

    binary = (mask > 127).astype(np.uint8)
    total_pixels = mask.shape[0] * mask.shape[1]
    gas_pixels = binary.sum()

    gas_area_pct = (gas_pixels / total_pixels) * 100.0

    ys, xs = np.where(binary == 1)
    if len(xs) > 0:
        gas_centroid_x = float(np.mean(xs))
        gas_centroid_y = float(np.mean(ys))
        gas_dispersion = float(np.std(np.stack([xs, ys], axis=1)))
    else:
        gas_centroid_x = 0.0
        gas_centroid_y = 0.0
        gas_dispersion = 0.0

    num_labels, _ = cv2.connectedComponents(binary)
    gas_connected_components = max(0, num_labels - 1)

    overlay_gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    gas_region = overlay_gray[binary == 1]
    if len(gas_region) > 0:
        gas_intensity_mean = float(np.mean(gas_region))
        gas_intensity_max = float(np.max(gas_region))
    else:
        gas_intensity_mean = 0.0
        gas_intensity_max = 0.0

    return {
        "gas_area_pct": round(gas_area_pct, 4),
        "gas_centroid_x": round(gas_centroid_x, 2),
        "gas_centroid_y": round(gas_centroid_y, 2),
        "gas_dispersion": round(gas_dispersion, 4),
        "gas_connected_components": gas_connected_components,
        "gas_intensity_mean": round(gas_intensity_mean, 2),
        "gas_intensity_max": round(gas_intensity_max, 2),
    }


def main():
    rows = []
    for seq_dir in sorted(DATASET_ROOT.iterdir()):
        if not seq_dir.is_dir() or not seq_dir.name.startswith("SEQ_"):
            continue

        seq_id = seq_dir.name.replace("SEQ_", "")
        masks_dir = seq_dir / "masks"
        overlays_dir = seq_dir / "overlays"
        images_dir = seq_dir / "images"

        if not masks_dir.exists():
            continue

        mask_files = sorted(masks_dir.glob("*.png"))
        print(f"Processing {seq_dir.name}: {len(mask_files)} frames...")

        for mask_file in mask_files:
            frame_name = mask_file.stem
            overlay_file = overlays_dir / f"{frame_name}.png"
            image_file = images_dir / f"{frame_name}.png"

            if not overlay_file.exists():
                continue

            features = extract_frame_features(mask_file, overlay_file)
            if features is None:
                continue

            row = {
                "frame_id": frame_name,
                "seq_id": seq_id,
                "image_path": str(image_file.relative_to(DATASET_ROOT.parent)),
                "mask_path": str(mask_file.relative_to(DATASET_ROOT.parent)),
                "overlay_path": str(overlay_file.relative_to(DATASET_ROOT.parent)),
                "is_interpolated": False,
            }
            row.update(features)
            rows.append(row)

    df = pd.DataFrame(rows)
    out_path = ANNOTATIONS_DIR / "frame_features.csv"
    df.to_csv(out_path, index=False)
    print(f"\nExtracted features for {len(df)} frames -> {out_path}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
