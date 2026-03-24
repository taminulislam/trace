"""Step 1.4: Temporal interpolation for short sequences.

Uses cv2.addWeighted linear interpolation between consecutive frames.
Applies to images/, masks/, overlays/ consistently.
Masks are rounded to 0 or 255 after interpolation.
Interpolated frames are saved with naming: SEQ_XXXX_frame_interp_YYYYY.png
"""

import os
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", _ROOT / "dataset"))
ANNOTATIONS_DIR = Path(os.environ.get("ANNOTATIONS_DIR", _ROOT / "annotations"))

INTERPOLATION_TARGETS = {
    "0483": 60,
    "0486": 60,
    "0490": 60,
    "0484": 80,
    "0488": 80,
    "0491": 80,
    "0493": 60,
    "0505": 160,
}


def get_sorted_frames(directory: Path):
    return sorted(directory.glob("*.png"))


def interpolate_pair(img_a, img_b, alpha):
    return cv2.addWeighted(img_a, 1.0 - alpha, img_b, alpha, 0)


def binarize_mask(mask):
    return ((mask > 127).astype(np.uint8)) * 255


def interpolate_sequence(seq_id: str, target_count: int):
    seq_dir = DATASET_ROOT / f"SEQ_{seq_id}"
    images_dir = seq_dir / "images"
    masks_dir = seq_dir / "masks"
    overlays_dir = seq_dir / "overlays"

    existing_images = get_sorted_frames(images_dir)
    existing_masks = get_sorted_frames(masks_dir)
    existing_overlays = get_sorted_frames(overlays_dir)

    n_existing = len(existing_images)
    if n_existing == 0:
        print(f"  [WARN] SEQ_{seq_id}: no images found, skipping")
        return []

    n_to_generate = target_count - n_existing
    if n_to_generate <= 0:
        print(f"  [SKIP] SEQ_{seq_id}: already has {n_existing} >= {target_count}")
        return []

    n_gaps = n_existing - 1
    if n_gaps == 0:
        print(f"  [WARN] SEQ_{seq_id}: only 1 frame, cannot interpolate")
        return []

    frames_per_gap = [n_to_generate // n_gaps] * n_gaps
    remainder = n_to_generate % n_gaps
    for i in range(remainder):
        frames_per_gap[i] += 1

    interp_log = []
    interp_counter = 0

    for gap_idx in range(n_gaps):
        img_a = cv2.imread(str(existing_images[gap_idx]), cv2.IMREAD_COLOR)
        img_b = cv2.imread(str(existing_images[gap_idx + 1]), cv2.IMREAD_COLOR)
        mask_a = cv2.imread(str(existing_masks[gap_idx]), cv2.IMREAD_GRAYSCALE)
        mask_b = cv2.imread(str(existing_masks[gap_idx + 1]), cv2.IMREAD_GRAYSCALE)
        ovl_a = cv2.imread(str(existing_overlays[gap_idx]), cv2.IMREAD_COLOR)
        ovl_b = cv2.imread(str(existing_overlays[gap_idx + 1]), cv2.IMREAD_COLOR)

        n_interp = frames_per_gap[gap_idx]
        for j in range(n_interp):
            alpha = (j + 1) / (n_interp + 1)
            frame_name = f"SEQ_{seq_id}_frame_interp_{interp_counter:05d}"

            interp_img = interpolate_pair(img_a, img_b, alpha)
            cv2.imwrite(str(images_dir / f"{frame_name}.png"), interp_img)

            interp_mask = interpolate_pair(mask_a, mask_b, alpha)
            interp_mask = binarize_mask(interp_mask)
            cv2.imwrite(str(masks_dir / f"{frame_name}.png"), interp_mask)

            interp_ovl = interpolate_pair(ovl_a, ovl_b, alpha)
            cv2.imwrite(str(overlays_dir / f"{frame_name}.png"), interp_ovl)

            interp_log.append(f"SEQ_{seq_id}/{frame_name}.png "
                              f"(between {existing_images[gap_idx].stem} "
                              f"and {existing_images[gap_idx + 1].stem}, "
                              f"alpha={alpha:.3f})")
            interp_counter += 1

    print(f"  SEQ_{seq_id}: {n_existing} -> {n_existing + interp_counter} "
          f"(+{interp_counter} interpolated)")
    return interp_log


def main():
    all_logs = []
    print("Temporal Interpolation - Step 1.4")
    print("=" * 50)

    for seq_id, target in INTERPOLATION_TARGETS.items():
        logs = interpolate_sequence(seq_id, target)
        all_logs.extend(logs)

    log_path = ANNOTATIONS_DIR / "interpolated_frames.log"
    with open(log_path, "w") as f:
        f.write(f"Total interpolated frames: {len(all_logs)}\n")
        f.write("=" * 50 + "\n")
        for line in all_logs:
            f.write(line + "\n")

    print(f"\nTotal interpolated frames: {len(all_logs)}")
    print(f"Log written to {log_path}")

    print("\nPost-interpolation frame counts:")
    for seq_id in INTERPOLATION_TARGETS:
        seq_dir = DATASET_ROOT / f"SEQ_{seq_id}"
        n = len(list((seq_dir / "images").glob("*.png")))
        print(f"  SEQ_{seq_id}: {n} frames")


if __name__ == "__main__":
    main()
