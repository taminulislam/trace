"""Step 1.2: Fix dataset inconsistencies.

Actions:
1. Rename frames/ -> images/ for SEQ_0491, SEQ_0492, SEQ_0493
2. SEQ_0490: remove orphaned mask (13 masks vs 12 images)
3. SEQ_0498: remove orphaned image (529 images vs 528 masks/overlays)
4. SEQ_0506: remove orphaned images (323 images vs 321 masks/overlays)
5. Binarize all masks: (mask > 127) * 255
6. Write annotations/dataset_audit.log
"""

import os
import shutil
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", _ROOT / "dataset"))
ANNOTATIONS_DIR = Path(os.environ.get("ANNOTATIONS_DIR", _ROOT / "annotations"))

log_lines = []


def log(msg: str):
    print(msg)
    log_lines.append(msg)


def count_files(directory: Path) -> int:
    if not directory.exists():
        return 0
    return len([f for f in directory.iterdir() if f.is_file()])


def get_stems(directory: Path) -> set:
    if not directory.exists():
        return set()
    return {f.stem for f in directory.iterdir() if f.is_file()}


def rename_frames_to_images():
    seqs_to_rename = ["SEQ_0491", "SEQ_0492", "SEQ_0493"]
    for seq in seqs_to_rename:
        frames_dir = DATASET_ROOT / seq / "frames"
        images_dir = DATASET_ROOT / seq / "images"
        if frames_dir.exists() and not images_dir.exists():
            shutil.move(str(frames_dir), str(images_dir))
            log(f"[RENAME] {seq}: frames/ -> images/")
        elif images_dir.exists():
            log(f"[SKIP] {seq}: images/ already exists")
        else:
            log(f"[WARN] {seq}: frames/ not found")


def fix_orphaned_masks_0490():
    seq = "SEQ_0490"
    images_dir = DATASET_ROOT / seq / "images"
    masks_dir = DATASET_ROOT / seq / "masks"

    img_stems = get_stems(images_dir)
    mask_stems = get_stems(masks_dir)

    orphaned = mask_stems - img_stems
    if orphaned:
        for stem in sorted(orphaned):
            mask_file = masks_dir / f"{stem}.png"
            if mask_file.exists():
                os.remove(mask_file)
                log(f"[DELETE] {seq}: removed orphaned mask {stem}.png")
    else:
        log(f"[SKIP] {seq}: no orphaned masks found")

    log(f"[COUNT] {seq} after fix: images={count_files(images_dir)} masks={count_files(masks_dir)}")


def fix_orphaned_images(seq: str):
    images_dir = DATASET_ROOT / seq / "images"
    masks_dir = DATASET_ROOT / seq / "masks"
    overlays_dir = DATASET_ROOT / seq / "overlays"

    img_stems = get_stems(images_dir)
    mask_stems = get_stems(masks_dir)

    orphaned = img_stems - mask_stems
    if orphaned:
        for stem in sorted(orphaned):
            img_file = images_dir / f"{stem}.png"
            if img_file.exists():
                os.remove(img_file)
                log(f"[DELETE] {seq}: removed orphaned image {stem}.png")
            overlay_file = overlays_dir / f"{stem}.png"
            if overlay_file.exists():
                os.remove(overlay_file)
                log(f"[DELETE] {seq}: removed orphaned overlay {stem}.png")
    else:
        log(f"[SKIP] {seq}: no orphaned images found")

    log(f"[COUNT] {seq} after fix: images={count_files(images_dir)} "
        f"masks={count_files(masks_dir)} overlays={count_files(overlays_dir)}")


def binarize_masks():
    fixed_count = 0
    total_count = 0
    for seq_dir in sorted(DATASET_ROOT.iterdir()):
        if not seq_dir.is_dir() or not seq_dir.name.startswith("SEQ_"):
            continue
        masks_dir = seq_dir / "masks"
        if not masks_dir.exists():
            continue
        for mask_file in sorted(masks_dir.iterdir()):
            if not mask_file.suffix == ".png":
                continue
            total_count += 1
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                log(f"[WARN] Could not read {mask_file}")
                continue
            unique_vals = np.unique(mask)
            needs_fix = not all(v in (0, 255) for v in unique_vals)
            if needs_fix:
                binarized = ((mask > 127).astype(np.uint8)) * 255
                cv2.imwrite(str(mask_file), binarized)
                fixed_count += 1

    log(f"[BINARIZE] Checked {total_count} masks, fixed {fixed_count} with intermediate values")


def audit_all_sequences():
    log("\n--- Final Dataset Audit ---")
    total_images = 0
    total_masks = 0
    total_overlays = 0
    for seq_dir in sorted(DATASET_ROOT.iterdir()):
        if not seq_dir.is_dir() or not seq_dir.name.startswith("SEQ_"):
            continue
        images_dir = seq_dir / "images"
        masks_dir = seq_dir / "masks"
        overlays_dir = seq_dir / "overlays"
        ni = count_files(images_dir)
        nm = count_files(masks_dir)
        no = count_files(overlays_dir)
        total_images += ni
        total_masks += nm
        total_overlays += no
        status = "OK" if ni == nm == no else "MISMATCH"
        log(f"  {seq_dir.name}: images={ni} masks={nm} overlays={no} [{status}]")
    log(f"  TOTAL: images={total_images} masks={total_masks} overlays={total_overlays}")


def main():
    log("=" * 60)
    log("TRACE Dataset Fix - Step 1.2")
    log("=" * 60)

    log("\n--- Before Fixes ---")
    audit_all_sequences()

    log("\n--- Renaming frames/ -> images/ ---")
    rename_frames_to_images()

    log("\n--- Fixing SEQ_0490 orphaned masks ---")
    fix_orphaned_masks_0490()

    log("\n--- Fixing SEQ_0498 orphaned images ---")
    fix_orphaned_images("SEQ_0498")

    log("\n--- Fixing SEQ_0506 orphaned images ---")
    fix_orphaned_images("SEQ_0506")

    log("\n--- Binarizing masks ---")
    binarize_masks()

    log("\n--- After Fixes ---")
    audit_all_sequences()

    audit_log_path = ANNOTATIONS_DIR / "dataset_audit.log"
    audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\nAudit log written to {audit_log_path}")


if __name__ == "__main__":
    main()
