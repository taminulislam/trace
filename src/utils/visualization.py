"""Visualization utilities for gas overlays, attention maps, and training metrics."""

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", Path(__file__).resolve().parents[2] / "outputs" / "visualizations"))


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray,
                          alpha: float = 0.4,
                          color: tuple = (0, 255, 0)) -> np.ndarray:
    """Overlay a binary mask on an image with a colored tint.

    Args:
        image: (H, W, 3) BGR image
        mask: (H, W) binary mask (0 or 255 / 0 or 1)
        alpha: transparency of the overlay
        color: BGR color for the mask region

    Returns:
        (H, W, 3) blended image
    """
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)

    colored = np.zeros_like(image)
    colored[:] = color
    binary = mask > 127
    result = image.copy()
    result[binary] = cv2.addWeighted(image[binary], 1 - alpha, colored[binary], alpha, 0)
    return result


def save_segmentation_comparison(image: np.ndarray, gt_mask: np.ndarray,
                                  pred_mask: np.ndarray,
                                  save_path: str,
                                  frame_id: str = "") -> None:
    """Save a side-by-side comparison of GT vs predicted segmentation.

    Layout: [Original | GT Overlay | Pred Overlay | Difference]
    """
    h, w = image.shape[:2]

    gt_overlay = overlay_mask_on_image(image, gt_mask, color=(0, 255, 0))
    pred_overlay = overlay_mask_on_image(image, pred_mask, color=(0, 0, 255))

    gt_bin = (gt_mask > 127).astype(np.uint8) if gt_mask.max() > 1 else gt_mask.astype(np.uint8)
    pred_bin = (pred_mask > 127).astype(np.uint8) if pred_mask.max() > 1 else pred_mask.astype(np.uint8)

    diff = np.zeros((h, w, 3), dtype=np.uint8)
    tp = (gt_bin == 1) & (pred_bin == 1)
    fp = (gt_bin == 0) & (pred_bin == 1)
    fn = (gt_bin == 1) & (pred_bin == 0)
    diff[tp] = (255, 255, 255)  # white = correct
    diff[fp] = (0, 0, 255)     # red = false positive
    diff[fn] = (255, 0, 0)     # blue = false negative

    canvas = np.concatenate([image, gt_overlay, pred_overlay, diff], axis=1)

    if frame_id:
        cv2.putText(canvas, frame_id, (10, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(save_path, canvas)


def visualize_attention_map(attention_weights: np.ndarray, image: np.ndarray,
                            save_path: str, title: str = "") -> None:
    """Overlay attention weights as a heatmap on the original image.

    Args:
        attention_weights: (H, W) attention map in [0, 1]
        image: (H, W, 3) BGR image
        save_path: output file path
        title: optional title text
    """
    h, w = image.shape[:2]
    attn_resized = cv2.resize(attention_weights, (w, h))
    attn_uint8 = (attn_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(attn_uint8, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)

    if title:
        cv2.putText(blended, title, (10, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(save_path, blended)


def visualize_gas_features(overlay: np.ndarray, mask: np.ndarray,
                            features: dict, save_path: str) -> None:
    """Visualize gas features (centroid, dispersion, area) on an overlay.

    Args:
        overlay: (H, W, 3) BGR overlay
        mask: (H, W) binary mask
        features: dict with gas_centroid_x, gas_centroid_y, gas_area_pct, etc.
        save_path: output file path
    """
    vis = overlay_mask_on_image(overlay, mask, alpha=0.3, color=(0, 255, 255))

    cx = int(features.get("gas_centroid_x", 0))
    cy = int(features.get("gas_centroid_y", 0))
    if cx > 0 and cy > 0:
        cv2.drawMarker(vis, (cx, cy), (0, 0, 255),
                        cv2.MARKER_CROSS, markerSize=20, thickness=2)

    area = features.get("gas_area_pct", 0)
    disp = features.get("gas_dispersion", 0)
    text = f"Area: {area:.1f}% | Disp: {disp:.1f}"
    cv2.putText(vis, text, (10, vis.shape[0] - 10),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(save_path, vis)


def plot_training_curves(metrics_log: dict, save_path: str) -> None:
    """Plot training curves from a metrics dictionary.

    Uses matplotlib if available, otherwise saves as CSV.

    Args:
        metrics_log: dict of {metric_name: list_of_values}
        save_path: output file path (.png or .csv)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_metrics = len(metrics_log)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]

        for ax, (name, values) in zip(axes, metrics_log.items()):
            ax.plot(values)
            ax.set_title(name)
            ax.set_xlabel("Step")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    except ImportError:
        import csv
        with open(save_path.replace(".png", ".csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(metrics_log.keys()))
            max_len = max(len(v) for v in metrics_log.values())
            for i in range(max_len):
                row = [metrics_log[k][i] if i < len(metrics_log[k]) else ""
                       for k in metrics_log]
                writer.writerow(row)
