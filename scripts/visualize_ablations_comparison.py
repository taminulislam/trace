"""Ablation comparison visualization.

Creates a figure with 4 rows x N columns:
  Col 0: Original image
  Col 1: Ground truth overlay
  Col 2+: One column per model (prediction overlay)

Each row uses the same test frame, so comparisons are apples-to-apples.

Usage:
    python scripts/visualize_ablations_comparison.py
    python scripts/visualize_ablations_comparison.py --n_rows 4 --out comparison_ablations.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ThermalFrameDataset
from src.models.model_factory import create_model

# ── Model registry ────────────────────────────────────────────────────────
# (label, model_name, seg_checkpoint_path)
MODELS = [
    (
        "TRACE",
        "trace",
        PROJECT_ROOT / "Exp1_TRACE/checkpoints/segmentation/segmentation_latest.pt",
    ),
    (
        "TRACE-Tiny\n(A6)",
        "trace_tiny",
        PROJECT_ROOT / "Exp_Ablations/A6_trace_tiny/checkpoints/segmentation/segmentation_latest.pt",
    ),
    (
        "No TGAA\n(A4)",
        "segformer_b2",
        PROJECT_ROOT / "Exp_Ablations/A4_no_tgaa/checkpoints/segmentation/segmentation_latest.pt",
    ),
    (
        "Simple Concat\n(A2)",
        "trace",
        PROJECT_ROOT / "Exp_Ablations/A2_simple_concat/checkpoints/segmentation/segmentation_latest.pt",
    ),
    (
        "No Temporal\n(A3)",
        "trace",
        PROJECT_ROOT / "Exp_Ablations/A3_no_temporal/checkpoints/segmentation/segmentation_latest.pt",
    ),
    (
        "No E2E\n(A8)",
        "trace",
        PROJECT_ROOT / "Exp_Ablations/shared_baseline/checkpoints/segmentation/segmentation_latest.pt",
    ),
]

# CVPR style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def load_model(model_name, ckpt_path, device):
    """Load segmentation model from checkpoint. Returns None if ckpt missing."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"  [SKIP] checkpoint not found: {ckpt_path}")
        return None
    print(f"  Loading {model_name} from {ckpt_path.name} ...")
    model = create_model(model_name, num_seg_classes=1, decode_dim=256, use_aux_mask=True).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def predict(model, sample, device):
    """Run inference and return binary prediction mask (H, W numpy bool)."""
    overlay = sample["overlay"].unsqueeze(0).to(device)
    mask    = sample["mask"].unsqueeze(0).to(device)
    intensity = sample["thermal_intensity"].unsqueeze(0).to(device)
    with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
        out = model(overlay, thermal_intensity=intensity, binary_mask=mask)
    pred = (torch.sigmoid(out["seg_logits"]) > 0.5).squeeze().cpu().numpy().astype(bool)
    return pred


def make_overlay(img_rgb, mask, color):
    """Blend a binary mask onto img_rgb with given RGB color (values 0-1)."""
    vis = img_rgb.copy()
    color = np.array(color, dtype=np.float32)
    vis[mask] = vis[mask] * 0.45 + color * 0.55
    return np.clip(vis, 0, 1)


def placeholder_image(shape, text, color=(0.15, 0.15, 0.15)):
    """Return a dark placeholder image with centered text."""
    img = np.full((*shape[:2], 3), color, dtype=np.float32)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rows", type=int, default=4, help="Number of sample rows")
    parser.add_argument("--split", default="test", help="Dataset split to sample from")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "all_figures" / "ablation_comparison_grid.png"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────
    ds = ThermalFrameDataset(split=args.split, img_size=(256, 320), augment=False)
    print(f"Dataset '{args.split}': {len(ds)} frames")
    if len(ds) == 0:
        print("No frames in split — trying 'val'")
        ds = ThermalFrameDataset(split="val", img_size=(256, 320), augment=False)

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=min(args.n_rows, len(ds)), replace=False)
    indices = sorted(indices.tolist())
    print(f"Selected frame indices: {indices}")

    # ── Load models ──────────────────────────────────────────────────────
    loaded_models = []  # (label, model_or_None)
    for label, model_name, ckpt_path in MODELS:
        m = load_model(model_name, ckpt_path, device)
        loaded_models.append((label, m))

    available = [(lbl, m) for lbl, m in loaded_models if m is not None]
    missing   = [(lbl, None) for lbl, m in loaded_models if m is None]

    if not available:
        print("\nNo checkpoints found — showing dataset images only (no predictions).")
    else:
        print(f"\nLoaded {len(available)} model(s). Missing: {[l for l,_ in missing]}")

    # ── Layout ───────────────────────────────────────────────────────────
    # Columns: Original | GT | [model_0] | [model_1] | ...
    col_labels = ["Original", "Ground\nTruth"] + [lbl for lbl, _ in loaded_models]
    n_cols = len(col_labels)
    n_rows = len(indices)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.6 * n_cols, 2.6 * n_rows),
        squeeze=False,
    )

    # Header labels (top row only)
    for c, col_lbl in enumerate(col_labels):
        axes[0, c].set_title(col_lbl, fontsize=8, fontweight="bold", pad=3)

    for row_i, ds_idx in enumerate(indices):
        sample  = ds[ds_idx]
        img_bgr = sample["overlay"].permute(1, 2, 0).numpy()   # already [0,1]
        img_rgb = img_bgr[..., ::-1].copy()
        gt_mask = sample["mask"].squeeze().numpy().astype(bool)
        frame_id = sample.get("frame_id", str(ds_idx))

        # Row label on first column
        axes[row_i, 0].set_ylabel(f"Frame {frame_id}", fontsize=7, labelpad=3)

        # Col 0: Original
        axes[row_i, 0].imshow(img_rgb)

        # Col 1: Ground truth overlay (blue)
        gt_vis = make_overlay(img_rgb, gt_mask, color=[0.1, 0.5, 1.0])
        axes[row_i, 1].imshow(gt_vis)

        # Cols 2+: Model predictions
        for col_i, (label, model) in enumerate(loaded_models):
            ax = axes[row_i, col_i + 2]
            if model is None:
                ax.imshow(placeholder_image(img_rgb.shape))
                ax.text(0.5, 0.5, "No\nCheckpoint", ha="center", va="center",
                        fontsize=7, color="white", transform=ax.transAxes)
            else:
                pred = predict(model, sample, device)
                pred_vis = make_overlay(img_rgb, pred, color=[0.1, 0.9, 0.2])
                ax.imshow(pred_vis)

    for ax in axes.flat:
        ax.axis("off")

    plt.suptitle("Segmentation Comparison: TRACE vs Ablations", fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
