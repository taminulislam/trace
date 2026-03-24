"""CVPR-quality visualizations for TRACE.

Generates 9 types of figures:
  1. Segmentation overlays (Input | Pred | GT | Diff)
  2. Confusion matrix heatmap
  3. ROC curves (per-class + macro)
  4. Precision-Recall curves
  5. Training curves from JSONL logs
  6. Per-class performance grouped bars
  7. Boundary quality overlay (pred vs GT boundaries)
  8. GradCAM / attention heatmaps
  9. Qualitative grid (thermal → mask → class → description)

Usage:
    python src/eval/visualize.py \
        --seg_checkpoint outputs/checkpoints/segmentation/segmentation_latest.pt \
        --raw_npz outputs/eval_results/classification_raw.npz
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import binary_erosion
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                              precision_recall_curve, roc_curve, auc)
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import ThermalFrameDataset
from src.models.model_factory import create_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
CLASS_NAMES = ["High-Flux", "Control", "Low-Flux"]

# CVPR-quality style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


# ══════════════════════════════════════════════════════════════════════
# 1. Segmentation overlays
# ══════════════════════════════════════════════════════════════════════
def plot_segmentation_grid(model, device, n_samples=8, save_dir=None):
    """Create a grid of segmentation results: Input | Pred | GT | Difference."""
    ds = ThermalFrameDataset(split="val", img_size=(256, 320), augment=False)
    save_dir = save_dir or FIG_DIR
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 3 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    cols = ["Input Thermal", "Predicted Mask", "Ground Truth", "TP/FP/FN"]

    model.eval()
    indices = np.linspace(0, len(ds) - 1, n_samples, dtype=int)

    for row, idx in enumerate(indices):
        sample = ds[idx]
        overlay = sample["overlay"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)
        intensity = sample["thermal_intensity"].unsqueeze(0).to(device)

        with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
            out = model(overlay, thermal_intensity=intensity, binary_mask=mask)
        pred = (torch.sigmoid(out["seg_logits"]) > 0.5).squeeze().cpu().numpy()
        gt = mask.squeeze().cpu().numpy().astype(bool)
        img = overlay.squeeze().permute(1, 2, 0).cpu().numpy()

        # Input
        axes[row, 0].imshow(img)
        axes[row, 0].set_ylabel(f"Sample {idx}", fontsize=9)

        # Predicted mask (green overlay)
        pred_vis = img.copy()
        pred_vis[pred] = pred_vis[pred] * 0.5 + np.array([0, 1, 0]) * 0.5
        axes[row, 1].imshow(pred_vis)

        # GT mask (blue overlay)
        gt_vis = img.copy()
        gt_vis[gt] = gt_vis[gt] * 0.5 + np.array([0, 0.5, 1]) * 0.5
        axes[row, 2].imshow(gt_vis)

        # Difference: TP=white, FP=red, FN=blue
        diff = np.zeros((*pred.shape, 3))
        diff[pred & gt] = [1, 1, 1]
        diff[pred & ~gt] = [1, 0, 0]
        diff[~pred & gt] = [0, 0.3, 1]
        axes[row, 3].imshow(diff)

    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontweight="bold")
    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    path = Path(save_dir) / "segmentation_grid.png"
    plt.savefig(path)
    plt.close()
    print(f"  ✅ Segmentation grid → {path}")


# ══════════════════════════════════════════════════════════════════════
# 2. Confusion matrix
# ══════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred, save_dir=None):
    save_dir = save_dir or FIG_DIR
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix", fontweight="bold")

    for i in range(3):
        for j in range(3):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    path = Path(save_dir) / "confusion_matrix.png"
    plt.savefig(path)
    plt.close()
    print(f"  ✅ Confusion matrix → {path}")


# ══════════════════════════════════════════════════════════════════════
# 3. ROC curves
# ══════════════════════════════════════════════════════════════════════
def plot_roc_curves(y_true, y_prob, save_dir=None):
    save_dir = save_dir or FIG_DIR
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#e74c3c", "#2ecc71", "#3498db"]

    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        binary = (y_true == i).astype(int)
        if binary.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(binary, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (per-class)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path = Path(save_dir) / "roc_curves.png"
    plt.savefig(path)
    plt.close()
    print(f"  ✅ ROC curves → {path}")


# ══════════════════════════════════════════════════════════════════════
# 4. Precision-Recall curves
# ══════════════════════════════════════════════════════════════════════
def plot_pr_curves(y_true, y_prob, save_dir=None):
    save_dir = save_dir or FIG_DIR
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#e74c3c", "#2ecc71", "#3498db"]

    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        binary = (y_true == i).astype(int)
        if binary.sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(binary, y_prob[:, i])
        ap = auc(recall, precision)
        ax.plot(recall, precision, color=color, lw=2, label=f"{name} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves", fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path = Path(save_dir) / "pr_curves.png"
    plt.savefig(path)
    plt.close()
    print(f"  ✅ PR curves → {path}")


# ══════════════════════════════════════════════════════════════════════
# 5. Training curves from JSONL logs
# ══════════════════════════════════════════════════════════════════════
def plot_training_curves(log_dir=None, save_dir=None):
    log_dir = log_dir or PROJECT_ROOT / "outputs" / "logs"
    save_dir = save_dir or FIG_DIR
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    stages = {"segmentation": "Stage 1: Segmentation",
              "temporal": "Stage 2: Temporal",
              "fusion": "Stage 3: Fusion"}
    found = False

    for stage_key, stage_title in stages.items():
        jsonl = Path(log_dir) / f"{stage_key}_metrics.jsonl"
        if not jsonl.exists():
            continue
        found = True

        steps, losses, val_metrics = [], [], {}
        with open(jsonl) as f:
            for line in f:
                d = json.loads(line.strip())
                step = d.get("step", len(steps))
                steps.append(step)
                for k, v in d.items():
                    if k == "step":
                        continue
                    if k not in val_metrics:
                        val_metrics[k] = []
                    val_metrics[k].append(v)

        n_metrics = len(val_metrics)
        if n_metrics == 0:
            continue
        cols = min(n_metrics, 4)
        rows = (n_metrics + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
        if n_metrics == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for ax, (name, values) in zip(axes, val_metrics.items()):
            ax.plot(steps[:len(values)], values, lw=1.5, color="#3498db")
            ax.set_title(name.replace("_", " ").title(), fontsize=10)
            ax.grid(True, alpha=0.2)
            ax.set_xlabel("Step", fontsize=9)
        for ax in axes[len(val_metrics):]:
            ax.set_visible(False)

        fig.suptitle(stage_title, fontweight="bold", fontsize=14)
        plt.tight_layout()
        path = Path(save_dir) / f"training_curves_{stage_key}.png"
        plt.savefig(path)
        plt.close()
        print(f"  ✅ Training curves ({stage_key}) → {path}")

    if not found:
        print("  ⚠ No JSONL logs found, skipping training curves")


# ══════════════════════════════════════════════════════════════════════
# 6. Per-class performance bars
# ══════════════════════════════════════════════════════════════════════
def plot_per_class_bars(results_json_path, save_dir=None):
    save_dir = save_dir or FIG_DIR
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if not Path(results_json_path).exists():
        print("  ⚠ No eval_results.json, skipping performance bars")
        return

    with open(results_json_path) as f:
        r = json.load(f)

    seg = r.get("segmentation", {})
    if not seg:
        print("  ⚠ No segmentation metrics in eval_results.json, skipping bars")
        return

    metrics = ["iou", "dice", "precision", "recall", "boundary_f1"]
    labels  = ["IoU", "Dice", "Precision", "Recall", "Boundary F1"]
    values  = [seg.get(m, 0) for m in metrics]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"]
    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor="white", linewidth=1)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    model_name = r.get("model", "")
    ax.set_title(f"Segmentation Performance — {model_name}", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    path = Path(save_dir) / "segmentation_bars.png"
    plt.savefig(path)
    plt.close()
    print(f"  ✅ Segmentation bars → {path}")


# ══════════════════════════════════════════════════════════════════════
# 7. Boundary quality visualization
# ══════════════════════════════════════════════════════════════════════
def plot_boundary_quality(model, device, n_samples=4, save_dir=None):
    save_dir = save_dir or FIG_DIR
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    ds = ThermalFrameDataset(split="val", img_size=(256, 320), augment=False)
    model.eval()

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    cols = ["Input", "Boundaries (Green=GT, Red=Pred)", "Overlap"]

    for row, idx in enumerate(np.linspace(0, len(ds) - 1, n_samples, dtype=int)):
        sample = ds[idx]
        overlay = sample["overlay"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)
        with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
            out = model(overlay, thermal_intensity=sample["thermal_intensity"].unsqueeze(0).to(device),
                        binary_mask=mask)
        pred = (torch.sigmoid(out["seg_logits"]) > 0.5).squeeze().cpu().numpy()
        gt = mask.squeeze().cpu().numpy().astype(bool)
        img = overlay.squeeze().permute(1, 2, 0).cpu().numpy()

        pred_bd = pred.astype(bool) & ~binary_erosion(pred, iterations=2)
        gt_bd = gt.astype(bool) & ~binary_erosion(gt, iterations=2)

        axes[row, 0].imshow(img)

        bd_vis = img.copy()
        bd_vis[gt_bd] = [0, 1, 0]
        bd_vis[pred_bd] = [1, 0, 0]
        axes[row, 1].imshow(bd_vis)

        overlap_vis = img.copy()
        overlap_vis[gt_bd & pred_bd] = [1, 1, 0]
        overlap_vis[gt_bd & ~pred_bd] = [0, 1, 0]
        overlap_vis[~gt_bd & pred_bd] = [1, 0, 0]
        axes[row, 2].imshow(overlap_vis)

    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontweight="bold")
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    path = Path(save_dir) / "boundary_quality.png"
    plt.savefig(path)
    plt.close()
    print(f"  ✅ Boundary quality → {path}")


# ══════════════════════════════════════════════════════════════════════
# 8. GradCAM / Attention heatmaps
# ══════════════════════════════════════════════════════════════════════
class GradCAM:
    """GradCAM for TRACE segmentation."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def __call__(self, overlay, thermal_intensity, binary_mask):
        self.model.zero_grad()
        out = self.model(overlay, thermal_intensity=thermal_intensity,
                         binary_mask=binary_mask)
        logits = out["seg_logits"]
        # Target: mean of positive logits
        target = torch.sigmoid(logits).mean()
        target.backward()

        # Weights: global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=overlay.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def plot_attention_heatmaps(model, device, n_samples=6, save_dir=None):
    """Generate GradCAM heatmaps like the reference image."""
    save_dir = save_dir or FIG_DIR
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    ds = ThermalFrameDataset(split="val", img_size=(256, 320), augment=False)

    model.eval()
    # Use stage 3 (deepest) and stage 4 for GradCAM
    try:
        target_stage3 = model.stage3
        target_stage4 = model.stage4
    except AttributeError:
        print("  ⚠ Could not find model stages, skipping heatmaps")
        return

    gradcam_deep = GradCAM(model, target_stage4)
    gradcam_mid = GradCAM(model, target_stage3)

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    cols = ["Input Thermal", "Mid-level Attention", "Deep Attention (GradCAM)"]

    for row, idx in enumerate(np.linspace(0, len(ds) - 1, n_samples, dtype=int)):
        sample = ds[idx]
        overlay = sample["overlay"].unsqueeze(0).to(device).requires_grad_(True)
        mask = sample["mask"].unsqueeze(0).to(device)
        intensity = sample["thermal_intensity"].unsqueeze(0).to(device)
        img_np = overlay.squeeze().detach().permute(1, 2, 0).cpu().numpy()

        # GradCAM at mid level (stage 3)
        cam_mid = gradcam_mid(overlay, intensity, mask)

        # Recompute for deep level
        overlay2 = sample["overlay"].unsqueeze(0).to(device).requires_grad_(True)
        cam_deep = gradcam_deep(overlay2, intensity, mask)

        axes[row, 0].imshow(img_np)

        # Mid-level heatmap overlay
        hm_mid = plt.cm.jet(cam_mid)[:, :, :3]
        blend_mid = img_np * 0.4 + hm_mid * 0.6
        axes[row, 1].imshow(np.clip(blend_mid, 0, 1))

        # Deep attention
        hm_deep = plt.cm.jet(cam_deep)[:, :, :3]
        blend_deep = img_np * 0.4 + hm_deep * 0.6
        axes[row, 2].imshow(np.clip(blend_deep, 0, 1))

    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontweight="bold")
    for ax in axes.flat:
        ax.axis("off")

    plt.suptitle("GradCAM Attention Heatmaps", fontweight="bold", fontsize=14)
    plt.tight_layout()
    path = Path(save_dir) / "attention_heatmaps.png"
    plt.savefig(path)
    plt.close()
    print(f"  ✅ Attention heatmaps → {path}")


# ══════════════════════════════════════════════════════════════════════
# 9. Qualitative result grid
# ══════════════════════════════════════════════════════════════════════
def plot_qualitative_grid(model, device, n_samples=6, save_dir=None):
    """Grid showing full pipeline: thermal → seg mask → class prediction."""
    save_dir = save_dir or FIG_DIR
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    ds = ThermalFrameDataset(split="val", img_size=(256, 320), augment=False)
    model.eval()

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 3 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    cols = ["Input Thermal", "Predicted Seg Mask", "GT Seg Mask", "Classification"]

    for row, idx in enumerate(np.linspace(0, len(ds) - 1, n_samples, dtype=int)):
        sample = ds[idx]
        overlay = sample["overlay"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)
        intensity = sample["thermal_intensity"].unsqueeze(0).to(device)
        class_id = sample["class_id"]

        with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
            out = model(overlay, thermal_intensity=intensity, binary_mask=mask)

        pred_mask = (torch.sigmoid(out["seg_logits"]) > 0.5).squeeze().cpu().numpy()
        img = overlay.squeeze().permute(1, 2, 0).cpu().numpy()
        gt = mask.squeeze().cpu().numpy()

        axes[row, 0].imshow(img)
        axes[row, 1].imshow(pred_mask, cmap="gray")
        axes[row, 2].imshow(gt, cmap="gray")

        # Classification label
        axes[row, 3].text(0.5, 0.5, f"GT: {CLASS_NAMES[class_id]}",
                          ha="center", va="center", fontsize=14,
                          fontweight="bold", transform=axes[row, 3].transAxes,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[row, 3].set_facecolor("#f0f0f0")

    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontweight="bold")
    for ax in axes.flat:
        ax.axis("off")

    plt.suptitle("TRACE Qualitative Results", fontweight="bold", fontsize=14)
    plt.tight_layout()
    path = Path(save_dir) / "qualitative_grid.png"
    plt.savefig(path)
    plt.close()
    print(f"  ✅ Qualitative grid → {path}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="TRACE Visualization")
    parser.add_argument("--seg_checkpoint", type=str, default=None)
    parser.add_argument("--raw_npz", type=str, default=None,
                        help="Path to classification_raw.npz from evaluate.py")
    parser.add_argument("--results_json", type=str,
                        default=str(PROJECT_ROOT / "outputs" / "eval_results" / "eval_results.json"))
    parser.add_argument("--output_dir", type=str, default=str(FIG_DIR))
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name (overrides MODEL_NAME env var)")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory containing *_metrics.jsonl training logs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig_dir = Path(args.output_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  TRACE — Generating CVPR Visualizations")
    print("=" * 60)

    # 5. Training curves (no checkpoint needed)
    print("\n📈 Training curves...")
    plot_training_curves(log_dir=args.log_dir, save_dir=fig_dir)

    # 6. Per-class performance bars
    print("\n📊 Per-class performance bars...")
    plot_per_class_bars(args.results_json, save_dir=fig_dir)

    # Segmentation-based visualizations
    if args.seg_checkpoint:
        print("\n🔬 Loading segmentation model...")
        _model_name = args.model_name or os.environ.get("MODEL_NAME", "trace")
        model = create_model(_model_name, num_seg_classes=1, decode_dim=256, use_aux_mask=True).to(device)
        ckpt = torch.load(args.seg_checkpoint, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        model.eval()

        print("\n🖼️  Segmentation grid...")
        plot_segmentation_grid(model, device, save_dir=fig_dir)

        print("\n🔲 Boundary quality...")
        plot_boundary_quality(model, device, save_dir=fig_dir)

        print("\n🔥 GradCAM attention heatmaps...")
        plot_attention_heatmaps(model, device, save_dir=fig_dir)

        print("\n📋 Qualitative grid...")
        plot_qualitative_grid(model, device, save_dir=fig_dir)

        del model
        torch.cuda.empty_cache()

    # Classification-based visualizations
    if args.raw_npz and Path(args.raw_npz).exists():
        print("\n📊 Loading classification results...")
        raw = np.load(args.raw_npz)
        y_true, y_pred, y_prob = raw["y_true"], raw["y_pred"], raw["y_prob"]

        print("  Confusion matrix...")
        plot_confusion_matrix(y_true, y_pred, save_dir=fig_dir)

        print("  ROC curves...")
        plot_roc_curves(y_true, y_prob, save_dir=fig_dir)

        print("  PR curves...")
        plot_pr_curves(y_true, y_prob, save_dir=fig_dir)

    print(f"\n🎉 All figures saved to: {fig_dir}")
    print(f"   Total figures: {len(list(fig_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
