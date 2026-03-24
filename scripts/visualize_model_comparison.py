"""Multi-model segmentation comparison visualization — ALL test frames.

Produces one PNG per 4 test frames. Each PNG has 4 rows x (2 + n_models) cols:
  Col 0 : Original image  [filename shown as row label]
  Col 1 : Ground truth overlay  (blue)
  Col 2+ : Per-model prediction overlay  (green)

Output: all_figures/model_comparison_all/page_0001.png, page_0002.png, ...

Usage:
    
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \\
        python scripts/visualize_model_comparison.py
    # options:
    #   --split test|val   --frames_per_page 4   --out_dir all_figures/model_comparison_all
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ThermalFrameDataset
from src.models.model_factory import create_model

# ── Model registry ────────────────────────────────────────────────────────
# (display_label, factory_name, checkpoint_path)
MODELS = [
    (
        "Mask2Former",
        "mask2former",
        PROJECT_ROOT / "Exp2_Mask2Former/checkpoints/segmentation/segmentation_latest.pt",
    ),
    (
        "SegFormer-B0",
        "segformer_b0",
        PROJECT_ROOT / "Exp4_SegFormerB0/checkpoints/segmentation/segmentation_latest.pt",
    ),
    (
        "Prior2Former",
        "prior2former",
        PROJECT_ROOT / "Exp5_Prior2Former/checkpoints/segmentation/segmentation_latest.pt",
    ),
    (
        "RepViT-M1",
        "repvit_m1",
        PROJECT_ROOT / "Exp08_repvit_m1/checkpoints/segmentation/segmentation_latest.pt",
    ),
    (
        "SHViT-S4",
        "shvit_s4",
        PROJECT_ROOT / "Exp09_shvit_s4/checkpoints/segmentation/segmentation_latest.pt",
    ),
    (
        "StarNet-S2",
        "starnet_s2",
        PROJECT_ROOT / "Exp10_starnet_s2/checkpoints/segmentation/segmentation_latest.pt",
    ),
    (
        "MobileNetV4-S",
        "mobilenetv4_conv_s",
        PROJECT_ROOT / "Exp11_mobilenetv4_conv_s/checkpoints/segmentation/segmentation_latest.pt",
    ),
]

# CVPR-quality style
plt.rcParams.update({
    "font.family": "serif",
    "font.size":   8,
    "axes.titlesize": 8,
    "figure.dpi":  150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

PRED_COLOR = np.array([0.08, 0.85, 0.20], dtype=np.float32)   # green
GT_COLOR   = np.array([0.10, 0.50, 1.00], dtype=np.float32)   # blue


# ── Helpers ───────────────────────────────────────────────────────────────

def load_model(label, factory_name, ckpt_path, device):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"  [SKIP] {label}: checkpoint not found")
        return None
    print(f"  Loading {label} ...", flush=True)
    model = create_model(factory_name, num_seg_classes=1, decode_dim=256,
                         use_aux_mask=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def predict(model, sample, device):
    overlay   = sample["overlay"].unsqueeze(0).to(device)
    mask      = sample["mask"].unsqueeze(0).to(device)
    intensity = sample["thermal_intensity"].unsqueeze(0).to(device)
    try:
        with autocast("cuda", dtype=torch.bfloat16):
            out = model(overlay, thermal_intensity=intensity, binary_mask=mask)
    except Exception:
        out = model(overlay, thermal_intensity=intensity, binary_mask=mask)
    return (torch.sigmoid(out["seg_logits"]) > 0.5).squeeze().cpu().numpy().astype(bool)


def blend(img_rgb, mask, color, alpha=0.55):
    vis = img_rgb.copy()
    vis[mask] = vis[mask] * (1 - alpha) + color * alpha
    return np.clip(vis, 0, 1)


def render_page(rows_data, loaded_models, out_path):
    """
    rows_data: list of (frame_id, img_rgb, gt_mask, list_of_preds)
    loaded_models: list of (label, model_or_None)
    """
    col_headers = ["Original", "Ground\nTruth"] + [lbl for lbl, _ in loaded_models]
    n_cols = len(col_headers)
    n_rows = len(rows_data)

    col_w = 2.2
    row_h = 2.2
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(col_w * n_cols, row_h * n_rows),
                             squeeze=False)

    # Column headers on first row only
    for c, hdr in enumerate(col_headers):
        axes[0, c].set_title(hdr, fontsize=7, fontweight="bold", pad=3)

    for r, (frame_id, img_rgb, gt_mask, preds) in enumerate(rows_data):
        # Row label = filename on the original image column
        axes[r, 0].set_ylabel(frame_id, fontsize=5.5, labelpad=3,
                               rotation=0, ha="right", va="center")

        axes[r, 0].imshow(img_rgb)
        axes[r, 1].imshow(blend(img_rgb, gt_mask, GT_COLOR))

        for c, ((_label, model), pred) in enumerate(zip(loaded_models, preds)):
            ax = axes[r, c + 2]
            if pred is None:
                ax.imshow(np.zeros_like(img_rgb))
                ax.text(0.5, 0.5, "No ckpt", ha="center", va="center",
                        fontsize=6, color="white", transform=ax.transAxes)
            else:
                ax.imshow(blend(img_rgb, pred, PRED_COLOR))

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout(pad=0.2, h_pad=0.4, w_pad=0.2)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",           default="test")
    parser.add_argument("--frames_per_page", type=int, default=4)
    parser.add_argument("--out_dir",         default=str(PROJECT_ROOT / "all_figures" / "model_comparison_all"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    ds = ThermalFrameDataset(split=args.split, img_size=(256, 320), augment=False)
    n_total = len(ds)
    print(f"Split '{args.split}': {n_total} frames")
    n_pages = math.ceil(n_total / args.frames_per_page)
    print(f"Frames per page: {args.frames_per_page}  →  {n_pages} pages\n")

    # Load all models once
    print("Loading models...")
    loaded_models = []
    for label, factory, ckpt in MODELS:
        m = load_model(label, factory, ckpt, device)
        loaded_models.append((label, m))
    print()

    # Iterate pages
    for page_idx in range(n_pages):
        start = page_idx * args.frames_per_page
        end   = min(start + args.frames_per_page, n_total)
        page_indices = list(range(start, end))

        rows_data = []
        for ds_idx in page_indices:
            sample   = ds[ds_idx]
            img_bgr  = sample["overlay"].permute(1, 2, 0).numpy()
            img_rgb  = img_bgr[..., ::-1].copy()
            gt_mask  = sample["mask"].squeeze().numpy().astype(bool)
            frame_id = sample.get("frame_id", str(ds_idx))

            preds = []
            for _label, model in loaded_models:
                if model is None:
                    preds.append(None)
                else:
                    preds.append(predict(model, sample, device))

            rows_data.append((frame_id, img_rgb, gt_mask, preds))

        out_path = out_dir / f"page_{page_idx + 1:04d}.png"
        render_page(rows_data, loaded_models, out_path)

        if (page_idx + 1) % 10 == 0 or (page_idx + 1) == n_pages:
            print(f"  Page {page_idx + 1}/{n_pages}  →  {out_path.name}", flush=True)

    print(f"\nDone. {n_pages} PNGs saved to: {out_dir}")


if __name__ == "__main__":
    main()
