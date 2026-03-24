"""Segmentation comparison visualization with thermal-colored predictions.

Layout (one row per frame):
  Col 0 : Original image
  Col 1 : Thermal intensity map  (inferno colormap)
  Col 2 : Ground truth overlay   (blue)
  Col 3+ : Per-model prediction  (inferno/lava colormap on predicted pixels)

HOW TO USE ON A NEW DEVICE
---------------------------
1. Set PROJECT_ROOT to your project directory.
2. Edit the MODELS list — add/remove (label, factory_name, checkpoint_path) tuples.
3. Edit TARGET_FRAMES to the frame IDs you want.
4. Run:
       python visualize_selected_frames.py
   or with options:
       python visualize_selected_frames.py --out my_output.png --dpi 200

ADDING A NEW MODEL
------------------
Append a tuple to MODELS:
    ("My Model", "factory_name", "/path/to/segmentation_latest.pt"),

The factory_name must be one registered in src/models/model_factory.py:
    trace, trace_tiny, segformer_b2, segformer_b0,
    mask2former, prior2former, iformer, lactnet, sam2,
    repvit_m1, shvit_s4, starnet_s2, mobilenetv4_conv_s
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch

# ── Configure these for your device ──────────────────────────────────────

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))

# (display label, model factory name, path to segmentation checkpoint)
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

# Frame IDs to visualize (must exist in annotations.csv)
TARGET_FRAMES = [
    "SEQ_0492_frame_00156",
    "SEQ_0501_frame_00062",
    "SEQ_0501_frame_00068",
    "SEQ_0501_frame_00450",
    "SEQ_0501_frame_01566",
    "SEQ_0501_frame_01767",
    "SEQ_0501_frame_02423",
]

# ─────────────────────────────────────────────────────────────────────────

GT_COLOR = np.array([0.10, 0.50, 1.00], dtype=np.float32)   # blue for GT


def load_model(label, factory_name, ckpt_path, device):
    """Load a segmentation model from checkpoint. Returns None if not found."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"  [SKIP] {label}: checkpoint not found at {ckpt_path}")
        return None

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.models.model_factory import create_model

    print(f"  Loading {label} ...", flush=True)
    model = create_model(factory_name, num_seg_classes=1,
                         decode_dim=256, use_aux_mask=False).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def predict(model, sample, device):
    """Run inference and return binary prediction mask (H, W bool)."""
    overlay   = sample["overlay"].unsqueeze(0).to(device)
    mask      = sample["mask"].unsqueeze(0).to(device)
    intensity = sample["thermal_intensity"].unsqueeze(0).to(device)
    try:
        from torch.amp import autocast
        with autocast("cuda", dtype=torch.bfloat16):
            out = model(overlay, thermal_intensity=intensity, binary_mask=mask)
    except Exception:
        out = model(overlay, thermal_intensity=intensity, binary_mask=mask)
    return (torch.sigmoid(out["seg_logits"]) > 0.5).squeeze().cpu().numpy().astype(bool)


def blend_gt(img_rgb, mask, color, alpha=0.55):
    """Flat color blend for ground truth."""
    vis = img_rgb.copy()
    vis[mask] = vis[mask] * (1 - alpha) + color * alpha
    return np.clip(vis, 0, 1)


def thermal_overlay(img_rgb, pred_mask, thermal, alpha=0.75):
    """
    Predicted pixels : colored by thermal intensity through inferno colormap (lava look).
    Non-predicted    : original image dimmed to make predictions pop.
    """
    vis = img_rgb.copy()
    vis[~pred_mask] = vis[~pred_mask] * 0.55      # dim background

    if pred_mask.any():
        lava = cm.inferno(thermal)[:, :, :3]       # (H, W, 3)
        vis[pred_mask] = (img_rgb[pred_mask] * (1 - alpha)
                          + lava[pred_mask] * alpha)
    return np.clip(vis, 0, 1)


def build_frame_index(splits, img_size=(256, 320)):
    """Build frame_id → (split, dataset_index) mapping across multiple splits."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.data.dataset import ThermalFrameDataset

    index = {}     # frame_id -> (dataset_object, local_idx)
    for split in splits:
        try:
            ds = ThermalFrameDataset(split=split, img_size=img_size, augment=False)
            for i in range(len(ds)):
                fid = ds[i]["frame_id"]
                if fid not in index:
                    index[fid] = (ds, i)
        except Exception as e:
            print(f"  Warning: could not load split '{split}': {e}")
    return index


def main():
    parser = argparse.ArgumentParser(description="Thermal segmentation comparison visualization")
    parser.add_argument("--out",    default=str(PROJECT_ROOT / "all_figures" / "selected_frames_thermal.png"),
                        help="Output PNG path")
    parser.add_argument("--dpi",    type=int, default=200, help="Output DPI")
    parser.add_argument("--alpha",  type=float, default=0.75,
                        help="Blend strength for thermal overlay (0=original, 1=full lava)")
    parser.add_argument("--splits", nargs="+", default=["test", "val", "train"],
                        help="Dataset splits to search for frame IDs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Output : {args.out}\n")

    # Build frame index
    print("Indexing frames ...")
    frame_index = build_frame_index(args.splits)
    print(f"  {len(frame_index)} frames indexed across splits: {args.splits}\n")

    # Verify target frames exist
    missing = [f for f in TARGET_FRAMES if f not in frame_index]
    if missing:
        print(f"WARNING: {len(missing)} frame(s) not found in dataset: {missing}")

    # Load models
    print("Loading models ...")
    loaded_models = []
    for label, factory, ckpt in MODELS:
        m = load_model(label, factory, ckpt, device)
        loaded_models.append((label, m))
    n_available = sum(1 for _, m in loaded_models if m is not None)
    print(f"  {n_available}/{len(MODELS)} models loaded\n")

    # Figure setup
    plt.rcParams.update({
        "font.family": "serif",
        "font.size":   7,
        "axes.titlesize": 7,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    col_headers = ["Original", "Thermal\nIntensity", "Ground\nTruth"] \
                  + [lbl for lbl, _ in loaded_models]
    n_cols  = len(col_headers)
    n_rows  = len(TARGET_FRAMES)
    col_w   = 2.1
    row_h   = 2.1

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(col_w * n_cols, row_h * n_rows),
                             squeeze=False)

    for c, hdr in enumerate(col_headers):
        axes[0, c].set_title(hdr, fontsize=6, fontweight="bold", pad=3)

    for r, frame_id in enumerate(TARGET_FRAMES):
        if frame_id not in frame_index:
            print(f"  [SKIP] {frame_id} not found — filling with blank row")
            for ax in axes[r]:
                ax.imshow(np.zeros((256, 320, 3)))
                ax.axis("off")
            continue

        ds, idx = frame_index[frame_id]
        sample   = ds[idx]
        img_rgb  = sample["overlay"].permute(1, 2, 0).numpy()[..., ::-1].copy()
        gt_mask  = sample["mask"].squeeze().numpy().astype(bool)
        thermal  = sample["thermal_intensity"].squeeze().numpy()   # (H, W) in [0,1]

        axes[r, 0].set_ylabel(frame_id, fontsize=4.5, labelpad=3,
                               rotation=0, ha="right", va="center")

        # Col 0: Original
        axes[r, 0].imshow(img_rgb)

        # Col 1: Thermal intensity heatmap
        axes[r, 1].imshow(thermal, cmap="inferno", vmin=0, vmax=1)

        # Col 2: Ground truth
        axes[r, 2].imshow(blend_gt(img_rgb, gt_mask, GT_COLOR))

        # Col 3+: Model predictions (thermal/lava color)
        for c_idx, (label, model) in enumerate(loaded_models):
            ax = axes[r, c_idx + 3]
            if model is None:
                ax.imshow(np.zeros_like(img_rgb))
                ax.text(0.5, 0.5, "No\nCheckpoint",
                        ha="center", va="center", fontsize=5,
                        color="white", transform=ax.transAxes)
            else:
                pred = predict(model, sample, device)
                ax.imshow(thermal_overlay(img_rgb, pred, thermal, alpha=args.alpha))

        print(f"  Frame {r+1}/{n_rows}: {frame_id}", flush=True)

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(
        "Segmentation Comparison — Predicted Gas colored by Thermal Intensity (inferno)",
        fontsize=9, fontweight="bold", y=1.002,
    )
    plt.tight_layout(pad=0.15, h_pad=0.4, w_pad=0.2)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=args.dpi)
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
