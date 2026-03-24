"""Quantitative evaluation on the test split.

Runs all CVPR-quality metrics and saves results to JSON + console table.
Usage:
    python src/eval/evaluate.py \
        --seg_checkpoint outputs/checkpoints/segmentation/segmentation_latest.pt \
        --temporal_checkpoint outputs/checkpoints/temporal/temporal_latest.pt \
        --fusion_checkpoint outputs/checkpoints/fusion/fusion_latest.pt
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                              confusion_matrix, f1_score, precision_score,
                              recall_score, roc_auc_score, roc_curve,
                              precision_recall_curve, average_precision_score)
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import ThermalFrameDataset, ThermalClipDataset
from src.models.model_factory import create_model
from src.models.temporal_encoder import TemporalEncoder
from src.models.atf import AsymmetricThermalFusion
from src.utils.config import SegmentationConfig, TemporalConfig, FusionConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "eval_results"
CLASS_NAMES = ["High-Flux (HF)", "Control", "Low-Flux (LF)"]


# ── Segmentation metrics ─────────────────────────────────────────────
def _boundary(m):
    return m.astype(bool) & ~binary_erosion(m, iterations=2)


def compute_seg_metrics(pred, gt):
    """Compute all segmentation metrics for a single sample."""
    s = 1e-6
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    iou = tp / (tp + fp + fn + s)
    dice = 2 * tp / (2 * tp + fp + fn + s)
    prec = tp / (tp + fp + s)
    rec = tp / (tp + fn + s)

    pb, gb = _boundary(pred), _boundary(gt)
    bf1 = 2 * (pb & gb).sum() / (pb.sum() + gb.sum() + s)

    pp, gp = np.argwhere(pred), np.argwhere(gt)
    if len(pp) > 0 and len(gp) > 0:
        hd = max(directed_hausdorff(pp, gp)[0], directed_hausdorff(gp, pp)[0])
        ce = np.linalg.norm(pp.mean(0) - gp.mean(0))
    else:
        hd = ce = float(max(pred.shape))

    pred_area = pred.sum()
    gt_area = gt.sum() + s
    plume_rel_err = abs(pred_area - gt_area) / gt_area

    return {
        "iou": float(iou), "dice": float(dice), "precision": float(prec),
        "recall": float(rec), "boundary_f1": float(bf1),
        "hausdorff": float(hd), "centroid_err": float(ce),
        "plume_area_rel_err": float(plume_rel_err),
    }


# ── Main evaluation ──────────────────────────────────────────────────
@torch.no_grad()
def evaluate_segmentation(model, device, img_size=(256, 320)):
    """Evaluate segmentation on test split."""
    ds = ThermalFrameDataset(split="test", img_size=img_size, augment=False)
    if len(ds) == 0:
        ds = ThermalFrameDataset(split="val", img_size=img_size, augment=False)
        print("  ⚠ No test split, using val split")
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    all_metrics = {k: [] for k in ["iou", "dice", "precision", "recall",
                                    "boundary_f1", "hausdorff", "centroid_err",
                                    "plume_area_rel_err"]}
    for batch in loader:
        overlay = batch["overlay"].to(device)
        mask = batch["mask"].to(device)
        with autocast(dtype=torch.bfloat16):
            out = model(overlay, thermal_intensity=batch["thermal_intensity"].to(device),
                        binary_mask=mask)
        pred = (torch.sigmoid(out["seg_logits"]) > 0.5).squeeze(1).cpu().numpy().astype(bool)
        gt = mask.squeeze(1).cpu().numpy().astype(bool)
        for p, g in zip(pred, gt):
            m = compute_seg_metrics(p, g)
            for k in all_metrics:
                all_metrics[k].append(m[k])

    results = {f"seg/{k}": float(np.mean(v)) for k, v in all_metrics.items()}
    results["seg/n_samples"] = len(all_metrics["iou"])
    return results


@torch.no_grad()
def evaluate_classification(model, device, model_type="temporal"):
    """Evaluate classification on test split."""
    if model_type == "temporal":
        ds = ThermalClipDataset(split="test")
        if len(ds) == 0:
            ds = ThermalClipDataset(split="val")
            print("  ⚠ No test split for clips, using val split")
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)
    else:
        ds = ThermalFrameDataset(split="test", img_size=(256, 320), augment=False)
        if len(ds) == 0:
            ds = ThermalFrameDataset(split="val", img_size=(256, 320), augment=False)
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)

    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        if model_type == "temporal":
            clip = batch["clip_overlays"].to(device)
            labels = batch["class_id"]
            with autocast(dtype=torch.bfloat16):
                out = model(clip, return_cls_logits=True)
            logits = out["cls_logits"]
        else:
            overlay = batch["overlay"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["class_id"]
            with autocast(dtype=torch.bfloat16):
                out = model(overlay, binary_mask=mask)
            logits = out.get("cls_logits", out.get("seg_logits"))

        probs = torch.softmax(logits.float(), dim=1)
        all_preds.extend(probs.argmax(1).cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    results = {
        "cls/accuracy": float((y_pred == y_true).mean()),
        "cls/balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cls/macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cls/cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "cls/n_samples": len(y_true),
    }

    # Per-class metrics
    for i, name in enumerate(CLASS_NAMES):
        results[f"cls/precision_{name}"] = float(precision_score(y_true, y_pred,
                                                                  labels=[i], average="macro", zero_division=0))
        results[f"cls/recall_{name}"] = float(recall_score(y_true, y_pred,
                                                            labels=[i], average="macro", zero_division=0))

    try:
        results["cls/macro_auc_roc"] = float(roc_auc_score(y_true, y_prob,
                                                            multi_class="ovr", average="macro"))
    except ValueError:
        results["cls/macro_auc_roc"] = float("nan")

    # Save raw arrays for visualization
    results["_raw"] = {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}
    return results


def print_results(results: dict, title: str = ""):
    """Print results as a formatted table."""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
    for k, v in sorted(results.items()):
        if k.startswith("_"):
            continue
        if isinstance(v, float):
            print(f"  {k:35s} {v:.4f}")
        else:
            print(f"  {k:35s} {v}")
    print()


def main():
    parser = argparse.ArgumentParser(description="TRACE Evaluation")
    parser.add_argument("--seg_checkpoint", type=str, default=None)
    parser.add_argument("--temporal_checkpoint", type=str, default=None)
    parser.add_argument("--fusion_checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name for factory (reads MODEL_NAME env if not set)")
    args = parser.parse_args()

    import os
    model_name = args.model_name or os.environ.get("MODEL_NAME", "trace")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # ── Segmentation ──
    if args.seg_checkpoint:
        print(f"\n🔬 Evaluating Segmentation ({model_name})...")
        model = create_model(model_name, num_seg_classes=1, decode_dim=256,
                             use_aux_mask=True).to(device)
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        ckpt = torch.load(args.seg_checkpoint, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        model.eval()
        seg_results = evaluate_segmentation(model, device)
        print_results(seg_results, "Segmentation Metrics")
        all_results.update(seg_results)
        all_results["params_m"] = round(total_params, 1)
        del model
        torch.cuda.empty_cache()

    # ── Temporal Classification ──
    if args.temporal_checkpoint:
        print("\n🕐 Evaluating Temporal Classification...")
        model = TemporalEncoder(output_dim=256).to(device)
        ckpt = torch.load(args.temporal_checkpoint, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        model.eval()
        cls_results = evaluate_classification(model, device, model_type="temporal")
        raw = cls_results.pop("_raw", None)
        print_results(cls_results, "Temporal Classification Metrics")
        all_results.update(cls_results)

        # Save confusion matrix and ROC data
        if raw is not None:
            np.savez(str(output_dir / "classification_raw.npz"),
                     y_true=raw["y_true"], y_pred=raw["y_pred"], y_prob=raw["y_prob"])
        del model
        torch.cuda.empty_cache()

    # ── Save JSON (structured for comparison table) ──
    serializable = {k: v for k, v in all_results.items() if not k.startswith("_")}
    structured = {"model": model_name, "params_m": serializable.get("params_m", "?")}
    seg_dict, cls_dict = {}, {}
    for k, v in serializable.items():
        if k.startswith("seg/"):
            seg_dict[k.replace("seg/", "")] = v
        elif k.startswith("cls/"):
            cls_dict[k.replace("cls/", "")] = v
    structured["segmentation"] = seg_dict
    structured["classification"] = cls_dict
    json_path = output_dir / "eval_results.json"
    with open(json_path, "w") as f:
        json.dump(structured, f, indent=2)
    print(f"\n📄 Results saved to: {json_path}")


if __name__ == "__main__":
    main()
