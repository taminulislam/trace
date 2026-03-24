"""Stage 1: TGAA-TRACE Segmentation Pretraining.

Sub-stage 1a: Freeze backbone, train TGAA gates + decode head (20 epochs, lr=6e-5)
Sub-stage 1b: Unfreeze all, full fine-tune (30 epochs, lr=1e-5)

Loss: 0.5*BCE + 0.5*Dice
Input: thermal overlay (3, 256, 320) + thermal_intensity (1, 256, 320)
Target: binary gas mask (1, 256, 320)

Metrics (CVPR-quality):
  Spatial:     mIoU, Dice/F1, Boundary-F1, Hausdorff Distance
  Detection:   Precision, Recall
  Domain:      Gas Centroid Error (px)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import directed_hausdorff
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import ThermalFrameDataset
from src.models.trace_model import TRACE
from src.utils.config import SegmentationConfig
from src.utils.trainer import (CheckpointManager, ETATracker, MetricsLogger,
                                cleanup_ddp, get_cosine_schedule_with_warmup,
                                get_sampler, is_main_process, print_header,
                                set_seed, setup_ddp, unwrap_model,
                                wrap_model_ddp)


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0):
    pred_sig = torch.sigmoid(pred)
    intersection = (pred_sig * target).sum(dim=(2, 3))
    union = pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1.0 - ((2.0 * intersection + smooth) / (union + smooth)).mean()


def combined_loss(pred, target, bce_w=0.5, dice_w=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dl = dice_loss(pred, target)
    return bce_w * bce + dice_w * dl, {"bce": bce.item(), "dice": dl.item()}


def compute_boundary(mask_np: np.ndarray) -> np.ndarray:
    """Return boundary pixels of a binary mask (erosion diff)."""
    eroded = binary_erosion(mask_np, iterations=1)
    return mask_np.astype(bool) & ~eroded


def compute_seg_metrics(pred_np: np.ndarray, gt_np: np.ndarray):
    """Compute rich segmentation metrics on CPU numpy arrays (H,W) binary.

    Returns dict with: dice, precision, recall, boundary_f1, hausdorff, centroid_err
    """
    smooth = 1e-6
    tp = (pred_np & gt_np).sum()
    fp = (pred_np & ~gt_np).sum()
    fn = (~pred_np & gt_np).sum()

    precision = tp / (tp + fp + smooth)
    recall    = tp / (tp + fn + smooth)
    dice      = 2 * tp / (2 * tp + fp + fn + smooth)

    # Boundary F1
    pred_b = compute_boundary(pred_np)
    gt_b   = compute_boundary(gt_np)
    tp_b   = (pred_b & gt_b).sum()
    boundary_f1 = (2 * tp_b / (pred_b.sum() + gt_b.sum() + smooth))

    # Hausdorff distance (95th percentile approximation via max directed)
    pred_pts = np.argwhere(pred_np)
    gt_pts   = np.argwhere(gt_np)
    if len(pred_pts) > 0 and len(gt_pts) > 0:
        hd = max(directed_hausdorff(pred_pts, gt_pts)[0],
                 directed_hausdorff(gt_pts, pred_pts)[0])
    else:
        hd = float(pred_np.shape[0])  # penalise empty predictions

    # Gas centroid error (Euclidean distance in pixels)
    if gt_np.sum() > 0 and pred_np.sum() > 0:
        gt_cy, gt_cx = np.argwhere(gt_np).mean(axis=0)
        pr_cy, pr_cx = np.argwhere(pred_np).mean(axis=0)
        centroid_err = float(np.sqrt((gt_cy - pr_cy)**2 + (gt_cx - pr_cx)**2))
    else:
        centroid_err = float(max(pred_np.shape))

    return {
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "boundary_f1": float(boundary_f1),
        "hausdorff": float(hd),
        "centroid_err": float(centroid_err),
    }


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device,
                    config, epoch, global_step, eta, logger, sampler):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)
    total_loss = 0.0
    for i, batch in enumerate(loader):
        overlay = batch["overlay"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        intensity = batch["thermal_intensity"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(dtype=torch.bfloat16):
            out = model(overlay, thermal_intensity=intensity,
                        binary_mask=mask if config.use_aux_mask else None)
            loss, loss_parts = combined_loss(out["seg_logits"], mask,
                                             config.bce_weight, config.dice_weight)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        global_step += 1
        total_loss += loss.item()

        if global_step % config.log_every_n_steps == 0 and is_main_process():
            eta_str, elapsed = eta.step(global_step)
            lr = optimizer.param_groups[0]["lr"]
            print(f"  [E{epoch:02d} S{global_step:05d}] "
                  f"loss={loss.item():.4f} bce={loss_parts['bce']:.4f} "
                  f"dice={loss_parts['dice']:.4f} lr={lr:.2e} "
                  f"ETA={eta_str} elapsed={elapsed}")
            logger.log({"train/loss": loss.item(),
                         **{f"train/{k}": v for k, v in loss_parts.items()},
                         "train/lr": lr}, step=global_step)

    avg_loss = total_loss / len(loader)
    return avg_loss, global_step


@torch.no_grad()
def validate(model, loader, device, config):
    model.eval()
    total_loss = 0.0
    accum = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0,
             "boundary_f1": 0.0, "hausdorff": 0.0, "centroid_err": 0.0}
    n = 0
    for batch in loader:
        overlay = batch["overlay"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        intensity = batch["thermal_intensity"].to(device, non_blocking=True)

        with autocast(dtype=torch.bfloat16):
            raw = unwrap_model(model)
            out = raw(overlay, thermal_intensity=intensity,
                      binary_mask=mask if config.use_aux_mask else None)
            loss, _ = combined_loss(out["seg_logits"], mask,
                                    config.bce_weight, config.dice_weight)

        pred = (torch.sigmoid(out["seg_logits"]) > 0.5).float()

        # mIoU (GPU)
        intersection = (pred * mask).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + mask.sum(dim=(2, 3)) - intersection
        accum["iou"] += (intersection / (union + 1e-6)).mean().item()

        # Rich metrics (CPU, per image in batch then averaged)
        pred_np = pred.squeeze(1).cpu().numpy().astype(bool)  # (B,H,W)
        gt_np   = mask.squeeze(1).cpu().numpy().astype(bool)
        batch_metrics = {k: 0.0 for k in ("dice","precision","recall",
                                           "boundary_f1","hausdorff","centroid_err")}
        for p, g in zip(pred_np, gt_np):
            m = compute_seg_metrics(p, g)
            for k in batch_metrics:
                batch_metrics[k] += m[k]
        B = len(pred_np)
        for k in batch_metrics:
            accum[k] += batch_metrics[k] / B

        total_loss += loss.item()
        n += 1

    return {
        "val/loss":         total_loss / n,
        "val/mIoU":         accum["iou"] / n,
        "val/Dice":         accum["dice"] / n,
        "val/Precision":    accum["precision"] / n,
        "val/Recall":       accum["recall"] / n,
        "val/BoundaryF1":   accum["boundary_f1"] / n,
        "val/Hausdorff":    accum["hausdorff"] / n,
        "val/CentroidErr":  accum["centroid_err"] / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()

    rank, world_size, device = setup_ddp()
    config = SegmentationConfig()
    set_seed(config.seed + rank)
    print_header("Stage 1: Segmentation Pretraining", config)
    if is_main_process():
        print(f"World size: {world_size} GPUs")

    # Data
    train_ds = ThermalFrameDataset(split="train", img_size=config.img_size, augment=True)
    val_ds = ThermalFrameDataset(split="val", img_size=config.img_size, augment=False)
    train_sampler = get_sampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=config.num_workers, pin_memory=config.pin_memory,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=config.pin_memory)
    if is_main_process():
        print(f"Train: {len(train_ds)} frames, Val: {len(val_ds)} frames")

    # Model — use factory to support different backbones
    import os
    model_name = os.environ.get("MODEL_NAME", "trace")
    from src.models.model_factory import create_model
    model = create_model(model_name, num_seg_classes=1, decode_dim=config.decode_dim,
                         use_aux_mask=config.use_aux_mask)
    if is_main_process():
        total_p = sum(p.numel() for p in model.parameters())
        print(f"Model: {model_name} ({total_p/1e6:.1f}M params)")
    model = wrap_model_ddp(model, device)
    scaler = GradScaler()
    ckpt_mgr = CheckpointManager(config.checkpoint_dir, config.stage_name,
                                  keep_last_n=config.keep_last_n_checkpoints)
    logger = MetricsLogger(config.log_dir, config.stage_name,
                           use_wandb=config.use_wandb,
                           wandb_project=config.wandb_project)

    # ---- Sub-stage 1a: freeze backbone, train TGAA gates + decode head ----
    if is_main_process():
        print("\n--- Sub-stage 1a: Freeze backbone ---")
    raw_model = unwrap_model(model)
    for name, param in raw_model.named_parameters():
        # For TRACE: keep TGAA + decode_head + mask_encoder trainable
        # For other models: keep only decode_head trainable (freeze backbone)
        if model_name == "trace":
            if "tgaa" not in name.lower() and "decode_head" not in name and "mask_encoder" not in name:
                param.requires_grad = False
        else:
            if "decode_head" not in name:
                param.requires_grad = False
    trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in raw_model.parameters())
    if is_main_process():
        print(f"Trainable: {trainable:,} / {total:,} params")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.substage_1a_lr, weight_decay=config.weight_decay)
    total_steps_1a = config.substage_1a_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, total_steps_1a)
    eta = ETATracker(total_steps_1a)

    global_step = 0
    start_epoch = 0

    if args.resume or args.resume_from:
        path = args.resume_from
        info = ckpt_mgr.load(path, model, optimizer, scheduler, scaler) if path else ckpt_mgr.load_latest(model, optimizer, scheduler, scaler)
        if info:
            start_epoch = info["epoch"] + 1
            global_step = info["global_step"]
            eta = ETATracker(total_steps_1a)

    best_miou_1a = -1.0
    for epoch in range(start_epoch, config.substage_1a_epochs):
        avg_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            config, epoch, global_step, eta, logger, train_sampler)
        if is_main_process():
            val_metrics = validate(model, val_loader, device, config)
            print(f"  [Epoch {epoch:02d}] train_loss={avg_loss:.4f} "
                  f"val_loss={val_metrics['val/loss']:.4f} "
                  f"mIoU={val_metrics['val/mIoU']:.4f} "
                  f"Dice={val_metrics['val/Dice']:.4f} "
                  f"P={val_metrics['val/Precision']:.4f} "
                  f"R={val_metrics['val/Recall']:.4f} "
                  f"BF1={val_metrics['val/BoundaryF1']:.4f} "
                  f"HD={val_metrics['val/Hausdorff']:.1f}px "
                  f"CE={val_metrics['val/CentroidErr']:.1f}px")
            logger.log(val_metrics, step=global_step)
            is_last = (epoch == config.substage_1a_epochs - 1)
            is_best = val_metrics["val/mIoU"] > best_miou_1a
            if is_best:
                best_miou_1a = val_metrics["val/mIoU"]
                print(f"  ★ New best mIoU={best_miou_1a:.4f}")
            if is_best or is_last:
                ckpt_mgr.save(model, optimizer, scheduler, scaler, epoch,
                              global_step, val_metrics, substage="1a")

    # ---- Sub-stage 1b: unfreeze all, full fine-tune ----
    if is_main_process():
        print("\n--- Sub-stage 1b: Full fine-tune ---")
    for param in model.parameters():
        param.requires_grad = True
    if is_main_process():
        trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        print(f"Trainable: {trainable:,} / {total:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.substage_1b_lr,
                                   weight_decay=config.weight_decay)
    total_steps_1b = config.substage_1b_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, total_steps_1b)
    eta = ETATracker(total_steps_1b)

    best_miou_1b = -1.0
    for epoch in range(config.substage_1b_epochs):
        e = epoch + config.substage_1a_epochs
        avg_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            config, e, global_step, eta, logger, train_sampler)
        if is_main_process():
            val_metrics = validate(model, val_loader, device, config)
            print(f"  [Epoch {e:02d}] train_loss={avg_loss:.4f} "
                  f"val_loss={val_metrics['val/loss']:.4f} "
                  f"mIoU={val_metrics['val/mIoU']:.4f} "
                  f"Dice={val_metrics['val/Dice']:.4f} "
                  f"P={val_metrics['val/Precision']:.4f} "
                  f"R={val_metrics['val/Recall']:.4f} "
                  f"BF1={val_metrics['val/BoundaryF1']:.4f} "
                  f"HD={val_metrics['val/Hausdorff']:.1f}px "
                  f"CE={val_metrics['val/CentroidErr']:.1f}px")
            logger.log(val_metrics, step=global_step)
            is_last = (epoch == config.substage_1b_epochs - 1)
            is_best = val_metrics["val/mIoU"] > best_miou_1b
            if is_best:
                best_miou_1b = val_metrics["val/mIoU"]
                print(f"  ★ New best mIoU={best_miou_1b:.4f}")
            if is_best or is_last:
                ckpt_mgr.save(model, optimizer, scheduler, scaler, e,
                              global_step, val_metrics, substage="1b")

    logger.finish()
    cleanup_ddp()
    if is_main_process():
        print("\nStage 1 complete.")


if __name__ == "__main__":
    main()
