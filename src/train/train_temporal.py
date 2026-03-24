"""Stage 2: Temporal Encoder Training — VideoMAE-Small.

Sub-stage 2a: Freeze VideoMAE backbone, train adapter + classifier (15 epochs, lr=5e-5)
Sub-stage 2b: Unfreeze backbone, full fine-tune (25 epochs, lr=1e-5)

Loss: CrossEntropy with label_smoothing=0.1
Input: clip of 16 overlay frames (B, T, C, H, W) at 224x224
Target: 3-class feed-type label

Metrics (CVPR-quality):
  Accuracy, Balanced Accuracy, Macro-F1, Cohen's Kappa, per-class AUC-ROC
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                              f1_score, roc_auc_score)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import ThermalClipDataset
from src.models.temporal_encoder import TemporalEncoder
from src.utils.config import TemporalConfig
from src.utils.trainer import (CheckpointManager, ETATracker, MetricsLogger,
                                cleanup_ddp, get_cosine_schedule_with_warmup,
                                get_sampler, is_main_process, print_header,
                                set_seed, setup_ddp, unwrap_model,
                                wrap_model_ddp)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device,
                    config, epoch, global_step, eta, logger, criterion, sampler):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)
    total_loss = 0.0
    correct = 0
    total = 0

    for i, batch in enumerate(loader):
        clips = batch["clip_overlays"].to(device, non_blocking=True)
        labels = batch["class_id"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(dtype=torch.bfloat16):
            out = model(clips, return_cls_logits=True)
            loss = criterion(out["cls_logits"], labels)

        scaler.scale(loss).backward()
        if (i + 1) % config.gradient_accumulation == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        global_step += 1
        total_loss += loss.item()
        pred = out["cls_logits"].argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        if global_step % config.log_every_n_steps == 0 and is_main_process():
            eta_str, elapsed = eta.step(global_step)
            lr = optimizer.param_groups[0]["lr"]
            acc = correct / max(total, 1)
            print(f"  [E{epoch:02d} S{global_step:05d}] "
                  f"loss={loss.item():.4f} acc={acc:.4f} lr={lr:.2e} "
                  f"ETA={eta_str} elapsed={elapsed}")
            logger.log({"train/loss": loss.item(), "train/acc": acc,
                         "train/lr": lr}, step=global_step)

    avg_loss = total_loss / len(loader)
    acc = correct / max(total, 1)
    return avg_loss, acc, global_step


@torch.no_grad()
def validate(model, loader, device, criterion):
    raw = unwrap_model(model)
    raw.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in loader:
        clips = batch["clip_overlays"].to(device, non_blocking=True)
        labels = batch["class_id"].to(device, non_blocking=True)
        with autocast(dtype=torch.bfloat16):
            out = raw(clips, return_cls_logits=True)
            loss = criterion(out["cls_logits"], labels)
        total_loss += loss.item()
        probs = torch.softmax(out["cls_logits"].float(), dim=1)
        pred = probs.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    n = len(loader)
    all_labels_np = np.array(all_labels)
    all_preds_np  = np.array(all_preds)
    all_probs_np  = np.array(all_probs)

    acc      = (all_preds_np == all_labels_np).mean()
    bal_acc  = balanced_accuracy_score(all_labels_np, all_preds_np)
    macro_f1 = f1_score(all_labels_np, all_preds_np, average="macro", zero_division=0)
    kappa    = cohen_kappa_score(all_labels_np, all_preds_np)
    try:
        auc = roc_auc_score(all_labels_np, all_probs_np, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    return {
        "val/loss":         total_loss / n,
        "val/acc":          float(acc),
        "val/BalancedAcc":  float(bal_acc),
        "val/MacroF1":      float(macro_f1),
        "val/CohenKappa":   float(kappa),
        "val/AUC_ROC":      float(auc),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()

    rank, world_size, device = setup_ddp()
    config = TemporalConfig()
    set_seed(config.seed + rank)
    print_header("Stage 2: Temporal Encoder Training", config)
    if is_main_process():
        print(f"World size: {world_size} GPUs")

    # Data
    train_ds = ThermalClipDataset(split="train", clip_img_size=config.clip_img_size)
    val_ds = ThermalClipDataset(split="val", clip_img_size=config.clip_img_size)
    train_sampler = get_sampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=config.num_workers, pin_memory=config.pin_memory,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=config.pin_memory)
    if is_main_process():
        print(f"Train: {len(train_ds)} clips, Val: {len(val_ds)} clips")

    # Model
    model = TemporalEncoder(output_dim=config.temporal_output_dim,
                            pretrained_name=config.videomae_pretrained,
                            freeze_backbone=True)
    if config.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable()
    model = wrap_model_ddp(model, device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scaler = GradScaler()
    ckpt_mgr = CheckpointManager(config.checkpoint_dir, config.stage_name,
                                  keep_last_n=config.keep_last_n_checkpoints)
    logger = MetricsLogger(config.log_dir, config.stage_name,
                           use_wandb=config.use_wandb,
                           wandb_project=config.wandb_project)

    # ---- Sub-stage 2a: Freeze backbone ----
    if is_main_process():
        print("\n--- Sub-stage 2a: Freeze backbone ---")
        raw = unwrap_model(model)
        trainable = sum(p.numel() for p in raw.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in raw.parameters())
        print(f"Trainable: {trainable:,} / {total_params:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.substage_2a_lr, weight_decay=config.weight_decay)
    total_steps_2a = config.substage_2a_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, total_steps_2a)
    eta = ETATracker(total_steps_2a)

    global_step = 0
    start_epoch = 0

    if args.resume or args.resume_from:
        path = args.resume_from
        info = ckpt_mgr.load(path, model, optimizer, scheduler, scaler) if path else ckpt_mgr.load_latest(model, optimizer, scheduler, scaler)
        if info:
            start_epoch = info["epoch"] + 1
            global_step = info["global_step"]

    for epoch in range(start_epoch, config.substage_2a_epochs):
        avg_loss, acc, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            config, epoch, global_step, eta, logger, criterion, train_sampler)
        if is_main_process():
            val_metrics = validate(model, val_loader, device, criterion)
            print(f"  [Epoch {epoch:02d}] train_loss={avg_loss:.4f} train_acc={acc:.4f} "
                  f"val_loss={val_metrics['val/loss']:.4f} "
                  f"acc={val_metrics['val/acc']:.4f} "
                  f"BalAcc={val_metrics['val/BalancedAcc']:.4f} "
                  f"MacroF1={val_metrics['val/MacroF1']:.4f} "
                  f"Kappa={val_metrics['val/CohenKappa']:.4f} "
                  f"AUC={val_metrics['val/AUC_ROC']:.4f}")
            logger.log(val_metrics, step=global_step)
            if (epoch + 1) % config.save_every_n_epochs == 0:
                ckpt_mgr.save(model, optimizer, scheduler, scaler, epoch,
                              global_step, val_metrics, substage="2a")

    # ---- Sub-stage 2b: Unfreeze all ----
    if is_main_process():
        print("\n--- Sub-stage 2b: Full fine-tune ---")
    unwrap_model(model).unfreeze_backbone()
    if is_main_process():
        raw = unwrap_model(model)
        trainable = sum(p.numel() for p in raw.parameters() if p.requires_grad)
        print(f"Trainable: {trainable:,} / {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.substage_2b_lr,
                                   weight_decay=config.weight_decay)
    total_steps_2b = config.substage_2b_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, total_steps_2b)
    eta = ETATracker(total_steps_2b)

    for epoch in range(config.substage_2b_epochs):
        e = epoch + config.substage_2a_epochs
        avg_loss, acc, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            config, e, global_step, eta, logger, criterion, train_sampler)
        if is_main_process():
            val_metrics = validate(model, val_loader, device, criterion)
            print(f"  [Epoch {e:02d}] train_loss={avg_loss:.4f} train_acc={acc:.4f} "
                  f"val_loss={val_metrics['val/loss']:.4f} "
                  f"acc={val_metrics['val/acc']:.4f} "
                  f"BalAcc={val_metrics['val/BalancedAcc']:.4f} "
                  f"MacroF1={val_metrics['val/MacroF1']:.4f} "
                  f"Kappa={val_metrics['val/CohenKappa']:.4f} "
                  f"AUC={val_metrics['val/AUC_ROC']:.4f}")
            logger.log(val_metrics, step=global_step)
            if (epoch + 1) % config.save_every_n_epochs == 0:
                ckpt_mgr.save(model, optimizer, scheduler, scaler, e,
                              global_step, val_metrics, substage="2b")

    logger.finish()
    cleanup_ddp()
    if is_main_process():
        print("\nStage 2 complete.")


if __name__ == "__main__":
    main()
