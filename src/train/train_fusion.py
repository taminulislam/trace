"""Stage 3: ATF Fusion Training (4-GPU DDP).

Freeze TRACE segmentation branch (Stage 1) and TemporalEncoder (Stage 2).
Train ATF module + lightweight classification head.
Loss: CE + 0.01 * L2(conf_B, conf_C)
30 epochs, lr=1e-4, batch_size=64.
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

from src.data.dataset import ThermalFrameDataset
from src.models.atf import AsymmetricThermalFusion
from src.models.model_factory import create_model
from src.utils.config import FusionConfig
from src.utils.trainer import (CheckpointManager, ETATracker, MetricsLogger,
                                cleanup_ddp, get_cosine_schedule_with_warmup,
                                get_sampler, is_main_process, print_header,
                                set_seed, setup_ddp, unwrap_model,
                                wrap_model_ddp)


class FusionClassifier(nn.Module):
    """ATF fusion + classification head wrapper."""

    def __init__(self, feature_dim=256, stage4_channels=512, num_classes=3):
        super().__init__()
        self.atf = AsymmetricThermalFusion(feature_dim=feature_dim,
                                            stage4_channels=stage4_channels)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, binary_mask, stage4_features, background_overlay):
        atf_out = self.atf(binary_mask, stage4_features, background_overlay)
        logits = self.classifier(atf_out["fused"])
        return logits, atf_out["confidence_scores"]


def make_background(overlay, mask):
    bg = overlay.clone()
    bg = bg * (1.0 - mask)
    return bg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--seg_checkpoint", type=str, default=None)
    args = parser.parse_args()

    rank, world_size, device = setup_ddp()
    config = FusionConfig()
    set_seed(config.seed + rank)
    print_header("Stage 3: ATF Fusion Training", config)
    if is_main_process():
        print(f"World size: {world_size} GPUs")

    # Data
    train_ds = ThermalFrameDataset(split="train", img_size=(256, 320), augment=False)
    val_ds = ThermalFrameDataset(split="val", img_size=(256, 320), augment=False)
    train_sampler = get_sampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=config.num_workers, pin_memory=config.pin_memory,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=config.pin_memory)
    if is_main_process():
        print(f"Train: {len(train_ds)} frames, Val: {len(val_ds)} frames")

    # Frozen segmentation model (from Stage 1)
    import os as _os
    _model_name = _os.environ.get("MODEL_NAME", "trace")
    _use_simple_fusion = config.use_simple_fusion
    if is_main_process():
        fusion_type = "SimpleConcatFusion" if _use_simple_fusion else "ATF (AsymmetricThermalFusion)"
        print(f"Seg backbone: {_model_name} | Fusion: {fusion_type} | stage4_channels: {config.stage4_channels}")
    seg_model = create_model(_model_name, num_seg_classes=1, decode_dim=256,
                             use_aux_mask=True).to(device)
    seg_ckpt = args.seg_checkpoint or config.segmentation_checkpoint
    if seg_ckpt:
        ckpt = torch.load(seg_ckpt, map_location=device, weights_only=False)
        seg_model.load_state_dict(ckpt["model_state_dict"])
        if is_main_process():
            print(f"Loaded segmentation checkpoint: {seg_ckpt}")
    seg_model.eval()
    for p in seg_model.parameters():
        p.requires_grad = False

    # Fusion model (this is what we train) — A2 ablation: swap ATF for simple concat
    if _use_simple_fusion:
        from src.models.simple_concat_fusion import SimpleConcatFusion

        class _SimpleFusionClassifier(nn.Module):
            def __init__(self, feature_dim, stage4_channels, num_classes):
                super().__init__()
                self.atf = SimpleConcatFusion(feature_dim, stage4_channels)
                self.classifier = nn.Sequential(
                    nn.Linear(feature_dim, feature_dim), nn.ReLU(inplace=True),
                    nn.Dropout(0.1), nn.Linear(feature_dim, num_classes),
                )
            def forward(self, binary_mask, stage4_features, background_overlay):
                atf_out = self.atf(binary_mask, stage4_features, background_overlay)
                return self.classifier(atf_out["fused"]), atf_out["confidence_scores"]

        fusion_model = _SimpleFusionClassifier(
            feature_dim=config.feature_dim,
            stage4_channels=config.stage4_channels,
            num_classes=config.num_classes,
        )
    else:
        fusion_model = FusionClassifier(
            feature_dim=config.feature_dim,
            stage4_channels=config.stage4_channels,
            num_classes=config.num_classes,
        )
    fusion_model = wrap_model_ddp(fusion_model, device)

    # Class-weighted loss to handle imbalance (train: HF=1895, Ctrl=1314, LF=1096)
    class_counts = torch.tensor([1895.0, 1314.0, 1096.0], device=device)
    class_weights = (1.0 / class_counts) / (1.0 / class_counts).sum() * config.num_classes
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    scaler = GradScaler()
    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=config.lr,
                                   weight_decay=config.weight_decay)
    total_steps = config.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 200, total_steps)
    eta = ETATracker(total_steps)

    ckpt_mgr = CheckpointManager(config.checkpoint_dir, config.stage_name,
                                  keep_last_n=config.keep_last_n_checkpoints)
    logger = MetricsLogger(config.log_dir, config.stage_name,
                           use_wandb=config.use_wandb,
                           wandb_project=config.wandb_project)

    global_step = 0
    start_epoch = 0

    if args.resume or args.resume_from:
        path = args.resume_from
        info = ckpt_mgr.load(path, fusion_model, optimizer, scheduler, scaler) if path else ckpt_mgr.load_latest(fusion_model, optimizer, scheduler, scaler)
        if info:
            start_epoch = info["epoch"] + 1
            global_step = info["global_step"]

    for epoch in range(start_epoch, config.epochs):
        fusion_model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            overlay = batch["overlay"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            labels = batch["class_id"].clone().detach().to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(dtype=torch.bfloat16):
                with torch.no_grad():
                    seg_out = seg_model(overlay,
                                        thermal_intensity=batch["thermal_intensity"].to(device),
                                        binary_mask=mask)
                    stage4_feat = seg_out["stage4_features"]

                bg_overlay = make_background(overlay, mask)
                logits, conf_scores = fusion_model(mask, stage4_feat, bg_overlay)
                ce_loss = criterion(logits, labels)
                conf_reg = config.conf_l2_weight * (conf_scores[1] ** 2 + conf_scores[2] ** 2)
                loss = ce_loss + conf_reg

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            if global_step % config.log_every_n_steps == 0 and is_main_process():
                eta_str, elapsed = eta.step(global_step)
                lr = optimizer.param_groups[0]["lr"]
                print(f"  [E{epoch:02d} S{global_step:05d}] "
                      f"loss={loss.item():.4f} ce={ce_loss.item():.4f} "
                      f"conf_B={conf_scores[1]:.3f} conf_C={conf_scores[2]:.3f} "
                      f"lr={lr:.2e} ETA={eta_str}")
                logger.log({"train/loss": loss.item(), "train/ce": ce_loss.item(),
                             "train/conf_B": conf_scores[1], "train/conf_C": conf_scores[2],
                             "train/lr": lr}, step=global_step)

        avg_loss = total_loss / len(train_loader)
        acc = correct / max(total, 1)

        # Validation (rank 0 only)
        if is_main_process():
            raw_fusion = unwrap_model(fusion_model)
            raw_fusion.eval()
            val_loss = 0.0
            all_preds, all_labels, all_probs = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    overlay = batch["overlay"].to(device, non_blocking=True)
                    mask = batch["mask"].to(device, non_blocking=True)
                    labels = batch["class_id"].clone().detach().to(device)
                    with autocast(dtype=torch.bfloat16):
                        seg_out = seg_model(overlay,
                                            thermal_intensity=batch["thermal_intensity"].to(device),
                                            binary_mask=mask)
                        bg_overlay = make_background(overlay, mask)
                        logits, _ = raw_fusion(mask, seg_out["stage4_features"], bg_overlay)
                        loss = criterion(logits, labels)
                    val_loss += loss.item()
                    probs = torch.softmax(logits.float(), dim=1)
                    pred = probs.argmax(dim=1)
                    all_preds.extend(pred.cpu().numpy().tolist())
                    all_labels.extend(labels.cpu().numpy().tolist())
                    all_probs.extend(probs.cpu().numpy().tolist())

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

            val_metrics = {
                "val/loss":        val_loss / len(val_loader),
                "val/acc":         float(acc),
                "val/BalancedAcc": float(bal_acc),
                "val/MacroF1":     float(macro_f1),
                "val/CohenKappa":  float(kappa),
                "val/AUC_ROC":     float(auc),
            }
            print(f"  [Epoch {epoch:02d}] train_loss={avg_loss:.4f} train_acc={acc:.4f} "
                  f"val_loss={val_metrics['val/loss']:.4f} "
                  f"acc={val_metrics['val/acc']:.4f} "
                  f"BalAcc={val_metrics['val/BalancedAcc']:.4f} "
                  f"MacroF1={val_metrics['val/MacroF1']:.4f} "
                  f"Kappa={val_metrics['val/CohenKappa']:.4f} "
                  f"AUC={val_metrics['val/AUC_ROC']:.4f}")
            logger.log(val_metrics, step=global_step)

            if (epoch + 1) % config.save_every_n_epochs == 0:
                ckpt_mgr.save(fusion_model, optimizer, scheduler, scaler, epoch,
                              global_step, val_metrics)

    logger.finish()
    cleanup_ddp()
    if is_main_process():
        print("\nStage 3 complete.")


if __name__ == "__main__":
    main()
