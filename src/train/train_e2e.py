"""Stage 5: End-to-End Joint Fine-tuning (4-GPU DDP).

Unfreeze all components with differential learning rates:
  TGAA encoder: 1e-5,  ATF: 1e-4,  Temporal adapter: 5e-6,
  LoRA: 2e-4,  Projection MLP: 1e-4

Joint loss: seg_loss + 0.5*cls_loss + 0.5*lm_loss
15 epochs, batch_size=8, gradient checkpointing enabled.
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
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                              f1_score, roc_auc_score)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import ThermalFrameDataset
from src.models.atf import AsymmetricThermalFusion
from src.models.model_factory import create_model
from src.models.temporal_encoder import TemporalEncoder
from src.utils.config import EndToEndConfig
from src.utils.trainer import (CheckpointManager, ETATracker, MetricsLogger,
                                cleanup_ddp, get_cosine_schedule_with_warmup,
                                get_sampler, is_main_process, print_header,
                                set_seed, setup_ddp, unwrap_model,
                                wrap_model_ddp)


class EndToEndModel(nn.Module):
    """Wraps all components for end-to-end training."""

    def __init__(self, seg_model, atf_module, temporal_model, llava_model=None,
                 num_classes=3, feature_dim=256):
        super().__init__()
        self.seg_model = seg_model
        self.atf = atf_module
        self.temporal = temporal_model
        self.llava = llava_model  # Optional — can be None
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, overlay, mask, intensity, input_ids=None,
                attention_mask=None, labels_text=None):
        seg_out = self.seg_model(overlay, thermal_intensity=intensity,
                                  binary_mask=mask)
        seg_logits = seg_out["seg_logits"]
        stage4_feat = seg_out["stage4_features"]

        bg_overlay = overlay * (1.0 - mask)
        atf_out = self.atf(mask, stage4_feat, bg_overlay)
        fused = atf_out["fused"]

        # Temporal embedding is not available at frame level — use zeros as placeholder
        temporal_emb = torch.zeros(overlay.size(0), 256, device=overlay.device)

        cls_logits = self.classifier(fused)

        result = {
            "seg_logits": seg_logits,
            "cls_logits": cls_logits,
            "fused": fused,
            "temporal_emb": temporal_emb,
        }

        if input_ids is not None and self.llava is not None:
            llava_out = self.llava(fused, temporal_emb, input_ids,
                                   attention_mask, labels_text)
            result["lm_loss"] = llava_out["loss"]

        return result


def dice_loss(pred, target, smooth=1.0):
    pred_sig = torch.sigmoid(pred)
    intersection = (pred_sig * target).sum(dim=(2, 3))
    union = pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1.0 - ((2.0 * intersection + smooth) / (union + smooth)).mean()


def _boundary(m):
    return m.astype(bool) & ~binary_erosion(m, iterations=1)


def compute_seg_metrics(pred_np, gt_np):
    smooth = 1e-6
    tp = (pred_np & gt_np).sum()
    fp = (pred_np & ~gt_np).sum()
    fn = (~pred_np & gt_np).sum()
    prec = tp / (tp + fp + smooth)
    rec  = tp / (tp + fn + smooth)
    dice = 2*tp / (2*tp + fp + fn + smooth)
    pb, gb = _boundary(pred_np), _boundary(gt_np)
    bf1 = 2*(pb & gb).sum() / (pb.sum() + gb.sum() + smooth)
    pp, gp = np.argwhere(pred_np), np.argwhere(gt_np)
    if len(pp) > 0 and len(gp) > 0:
        hd = max(directed_hausdorff(pp, gp)[0], directed_hausdorff(gp, pp)[0])
        cce = np.linalg.norm(pp.mean(0) - gp.mean(0))
    else:
        hd = cce = float(max(pred_np.shape))
    return {"dice": float(dice), "prec": float(prec), "rec": float(rec),
            "bf1": float(bf1), "hd": float(hd), "cce": float(cce)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--seg_checkpoint", type=str, default=None)
    parser.add_argument("--temporal_checkpoint", type=str, default=None)
    parser.add_argument("--fusion_checkpoint", type=str, default=None)
    parser.add_argument("--llava_checkpoint", type=str, default=None)
    args = parser.parse_args()

    rank, world_size, device = setup_ddp()
    config = EndToEndConfig()
    set_seed(config.seed + rank)
    print_header("Stage 5: End-to-End Joint Fine-tuning", config)
    if is_main_process():
        print(f"World size: {world_size} GPUs")

    # Load all component models
    import os as _os
    _model_name = _os.environ.get("MODEL_NAME", "trace")
    _stage4_ch  = int(_os.environ.get("STAGE4_CHANNELS", "512"))
    if is_main_process():
        print(f"Seg backbone: {_model_name}, stage4_channels: {_stage4_ch}")
    seg_model = create_model(_model_name, num_seg_classes=1, decode_dim=256,
                             use_aux_mask=True)
    if args.seg_checkpoint or config.segmentation_checkpoint:
        ckpt_path = args.seg_checkpoint or config.segmentation_checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        seg_model.load_state_dict(ckpt["model_state_dict"])
        if is_main_process():
            print(f"Loaded seg: {ckpt_path}")

    temporal_model = TemporalEncoder(output_dim=256)
    if args.temporal_checkpoint or config.temporal_checkpoint:
        ckpt_path = args.temporal_checkpoint or config.temporal_checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        temporal_model.load_state_dict(ckpt["model_state_dict"])
        if is_main_process():
            print(f"Loaded temporal: {ckpt_path}")

    atf_module = AsymmetricThermalFusion(feature_dim=256, stage4_channels=_stage4_ch)
    if args.fusion_checkpoint or config.fusion_checkpoint:
        ckpt_path = args.fusion_checkpoint or config.fusion_checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        atf_state = {k.replace("atf.", ""): v for k, v in ckpt["model_state_dict"].items()
                     if k.startswith("atf.")}
        atf_module.load_state_dict(atf_state)
        if is_main_process():
            print(f"Loaded ATF: {ckpt_path}")

    # LLaVA is optional — skip if not available
    llava_model = None
    if args.llava_checkpoint or config.llava_checkpoint:
        try:
            from src.models.llava_lora import TRACELLaVA
            llava_model = TRACELLaVA(lora_rank=16, lora_alpha=32)
            ckpt_path = args.llava_checkpoint or config.llava_checkpoint
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            llava_model.load_state_dict(ckpt["model_state_dict"])
            if is_main_process():
                print(f"Loaded LLaVA: {ckpt_path}")
        except (ImportError, ModuleNotFoundError):
            if is_main_process():
                print("  ⚠ LLaVA not available, running without language model")
            llava_model = None
    else:
        if is_main_process():
            print("  ℹ Running without LLaVA (seg + classification only)")

    e2e_model = EndToEndModel(seg_model, atf_module, temporal_model, llava_model)
    e2e_model = wrap_model_ddp(e2e_model, device)

    # Differential learning rates
    raw = unwrap_model(e2e_model)
    param_groups = [
        {"params": list(raw.seg_model.parameters()), "lr": config.tgaa_encoder_lr},
        {"params": list(raw.atf.parameters()), "lr": config.atf_lr},
        {"params": [p for n, p in raw.temporal.named_parameters()
                    if "adapter" in n], "lr": config.temporal_adapter_lr},
        {"params": list(raw.classifier.parameters()), "lr": config.atf_lr},
    ]
    if raw.llava is not None:
        param_groups.extend([
            {"params": [p for n, p in raw.llava.named_parameters()
                        if "lora" in n.lower() and p.requires_grad], "lr": config.lora_lr},
            {"params": [p for n, p in raw.llava.named_parameters()
                        if "projection" in n.lower() and p.requires_grad], "lr": config.projection_lr},
        ])
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)

    # Data
    train_ds = ThermalFrameDataset(split="train", img_size=(256, 320), augment=True)
    val_ds = ThermalFrameDataset(split="val", img_size=(256, 320), augment=False)
    train_sampler = get_sampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=config.num_workers, pin_memory=config.pin_memory,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=config.pin_memory)
    if is_main_process():
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    scaler = GradScaler()
    total_steps = config.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 200, total_steps)
    eta = ETATracker(total_steps)
    ckpt_mgr = CheckpointManager(config.checkpoint_dir, config.stage_name,
                                  keep_last_n=config.keep_last_n_checkpoints)
    logger = MetricsLogger(config.log_dir, config.stage_name,
                           use_wandb=config.use_wandb,
                           wandb_project=config.wandb_project)
    # Class-weighted loss to handle imbalance (train: HF=1895, Ctrl=1314, LF=1096)
    class_counts = torch.tensor([1895.0, 1314.0, 1096.0], device=device)
    class_weights = (1.0 / class_counts) / (1.0 / class_counts).sum() * 3
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    global_step = 0
    start_epoch = 0

    if args.resume or args.resume_from:
        path = args.resume_from
        info = ckpt_mgr.load(path, e2e_model, optimizer, scheduler, scaler) if path else ckpt_mgr.load_latest(e2e_model, optimizer, scheduler, scaler)
        if info:
            start_epoch = info["epoch"] + 1
            global_step = info["global_step"]

    for epoch in range(start_epoch, config.epochs):
        e2e_model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        total_loss = 0.0

        for batch in train_loader:
            overlay = batch["overlay"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            intensity = batch["thermal_intensity"].to(device, non_blocking=True)
            class_ids = batch["class_id"].clone().detach().to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(dtype=torch.bfloat16):
                out = e2e_model(overlay, mask, intensity)
                seg_loss = (0.5 * F.binary_cross_entropy_with_logits(out["seg_logits"], mask)
                            + 0.5 * dice_loss(out["seg_logits"], mask))
                cls_loss = ce_criterion(out["cls_logits"], class_ids)
                loss = (config.seg_loss_weight * seg_loss
                        + config.cls_loss_weight * cls_loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            total_loss += loss.item()

            if global_step % config.log_every_n_steps == 0 and is_main_process():
                eta_str, elapsed = eta.step(global_step)
                print(f"  [E{epoch:02d} S{global_step:05d}] "
                      f"loss={loss.item():.4f} seg={seg_loss.item():.4f} "
                      f"cls={cls_loss.item():.4f} ETA={eta_str}")
                logger.log({"train/loss": loss.item(), "train/seg_loss": seg_loss.item(),
                             "train/cls_loss": cls_loss.item()}, step=global_step)

        avg_loss = total_loss / len(train_loader)

        # Validation (rank 0)
        if is_main_process():
            raw_e2e = unwrap_model(e2e_model)
            raw_e2e.eval()
            val_loss = 0.0
            all_preds, all_labels, all_probs = [], [], []
            seg_acc = {"iou": 0., "dice": 0., "prec": 0., "rec": 0.,
                       "bf1": 0., "hd": 0., "cce": 0.}
            n_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    overlay    = batch["overlay"].to(device, non_blocking=True)
                    mask       = batch["mask"].to(device, non_blocking=True)
                    intensity  = batch["thermal_intensity"].to(device, non_blocking=True)
                    class_ids  = batch["class_id"].clone().detach().to(device)
                    with autocast(dtype=torch.bfloat16):
                        out = raw_e2e(overlay, mask, intensity)
                    # Classification
                    probs = torch.softmax(out["cls_logits"].float(), dim=1)
                    pred_cls = probs.argmax(dim=1)
                    all_preds.extend(pred_cls.cpu().numpy().tolist())
                    all_labels.extend(class_ids.cpu().numpy().tolist())
                    all_probs.extend(probs.cpu().numpy().tolist())
                    # Segmentation — GPU fast
                    pred_mask = (torch.sigmoid(out["seg_logits"]) > 0.5).float()
                    inter = (pred_mask * mask).sum(dim=(2, 3))
                    uni   = pred_mask.sum(dim=(2, 3)) + mask.sum(dim=(2, 3)) - inter
                    seg_acc["iou"] += (inter / (uni + 1e-6)).mean().item()
                    # Seg rich — CPU
                    pnp = pred_mask.squeeze(1).cpu().numpy().astype(bool)
                    gnp = mask.squeeze(1).cpu().numpy().astype(bool)
                    for p, g in zip(pnp, gnp):
                        m = compute_seg_metrics(p, g)
                        for k in ("dice","prec","rec","bf1","hd","cce"):
                            seg_acc[k] += m[k]
                    val_loss += (0.5 * F.binary_cross_entropy_with_logits(out["seg_logits"], mask)
                                 + 0.5 * dice_loss(out["seg_logits"], mask)).item()
                    n_val += len(pnp)

            nb = len(val_loader)
            all_labels_np = np.array(all_labels)
            all_preds_np  = np.array(all_preds)
            all_probs_np  = np.array(all_probs)
            bal_acc  = balanced_accuracy_score(all_labels_np, all_preds_np)
            macro_f1 = f1_score(all_labels_np, all_preds_np, average="macro", zero_division=0)
            kappa    = cohen_kappa_score(all_labels_np, all_preds_np)
            try:
                auc = roc_auc_score(all_labels_np, all_probs_np, multi_class="ovr", average="macro")
            except ValueError:
                auc = float("nan")

            val_metrics = {
                "val/loss":        val_loss / nb,
                "val/acc":         (all_preds_np == all_labels_np).mean(),
                "val/BalancedAcc": float(bal_acc),
                "val/MacroF1":     float(macro_f1),
                "val/CohenKappa":  float(kappa),
                "val/AUC_ROC":     float(auc),
                "val/mIoU":        seg_acc["iou"] / nb,
                "val/Dice":        seg_acc["dice"] / n_val,
                "val/Precision":   seg_acc["prec"] / n_val,
                "val/Recall":      seg_acc["rec"]  / n_val,
                "val/BoundaryF1":  seg_acc["bf1"]  / n_val,
                "val/Hausdorff":   seg_acc["hd"]   / n_val,
                "val/CentroidErr": seg_acc["cce"]  / n_val,
            }
            print(f"  [Epoch {epoch:02d}] train={avg_loss:.4f} "
                  f"val_loss={val_metrics['val/loss']:.4f} "
                  f"mIoU={val_metrics['val/mIoU']:.4f} Dice={val_metrics['val/Dice']:.4f} "
                  f"BF1={val_metrics['val/BoundaryF1']:.4f} HD={val_metrics['val/Hausdorff']:.1f}px \n"
                  f"           acc={val_metrics['val/acc']:.4f} "
                  f"BalAcc={val_metrics['val/BalancedAcc']:.4f} "
                  f"MacroF1={val_metrics['val/MacroF1']:.4f} "
                  f"Kappa={val_metrics['val/CohenKappa']:.4f} "
                  f"AUC={val_metrics['val/AUC_ROC']:.4f}")
            logger.log(val_metrics, step=global_step)

            if (epoch + 1) % config.save_every_n_epochs == 0:
                ckpt_mgr.save(e2e_model, optimizer, scheduler, scaler, epoch,
                              global_step, val_metrics)

    logger.finish()
    cleanup_ddp()
    if is_main_process():
        print("\nStage 5 complete.")


if __name__ == "__main__":
    main()
