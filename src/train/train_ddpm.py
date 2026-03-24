"""Stage 6: DDPM Augmentation Training (4-GPU DDP).

Train conditional U-Net to generate synthetic binary gas masks.
200 epochs, lr=1e-4, batch_size=64, T=100 diffusion steps.
After training, generate 200 synthetic masks per class.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.ddpm_augment import (ConditionalUNet, generate_masks,
                                    get_diffusion_params, q_sample,
                                    DIFFUSION_STEPS, TRAIN_RESOLUTION)
from src.utils.config import DDPMConfig
from src.utils.trainer import (CheckpointManager, ETATracker, MetricsLogger,
                                cleanup_ddp, get_sampler, is_main_process,
                                print_header, set_seed, setup_ddp,
                                unwrap_model, wrap_model_ddp)

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))


class MaskDataset(Dataset):
    """Dataset of binary masks for DDPM training."""

    def __init__(self, annotations_csv: str = None, split: str = "train",
                 resolution: tuple = TRAIN_RESOLUTION):
        if annotations_csv is None:
            annotations_csv = str(PROJECT_ROOT / "annotations" / "annotations.csv")
        df = pd.read_csv(annotations_csv)
        df = df[df["split"] == split]
        df = df[~df["excluded"]]
        self.df = df.reset_index(drop=True)
        self.resolution = resolution

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mask = cv2.imread(str(PROJECT_ROOT / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.resolution[1], self.resolution[0]),
                          interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        return {"mask": mask_t, "class_id": int(row["class_id"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--generate_only", action="store_true",
                        help="Skip training, just generate masks from checkpoint")
    args = parser.parse_args()

    rank, world_size, device = setup_ddp()
    config = DDPMConfig()
    set_seed(config.seed + rank)
    print_header("Stage 6: DDPM Augmentation", config)
    if is_main_process():
        print(f"World size: {world_size} GPUs")

    model = ConditionalUNet(in_channels=1, base_channels=config.base_channels,
                             num_classes=3)
    model = wrap_model_ddp(model, device)
    if is_main_process():
        raw = unwrap_model(model)
        params = sum(p.numel() for p in raw.parameters())
        print(f"DDPM U-Net params: {params:,}")

    diffusion_params = get_diffusion_params(config.diffusion_steps)
    scaler = GradScaler()
    ckpt_mgr = CheckpointManager(config.checkpoint_dir, config.stage_name,
                                  keep_last_n=config.keep_last_n_checkpoints)
    logger = MetricsLogger(config.log_dir, config.stage_name,
                           use_wandb=config.use_wandb,
                           wandb_project=config.wandb_project)

    if not args.generate_only:
        train_ds = MaskDataset(split="train", resolution=config.train_resolution)
        train_sampler = get_sampler(train_ds, shuffle=True)
        train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                                  shuffle=(train_sampler is None), sampler=train_sampler,
                                  num_workers=config.num_workers, pin_memory=config.pin_memory,
                                  drop_last=True)
        if is_main_process():
            print(f"Train: {len(train_ds)} masks")

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                       weight_decay=config.weight_decay)
        total_steps = config.epochs * len(train_loader)
        eta = ETATracker(total_steps)

        global_step = 0
        start_epoch = 0

        if args.resume or args.resume_from:
            path = args.resume_from
            info = ckpt_mgr.load(path, model, optimizer) if path else ckpt_mgr.load_latest(model, optimizer)
            if info:
                start_epoch = info["epoch"] + 1
                global_step = info["global_step"]

        for epoch in range(start_epoch, config.epochs):
            model.train()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            total_loss = 0.0

            for batch in train_loader:
                masks = batch["mask"].to(device, non_blocking=True)
                class_ids = batch["class_id"].to(device, non_blocking=True)

                t = torch.randint(0, config.diffusion_steps, (masks.size(0),),
                                  device=device)

                optimizer.zero_grad(set_to_none=True)
                with autocast(dtype=torch.bfloat16):
                    x_t, noise = q_sample(masks, t, diffusion_params)
                    pred_noise = model(x_t, t, class_ids)
                    loss = torch.nn.functional.mse_loss(pred_noise, noise)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                global_step += 1
                total_loss += loss.item()

                if global_step % config.log_every_n_steps == 0 and is_main_process():
                    eta_str, elapsed = eta.step(global_step)
                    print(f"  [E{epoch:03d} S{global_step:06d}] "
                          f"loss={loss.item():.6f} ETA={eta_str}")
                    logger.log({"train/loss": loss.item()}, step=global_step)

            avg_loss = total_loss / len(train_loader)
            if is_main_process():
                print(f"  [Epoch {epoch:03d}] avg_loss={avg_loss:.6f}")
                if (epoch + 1) % config.save_every_n_epochs == 0:
                    ckpt_mgr.save(model, optimizer, None, scaler, epoch,
                                  global_step, {"train/loss": avg_loss})
    else:
        info = ckpt_mgr.load_latest(model)
        if info is None and is_main_process():
            print("ERROR: No checkpoint found for generation.")
            cleanup_ddp()
            return

    # ---- Generate synthetic masks (rank 0 only) ----
    if is_main_process():
        print(f"\nGenerating {config.n_synthetic_masks} synthetic masks per class...")
        output_dir = PROJECT_ROOT / "dataset" / "synthetic_masks"
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_model = unwrap_model(model)
        for class_id in range(3):
            class_names = {0: "high_forage", 1: "control", 2: "low_forage"}
            masks = generate_masks(raw_model, config.n_synthetic_masks, class_id, device,
                                   config.train_resolution)
            class_dir = output_dir / class_names[class_id]
            class_dir.mkdir(parents=True, exist_ok=True)

            for i in range(config.n_synthetic_masks):
                mask_np = (masks[i, 0].cpu().numpy() * 255).astype(np.uint8)
                mask_np = cv2.resize(mask_np, (320, 240), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(class_dir / f"synth_{i:04d}.png"), mask_np)

            print(f"  {class_names[class_id]}: {config.n_synthetic_masks} masks saved")

    logger.finish()
    cleanup_ddp()
    if is_main_process():
        print("\nStage 6 complete.")


if __name__ == "__main__":
    main()
