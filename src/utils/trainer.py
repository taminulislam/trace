"""Training utilities: checkpoint save/load, resume, ETA tracking, logging, DDP."""

import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist


# ------------------------------------------------------------------ #
#  Distributed Data Parallel helpers
# ------------------------------------------------------------------ #

def setup_ddp():
    """Initialize DDP process group. Call once at the start of main()."""
    if "RANK" not in os.environ:
        # Single-GPU fallback
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device


def cleanup_ddp():
    """Destroy DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Return True if this is rank 0 (or single-GPU)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def wrap_model_ddp(model, device):
    """Wrap model in DDP if distributed, else just move to device."""
    model = model.to(device)
    if dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True,
        )
    return model


def get_sampler(dataset, shuffle=True):
    """Return DistributedSampler if DDP, else None."""
    if dist.is_initialized():
        return torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=shuffle)
    return None


def unwrap_model(model):
    """Get the underlying model from DDP wrapper."""
    if hasattr(model, "module"):
        return model.module
    return model


# ------------------------------------------------------------------ #
#  Checkpoint Manager
# ------------------------------------------------------------------ #

class CheckpointManager:
    """Manages checkpoint saving, loading, and cleanup.

    Only rank 0 saves/deletes; all ranks load.
    """

    def __init__(self, checkpoint_dir: str, stage_name: str,
                 keep_last_n: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir) / stage_name
        if is_main_process():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.stage_name = stage_name
        self.keep_last_n = keep_last_n

    def save(self, model, optimizer, scheduler, scaler, epoch: int,
             global_step: int, metrics: dict = None, substage: str = ""):
        if not is_main_process():
            return None

        tag = f"{self.stage_name}"
        if substage:
            tag += f"_{substage}"
        tag += f"_epoch{epoch:04d}_step{global_step:06d}"
        path = self.checkpoint_dir / f"{tag}.pt"

        raw_model = unwrap_model(model)
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics or {},
            "stage_name": self.stage_name,
            "substage": substage,
        }
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None:
            state["scaler_state_dict"] = scaler.state_dict()

        torch.save(state, path)
        print(f"[Checkpoint] Saved: {path.name}")

        latest = self.checkpoint_dir / f"{self.stage_name}_latest.pt"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(path.name)

        self._cleanup()
        return str(path)

    def _cleanup(self):
        pts = sorted(self.checkpoint_dir.glob(f"{self.stage_name}*.pt"),
                     key=lambda p: p.stat().st_mtime)
        pts = [p for p in pts if not p.is_symlink()]
        while len(pts) > self.keep_last_n:
            old = pts.pop(0)
            old.unlink()
            print(f"[Checkpoint] Deleted old: {old.name}")

    def load_latest(self, model, optimizer=None, scheduler=None, scaler=None):
        latest = self.checkpoint_dir / f"{self.stage_name}_latest.pt"
        if not latest.exists():
            return None
        return self.load(str(latest), model, optimizer, scheduler, scaler)

    def load(self, path: str, model, optimizer=None, scheduler=None,
             scaler=None):
        raw_model = unwrap_model(model)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        info = {
            "epoch": ckpt["epoch"],
            "global_step": ckpt["global_step"],
            "metrics": ckpt.get("metrics", {}),
            "substage": ckpt.get("substage", ""),
        }
        if is_main_process():
            print(f"[Checkpoint] Resumed from {Path(path).name} "
                  f"(epoch={info['epoch']}, step={info['global_step']})")
        return info


# ------------------------------------------------------------------ #
#  ETA Tracker
# ------------------------------------------------------------------ #

class ETATracker:
    """Tracks elapsed time and estimates time remaining."""

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.start_time = time.time()

    def step(self, current_step: int):
        elapsed = time.time() - self.start_time
        remaining_steps = self.total_steps - current_step
        if current_step > 0:
            avg_time_per_step = elapsed / current_step
            eta_seconds = remaining_steps * avg_time_per_step
        else:
            eta_seconds = 0
        return self._format_time(eta_seconds), self._format_time(elapsed)

    @staticmethod
    def _format_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}h {m:02d}m {s:02d}s"
        return f"{m}m {s:02d}s"


# ------------------------------------------------------------------ #
#  Metrics Logger
# ------------------------------------------------------------------ #

class MetricsLogger:
    """Logs metrics to console, JSON file, and optionally wandb.
    Only rank 0 writes logs.
    """

    def __init__(self, log_dir: str, stage_name: str, use_wandb: bool = False,
                 wandb_project: str = "trace", wandb_run_name: str = None):
        self.log_dir = Path(log_dir)
        self.stage_name = stage_name
        self.use_wandb = use_wandb
        self._wandb_initialized = False

        if not is_main_process():
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{stage_name}_metrics.jsonl"

        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name or stage_name,
                    config={"stage": stage_name},
                    resume="allow",
                )
                self._wandb_initialized = True
            except Exception as e:
                print(f"[Warning] wandb init failed: {e}. Logging to file only.")
                self.use_wandb = False

    def log(self, metrics: dict, step: int):
        if not is_main_process():
            return
        record = {"step": step, **metrics}
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        if self._wandb_initialized:
            import wandb
            wandb.log(metrics, step=step)

    def finish(self):
        if not is_main_process():
            return
        if self._wandb_initialized:
            import wandb
            wandb.finish()


# ------------------------------------------------------------------ #
#  Utilities
# ------------------------------------------------------------------ #

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int,
                                     num_training_steps: int):
    """Cosine annealing LR scheduler with linear warmup."""
    from torch.optim.lr_scheduler import LambdaLR
    import math

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_header(stage_name: str, config=None):
    """Print a formatted header for a training stage (rank 0 only)."""
    if not is_main_process():
        return
    print("=" * 60)
    print(f"  TRACE — {stage_name}")
    print("=" * 60)
    if config is not None:
        for k, v in vars(config).items():
            if not k.startswith("_"):
                print(f"  {k}: {v}")
        print("=" * 60)
