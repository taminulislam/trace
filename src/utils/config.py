"""Dataclass configs for all training stages.

Each config includes resume, checkpoint saving, and ETA tracking settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))


@dataclass
class BaseTrainConfig:
    """Common training configuration shared across all stages."""
    # Paths
    project_root: str = str(PROJECT_ROOT)
    annotations_csv: str = str(PROJECT_ROOT / "annotations" / "annotations.csv")
    clips_csv: str = str(PROJECT_ROOT / "annotations" / "clips.csv")
    checkpoint_dir: str = str(PROJECT_ROOT / "outputs" / "checkpoints")
    log_dir: str = str(PROJECT_ROOT / "outputs" / "logs")

    def __post_init__(self):
        """Override paths from environment variables for experiment isolation."""
        if os.environ.get("ANNOTATIONS_CSV"):
            self.annotations_csv = os.environ["ANNOTATIONS_CSV"]
        if os.environ.get("CLIPS_CSV"):
            self.clips_csv = os.environ["CLIPS_CSV"]
        if os.environ.get("CHECKPOINT_DIR"):
            self.checkpoint_dir = os.environ["CHECKPOINT_DIR"]
        if os.environ.get("LOG_DIR"):
            self.log_dir = os.environ["LOG_DIR"]
        if os.environ.get("WANDB_DISABLED") == "true":
            self.use_wandb = False

    # Training basics
    seed: int = 42
    precision: str = "bf16"
    num_workers: int = 8
    pin_memory: bool = True
    use_compile: bool = True
    gradient_checkpointing: bool = False

    # Optimizer
    optimizer: str = "adamw"
    weight_decay: float = 0.01

    # Resume
    resume_from: Optional[str] = None  # path to checkpoint to resume from

    # Checkpoint saving
    save_every_n_epochs: int = 1
    save_every_n_steps: Optional[int] = None  # if set, save every N steps too
    keep_last_n_checkpoints: int = 3  # delete older checkpoints

    # Logging
    use_wandb: bool = True
    wandb_project: str = "trace"
    wandb_run_name: Optional[str] = None
    log_every_n_steps: int = 10
    show_eta: bool = True  # show estimated time remaining


@dataclass
class SegmentationConfig(BaseTrainConfig):
    """Stage 1: TGAA-TRACE segmentation pretraining."""
    stage_name: str = "segmentation"

    # Sub-stages
    # 1a: freeze backbone, train TGAA gates + decode head
    # 1b: unfreeze all, full fine-tune
    substage_1a_epochs: int = 10
    substage_1a_lr: float = 6e-5
    substage_1b_epochs: int = 15
    substage_1b_lr: float = 1e-5

    def __post_init__(self):
        super().__post_init__()
        if os.environ.get("SEG_1A_EPOCHS"):
            self.substage_1a_epochs = int(os.environ["SEG_1A_EPOCHS"])
        if os.environ.get("SEG_1B_EPOCHS"):
            self.substage_1b_epochs = int(os.environ["SEG_1B_EPOCHS"])

    # Data
    img_size: tuple = (256, 320)
    batch_size: int = 32

    # Loss
    bce_weight: float = 0.5
    dice_weight: float = 0.5

    # Scheduler
    scheduler: str = "cosine"
    warmup_steps: int = 500

    # Model
    decode_dim: int = 256
    use_aux_mask: bool = True


@dataclass
class TemporalConfig(BaseTrainConfig):
    """Stage 2: Temporal encoder training."""
    stage_name: str = "temporal"

    # Sub-stages
    substage_2a_epochs: int = 8
    substage_2a_lr: float = 5e-5
    substage_2b_epochs: int = 12
    substage_2b_lr: float = 1e-5

    def __post_init__(self):
        super().__post_init__()
        if os.environ.get("TEMP_2A_EPOCHS"):
            self.substage_2a_epochs = int(os.environ["TEMP_2A_EPOCHS"])
        if os.environ.get("TEMP_2B_EPOCHS"):
            self.substage_2b_epochs = int(os.environ["TEMP_2B_EPOCHS"])

    # Data
    clip_img_size: tuple = (224, 224)
    clip_length: int = 16
    batch_size: int = 8
    gradient_accumulation: int = 4

    # Loss
    label_smoothing: float = 0.1
    num_classes: int = 3
    warmup_steps: int = 200

    # Model
    temporal_output_dim: int = 256
    videomae_pretrained: str = "MCG-NJU/videomae-small-finetuned-kinetics"

    gradient_checkpointing: bool = True


@dataclass
class FusionConfig(BaseTrainConfig):
    """Stage 3: ATF fusion training."""
    stage_name: str = "fusion"

    epochs: int = 30
    lr: float = 1e-4
    batch_size: int = 16

    def __post_init__(self):
        super().__post_init__()
        if os.environ.get("FUSION_EPOCHS"):
            self.epochs = int(os.environ["FUSION_EPOCHS"])
        if os.environ.get("STAGE4_CHANNELS"):
            self.stage4_channels = int(os.environ["STAGE4_CHANNELS"])
        if os.environ.get("USE_SIMPLE_FUSION"):
            self.use_simple_fusion = os.environ["USE_SIMPLE_FUSION"].lower() == "true"

    # Loss
    ce_weight: float = 1.0
    conf_l2_weight: float = 0.01  # L2 regularization on conf_B, conf_C

    # Model
    feature_dim: int = 256
    stage4_channels: int = 512
    num_classes: int = 3
    use_simple_fusion: bool = False  # A2 ablation: replace ATF with naive concat

    # Frozen model checkpoints (from Stage 1 and 2)
    segmentation_checkpoint: Optional[str] = None
    temporal_checkpoint: Optional[str] = None


@dataclass
class LLaVAConfig(BaseTrainConfig):
    """Stage 4: LoRA LLaVA fine-tuning."""
    stage_name: str = "llava"

    epochs: int = 5
    lora_lr: float = 2e-4
    projection_lr: float = 1e-4
    batch_size: int = 4
    gradient_accumulation: int = 8

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_targets: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj"])

    # Model
    llava_model_name: str = "liuhaotian/llava-v1.5-7b"
    num_visual_tokens: int = 32
    max_text_len: int = 256

    # Data
    descriptions_csv: str = str(PROJECT_ROOT / "annotations" / "behaviour_descriptions.csv")

    gradient_checkpointing: bool = True

    # Frozen model checkpoints
    segmentation_checkpoint: Optional[str] = None
    temporal_checkpoint: Optional[str] = None
    fusion_checkpoint: Optional[str] = None


@dataclass
class EndToEndConfig(BaseTrainConfig):
    """Stage 5: End-to-end fine-tuning."""
    stage_name: str = "e2e"

    epochs: int = 20
    batch_size: int = 8

    def __post_init__(self):
        super().__post_init__()
        if os.environ.get("E2E_EPOCHS"):
            self.epochs = int(os.environ["E2E_EPOCHS"])
        if os.environ.get("SEG_LOSS_WEIGHT"):
            self.seg_loss_weight = float(os.environ["SEG_LOSS_WEIGHT"])
        if os.environ.get("CLS_LOSS_WEIGHT"):
            self.cls_loss_weight = float(os.environ["CLS_LOSS_WEIGHT"])

    # Differential learning rates
    tgaa_encoder_lr: float = 1e-5
    atf_lr: float = 1e-4
    temporal_adapter_lr: float = 5e-6
    lora_lr: float = 2e-4
    projection_lr: float = 1e-4

    # Joint loss weights — cls upweighted since seg is already near-perfect
    seg_loss_weight: float = 0.3
    cls_loss_weight: float = 2.0
    lm_loss_weight: float = 0.5

    gradient_checkpointing: bool = True

    # Checkpoints from prior stages
    segmentation_checkpoint: Optional[str] = None
    temporal_checkpoint: Optional[str] = None
    fusion_checkpoint: Optional[str] = None
    llava_checkpoint: Optional[str] = None


@dataclass
class DDPMConfig(BaseTrainConfig):
    """Stage 6: DDPM augmentation training."""
    stage_name: str = "ddpm"

    epochs: int = 80
    lr: float = 1e-4
    batch_size: int = 64

    # Model
    base_channels: int = 64
    diffusion_steps: int = 100
    train_resolution: tuple = (48, 64)

    # Generation
    n_synthetic_masks: int = 200
    target_class: int = 0  # HF class
