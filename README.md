# TRACE

**TRACE** is a multi-stage deep learning pipeline for thermal gas segmentation and flux classification in ruminant CO₂ emission monitoring. It combines a TGAA-augmented SegFormer backbone with a VideoMAE temporal encoder, fused via an Attention Temporal Fusion (ATF) module and jointly fine-tuned end-to-end.

<p align="center">
  <img src="TRACE.png" alt="TRACE Architecture" width="800"/>
</p>

---

## Highlights

- **State-of-the-art segmentation** on thermal gas imagery: mIoU 0.9865, Dice 0.9914
- **Multi-stage pipeline**: segmentation → temporal → fusion → language → end-to-end
- **TGAA blocks** replace standard MiT self-attention for temporally-aware feature extraction
- **ATF module** fuses spatial (SegFormer) and temporal (VideoMAE) representations
- **LLaVA-LoRA** stage for language-grounded visual understanding
- Supports multi-GPU DDP training out of the box

---

## Results

### Segmentation (Test Set, 789 samples)

| Model | Params | mIoU | Dice | BF1 | HD (px) |
|:------|-------:|-----:|-----:|----:|--------:|
| **TRACE** | 27.7M | **0.9865** | **0.9914** | **0.8951** | **0.0012** |
| LACTNet | 12.7M | 0.9626 | 0.9807 | 0.7020 | 3.1994 |
| iFormer | 5.5M | 0.9608 | 0.9796 | 0.6933 | 3.1145 |
| SegFormer-B2 | 24.7M | 0.9533 | 0.9754 | 0.6421 | 3.6415 |
| SegFormer-B0 | 3.7M | 0.9537 | 0.9758 | 0.6386 | 3.5355 |

### Classification (Test Set, 104 samples)

| Model | Params | Accuracy | Bal. Acc | Macro-F1 | AUC-ROC |
|:------|-------:|---------:|---------:|---------:|--------:|
| **TRACE** | 27.7M | **0.8173** | **0.7845** | **0.7792** | **0.9412** |
| SegFormer-B0 | 3.7M | 0.7692 | 0.7334 | 0.7296 | 0.8931 |
| Mask2Former | 27.9M | 0.7596 | 0.6154 | 0.5623 | 0.7106 |
| Prior2Former | 21.4M | 0.7404 | 0.5983 | 0.5475 | 0.7183 |

---

## Pipeline Overview

TRACE trains in 5 sequential stages:

```
Stage 1 — Segmentation Pretraining
  └─ TGAA-SegFormer on thermal overlay images
     1a: freeze backbone, train TGAA gates + decode head
     1b: full fine-tune

Stage 2 — Temporal Encoder Training
  └─ VideoMAE-Small fine-tuned on 16-frame thermal clips

Stage 3 — ATF Fusion
  └─ Attention Temporal Fusion joins spatial + temporal embeddings

Stage 4 — LLaVA LoRA
  └─ Language-grounded visual fine-tuning with LoRA adapters

Stage 5 — End-to-End Fine-tuning
  └─ Joint optimization with differential learning rates
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/taminulislam/trace.git
cd trace
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate trace
```

### 3. Install pip dependencies

```bash
pip install -r requirements.txt
```

> **Requirements:** Python 3.11, PyTorch >= 2.8, CUDA 12.8, transformers >= 4.40, timm >= 0.9

---

## Dataset

The CO₂ Farm Thermal Gas Dataset is not publicly available yet. Please contact the authors to request access.

Expected dataset structure:

```
dataset/
├── SEQ_XXXX/
│   ├── images/          # Thermal overlay frames (.png)
│   ├── masks/           # Binary gas segmentation masks (.png)
│   └── overlays/        # Visualization overlays (.png)
annotations/
├── annotations.csv      # Frame-level labels (sequence, frame, class)
├── clips.csv            # Clip-level metadata (sequence, class, split)
└── split_train_val_test.csv
```

**Classes:** High-Flux (HF), Control (Ctrl), Low-Flux (LF)
**Split:** 18 train / 3 val / 3 test sequences

---

## Training

### Full pipeline (all stages)

```bash
bash run_all.sh
```

### Individual stages

```bash
# Stage 1: Segmentation pretraining
python src/train/train_segmentation.py

# Stage 2: Temporal encoder
python src/train/train_temporal.py

# Stage 3: ATF Fusion (requires Stage 1 + 2 checkpoints)
python src/train/train_fusion.py \
    --seg_checkpoint outputs/checkpoints/segmentation/segmentation_latest.pt

# Stage 4: LLaVA LoRA (requires Stage 1 + 2 + 3 checkpoints)
python src/train/train_llava.py \
    --seg_checkpoint outputs/checkpoints/segmentation/segmentation_latest.pt \
    --temporal_checkpoint outputs/checkpoints/temporal/temporal_latest.pt \
    --fusion_checkpoint outputs/checkpoints/fusion/fusion_latest.pt

# Stage 5: End-to-end fine-tuning
python src/train/train_e2e.py \
    --seg_checkpoint outputs/checkpoints/segmentation/segmentation_latest.pt \
    --temporal_checkpoint outputs/checkpoints/temporal/temporal_latest.pt \
    --fusion_checkpoint outputs/checkpoints/fusion/fusion_latest.pt \
    --llava_checkpoint outputs/checkpoints/llava/llava_latest.pt
```

### Key environment variables

| Variable | Default | Description |
|:---------|:--------|:------------|
| `ANNOTATIONS_CSV` | `annotations/annotations.csv` | Path to annotations file |
| `CLIPS_CSV` | `annotations/clips.csv` | Path to clips file |
| `CHECKPOINT_DIR` | `outputs/checkpoints` | Checkpoint save directory |
| `LOG_DIR` | `outputs/logs` | Log directory |
| `WANDB_DISABLED` | `false` | Set to `true` to disable W&B logging |

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=2 src/train/train_segmentation.py
```

---

## Evaluation

```bash
python src/eval/evaluate.py \
    --checkpoint outputs/checkpoints/e2e/e2e_latest.pt \
    --annotations_csv annotations/annotations.csv \
    --clips_csv annotations/clips.csv
```

---

## Pre-trained Models

Pre-trained checkpoints will be available on HuggingFace Hub:

> [HuggingFace model page — coming soon]

| Checkpoint | Stage | Description |
|:-----------|:------|:------------|
| `trace-segmentation` | Stage 1 | TGAA-SegFormer segmentation |
| `trace-temporal` | Stage 2 | VideoMAE temporal encoder |
| `trace-fusion` | Stage 3 | ATF fusion model |
| `trace-e2e` | Stage 5 | Full end-to-end TRACE model |

---

## Project Structure

```
TRACE/
├── src/
│   ├── data/            # Dataset, augmentation, clip sampling
│   ├── models/          # TRACE, TGAA, ATF, temporal encoder, LLaVA-LoRA
│   ├── train/           # Per-stage training scripts
│   ├── eval/            # Evaluation and visualization
│   └── utils/           # Config dataclasses, trainer utilities
├── scripts/             # Visualization and comparison scripts
├── annotations/         # Dataset split CSVs
├── run_all.sh           # Full pipeline runner
├── requirements.txt
├── environment.yml
└── TRACE.png            # Architecture diagram
```

---

## Citation

```bibtex
@article{trace2026,
  title   = {TRACE},
  author  = {},
  year    = {2026},
}
```

> Citation will be updated upon publication.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
