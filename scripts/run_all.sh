#!/bin/bash
# TRACE Full Training Pipeline
# Runs all stages sequentially: 1 → 2 → 3 → 4
# Usage: nohup bash run_all.sh > logs/pipeline.log 2>&1 &
set -e

export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export HF_HUB_DISABLE_XET=1

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate trace

mkdir -p logs

echo "========================================="
echo "TRACE Training Pipeline"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "========================================="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)')
"
echo "========================================="

# ---- Stage 1: Segmentation ----
echo ""
echo ">>>>> STAGE 1: Segmentation Pretraining <<<<<"
echo "Started at: $(date)"
python src/train/train_segmentation.py 2>&1 | tee logs/segmentation_stdout.log
echo "Stage 1 finished at: $(date)"

# ---- Stage 2: Temporal ----
echo ""
echo ">>>>> STAGE 2: Temporal Encoder <<<<<"
echo "Started at: $(date)"
python src/train/train_temporal.py 2>&1 | tee logs/temporal_stdout.log
echo "Stage 2 finished at: $(date)"

# ---- Stage 3: ATF Fusion ----
SEG_CKPT="outputs/checkpoints/segmentation/segmentation_latest.pt"
echo ""
echo ">>>>> STAGE 3: ATF Fusion <<<<<"
echo "Started at: $(date)"
python src/train/train_fusion.py --seg_checkpoint "$SEG_CKPT" 2>&1 | tee logs/fusion_stdout.log
echo "Stage 3 finished at: $(date)"

# ---- Stage 4: End-to-End ----
TEMP_CKPT="outputs/checkpoints/temporal/temporal_latest.pt"
FUSION_CKPT="outputs/checkpoints/fusion/fusion_latest.pt"
echo ""
echo ">>>>> STAGE 4: End-to-End Fine-tuning <<<<<"
echo "Started at: $(date)"
python src/train/train_e2e.py \
    --seg_checkpoint "$SEG_CKPT" \
    --temporal_checkpoint "$TEMP_CKPT" \
    --fusion_checkpoint "$FUSION_CKPT" 2>&1 | tee logs/e2e_stdout.log
echo "Stage 4 finished at: $(date)"

echo ""
echo "========================================="
echo "ALL STAGES COMPLETE"
echo "Finished at: $(date)"
echo "========================================="
