#!/bin/bash
# Run Stage 4 (E2E) + Evaluation
set -e

export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export HF_HUB_DISABLE_XET=1

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate trace
cd "$(dirname "$0")"

SEG_CKPT=$(ls -t outputs/checkpoints/segmentation/*.pt 2>/dev/null | head -1)
TEMP_CKPT=$(ls -t outputs/checkpoints/temporal/*.pt 2>/dev/null | head -1)
FUSION_CKPT=$(ls -t outputs/checkpoints/fusion/*.pt 2>/dev/null | head -1)

echo "========================================="
echo "  Stage 4: E2E + Eval"
echo "========================================="

# Stage 4
echo ""
echo ">>>>> STAGE 4: E2E Fine-tuning <<<<<"
python src/train/train_e2e.py \
    --seg_checkpoint "$SEG_CKPT" \
    --temporal_checkpoint "$TEMP_CKPT" \
    --fusion_checkpoint "$FUSION_CKPT" 2>&1 | tee logs/e2e_stdout.log
echo "Stage 4 finished at: $(date)"

# Evaluation
echo ""
echo ">>>>> EVALUATION <<<<<"
bash run_eval.sh

echo ""
echo "========================================="
echo "PIPELINE COMPLETE"
echo "========================================="
