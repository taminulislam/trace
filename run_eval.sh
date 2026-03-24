#!/bin/bash
# One-command evaluation and visualization
# Runs after training to generate all CVPR figures + metrics
set -e

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate trace
cd "$(dirname "$0")"

mkdir -p outputs/eval_results outputs/figures

# Auto-detect latest checkpoints
SEG_CKPT=$(ls -t outputs/checkpoints/segmentation/*.pt 2>/dev/null | head -1)
TEMP_CKPT=$(ls -t outputs/checkpoints/temporal/*.pt 2>/dev/null | head -1)
FUSION_CKPT=$(ls -t outputs/checkpoints/fusion/*.pt 2>/dev/null | head -1)

echo "========================================="
echo "  TRACE Evaluation Pipeline"
echo "========================================="
echo "  Seg checkpoint: $SEG_CKPT"
echo "  Temporal checkpoint: $TEMP_CKPT"
echo "  Fusion checkpoint: $FUSION_CKPT"
echo "========================================="

# Step 1: Quantitative evaluation
echo ""
echo ">>> Step 1: Quantitative Evaluation <<<"
EVAL_ARGS=""
[ -n "$SEG_CKPT" ] && EVAL_ARGS="$EVAL_ARGS --seg_checkpoint $SEG_CKPT"
[ -n "$TEMP_CKPT" ] && EVAL_ARGS="$EVAL_ARGS --temporal_checkpoint $TEMP_CKPT"
python src/eval/evaluate.py $EVAL_ARGS

# Step 2: Visualizations
echo ""
echo ">>> Step 2: Generating Visualizations <<<"
VIS_ARGS=""
[ -n "$SEG_CKPT" ] && VIS_ARGS="$VIS_ARGS --seg_checkpoint $SEG_CKPT"
[ -f "outputs/eval_results/classification_raw.npz" ] && \
    VIS_ARGS="$VIS_ARGS --raw_npz outputs/eval_results/classification_raw.npz"
python src/eval/visualize.py $VIS_ARGS

echo ""
echo "========================================="
echo "  EVALUATION COMPLETE"
echo "  Results: outputs/eval_results/"
echo "  Figures: outputs/figures/"
echo "========================================="
