#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Generic Experiment Runner — Segmentation Training + Evaluation
#  Set MODEL_NAME and EXP_DIR before calling this script
# ═══════════════════════════════════════════════════════════════
set -e

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate trace

PROJECT="$(dirname "$0")"
cd $PROJECT

# Validate env vars
if [ -z "$MODEL_NAME" ] || [ -z "$EXP_DIR" ]; then
    echo "ERROR: Set MODEL_NAME and EXP_DIR before running"
    exit 1
fi

# Override paths
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export HF_HUB_DISABLE_XET=1
export WANDB_DISABLED="true"
export CHECKPOINT_DIR=$EXP_DIR/checkpoints
export LOG_DIR=$EXP_DIR/logs

# Epochs
export SEG_1A_EPOCHS=${SEG_1A_EPOCHS:-8}
export SEG_1B_EPOCHS=${SEG_1B_EPOCHS:-12}

mkdir -p $CHECKPOINT_DIR/segmentation $LOG_DIR $EXP_DIR/figures $EXP_DIR/eval_results

echo "═══════════════════════════════════════════════════════════"
echo "  Experiment: $MODEL_NAME"
echo "  Epochs: 1a=${SEG_1A_EPOCHS} 1b=${SEG_1B_EPOCHS}"
echo "  Output: $EXP_DIR"
echo "  Started: $(date)"
echo "═══════════════════════════════════════════════════════════"

# ── STAGE 1: Segmentation ──
echo ""
echo ">>>>> STAGE 1: Segmentation (${SEG_1A_EPOCHS}+${SEG_1B_EPOCHS} epochs) <<<<<"
python src/train/train_segmentation.py 2>&1 | tee $LOG_DIR/stage1_stdout.log
echo "Stage 1 finished at: $(date)"

SEG_CKPT=$(ls -t $CHECKPOINT_DIR/segmentation/*.pt 2>/dev/null | head -1)
echo "Seg checkpoint: $SEG_CKPT"

# ── EVALUATION ──
echo ""
echo ">>>>> EVALUATION <<<<<"
python src/eval/evaluate.py \
    --seg_checkpoint "$SEG_CKPT" \
    --output_dir $EXP_DIR/eval_results \
    --model_name "$MODEL_NAME" \
    2>&1 | tee $LOG_DIR/eval_stdout.log

# ── VISUALIZATIONS ──
echo ""
echo ">>>>> VISUALIZATIONS <<<<<"
python src/eval/visualize.py \
    --seg_checkpoint "$SEG_CKPT" \
    --raw_npz $EXP_DIR/eval_results/classification_raw.npz \
    --results_json $EXP_DIR/eval_results/eval_results.json \
    --output_dir $EXP_DIR/figures \
    --model_name "$MODEL_NAME" \
    2>&1 | tee $LOG_DIR/viz_stdout.log

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  $MODEL_NAME — COMPLETE"
echo "  Finished: $(date)"
echo "  Results:  $EXP_DIR/eval_results/"
echo "  Figures:  $EXP_DIR/figures/"
echo "═══════════════════════════════════════════════════════════"
