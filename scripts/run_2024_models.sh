#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  Sequential Training — CVPR/ECCV 2024 Lightweight Models
#
#  Runs all 4 models one after another. When one finishes (train + eval +
#  viz), the next starts automatically.
#
#  Models (in order):
#    1. RepViT-M1      (CVPR 2024, ~6.8M)
#    2. SHViT-S4       (CVPR 2024, ~16.6M)
#    3. StarNet-S2     (CVPR 2024, ~3.7M)
#    4. MobileNetV4-Conv-S (ECCV 2024, ~3.8M)
#
#  Usage:
#    bash scripts/run_2024_models.sh
#
#  Optional env overrides:
#    SEG_1A_EPOCHS=8 SEG_1B_EPOCHS=12 bash scripts/run_2024_models.sh
# ═══════════════════════════════════════════════════════════════════════════
set -e

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate trace

PROJECT="$(dirname "$0")"
cd $PROJECT

export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export HF_HUB_DISABLE_XET=1
export WANDB_DISABLED="true"

export SEG_1A_EPOCHS=${SEG_1A_EPOCHS:-5}
export SEG_1B_EPOCHS=${SEG_1B_EPOCHS:-8}

OVERALL_START=$(date +%s)

echo "═══════════════════════════════════════════════════════════════════════════"
echo "  2024 Lightweight Model Sweep"
echo "  Models: RepViT-M1 → SHViT-S4 → StarNet-S2 → MobileNetV4-Conv-S"
echo "  Epochs per model: 1a=${SEG_1A_EPOCHS}  1b=${SEG_1B_EPOCHS}"
echo "  Started: $(date)"
echo "═══════════════════════════════════════════════════════════════════════════"

# ─── Helper: run one full experiment ───────────────────────────────────────
run_experiment() {
    local MODEL_NAME=$1
    local EXP_TAG=$2           # short tag used in directory name, e.g. Exp08_repvit_m1

    local EXP_DIR=$PROJECT/outputs/$EXP_TAG
    local CKPT_DIR=$EXP_DIR/checkpoints
    local LOG_DIR=$EXP_DIR/logs

    export MODEL_NAME=$MODEL_NAME
    export CHECKPOINT_DIR=$CKPT_DIR
    export LOG_DIR=$LOG_DIR

    mkdir -p $CKPT_DIR/segmentation $LOG_DIR $EXP_DIR/figures $EXP_DIR/eval_results

    local START=$(date +%s)
    echo ""
    echo "┌──────────────────────────────────────────────────────────────────────"
    echo "│  START  $MODEL_NAME"
    echo "│  Dir:   $EXP_DIR"
    echo "│  Time:  $(date)"
    echo "└──────────────────────────────────────────────────────────────────────"

    # ── Stage 1: Segmentation training ──
    echo ""
    echo "  >>> Stage 1: Segmentation (${SEG_1A_EPOCHS}+${SEG_1B_EPOCHS} epochs)"
    python src/train/train_segmentation.py 2>&1 | tee $LOG_DIR/stage1_stdout.log
    echo "  Stage 1 done at: $(date)"

    SEG_CKPT=$(ls -t $CKPT_DIR/segmentation/*.pt 2>/dev/null | head -1)
    if [ -z "$SEG_CKPT" ]; then
        echo "  ERROR: No checkpoint found for $MODEL_NAME — skipping eval"
        return 1
    fi
    echo "  Seg checkpoint: $SEG_CKPT"

    # ── Evaluation ──
    echo ""
    echo "  >>> Evaluation"
    python src/eval/evaluate.py \
        --seg_checkpoint "$SEG_CKPT" \
        --output_dir $EXP_DIR/eval_results \
        --model_name "$MODEL_NAME" \
        2>&1 | tee $LOG_DIR/eval_stdout.log
    echo "  Evaluation done at: $(date)"

    # ── Visualizations ──
    echo ""
    echo "  >>> Visualizations"
    python src/eval/visualize.py \
        --seg_checkpoint "$SEG_CKPT" \
        --raw_npz $EXP_DIR/eval_results/classification_raw.npz \
        --results_json $EXP_DIR/eval_results/eval_results.json \
        --output_dir $EXP_DIR/figures \
        --model_name "$MODEL_NAME" \
        2>&1 | tee $LOG_DIR/viz_stdout.log
    echo "  Visualizations done at: $(date)"

    local END=$(date +%s)
    local ELAPSED=$(( (END - START) / 60 ))
    echo ""
    echo "┌──────────────────────────────────────────────────────────────────────"
    echo "│  DONE   $MODEL_NAME  (${ELAPSED} min)"
    echo "│  Results: $EXP_DIR/eval_results/"
    echo "│  Figures: $EXP_DIR/figures/"
    echo "└──────────────────────────────────────────────────────────────────────"
}

# ─── Run all 4 models sequentially ─────────────────────────────────────────

run_experiment "repvit_m1"         "Exp08_repvit_m1"
run_experiment "shvit_s4"          "Exp09_shvit_s4"
run_experiment "starnet_s2"        "Exp10_starnet_s2"
run_experiment "mobilenetv4_conv_s" "Exp11_mobilenetv4_conv_s"

# ─── Final summary ──────────────────────────────────────────────────────────
OVERALL_END=$(date +%s)
OVERALL_MIN=$(( (OVERALL_END - OVERALL_START) / 60 ))

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  ALL 4 MODELS COMPLETE"
echo "  Total time: ${OVERALL_MIN} min"
echo "  Finished:   $(date)"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "  Run comparison table:"
echo "    python scripts/compare_results.py"
echo ""
