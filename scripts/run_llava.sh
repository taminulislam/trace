#!/bin/bash
# Stage 4: LLaVA LoRA Fine-tuning (single GPU)
set -e

export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export HF_HUB_DISABLE_XET=1

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate trace
cd "$(dirname "$0")"

echo "========================================="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "========================================="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)')
"
echo "========================================="

SEG_CKPT="outputs/checkpoints/segmentation/segmentation_latest.pt"
TEMP_CKPT="outputs/checkpoints/temporal/temporal_latest.pt"
FUSION_CKPT="outputs/checkpoints/fusion/fusion_latest.pt"

python src/train/train_llava.py --resume \
    --seg_checkpoint "$SEG_CKPT" \
    --temporal_checkpoint "$TEMP_CKPT" \
    --fusion_checkpoint "$FUSION_CKPT" 2>&1 | tee logs/llava_stdout.log

echo "========================================="
echo "Stage 4 (LLaVA LoRA) finished at $(date)"
echo "========================================="
