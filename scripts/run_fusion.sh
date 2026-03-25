#!/bin/bash
# Stage 3: ATF Fusion Training (single GPU)
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

python src/train/train_fusion.py --resume --seg_checkpoint "$SEG_CKPT" 2>&1 | tee logs/fusion_stdout.log

echo "========================================="
echo "Stage 3 (ATF Fusion) finished at $(date)"
echo "========================================="
