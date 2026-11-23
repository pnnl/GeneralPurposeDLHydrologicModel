#!/bin/bash
#SBATCH --job-name=gnn-v24
# Submit from the repository root. Set --account and --partition for your cluster.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=900G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_v24_%j.out
#SBATCH --error=logs/train_v24_%j.err

# Deterministic LSTM-DirectedGAT training. New runs write outputs/checkpoints_v24 and leave weights/ in place.
#
# From the repository root, after mkdir -p logs:
#   sbatch code/training/submit_v24.sh pretrain
#   sbatch code/training/submit_v24.sh finetune
#
# Finetune loads outputs/checkpoints_v24/pretrain/best_model.pt.
# To start from the included point model:
#   sbatch code/training/submit_v24.sh finetune \
#       --pretrain-checkpoint weights/checkpoints_v24/pretrain/best_model.pt

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs outputs/checkpoints_v24/pretrain outputs/checkpoints_v24/finetune

if [[ -z "${1:-}" ]]; then
    echo "ERROR: Must specify stage"
    echo "Usage: sbatch code/training/submit_v24.sh {pretrain|finetune} [options]"
    exit 1
fi

STAGE="$1"
shift

case "$STAGE" in
    pretrain|finetune) ;;
    *) echo "ERROR: Stage must be pretrain or finetune"; exit 1 ;;
esac

TRAIN_ARGS=("$@")

echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-local}"
echo "Start time: $(date)"
echo "Training v24. Stage: ${STAGE}"
echo "=============================================="

export PYTHONPATH="$ROOT/code/training:$ROOT/code/evaluation:$ROOT/code/standalone${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000
export NCCL_TIMEOUT=3600000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TORCH_CUDNN_V8_API_ENABLED=1

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

NGPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NGPUS GPUs"

python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

echo ""
echo "Starting v24 ${STAGE} with $NGPUS GPUs"
echo "  Forcings: data/forcings_processed_hruid/forcings_f16.zarr"
echo "  Streamflow: data/streamflow/"

if [[ "$STAGE" == "pretrain" ]]; then
    echo "Defaults: --warmup-days 365 --seq-length 180 --lr 5e-4 --gauge-dropout-rate 0.0"
elif [[ "$STAGE" == "finetune" ]]; then
    echo "Defaults: --warmup-days 365 --seq-length 180 --lr 1e-4 --gauge-dropout-rate 0.15"
    echo "Auto-loading pretrain weights from: outputs/checkpoints_v24/pretrain/best_model.pt"
fi
echo ""

torchrun \
    --standalone \
    --nproc_per_node="$NGPUS" \
    --master_port=29513 \
    code/training/train_multi_gpu_v24.py \
    "$STAGE" \
    ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}

echo ""
echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
