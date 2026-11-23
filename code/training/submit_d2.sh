#!/bin/bash
#SBATCH --job-name=gnn-d2
# Submit from the repository root. Set --account and --partition for your cluster.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=900G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_d2_%j.out
#SBATCH --error=logs/train_d2_%j.err

# Probabilistic training: LSTM-DirectedGAT encoder with a log-normal MDN head.
# Warmup starts from weights/checkpoints_v24/finetune/best_model.pt.
# New checkpoints go to outputs/checkpoints_d2 and do not replace weights/.
#
# From the repository root, after mkdir -p logs:
#   sbatch code/training/submit_d2.sh warmup
#   sbatch code/training/submit_d2.sh pretrain
#   sbatch code/training/submit_d2.sh finetune

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs

if [[ -z "${1:-}" ]]; then
    echo "ERROR: must specify stage {warmup|pretrain|finetune}"
    exit 1
fi
STAGE="$1"
shift
case "$STAGE" in warmup|pretrain|finetune) ;; *) echo "bad stage"; exit 1 ;; esac
TRAIN_ARGS=("$@")

echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local} | Node: ${SLURM_NODELIST:-local} | $(date)"
echo "D2 MDN training. Stage: ${STAGE}"
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

nvidia-smi --query-gpu=index,name,memory.total --format=csv
NGPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NGPUS GPUs"

torchrun \
    --standalone \
    --nproc_per_node="$NGPUS" \
    --master_port=29519 \
    code/training/train_multi_gpu_d2.py \
    "$STAGE" \
    ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}

echo ""
echo "End time: $(date)"
