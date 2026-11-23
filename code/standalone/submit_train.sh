#!/bin/bash
#SBATCH --job-name=st-model
# Submit from zenodo_open_research. Set --account and --partition for your cluster.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=900G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/train_%x_%j.out
#SBATCH --error=logs/train_%x_%j.err

# Separately trained point model or mixture, from random weights.
# Checkpoints: outputs/standalone/checkpoints/{point,mdn}/best_model.pt
#
# From zenodo_open_research, after mkdir -p logs:
#   sbatch code/standalone/submit_train.sh point
#   sbatch code/standalone/submit_train.sh mdn

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs

MODEL="${1:?Use point or mdn}"
if [[ "$MODEL" != point && "$MODEL" != mdn ]]; then
    echo "Model must be point or mdn" >&2
    exit 2
fi

export PYTHONPATH="$ROOT/code/standalone:$ROOT/code/training:$ROOT/code/evaluation${PYTHONPATH:+:$PYTHONPATH}"
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

echo "Job ${SLURM_JOB_ID:-local} model=$MODEL node=${SLURM_NODELIST:-local} start=$(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
torchrun --standalone --nproc_per_node=8 code/standalone/train_standalone.py "$MODEL" --epochs 50
echo "Finished $(date)"
