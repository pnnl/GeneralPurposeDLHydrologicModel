#!/bin/bash
#SBATCH --job-name=d2-eval
# Submit from zenodo_open_research. Set --account and --partition for your cluster.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=900G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_d2_%j.out
#SBATCH --error=logs/eval_d2_%j.err

# Evaluate a D2 mixture checkpoint.
# From zenodo_open_research, after mkdir -p logs:
#   sbatch code/evaluation/submit_eval_d2.sh weights/checkpoints_d2/pretrain/best_model.pt
#   sbatch code/evaluation/submit_eval_d2.sh weights/checkpoints_d2/finetune_frozen/best_model.pt --basin-set heldout
# Extra arguments after the checkpoint are forwarded to evaluate_d2.py.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs

if [[ -z "${1:-}" ]]; then
    echo "ERROR: must pass a checkpoint path"
    exit 1
fi
CKPT="$1"
shift
EXTRA_ARGS=("$@")

export PYTHONPATH="$ROOT/code/training:$ROOT/code/evaluation:$ROOT/code/standalone${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

NGPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Eval D2 ckpt=$CKPT on $NGPUS GPUs | $(date)"

torchrun \
    --standalone \
    --nproc_per_node="$NGPUS" \
    --master_port=29520 \
    code/evaluation/evaluate_d2.py \
    --checkpoint "$CKPT" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "End: $(date)"
