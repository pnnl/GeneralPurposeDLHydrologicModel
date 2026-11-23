#!/bin/bash
#SBATCH --job-name=gnn-infer-test
# Submit from the repository root. Set --account and --partition for your cluster.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=logs/inference_test_%j.out
#SBATCH --error=logs/inference_test_%j.err

# Test-period (2020-2024) deterministic streamflow inference.
# From the repository root, after mkdir -p logs:
#   sbatch code/evaluation/submit_inference_test.sh v24 \
#       weights/checkpoints_v24/pretrain/best_model.pt \
#       outputs/predictions_v24_pretrain_test.parquet
#   sbatch code/evaluation/submit_inference_test.sh v24 \
#       weights/checkpoints_v24/finetune/best_model.pt \
#       outputs/predictions_v24_finetune_test.parquet

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs outputs

VERSION="${1:-}"
CHECKPOINT="${2:-}"
OUTPUT="${3:-}"
if [[ -z "$VERSION" || -z "$CHECKPOINT" || -z "$OUTPUT" ]]; then
    echo "Usage: sbatch code/evaluation/submit_inference_test.sh {v24|v25} CHECKPOINT OUTPUT.parquet"
    exit 1
fi
echo "Job ${SLURM_JOB_ID:-local} | $VERSION | $CHECKPOINT -> $OUTPUT | $(date)"

export PYTHONPATH="$ROOT/code/training:$ROOT/code/evaluation:$ROOT/code/standalone${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nproc_per_node=1 --master_port=29556 \
    code/evaluation/inference_v24_v25.py \
    --version "$VERSION" \
    --checkpoint "$CHECKPOINT" \
    --period test \
    --output "$OUTPUT"
echo "End: $(date)"
