#!/bin/bash
#SBATCH --job-name=st-eval-mdn
# Submit from the repository root. Set --account and --partition for your cluster.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval_mdn_%j.out
#SBATCH --error=logs/eval_mdn_%j.err

# 2020-2024 mixture scores for the separately trained model and the shared frozen head.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs outputs/standalone

export PYTHONPATH="$ROOT/code/standalone:$ROOT/code/training:$ROOT/code/evaluation${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

torchrun --standalone --nproc_per_node=1 code/evaluation/evaluate_d2.py \
    --checkpoint outputs/standalone/checkpoints/mdn/best_model.pt \
    --basin-set heldout --years test \
    --output outputs/standalone/mdn_standalone_test.csv
torchrun --standalone --nproc_per_node=1 code/evaluation/evaluate_d2.py \
    --checkpoint weights/checkpoints_d2/finetune_frozen/best_model.pt \
    --basin-set heldout --years test \
    --output outputs/standalone/mdn_paper_h3_reeval_test.csv
