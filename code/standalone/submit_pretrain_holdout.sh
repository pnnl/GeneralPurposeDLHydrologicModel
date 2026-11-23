#!/bin/bash
#SBATCH --job-name=st-pretrain-pub
# Submit from zenodo_open_research. Set --account and --partition for your cluster.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/pretrain_holdout_%j.out
#SBATCH --error=logs/pretrain_holdout_%j.err

# Ungauged predictions from the shared time-series checkpoint, 2020-2024 holdouts.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs outputs/standalone

export PYTHONPATH="$ROOT/code/standalone:$ROOT/code/training:$ROOT/code/evaluation${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

python code/standalone/prepare_eval_ids.py
torchrun --standalone --nproc_per_node=1 code/evaluation/inference_v24_v25.py \
    --version v24 \
    --checkpoint weights/checkpoints_v24/pretrain/best_model.pt \
    --period test \
    --nodes-csv outputs/standalone/heldout_comids.csv \
    --output outputs/standalone/point_pretrain_holdout_test.parquet
