#!/bin/bash
#SBATCH --job-name=st-eval-point
# Submit from zenodo_open_research. Set --account and --partition for your cluster.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval_point_%j.out
#SBATCH --error=logs/eval_point_%j.err

# 2020-2024 point inference for the separately trained model and the shared finetune.

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
    --checkpoint outputs/standalone/checkpoints/point/best_model.pt \
    --period test \
    --nodes-csv outputs/standalone/heldout_comids.csv \
    --output outputs/standalone/point_test.parquet
torchrun --standalone --nproc_per_node=1 code/evaluation/inference_v24_v25.py \
    --version v24 \
    --checkpoint weights/checkpoints_v24/finetune/best_model.pt \
    --period test \
    --nodes-csv outputs/standalone/heldout_comids.csv \
    --output outputs/standalone/point_paper_h2_reeval_test.parquet
python code/standalone/score_point.py
