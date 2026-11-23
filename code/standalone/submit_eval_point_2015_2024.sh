#!/bin/bash
#SBATCH --job-name=st-point-2015-2024
# Submit from the repository root. Set --account and --partition for your cluster.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_point_2015_2024_%j.out
#SBATCH --error=logs/eval_point_2015_2024_%j.err

# Point inference for 2015-2024, then the paired comparison.

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
    --start-year 2015 --end-year 2024 \
    --forcings-dir data/forcings_processed_hruid \
    --nodes-csv outputs/standalone/heldout_comids.csv \
    --output outputs/standalone/point_2015_2024_standalone.parquet
torchrun --standalone --nproc_per_node=1 code/evaluation/inference_v24_v25.py \
    --version v24 \
    --checkpoint weights/checkpoints_v24/finetune/best_model.pt \
    --start-year 2015 --end-year 2024 \
    --forcings-dir data/forcings_processed_hruid \
    --nodes-csv outputs/standalone/heldout_comids.csv \
    --output outputs/standalone/point_2015_2024_shared.parquet
python code/standalone/pair_span.py point \
    --standalone outputs/standalone/point_2015_2024_standalone.parquet \
    --shared outputs/standalone/point_2015_2024_shared.parquet \
    --years 2015-2024 \
    --output outputs/standalone/point_comparison_2015_2024.json \
    --paired-csv outputs/standalone/point_paired_2015_2024.csv
