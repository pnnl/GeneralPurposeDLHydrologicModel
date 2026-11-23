#!/bin/bash
#SBATCH --job-name=st-mdn-2015-2024
# Submit from zenodo_open_research. Set --account and --partition for your cluster.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_mdn_2015_2024_%j.out
#SBATCH --error=logs/eval_mdn_2015_2024_%j.err

# Mixture scores for 2015-2024, then the paired comparison.
# The separately trained checkpoint is outputs/standalone/checkpoints/mdn/best_model.pt.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs outputs/standalone

export PYTHONPATH="$ROOT/code/standalone:$ROOT/code/training:$ROOT/code/evaluation${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

torchrun --standalone --nproc_per_node=1 code/evaluation/evaluate_d2.py \
    --checkpoint outputs/standalone/checkpoints/mdn/best_model.pt \
    --basin-set heldout --start-year 2015 --end-year 2024 \
    --output outputs/standalone/mdn_standalone_2015_2024.csv
torchrun --standalone --nproc_per_node=1 code/evaluation/evaluate_d2.py \
    --checkpoint weights/checkpoints_d2/finetune_frozen/best_model.pt \
    --basin-set heldout --start-year 2015 --end-year 2024 \
    --output outputs/standalone/mdn_shared_2015_2024.csv
python code/standalone/pair_span.py mdn \
    --standalone outputs/standalone/mdn_standalone_2015_2024.csv \
    --shared outputs/standalone/mdn_shared_2015_2024.csv \
    --years 2015-2024 \
    --output outputs/standalone/mdn_comparison_2015_2024.json \
    --paired-csv outputs/standalone/mdn_paired_2015_2024.csv
