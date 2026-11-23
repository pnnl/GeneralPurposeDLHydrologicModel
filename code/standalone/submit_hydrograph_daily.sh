#!/bin/bash
#SBATCH --job-name=st-hydro-wy2022
# Submit from the repository root. Set --account and --partition for your cluster.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=logs/hydrograph_daily_%j.out
#SBATCH --error=logs/hydrograph_daily_%j.err

# Daily mixture means for the hydrograph catchments.
# Needs code/standalone/hydrograph_catchments.json, which is not included in this repository.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs outputs/standalone

export PYTHONPATH="$ROOT/code/standalone:$ROOT/code/training:$ROOT/code/evaluation${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

torchrun --standalone --nproc_per_node=1 code/standalone/save_hydrograph_daily.py
