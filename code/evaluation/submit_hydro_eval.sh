#!/bin/bash
#SBATCH --job-name=hydro-eval-3head
# Submit from the repository root. Set --account and --partition for your cluster.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/hydro_eval_%j.out
#SBATCH --error=logs/hydro_eval_%j.err

# CPU scoring of KGE parts and hydrologic signatures.
# Reads inference parquets from outputs/ and writes outputs/eval_hydro/.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs outputs/eval_hydro

export PYTHONPATH="$ROOT/code/training:$ROOT/code/evaluation:$ROOT/code/standalone${PYTHONPATH:+:$PYTHONPATH}"

echo "hydro eval start $(date)"
python code/evaluation/evaluate_hydro_signatures.py \
  --gnn-root "$ROOT" \
  --outdir "$ROOT/outputs/eval_hydro"
echo "hydro eval end $(date)"
