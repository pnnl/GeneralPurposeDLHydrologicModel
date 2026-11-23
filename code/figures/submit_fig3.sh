#!/bin/bash
#SBATCH --job-name=fig3-calib
# Submit from zenodo_open_research. Set --account and --partition for your cluster.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=logs/fig3_calib_%j.out
#SBATCH --error=logs/fig3_calib_%j.err

# Figure 7 (predictive calibration and sharpness). CPU only.
# Reads outputs/predictions_downscaled/ens100_wrftgw_2015_2024.zarr, which is not
# included in this bundle, and the observation parquets under outputs/.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs

export PYTHONPATH="$ROOT/code/training:$ROOT/code/evaluation:$ROOT/code/figures:$ROOT/code/standalone${PYTHONPATH:+:$PYTHONPATH}"

echo "fig3-calib job ${SLURM_JOB_ID:-local} on ${SLURM_NODELIST:-local}  start $(date)"
python code/figures/fig3_calibration_sharpness.py
echo "fig3-calib done $(date)"
ls -la outputs/figures/predictive_calibration_sharpness.pdf \
       outputs/tables/fig3_*.csv
