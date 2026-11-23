# General-Purpose Deep Learning Hydrologic Model

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24145836.svg)](https://doi.org/10.5281/zenodo.24145836)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A graph-based deep learning framework for daily streamflow prediction and uncertainty quantification across North American river basins. The model combines a Long Short-Term Memory (**LSTM**) sequence encoder with a Directed Graph Attention Network (**DirectedGAT**) that routes information downstream along river networks, supporting:

1. **Gauged Streamflow Prediction** — Daily deterministic streamflow forecasting at monitored catchments.
2. **Ungauged Basin Prediction** — Spatial generalization to unmonitored catchments via graph message passing.
3. **Probabilistic Streamflow Prediction** — Calibrated predictive distributions using a Mixture Density Network (**MDN**) head on top of the shared spatiotemporal encoder.

---

## Repository Structure

```text
GeneralPurposeDLHydrologicModel/
├── code/
│   ├── training/       # Model definitions and multi-GPU training scripts
│   ├── evaluation/     # Deterministic, probabilistic, and hydrologic signature evaluation
│   ├── standalone/     # Independently trained baseline models for comparison
│   └── figures/        # Visualization, spatial mapping, and summary table scripts
├── data/               # Input datasets from the companion Zenodo archive (placed locally)
├── weights/            # Pretrained model checkpoints from the companion archive (placed locally)
└── LICENSE
```

All scripts resolve paths relative to the repository root (`code/training/config.py` sets `base_dir` automatically), and the included Slurm submission scripts configure `PYTHONPATH` across `code/training`, `code/evaluation`, and `code/standalone`.

---

## Data and Pretrained Weights Setup

Processed input datasets (`data/`) and pretrained model checkpoints (`weights/`) are publicly archived on Zenodo at **[https://doi.org/10.5281/zenodo.24145836](https://doi.org/10.5281/zenodo.24145836)**. Download and place `data/` and `weights/` at the root of this repository alongside `code/` before running training or evaluation.

### Expected `data/` Layout

| Directory | Description | Primary Files |
| --- | --- | --- |
| `data/forcings_processed_hruid/` | Daily catchment meteorological forcings (1979-01-01 to 2024-12-31, `float16`) | `forcings_f16.zarr`, `catchment_index.parquet` |
| `data/catchment_network/` | River network topology for directed graph message passing | `catchment_network.gpkg` (layers: `catchments_network`, `network_edges`) |
| `data/static/` | Static physiographic, climatic, and land-surface catchment attributes (40 features) | `catchment_static_attributes.parquet` |
| `data/streamflow/` | Quality-controlled daily streamflow observations (USGS, HYDAT, and CONAGUA) | `{usgs,hydat,conagua}/comid_*.parquet`, `streamflow_index.csv`, `streamflow_master_index.csv` |

**Meteorological forcing variables (in input channel order):**
`precipitation_mm`, `temperature_mean_C`, `temperature_min_C`, `temperature_max_C`, `shortwave_radiation_Wm2`, `longwave_radiation_Wm2`, `wind_speed_ms`, and `vapor_pressure_Pa`.

### Expected `weights/` Layout

- **Deterministic model checkpoints:** `weights/checkpoints_v24/{pretrain,finetune}/best_model.pt`
  - `pretrain`: Temporally trained deterministic model
  - `finetune`: Spatially fine-tuned model for ungauged basin prediction
- **Probabilistic (MDN) checkpoints:** `weights/checkpoints_d2/{warmup,pretrain,finetune,finetune_frozen}/best_model.pt`
  - `finetune_frozen`: Probabilistic head trained with the pretrained encoder held fixed
- Each checkpoint directory includes `split_metadata.json` defining the training and held-out evaluation basin split (8,677 training / 965 held-out basins).

New training runs write outputs to `outputs/` without overwriting `weights/`.

---

## Usage

### Environment and Slurm Notes

Before submitting jobs on a Slurm cluster:
1. Activate a Python environment with PyTorch, PyTorch Geometric, Zarr, Xarray, GeoPandas, and Matplotlib installed.
2. Add your cluster's `#SBATCH --account` and `#SBATCH --partition` directives to the `.sh` scripts if required by your scheduler.
3. Create a `logs/` directory in the working directory where `sbatch` is invoked:
   ```bash
   mkdir -p logs
   ```

### 1. Training (`code/training/`)

Train the deterministic LSTM-DirectedGAT model (`train_multi_gpu_v24.py`) and the probabilistic MDN head (`train_multi_gpu_d2.py`):

```bash
# Deterministic model: temporal pretraining followed by spatial fine-tuning
sbatch code/training/submit_v24.sh pretrain
sbatch code/training/submit_v24.sh finetune

# Probabilistic MDN head: warmup, pretraining, and fine-tuning
sbatch code/training/submit_d2.sh warmup
sbatch code/training/submit_d2.sh pretrain
sbatch code/training/submit_d2.sh finetune
```

To fine-tune directly from the provided pretrained deterministic weights, pass `--pretrain-checkpoint weights/checkpoints_v24/pretrain/best_model.pt`.

### 2. Evaluation (`code/evaluation/`)

Generate streamflow predictions, probabilistic calibration metrics (CRPS, PIT, reliability), and hydrologic signatures:

```bash
# Deterministic inference on held-out basins
sbatch code/evaluation/submit_inference_test.sh v24 \
    weights/checkpoints_v24/finetune/best_model.pt \
    outputs/predictions_v24_finetune_test.parquet

# Probabilistic MDN evaluation on held-out basins
sbatch code/evaluation/submit_eval_d2.sh \
    weights/checkpoints_d2/finetune_frozen/best_model.pt --basin-set heldout

# Hydrologic signature evaluation
sbatch code/evaluation/submit_hydro_eval.sh
```

`build_3head_table.py` compiles summary performance tables across deterministic and probabilistic evaluation outputs in `outputs/`.

### 3. Standalone Baselines (`code/standalone/`)

Train and evaluate independent deterministic (`point`) and probabilistic (`mdn`) models from scratch on the same basin split for comparison with the shared-encoder model:

```bash
sbatch code/standalone/submit_train.sh point
sbatch code/standalone/submit_train.sh mdn

sbatch code/standalone/submit_eval_point.sh
sbatch code/standalone/submit_eval_mdn.sh
sbatch code/standalone/submit_pretrain_holdout.sh
sbatch code/standalone/submit_eval_point_2015_2024.sh
sbatch code/standalone/submit_eval_mdn_2015_2024.sh
```

Checkpoints and comparison tables are saved under `outputs/standalone/`.

### 4. Figures and Analysis (`code/figures/`)

Scripts in `code/figures/` generate spatial maps, calibration plots, hydroclimate-stratified summaries, and bootstrap confidence intervals, saving figures to `outputs/figures/`:

| Script | Output |
| --- | --- |
| `architecture/model_structure.tex` | Model architecture schematic (LaTeX/TikZ) |
| `plot_study_domain.py` | Study domain and stream gauge network map |
| `plot_spatial_kge.py` | Spatial Kling-Gupta Efficiency (KGE) performance maps |
| `plot_head12_hydro.py` | Deterministic streamflow and hydrologic signature comparisons |
| `plot_spatial_d2_uq.py` | Spatial probabilistic uncertainty quantification maps |
| `fig3_calibration_sharpness.py` | Probabilistic calibration and sharpness diagnostics |
| `fig2_hydroclimate_stratified.py` | Performance stratified by hydroclimatic regime and basin regulation |
| `plot_consistency_violin.py` | Cross-task consistency distributions |
| `plot_pretrain_consistency.py` | Pretrained vs. fine-tuning consistency diagnostics |
| `plot_within_mixture.py` | Mixture component diagnostics |
| `bootstrap_ci_tables.py` | Bootstrap confidence intervals for summary metrics |

Example:
```bash
python code/figures/plot_study_domain.py
sbatch code/figures/submit_fig3.sh
```

> **Note on optional figure inputs:** Certain figure scripts expect generated evaluation outputs in `outputs/` or optional auxiliary reference datasets (such as GAGES-II basin classifications at `data/gagesii/conterm_bas_classif.txt` for regulation stratification, downscaled ensemble outputs for ensemble comparison plots, or custom catchment lists in `code/standalone/hydrograph_catchments.json` for site-level hydrograph plots).

---

## Data Citation

> Konapala, G., Bracken, C., Son, Y., Birnbaum, A., Giovando, J., & Voisin, N. (2026). *MERIT catchment forcings, network, static attributes, and area-checked streamflow for GeneralPurposeDLHydrologicModel (Version 1.0)* [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.24145836

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
