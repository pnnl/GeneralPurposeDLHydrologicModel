# GeneralPurposeDLHydrologicModel

This repository is the code for the public GitHub repository
GeneralPurposeDLHydrologicModel. `data/` and `weights/` stay in the Zenodo
research bundle. Place those two folders next to `code/` before you run the
scripts. Add the open-source `LICENSE` file from the approved EIDR before
the repository is made public.

Code, trained weights, and the processed MERIT inputs for the shared
LSTM-DirectedGAT streamflow paper. Training, evaluation, and the figures
start from the four data folders below. Raw WRF files, yearly forcing
parquets, VIC parameter files, and the raw agency gauge databases are not
included.

The folders are separate so each one can be uploaded on its own. Zenodo's
default record limit is 50 GB. `code/`, `weights/`, `data/catchment_network/`,
`data/static/`, and `data/streamflow/` fit in one record. Upload
`data/forcings_processed_hruid/forcings_f16.zarr` as its own record (about
39 GB). The float32 store is not included. Training reads the float16 store
when that directory is present.

## Paths

Every path in this README is relative to the GeneralPurposeDLHydrologicModel folder. `code/training/config.py`
sets `base_dir` from the location of that file, so the same relative paths
work after the folder is moved. The shell scripts change to this folder from
their own location. They put `code/training`, `code/evaluation`, and
`code/standalone` on `PYTHONPATH`.

The scripts do not name a cluster account or partition, and they do not
activate an environment. Add `#SBATCH --account` and `#SBATCH --partition`
for your cluster, and activate an environment that provides PyTorch, before
you submit. Slurm writes `logs/` in the directory where you call `sbatch`,
so create it there first:

```bash
mkdir -p logs
sbatch code/training/submit_v24.sh pretrain
```

## Data

| Folder | What it is | What the code opens |
| --- | --- | --- |
| `data/forcings_processed_hruid/` | Daily MERIT catchment forcings, 1979-01-01 through 2024-12-31, 8 variables, float16 | `data/forcings_processed_hruid/forcings_f16.zarr`. `catchment_index.parquet` is the HRUID index. |
| `data/catchment_network/` | MERIT river network used for message passing | `data/catchment_network/catchment_network.gpkg`, layers `catchments_network` and `network_edges` |
| `data/static/` | Static catchment attributes. The curated 40 attributes are taken from this file. | `data/static/catchment_static_attributes.parquet` |
| `data/streamflow/` | Area-checked daily streamflow at MERIT COMID, USGS, HYDAT, and CONAGUA | `data/streamflow/{usgs,hydat,conagua}/comid_*.parquet`. `streamflow_index.csv` and `streamflow_master_index.csv` are in the same folder. |

The eight forcing variables, in model order, are `precipitation_mm`,
`temperature_mean_C`, `temperature_min_C`, `temperature_max_C`,
`shortwave_radiation_Wm2`, `longwave_radiation_Wm2`, `wind_speed_ms`, and
`vapor_pressure_Pa`.

## Weights

`weights/checkpoints_v24/{pretrain,finetune}/best_model.pt` are the temporal
point model and the spatially fine-tuned point model.
`weights/checkpoints_d2/{warmup,pretrain,finetune,finetune_frozen}/best_model.pt`
are the mixture stages. The paper mixture is `finetune_frozen`, trained with
the encoder held fixed. Each directory also has `split_metadata.json` for the
8,677 / 965 basin split.

A new training run writes `outputs/checkpoints_v24/` or
`outputs/checkpoints_d2/` and leaves these files in place. To fine-tune from
the included point model, pass
`--pretrain-checkpoint weights/checkpoints_v24/pretrain/best_model.pt`.
Mixture warmup starts from
`weights/checkpoints_v24/finetune/best_model.pt`.

## Code

`code/training/`

Temporal point training is `train_multi_gpu_v24.py` (`submit_v24.sh`).
Mixture training is `train_multi_gpu_d2.py` (`submit_d2.sh`). Those entry
points import `train_multi_gpu_v15.py`, `train_multi_gpu_v16.py`, and
`train_multi_gpu_v20.py`. `train_multi_gpu_v17.py` is imported by the point
inference script.

```bash
sbatch code/training/submit_v24.sh pretrain
sbatch code/training/submit_v24.sh finetune
sbatch code/training/submit_d2.sh warmup
sbatch code/training/submit_d2.sh pretrain
sbatch code/training/submit_d2.sh finetune
```

`code/evaluation/`

`inference_v24_v25.py` writes the gauged and ungauged point predictions.
`evaluate_d2.py` scores a mixture checkpoint.
`build_3head_table.py` builds the three-product table from files under
`outputs/`.
`hydro_signatures.py`, `evaluate_hydro_signatures.py`, and
`evaluate_hydro_from_ens_zarr.py` compute the hydrograph signatures.
`FILEMAP.txt` maps these scripts to the paper sections.

```bash
sbatch code/evaluation/submit_inference_test.sh v24 \
    weights/checkpoints_v24/finetune/best_model.pt \
    outputs/predictions_v24_finetune_test.parquet
sbatch code/evaluation/submit_eval_d2.sh \
    weights/checkpoints_d2/finetune_frozen/best_model.pt --basin-set heldout
sbatch code/evaluation/submit_hydro_eval.sh
```

`code/figures/`

| Script | Figure |
| --- | --- |
| `architecture/model_structure.tex` | Figure 1 drawing source |
| `plot_study_domain.py` | Figure 2, gauge map |
| `plot_spatial_kge.py` | Figure 3, KGE map |
| `plot_head12_hydro.py` | Figure 4 and Supporting Information Figures S1 and S2 |
| `plot_spatial_d2_uq.py` | Figures 5 and 6 |
| `fig3_calibration_sharpness.py` | Figure 7 |
| `fig2_hydroclimate_stratified.py` | Figure 8 and Supporting Information Figure S4 |
| `plot_consistency_violin.py` | Figure 9 |
| `plot_pretrain_consistency.py` | Supporting Information Figures S5 and S6 |
| `plot_within_mixture.py` | Supporting Information Figures S7-S14 |
| `bootstrap_ci_tables.py` | Bootstrap intervals reported with the tables |
| `plot_calibration_d2.py` | Earlier calibration panels kept with the evaluation output |

Figure scripts read network, static attributes, and split files from `data/`
and `weights/`. Score tables and prediction parquets are read from `outputs/`.
Drawn figures are written to `outputs/figures/`. Latin Modern Roman is used
when those fonts are installed, or when the four `lmroman10-*.otf` files are
placed in `fonts/` next to `code/figures/paper_mpl_font.py`.

```bash
python code/figures/plot_study_domain.py
sbatch code/figures/submit_fig3.sh
```

`code/standalone/`

Separately trained point model and mixture. `train_standalone.py` imports the
training modules from `code/training` and starts from random weights on the
paper basin split. Checkpoints go to
`outputs/standalone/checkpoints/{point,mdn}/`. The `submit_eval_*.sh` and
`score_*.py` scripts score those checkpoints with the same inference and
metric code as the shared model. Comparison tables are written to
`outputs/standalone/`.

```bash
sbatch code/standalone/submit_train.sh point
sbatch code/standalone/submit_train.sh mdn
sbatch code/standalone/submit_eval_point.sh
sbatch code/standalone/submit_eval_mdn.sh
sbatch code/standalone/submit_pretrain_holdout.sh
sbatch code/standalone/submit_eval_point_2015_2024.sh
sbatch code/standalone/submit_eval_mdn_2015_2024.sh
```

`submit_within_mixture.sh` and `submit_hydrograph_daily.sh` also read
`code/standalone/hydrograph_catchments.json`. That file is not in this repository.

## Not included

The scripts look for these relative paths and stop if a file is missing.

- `data/gagesii/conterm_bas_classif.txt`, the GAGES-II basin class used by the management panel in Figure 8.
- `data/delineations/conus_MERIT_v1_catch.nov25.gpkg`. The included forcing store is indexed by HRUID, so training does not open this file.
- `outputs/predictions_downscaled/ens100_wrftgw_2015_2024.zarr`, the 100-member ensemble used by Figure 7.
- Prediction parquets and `outputs/eval_hydro/`. Inference and `submit_hydro_eval.sh` write them.
- `code/standalone/hydrograph_catchments.json`, the catchment list for the daily hydrograph and within-mixture scripts.
- Separately trained checkpoints. `submit_train.sh` writes them under `outputs/standalone/checkpoints/`.
