#!/usr/bin/env python3
"""
Inference for v24 (DirectedGAT) and v25 (EA-LSTM) models on test period (2020-2024).

Loads the v24 or v25 finetune best_model.pt, runs predictions for all gauged nodes
across the test period, and saves to parquet.

Usage, from the bundle root:
    python code/evaluation/inference_v24_v25.py --version v24 --checkpoint weights/checkpoints_v24/finetune/best_model.pt

    # Optional: predict for specific nodes only (e.g., NWM-comparable subset)
    python inference_v24_v25.py --version v24 --checkpoint ... --nodes-csv nwm_comparable_comids.csv
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings('ignore')
os.environ['PYTHONUNBUFFERED'] = '1'

BUNDLE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BUNDLE / "code" / "training"))

from train_multi_gpu_v15 import (
    SequenceDataModule, log, build_date_list, cleanup_shared_memory,
)
from train_multi_gpu_v20 import SequenceDataModuleV20, DROP_FEATURES, LOG_TRANSFORM_FEATURES
from train_multi_gpu_v16 import V16Trainer
from train_multi_gpu_v17 import V17Trainer
from config import get_config_v24


def normalize_state_dict(state_dict):
    out = {}
    for k, v in state_dict.items():
        nk = k
        for pfx in ("module.", "_orig_mod."):
            while nk.startswith(pfx):
                nk = nk[len(pfx):]
        out[nk] = v
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', choices=['v24', 'v25'], required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seq-length', type=int, default=180)
    parser.add_argument('--warmup-days', type=int, default=365)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--nodes-csv', type=str, default=None,
                        help='Optional CSV with column "comid" to restrict prediction to subset')
    parser.add_argument('--period', choices=['val', 'test', 'all'], default='val',
                        help='val (2015-2019), test (2020-2024), or all (full '
                             'store range, first year reserved for warmup)')
    parser.add_argument('--start-year', type=int, default=None,
                        help='First prediction year (overrides --period). Needs '
                             '365 days of forcings before it for LSTM warmup.')
    parser.add_argument('--end-year', type=int, default=None,
                        help='Last prediction year (overrides --period).')
    parser.add_argument('--forcings-dir', type=str, default=None,
                        help='Override config.data.forcings_processed_dir, e.g. '
                             'data/forcings_processed_era5 or '
                             'data/forcings_processed_ace2_ic0. Store must have '
                             "forcings_f16.zarr with key 'data' + 8 vars.")
    args = parser.parse_args()

    if args.output is None:
        args.output = f'predictions_{args.version}_{args.period}.parquet'

    log("=" * 60)
    log(f"INFERENCE — {args.version.upper()} on {args.period} period")
    log("=" * 60)
    log(f"Checkpoint: {args.checkpoint}")
    log(f"Output: {args.output}")

    # Single-GPU inference
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')
    os.environ.setdefault('LOCAL_RANK', '0')
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29555')

    config = get_config_v24()

    # Optional: drive inference from a different forcings store (downscaled
    # ERA5/ACE2). The store is a drop-in (same catchment_ids, 8 vars, key
    # 'data'); only the directory changes.
    if args.forcings_dir:
        config.data.forcings_processed_dir = Path(args.forcings_dir)
        log(f"Forcings override: {config.data.forcings_processed_dir}")

    # Full-record / custom-range prediction. The data module sizes its in-RAM
    # forcing window from the union of train/val/test years, so widening
    # train_years makes it preload the whole span. eval_years_override then
    # drives the prediction date list.
    eval_years_override = None
    if args.period == 'all' or args.start_year or args.end_year:
        import zarr as _zarr
        _zpath = Path(config.data.forcings_processed_dir) / 'forcings_f16.zarr'
        _zdates = pd.to_datetime(_zarr.open(str(_zpath), mode='r')['dates'][:])
        _store_y0, _store_y1 = int(_zdates.min().year), int(_zdates.max().year)
        # First store year is reserved for the 365-day LSTM warmup.
        y0 = args.start_year if args.start_year else _store_y0 + 1
        y1 = args.end_year if args.end_year else _store_y1
        eval_years_override = list(range(y0, y1 + 1))
        config.data.train_years = list(range(_store_y0, y1 + 1))  # widen load window
        log(f"Full-record prediction: years {y0}-{y1} "
            f"(store {_store_y0}-{_store_y1}, {len(eval_years_override)} yrs)")

    # Init distributed (world_size=1)
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', rank=0, world_size=1)

    log("\nLoading data module...")
    data_module = SequenceDataModuleV20(config, rank=0, world_size=1, warmup_days=args.warmup_days)

    # Build evaluation dates
    if eval_years_override is not None:
        eval_years = eval_years_override
    elif args.period == 'val':
        eval_years = config.data.val_years
    else:
        eval_years = config.data.test_years
    test_dates = build_date_list(eval_years)
    log(f"{args.period.upper()} dates: {len(test_dates)} (years {min(eval_years)}-{max(eval_years)})")

    # Optional: subset gauged nodes
    if args.nodes_csv and Path(args.nodes_csv).exists():
        sub_df = pd.read_csv(args.nodes_csv)
        sub_comids = set(int(c) for c in sub_df['comid'])
        all_gauged = data_module.train_gauged_nodes + data_module.eval_gauged_nodes
        eval_nodes = [n for n in all_gauged
                      if data_module.idx_to_comid.get(n) in sub_comids]
        log(f"Restricted to {len(eval_nodes)} nodes from {args.nodes_csv}")
    else:
        # All gauged nodes (train + eval/PUB held-out)
        eval_nodes = sorted(set(data_module.train_gauged_nodes + data_module.eval_gauged_nodes))
        log(f"Predicting for all {len(eval_nodes)} gauged nodes")

    # Initialize trainer (used for forward/inference logic)
    if args.version == 'v24':
        trainer = V16Trainer(
            config=config,
            data_module=data_module,
            seq_length=args.seq_length,
            warmup_days=args.warmup_days,
            loss_type='kge',
            warmup_epochs=0,
            sequences_per_epoch=200,
            gauge_dropout_rate=0.0,
            checkpoint_dir='/tmp',
            version='v24-inference',
        )
    else:  # v25
        trainer = V17Trainer(
            config=config,
            data_module=data_module,
            seq_length=args.seq_length,
            warmup_days=args.warmup_days,
            loss_type='kge',
            warmup_epochs=0,
            sequences_per_epoch=200,
            gauge_dropout_rate=0.0,
            checkpoint_dir='/tmp',
            version='v25-inference',
            model_type='ea-lstm',
        )
    trainer.batch_size = args.batch_size

    # Load checkpoint
    log(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
    state_dict = normalize_state_dict(ckpt['model_state_dict'])

    raw_model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    raw_model.load_state_dict(state_dict, strict=True)
    log(f"  Loaded weights from epoch {ckpt.get('epoch', '?')} "
        f"(val KGE: {ckpt.get('val_kge', 'unknown')})")

    # CRITICAL: normalize inputs with the TRAINING forcing stats, not stats
    # recomputed from the (different) downscaled forcings. The trainer computes
    # forcing_mean/std from whatever forcings are loaded; for cross-source
    # inference that silently shifts the model's inputs. Restore the checkpoint's
    # training stats so ERA5/ACE2 inputs are normalized exactly as in training.
    # (edge_feat stats are graph-derived and identical — same network topology —
    #  so no override is needed there.)
    if 'forcing_mean' in ckpt and 'forcing_std' in ckpt:
        trainer.forcing_mean = ckpt['forcing_mean'].to(trainer.device).float()
        trainer.forcing_std = ckpt['forcing_std'].to(trainer.device).float()
        log(f"  Restored training forcing normalization from checkpoint "
            f"(mean[0]={float(trainer.forcing_mean[0]):.3f})")
    else:
        log("  WARNING: checkpoint has no forcing_mean/std — using stats "
            "recomputed from the loaded forcings (may bias cross-source inference)")

    trainer.model.eval()

    # Iterate test period in non-overlapping seq_length chunks
    all_records = []
    gauged_nodes_arr = np.array(eval_nodes, dtype=np.int64)
    test_start = test_dates[0]
    test_end = test_dates[-1]
    log(f"\nTest period: {test_start.date()} → {test_end.date()}")

    # Build chunk start dates (non-overlapping)
    chunk_starts = []
    cursor = test_start
    while cursor + pd.Timedelta(days=args.seq_length - 1) <= test_end:
        chunk_starts.append(cursor)
        cursor = cursor + pd.Timedelta(days=args.seq_length)

    # Last partial chunk
    if cursor <= test_end:
        chunk_starts.append(cursor)

    log(f"  {len(chunk_starts)} prediction chunks (seq_length={args.seq_length})")

    n_basins = len(gauged_nodes_arr)

    with torch.no_grad():
        for chunk_start in tqdm(chunk_starts, desc='Chunks'):
            chunk_records = []
            for batch_start in range(0, n_basins, args.batch_size):
                batch_end = min(batch_start + args.batch_size, n_basins)
                basin_nodes = gauged_nodes_arr[batch_start:batch_end].tolist()

                result = trainer._prepare_sequence_batch(
                    data_module, basin_nodes, chunk_start
                )
                if result is None:
                    continue

                batch_data, target_flows = result
                target_flows = target_flows.to(trainer.device)

                predictions = trainer._forward_sequence(batch_data, trainer.device)
                preds_np = predictions.float().cpu().numpy()  # [n_seeds, seq_len]
                obs_np = target_flows.cpu().numpy()           # [n_seeds, seq_len]

                # Map back to original basin_nodes via retained_basin_indices
                # _prepare_sequence_batch may filter basins; preds_np matches retained order
                # We need to reconstruct which basin each prediction came from.
                # Look at how target_flows was filtered; trainer keeps indices in seed_indices order.

                # Simple approach: predictions come in basin_nodes order BUT may be filtered.
                # The _prepare_sequence_batch returns (batch_data, target_flows) where
                # target_flows is filtered to retained_basin_indices.
                # For our purposes, iterate over preds.shape[0] and assume that maps to basin_nodes
                # in original order (the trainer's validate uses this assumption).

                n_actual = preds_np.shape[0]
                for i in range(n_actual):
                    if i >= len(basin_nodes):
                        break
                    node_idx = basin_nodes[i]
                    comid = data_module.idx_to_comid.get(node_idx)
                    if comid is None:
                        continue
                    area_m2 = data_module.node_areas.get(node_idx, np.nan)

                    for t in range(preds_np.shape[1]):
                        date = chunk_start + pd.Timedelta(days=t)
                        if date > test_end:
                            break
                        pred_mm_day = float(preds_np[i, t])
                        obs_mm_day = float(obs_np[i, t]) if not np.isnan(obs_np[i, t]) else np.nan
                        # Convert mm/day → m³/s using area
                        if not np.isnan(area_m2) and area_m2 > 0:
                            pred_cms = pred_mm_day * area_m2 / (86400.0 * 1000.0)
                            obs_cms = obs_mm_day * area_m2 / (86400.0 * 1000.0) if not np.isnan(obs_mm_day) else np.nan
                        else:
                            pred_cms = np.nan
                            obs_cms = np.nan

                        chunk_records.append({
                            'date': date,
                            'comid': comid,
                            'node_idx': node_idx,
                            'pred_mm_day': pred_mm_day,
                            'obs_mm_day': obs_mm_day,
                            'pred_cms': pred_cms,
                            'obs_cms': obs_cms,
                        })

            all_records.extend(chunk_records)

    log(f"\nTotal records: {len(all_records)}")
    df = pd.DataFrame(all_records)
    df.to_parquet(args.output, index=False)
    log(f"Saved: {args.output}")

    # Summary stats
    n_basins_pred = df['comid'].nunique()
    log(f"  Basins with predictions: {n_basins_pred}")
    log(f"  Date range: {df['date'].min()} → {df['date'].max()}")

    cleanup_shared_memory()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
