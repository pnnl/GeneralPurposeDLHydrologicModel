#!/usr/bin/env python3
"""
Multi-GPU training v20 — v16's LSTM+DirectedGAT with curated 40-feature attributes.

Changes from v19 (51 features):
  - Drops 11 problematic features (categoricals, redundant, near-constant)
  - Log-transforms 6 heavily skewed features before z-scoring
  - 40 features, LSTM input: forcings(8) + static(40) = 48

Two-stage PUB training (same as v16/v19):
  Stage 1 (pretrain): Temporal split on TRAIN basins only
  Stage 2 (finetune): Basin holdout PUB evaluation

Usage:
    sbatch submit_v20.sh pretrain
    sbatch submit_v20.sh finetune
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.distributed as dist
from pathlib import Path
from typing import List, Optional
import json
from datetime import datetime
import argparse
import warnings

warnings.filterwarnings('ignore')
os.environ['PYTHONUNBUFFERED'] = '1'

from train_multi_gpu_v15 import (
    SequenceDataModule,
    setup_distributed,
    is_main_process,
    sync_if_distributed,
    log,
    unique_years,
    build_date_list,
    cleanup_shared_memory,
)
from train_multi_gpu_v16 import V16Trainer
from config import get_config_v20


# Features to drop: categoricals, redundant (r>0.85), near-constant, zero-dominated
DROP_FEATURES = {
    'geol_class_1st',       # categorical integer (0-15)
    'dominant_veg_class',   # categorical integer
    'july_avg_temp_C',      # r=0.90 with avg_temperature_C
    'annual_precip_mm',     # r=0.95 with p_mean
    'bubbling_pressure_cm', # r=1.00 with brooks_corey_expt
    'pet_mean',             # r=0.98 with avg_temperature_C
    'lai_max',              # r=0.97 with mean_annual_lai
    'low_prec_dur',         # r=0.96 with aridity
    'silt_frac',            # sand+silt+clay~1, linearly dependent
    'carbonate_frac',       # 88.5% zeros
    'high_prec_dur',        # CV=0.08, near-constant
}

# Features to log-transform before z-scoring (heavy right skew)
LOG_TRANSFORM_FEATURES = {
    'area_m2',          # skew=3.45, 5 orders of magnitude
    'uparea',           # skew=20.87, 25 km² to 3M km²
    'strmDrop_t',       # skew=5.08, heavy tail
    'ksat_mm_day',      # skew=5.11, heavy tail
    'baseflow_Dsmax',   # skew=3.33, heavy tail
    'aridity',          # skew=5.59, heavy tail to 34.7
}


class SequenceDataModuleV20(SequenceDataModule):
    """V20 data module: curated features with log-transforms."""

    def _load_static_features(self, config):
        """Load static features with curation: drop problematic features, log-transform skewed ones."""
        log("Loading static features (V20 curated)...")

        v2_path = config.data.base_dir / "data" / "static" / "catchment_static_attributes_v2.parquet"
        v1_path = config.data.base_dir / "data" / "static" / "catchment_static_attributes.parquet"

        if v2_path.exists():
            static_path = v2_path
            log(f"  Using V2 attributes: {static_path}")
        else:
            static_path = v1_path
            log(f"  WARNING: V2 attributes not found, falling back to V1: {static_path}")

        attrs_df = pd.read_parquet(static_path).set_index('hruid')

        # Extract catchment areas for normalization
        self.node_areas = {}
        if 'area_m2' in attrs_df.columns:
            for idx, hruid in self.idx_to_hruid.items():
                if hruid in attrs_df.index:
                    area = attrs_df.loc[hruid, 'area_m2']
                    if pd.notna(area) and area > 0:
                        self.node_areas[idx] = float(area)
            log(f"  Catchment areas loaded for {len(self.node_areas)} nodes")

        # Extract elevation for edge feature computation
        self.node_elevation = {}
        if 'elevation_m' in attrs_df.columns:
            for idx, hruid in self.idx_to_hruid.items():
                if hruid in attrs_df.index:
                    elev = attrs_df.loc[hruid, 'elevation_m']
                    if pd.notna(elev):
                        self.node_elevation[idx] = float(elev)
            log(f"  Elevations loaded for {len(self.node_elevation)} nodes")

        # Select numeric feature columns, drop constants
        feature_cols = [c for c in attrs_df.columns
                       if attrs_df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
        non_constant_cols = []
        for col in feature_cols:
            col_std = attrs_df[col].std()
            if pd.notna(col_std) and col_std > 1e-10:
                non_constant_cols.append(col)
            else:
                log(f"  Dropping constant feature: {col}")
        feature_cols = non_constant_cols

        # V20: Drop problematic features
        dropped = [c for c in feature_cols if c in DROP_FEATURES]
        feature_cols = [c for c in feature_cols if c not in DROP_FEATURES]
        log(f"  Dropped {len(dropped)} features: {dropped}")
        log(f"  Using {len(feature_cols)} features: {feature_cols}")

        # Build feature matrix
        features = np.full((self.n_nodes, len(feature_cols)), np.nan, dtype=np.float32)
        for idx, hruid in self.idx_to_hruid.items():
            if hruid in attrs_df.index:
                features[idx] = attrs_df.loc[hruid, feature_cols].values

        # V20: Log-transform skewed features before normalization
        log_transformed = []
        for i, col in enumerate(feature_cols):
            if col in LOG_TRANSFORM_FEATURES:
                mask = ~np.isnan(features[:, i])
                features[mask, i] = np.log1p(np.abs(features[mask, i]))
                log_transformed.append(col)
        if log_transformed:
            log(f"  Log-transformed {len(log_transformed)} features: {log_transformed}")

        self.static_features = torch.tensor(features, dtype=torch.float32)

        # Impute missing with column means
        col_means = torch.nanmean(self.static_features, dim=0)
        for i in range(self.static_features.shape[1]):
            mask = torch.isnan(self.static_features[:, i])
            self.static_features[mask, i] = col_means[i] if not torch.isnan(col_means[i]) else 0.0

        # Z-score normalize static features
        self.static_mean = self.static_features.mean(dim=0)
        self.static_std = self.static_features.std(dim=0) + 1e-8
        self.static_features = (self.static_features - self.static_mean) / self.static_std

        self.n_static_features = len(feature_cols)
        config.model.static_input_dim = self.n_static_features
        config.model.lstm_input_dim = len(config.data.forcing_vars) + self.n_static_features
        log(f"  Static features shape: {self.static_features.shape}")
        log(f"  lstm_input_dim set to: {config.model.lstm_input_dim} ({len(config.data.forcing_vars)} forcing + {self.n_static_features} static)")


def main():
    parser = argparse.ArgumentParser(
        description='Train GNN v20 — v16 + curated 40 attributes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('stage', type=str, choices=['pretrain', 'finetune'],
                        help='Training stage: pretrain (temporal) or finetune (PUB basin holdout)')

    parser.add_argument('--seq-length', type=int, default=180)
    parser.add_argument('--warmup-days', type=int, default=365)
    parser.add_argument('--train-basin-fraction', type=float, default=0.9)
    parser.add_argument('--basin-split-seed', type=int, default=42)
    parser.add_argument('--loss', type=str, default='kge', choices=['nse', 'kge'])
    parser.add_argument('--warmup-epochs', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--sequences-per-epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--gauge-dropout-rate', type=float, default=None)
    parser.add_argument('--pretrain-checkpoint', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--resume-best', action='store_true')

    args = parser.parse_args()

    # Stage-specific defaults
    if args.gauge_dropout_rate is None:
        args.gauge_dropout_rate = 0.0 if args.stage == 'pretrain' else 0.15

    if args.lr is None:
        args.lr = 5e-4 if args.stage == 'pretrain' else 1e-4

    # Checkpoint directories
    if args.stage == 'pretrain':
        checkpoint_dir = 'checkpoints_v20/pretrain'
        version = 'v20-pretrain'
    else:
        checkpoint_dir = 'checkpoints_v20/finetune'
        version = 'v20-finetune'

    # Validation
    if not (0.0 < args.train_basin_fraction < 1.0):
        parser.error('--train-basin-fraction must be in (0, 1)')
    if args.stage == 'finetune' and args.resume is None and not args.resume_best:
        default_pretrain = args.pretrain_checkpoint or 'checkpoints_v20/pretrain/best_model.pt'
        if not Path(default_pretrain).exists():
            parser.error(
                f"Finetune stage requires a pretrain checkpoint.\n"
                f"  Default not found: {default_pretrain}\n"
                f"  Run pretrain stage first, or provide --pretrain-checkpoint"
            )

    # Logging
    log("=" * 60)
    log(f"GNN STREAMFLOW V20 — CURATED FEATURES + LOG-TRANSFORMS")
    log(f"Stage: {args.stage.upper()}")
    log("=" * 60)
    log(f"Start time: {datetime.now()}")

    rank, world_size, local_rank = setup_distributed()
    log(f"NCCL version {torch.cuda.nccl.version()}")
    log(f"World size: {world_size}")

    # Config
    config = get_config_v20()
    if args.epochs:
        config.train.max_epochs = args.epochs

    config.data.split_mode = "basin"
    config.data.train_basin_fraction = args.train_basin_fraction
    config.data.basin_split_seed = args.basin_split_seed
    config.train.learning_rate = args.lr

    log(f"\nV20 Config:")
    log(f"  Stage: {args.stage}")
    log(f"  LSTM input: {config.model.lstm_input_dim} (8 forcing + 40 static)")
    log(f"  LSTM: hidden={config.model.lstm_hidden_dim}, layers={config.model.lstm_num_layers}")
    log(f"  GNN: hidden={config.model.gnn_hidden_dim}, layers={config.model.gnn_num_layers}, type=DirectedGAT")
    log(f"  Edge features: {config.model.edge_feat_dim}-dim")
    log(f"  Static input dim: {config.model.static_input_dim} (curated 40 features)")
    log(f"  Learning rate: {args.lr}")
    log(f"  Warmup days: {args.warmup_days}")
    log(f"  Sequence length: {args.seq_length}")
    log(f"  Basin fraction: {args.train_basin_fraction}")
    log(f"  Loss: {args.loss}")
    log(f"  Gauge dropout: {args.gauge_dropout_rate}")
    log(f"  Checkpoint dir: {checkpoint_dir}")

    # Load data (V20 module with curated features)
    log("\n" + "=" * 60)
    log("LOADING DATA")
    log("=" * 60)

    data_module = SequenceDataModuleV20(config, rank, world_size, warmup_days=args.warmup_days)

    # Stage-specific date and basin logic (same as v16/v19)
    if args.stage == 'pretrain':
        train_dates = build_date_list(config.data.train_years)
        val_dates = build_date_list(config.data.val_years)
        train_basin_nodes = data_module.train_gauged_nodes
        eval_basin_nodes = data_module.train_gauged_nodes

        log(f"\n--- PRETRAIN STAGE ---")
        log(f"Train dates: {len(train_dates)} (years {min(config.data.train_years)}-{max(config.data.train_years)})")
        log(f"Val dates: {len(val_dates)} (years {min(config.data.val_years)}-{max(config.data.val_years)})")
        log(f"Train basins: {len(train_basin_nodes)}")
        log(f"Val basins: {len(eval_basin_nodes)} (SAME as train -- temporal validation)")
        log(f"Held-out basins (NEVER seen): {len(data_module.eval_gauged_nodes)}")

    else:  # finetune
        all_years = unique_years(
            config.data.train_years,
            config.data.val_years,
            config.data.test_years,
        )
        train_dates = build_date_list(all_years)
        val_dates = build_date_list(all_years)
        train_basin_nodes = data_module.train_gauged_nodes
        eval_basin_nodes = data_module.eval_gauged_nodes

        log(f"\n--- FINETUNE STAGE (PUB) ---")
        log(f"Train dates: {len(train_dates)} (all years)")
        log(f"Val dates: {len(val_dates)} (all years)")
        log(f"Train basins: {len(train_basin_nodes)}")
        log(f"Eval basins (held-out): {len(eval_basin_nodes)}")

    # Save split metadata
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if is_main_process():
        metadata = data_module.get_split_metadata()
        metadata['stage'] = args.stage
        metadata['model'] = 'v20-curated-features'
        metadata['dropped_features'] = sorted(DROP_FEATURES)
        metadata['log_transformed_features'] = sorted(LOG_TRANSFORM_FEATURES)
        with (Path(checkpoint_dir) / "split_metadata.json").open('w') as f:
            json.dump(metadata, f, indent=2)

    # Create trainer (uses standard StreamflowGNN via V16Trainer)
    trainer = V16Trainer(
        config=config,
        data_module=data_module,
        seq_length=args.seq_length,
        warmup_days=args.warmup_days,
        loss_type=args.loss,
        warmup_epochs=args.warmup_epochs,
        sequences_per_epoch=args.sequences_per_epoch,
        gauge_dropout_rate=args.gauge_dropout_rate,
        checkpoint_dir=checkpoint_dir,
        version=version,
    )

    # Load checkpoint
    start_epoch = 0

    if args.stage == 'finetune' and args.resume is None and not args.resume_best:
        pretrain_ckpt = args.pretrain_checkpoint or 'checkpoints_v20/pretrain/best_model.pt'
        start_epoch = trainer.load_pretrain_weights(pretrain_ckpt)
    elif args.resume_best:
        resume_path = f"{checkpoint_dir}/best_model.pt"
        start_epoch = trainer.load_checkpoint(resume_path)
    elif args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)

    # Override LR
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = args.lr
    trainer.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer, mode='max', factor=0.5, patience=5, verbose=is_main_process(),
    )
    log(f"  Learning rate set to: {args.lr}")

    # Train
    trainer.fit(
        data_module,
        train_dates,
        val_dates,
        train_basin_nodes=train_basin_nodes,
        eval_basin_nodes=eval_basin_nodes,
        start_epoch=start_epoch,
        stage_label=args.stage,
    )

    log(f"\nEnd time: {datetime.now()}")
    log(f"V20 {args.stage} complete!")


if __name__ == '__main__':
    import atexit
    atexit.register(cleanup_shared_memory)

    try:
        main()
    finally:
        cleanup_shared_memory()
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()
