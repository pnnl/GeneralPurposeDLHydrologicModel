#!/usr/bin/env python3
"""
Main multi-GPU training entry point for the deterministic LSTM-DirectedGAT model.

Uses the 8-variable daily catchment forcings, curated 40 static catchment attributes,
directed river-network graph topology, and quality-controlled streamflow observations
in data/streamflow/.

Two-stage training protocol:
  1. pretrain : Temporal training (1985-2014) and validation (2015-2019) on training basins
  2. finetune : Spatial fine-tuning with gauge dropout, validated on held-out basins

Usage (from the repository root):
    sbatch code/training/submit_v24.sh pretrain
    sbatch code/training/submit_v24.sh finetune
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
from train_multi_gpu_v20 import SequenceDataModuleV20, DROP_FEATURES, LOG_TRANSFORM_FEATURES
from config import get_config_v24

BUNDLE = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(
        description='Train GNN v24 — WRF TGW + area-verified streamflow',
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

    if args.gauge_dropout_rate is None:
        args.gauge_dropout_rate = 0.0 if args.stage == 'pretrain' else 0.15

    if args.lr is None:
        args.lr = 5e-4 if args.stage == 'pretrain' else 1e-4

    if args.stage == 'pretrain':
        checkpoint_dir = str(BUNDLE / 'outputs' / 'checkpoints_v24' / 'pretrain')
        version = 'v24-pretrain'
    else:
        checkpoint_dir = str(BUNDLE / 'outputs' / 'checkpoints_v24' / 'finetune')
        version = 'v24-finetune'

    if not (0.0 < args.train_basin_fraction < 1.0):
        parser.error('--train-basin-fraction must be in (0, 1)')
    if args.stage == 'finetune' and args.resume is None and not args.resume_best:
        default_pretrain = args.pretrain_checkpoint or str(
            BUNDLE / 'outputs' / 'checkpoints_v24' / 'pretrain' / 'best_model.pt')
        if not Path(default_pretrain).exists():
            parser.error(
                f"Finetune stage requires a pretrain checkpoint.\n"
                f"  Default not found: {default_pretrain}\n"
                f"  Run pretrain stage first, or provide --pretrain-checkpoint"
            )

    log("=" * 60)
    log(f"GNN STREAMFLOW V24 — WRF TGW + AREA-VERIFIED STREAMFLOW")
    log(f"Stage: {args.stage.upper()}")
    log("=" * 60)
    log(f"Start time: {datetime.now()}")

    rank, world_size, local_rank = setup_distributed()
    log(f"NCCL version {torch.cuda.nccl.version()}")
    log(f"World size: {world_size}")

    config = get_config_v24()
    if args.epochs:
        config.train.max_epochs = args.epochs

    config.data.split_mode = "basin"
    config.data.train_basin_fraction = args.train_basin_fraction
    config.data.basin_split_seed = args.basin_split_seed
    config.train.learning_rate = args.lr

    log(f"\nV24 Config:")
    log(f"  Stage: {args.stage}")
    log(f"  Forcings: WRF TGW (8 vars)")
    log(f"  Streamflow: data/streamflow (area-verified)")
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

    log("\n" + "=" * 60)
    log("LOADING DATA")
    log("=" * 60)

    data_module = SequenceDataModuleV20(config, rank, world_size, warmup_days=args.warmup_days)

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

    else:
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

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if is_main_process():
        metadata = data_module.get_split_metadata()
        metadata['stage'] = args.stage
        metadata['model'] = 'v24-gnn-area-verified'
        metadata['forcings'] = 'WRF TGW (8 vars)'
        metadata['streamflow'] = 'data/streamflow (area-verified 0.7-1.3x)'
        metadata['dropped_features'] = sorted(DROP_FEATURES)
        metadata['log_transformed_features'] = sorted(LOG_TRANSFORM_FEATURES)
        with (Path(checkpoint_dir) / "split_metadata.json").open('w') as f:
            json.dump(metadata, f, indent=2)

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

    start_epoch = 0

    if args.stage == 'finetune' and args.resume is None and not args.resume_best:
        pretrain_ckpt = args.pretrain_checkpoint or str(
            BUNDLE / 'outputs' / 'checkpoints_v24' / 'pretrain' / 'best_model.pt')
        start_epoch = trainer.load_pretrain_weights(pretrain_ckpt)
    elif args.resume_best:
        resume_path = f"{checkpoint_dir}/best_model.pt"
        start_epoch = trainer.load_checkpoint(resume_path)
    elif args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)

    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = args.lr
    trainer.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer, mode='max', factor=0.5, patience=5, verbose=is_main_process(),
    )
    log(f"  Learning rate set to: {args.lr}")

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
    log(f"V24 {args.stage} complete!")


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
