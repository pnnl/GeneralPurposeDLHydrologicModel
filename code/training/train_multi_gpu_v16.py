#!/usr/bin/env python3
"""
Multi-GPU training v16 — Two-stage PUB pretraining.

Addresses data leakage in v15: pre-training on all basins then evaluating on
held-out basins would let the model memorize held-out basin characteristics.

Two-stage approach:
  Stage 1 (pretrain): Temporal split on TRAIN basins only.
    - Basin split (seed=42, fraction=0.9) identifies train/eval basins
    - Training uses temporal dates (1985-2014) on train basins only
    - Validation uses temporal holdout (2015-2019) on TRAIN basins only
    - Eval basins are NEVER seen
    - No gauge dropout, standard KGE loss, LR=5e-4

  Stage 2 (finetune): Basin holdout PUB evaluation.
    - Resume model weights only from pretrain (fresh optimizer)
    - Train on train basins with gauge dropout (0.15)
    - Validate on held-out eval basins (true PUB evaluation)
    - All years available for both train/val
    - Lower LR (1e-4)

Architecture: Same as v15 (LSTM 512x3 + DirectedRiverGNN 256x3 + 6-dim edge features).

Usage:
    # Stage 1: pretrain (temporal split, train basins only)
    sbatch submit_v16.sh pretrain

    # Stage 2: finetune for PUB (auto-resumes from pretrain best)
    sbatch submit_v16.sh finetune

    # Manual finetune from specific pretrain checkpoint
    sbatch submit_v16.sh finetune --pretrain-checkpoint checkpoints_v16/pretrain_epoch_20.pt
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json
from datetime import datetime
import time
import argparse
import warnings

warnings.filterwarnings('ignore')
os.environ['PYTHONUNBUFFERED'] = '1'

# Reuse all core components from v15 (identical architecture and training logic)
from train_multi_gpu_v15 import (
    RollingKGELoss,
    RollingNSELoss,
    SequenceDataModule,
    RollingLossTrainer,
    setup_distributed,
    is_main_process,
    sync_if_distributed,
    log,
    unique_years,
    build_date_list,
    cleanup_shared_memory,
)
from config import get_config_v16
from model_v15 import StreamflowGNN, count_parameters

from torch.cuda.amp import autocast, GradScaler


# ============================================================================
# V16 Trainer — extends v15 with configurable checkpoint dir + weights-only loading
# ============================================================================

class V16Trainer(RollingLossTrainer):
    """
    Extends v15's RollingLossTrainer with:
    - Configurable checkpoint directory and version string
    - weights-only loading for finetune stage (fresh optimizer/scheduler)
    """

    def __init__(
        self,
        *args,
        checkpoint_dir: str = 'checkpoints_v16',
        version: str = 'v16',
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._checkpoint_dir = Path(checkpoint_dir)
        self._version = version

    def load_pretrain_weights(self, checkpoint_path: str) -> int:
        """
        Load ONLY model weights from a pretrain checkpoint.
        Does NOT restore optimizer, scheduler, scaler, or history.
        This ensures a clean start for the finetune stage.
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Pretrain checkpoint not found: {ckpt_path}")

        sync_if_distributed()
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location=self.device)

        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])

        # Restore forcing normalization (must match for correct predictions)
        if 'forcing_mean' in checkpoint:
            self.forcing_mean = checkpoint['forcing_mean'].to(self.device)
            self.forcing_std = checkpoint['forcing_std'].to(self.device)
        if 'edge_feat_mean' in checkpoint:
            self.edge_feat_mean = checkpoint['edge_feat_mean']
            self.edge_feat_std = checkpoint['edge_feat_std']

        # Fresh history — this is a new training stage
        self.history = {
            'train_loss': [], 'val_nse': [], 'val_kge': [],
            'best_epoch': 0, 'best_val_metric': -float('inf'),
        }
        self.patience_counter = 0

        sync_if_distributed()
        if is_main_process():
            log(f"Loaded pretrain weights from: {ckpt_path}")
            log(f"  Pretrain version: {checkpoint.get('version', 'unknown')}")
            log(f"  Pretrain epoch: {checkpoint.get('epoch', 'N/A')}")
            log(f"  Pretrain val KGE: {checkpoint.get('val_kge', 'N/A'):.4f}")
            log(f"  Pretrain val NSE: {checkpoint.get('val_nse', 'N/A'):.4f}")
            log(f"  Fresh optimizer, scheduler, and scaler (no state loaded)")

        return 0  # start from epoch 0 for finetune

    def fit(
        self,
        data_module: SequenceDataModule,
        train_dates: List[pd.Timestamp],
        val_dates: List[pd.Timestamp],
        train_basin_nodes: Optional[List[int]] = None,
        eval_basin_nodes: Optional[List[int]] = None,
        start_epoch: int = 0,
        stage_label: str = "",
    ):
        """
        Full training loop. Same as v15 but uses self._checkpoint_dir and self._version.
        Added stage_label for logging clarity.
        """
        train_basin_nodes = list(train_basin_nodes or data_module.train_gauged_nodes)
        eval_basin_nodes = list(eval_basin_nodes or data_module.eval_gauged_nodes)
        overlap = len(set(train_basin_nodes) & set(eval_basin_nodes))

        display_label = f" [{stage_label}]" if stage_label else ""
        log("\n" + "=" * 60)
        log(f"TRAINING {self._version.upper()}{display_label} - Larger LSTM (512) + Directed GNN ({self.world_size} GPUs)")
        log("=" * 60)
        log(f"Train dates: {len(train_dates)}")
        log(f"Val dates: {len(val_dates)}")
        log(f"Split mode: {data_module.split_mode}")
        log(f"Train basins: {len(train_basin_nodes)}")
        log(f"Eval basins: {len(eval_basin_nodes)}")
        log(f"Train/eval overlap: {overlap}")
        log(f"Warmup days: {self.warmup_days} (LSTM-only, no GNN)")
        log(f"Sequence length: {self.seq_length} days (LSTM+GNN prediction)")
        log(f"Sequences per epoch: {self.sequences_per_epoch}")
        log(f"Total gauged nodes: {len(data_module.all_gauged_nodes)}")
        log(f"Batch size (basins): {self.batch_size}")
        log(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        log(f"MSE warmup: {self.warmup_epochs} epochs")
        log(f"Gauge dropout rate: {self.gauge_dropout_rate}")
        log(f"Target units: specific_discharge_mm_day")
        log(f"Model output: Softplus → mm/day (real-space loss)")
        log(f"Forcing normalization: z-score (per-variable)")

        use_kge_metric = (self.loss_type == "kge")
        metric_name = "KGE" if use_kge_metric else "NSE"

        best_val_metric = float(self.history.get('best_val_metric', -float('inf')))
        if not np.isfinite(best_val_metric):
            best_val_metric = -float('inf')
        patience_counter = int(getattr(self, 'patience_counter', 0))
        patience = self.config.train.patience

        checkpoint_dir = self._checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if start_epoch > 0:
            log(f"Resuming from epoch {start_epoch + 1}/{self.config.train.max_epochs}")
            log(f"Current best val {metric_name}: {best_val_metric:.4f}")
        if start_epoch >= self.config.train.max_epochs:
            log(f"Start epoch ({start_epoch + 1}) is beyond max epochs ({self.config.train.max_epochs}); nothing to train.")
            return

        sync_if_distributed()
        if is_main_process():
            log("\nAll ranks synchronized. Starting training loop...")

        for epoch in range(start_epoch, self.config.train.max_epochs):
            epoch_start = time.time()
            log(f"\nEpoch {epoch + 1}/{self.config.train.max_epochs}")

            sync_if_distributed()

            # Train
            train_loss, train_stats = self.train_epoch(data_module, train_dates, train_basin_nodes, epoch)

            if train_stats.get('use_mse'):
                log(f"  Train loss (MSE warmup): {train_loss:.6f}")
            else:
                log(f"  Train loss (Rolling {self.loss_type.upper()}): {train_loss:.6f}")
                if train_stats.get('n_basins', 0) > 0:
                    log(f"    Basins: {train_stats['n_basins']}, "
                        f"median NSE: {train_stats.get('median_nse', 0):.4f}, "
                        f"median KGE: {train_stats.get('median_kge', 0):.4f}")
            if train_stats.get('supervised_basins', 0) > 0:
                log(f"    Supervised basins after dropout: {train_stats['supervised_basins']}")

            # Validate
            val_nse, val_kge, val_stats = self.validate(data_module, val_dates, eval_basin_nodes)

            epoch_time = time.time() - epoch_start
            log(f"  Epoch time: {epoch_time:.1f}s")

            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['val_nse'].append(val_nse)
            self.history['val_kge'].append(val_kge)

            val_metric = val_kge if use_kge_metric else val_nse
            self.scheduler.step(val_metric)

            # Save best model
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                self.history['best_epoch'] = epoch
                self.history['best_val_metric'] = val_metric
                patience_counter = 0

                if is_main_process():
                    log(f"  NEW BEST! Val {metric_name}: {val_metric:.4f}")
                    model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_state,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'scaler_state_dict': self.scaler.state_dict(),
                        'val_nse': val_nse,
                        'val_kge': val_kge,
                        'patience_counter': patience_counter,
                        'history': self.history,
                        'config': self.config,
                        'data_split': self.data_split_summary,
                        'version': self._version,
                        'stage': stage_label,
                        'target_units': 'specific_discharge_mm_day',
                        'forcing_mean': self.forcing_mean.cpu(),
                        'forcing_std': self.forcing_std.cpu(),
                        'edge_feat_mean': self.edge_feat_mean.cpu(),
                        'edge_feat_std': self.edge_feat_std.cpu(),
                    }, checkpoint_dir / 'best_model.pt')
                    log(f"  Saved: {checkpoint_dir}/best_model.pt")
            else:
                patience_counter += 1

            # Save periodic checkpoint
            if is_main_process():
                log(f"  Saving periodic checkpoint (epoch {epoch+1})...")
                model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict(),
                    'val_nse': val_nse,
                    'val_kge': val_kge,
                    'patience_counter': patience_counter,
                    'history': self.history,
                    'config': self.config,
                    'data_split': self.data_split_summary,
                    'version': self._version,
                    'stage': stage_label,
                    'forcing_mean': self.forcing_mean.cpu(),
                    'forcing_std': self.forcing_std.cpu(),
                    'edge_feat_mean': self.edge_feat_mean.cpu(),
                    'edge_feat_std': self.edge_feat_std.cpu(),
                }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt')
                log(f"  Saved: {checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt")

            sync_if_distributed()

            # Early stopping
            if patience_counter >= patience:
                log(f"\nEarly stopping at epoch {epoch + 1}")
                break

        sync_if_distributed()

        log(f"\nBest epoch: {self.history['best_epoch'] + 1}")
        log(f"Best val {metric_name}: {self.history['best_val_metric']:.4f}")


# ============================================================================
# Main — two-stage PUB pretraining
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train GNN v16 — Two-stage PUB pretraining',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stage 1: pretrain on temporal split (train basins only)
  sbatch submit_v16.sh pretrain

  # Stage 2: finetune for PUB evaluation
  sbatch submit_v16.sh finetune

  # Finetune from specific pretrain checkpoint
  sbatch submit_v16.sh finetune --pretrain-checkpoint checkpoints_v16/pretrain_epoch_20.pt

  # Override defaults
  sbatch submit_v16.sh pretrain --lr 3e-4 --warmup-days 180
  sbatch submit_v16.sh finetune --gauge-dropout-rate 0.1 --lr 5e-5
""",
    )

    parser.add_argument('stage', type=str, choices=['pretrain', 'finetune'],
                        help='Training stage: pretrain (temporal) or finetune (PUB basin holdout)')

    # Shared arguments
    parser.add_argument('--seq-length', type=int, default=180,
                        help='Prediction sequence length in days (default: 180)')
    parser.add_argument('--warmup-days', type=int, default=365,
                        help='LSTM warmup period in days (default: 365)')
    parser.add_argument('--train-basin-fraction', type=float, default=0.9,
                        help='Fraction of gauged basins used for training (default: 0.9)')
    parser.add_argument('--basin-split-seed', type=int, default=42,
                        help='Random seed for basin split (default: 42)')
    parser.add_argument('--loss', type=str, default='kge', choices=['nse', 'kge'],
                        help='Loss type (default: kge)')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='MSE warmup epochs before rolling loss (default: 0)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override max epochs')
    parser.add_argument('--sequences-per-epoch', type=int, default=200,
                        help='Number of sequence start dates per epoch (default: 200)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')

    # Stage-specific arguments
    parser.add_argument('--gauge-dropout-rate', type=float, default=None,
                        help='Gauge dropout rate (default: 0.0 for pretrain, 0.15 for finetune)')
    parser.add_argument('--pretrain-checkpoint', type=str, default=None,
                        help='[finetune only] Path to pretrain checkpoint (default: checkpoints_v16/pretrain/best_model.pt)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint (restores full optimizer state)')
    parser.add_argument('--resume-best', action='store_true',
                        help='Resume from best_model.pt of current stage')

    args = parser.parse_args()

    # ---- Stage-specific defaults ----
    if args.gauge_dropout_rate is None:
        args.gauge_dropout_rate = 0.0 if args.stage == 'pretrain' else 0.15

    if args.lr is None:
        args.lr = 5e-4 if args.stage == 'pretrain' else 1e-4

    # Checkpoint directories per stage
    if args.stage == 'pretrain':
        checkpoint_dir = 'checkpoints_v16/pretrain'
        version = 'v16-pretrain'
    else:
        checkpoint_dir = 'checkpoints_v16/finetune'
        version = 'v16-finetune'

    # Validation
    if not (0.0 < args.train_basin_fraction < 1.0):
        parser.error('--train-basin-fraction must be in (0, 1)')
    if not (0.0 <= args.gauge_dropout_rate < 1.0):
        parser.error('--gauge-dropout-rate must be in [0, 1)')
    if args.stage == 'finetune' and args.resume is None and not args.resume_best:
        # Default: load pretrain best
        default_pretrain = args.pretrain_checkpoint or 'checkpoints_v16/pretrain/best_model.pt'
        if not Path(default_pretrain).exists():
            parser.error(
                f"Finetune stage requires a pretrain checkpoint.\n"
                f"  Default not found: {default_pretrain}\n"
                f"  Run pretrain stage first, or provide --pretrain-checkpoint"
            )

    # ---- Logging ----
    log("=" * 60)
    log(f"GNN STREAMFLOW V16 — TWO-STAGE PUB PRETRAINING")
    log(f"Stage: {args.stage.upper()}")
    log("=" * 60)
    log(f"Start time: {datetime.now()}")

    rank, world_size, local_rank = setup_distributed()
    log(f"NCCL version {torch.cuda.nccl.version()}")
    log(f"World size: {world_size}")
    log(f"Rank {rank}, Local rank {local_rank}, Device cuda:{local_rank}")

    # ---- Config ----
    config = get_config_v16()
    if args.epochs:
        config.train.max_epochs = args.epochs

    # Always use basin split mode in the data module (to identify train/eval basins)
    config.data.split_mode = "basin"
    config.data.train_basin_fraction = args.train_basin_fraction
    config.data.basin_split_seed = args.basin_split_seed

    # Override LR
    config.train.learning_rate = args.lr

    log(f"\nV16 Config:")
    log(f"  Stage: {args.stage}")
    log(f"  LSTM input: {config.model.lstm_input_dim} (will be overridden by actual data)")
    log(f"  LSTM: hidden={config.model.lstm_hidden_dim}, layers={config.model.lstm_num_layers}")
    log(f"  GNN: hidden={config.model.gnn_hidden_dim}, layers={config.model.gnn_num_layers}, type=DirectedGAT")
    log(f"  Edge features: {config.model.edge_feat_dim}-dim")
    log(f"  Learning rate: {args.lr}")
    log(f"  Warmup days: {args.warmup_days}")
    log(f"  Sequence length: {args.seq_length}")
    log(f"  Train basin fraction: {args.train_basin_fraction}")
    log(f"  Basin split seed: {args.basin_split_seed}")
    log(f"  Loss type: {args.loss} (real-space)")
    log(f"  MSE warmup epochs: {args.warmup_epochs}")
    log(f"  Gauge dropout rate: {args.gauge_dropout_rate}")
    log(f"  Checkpoint dir: {checkpoint_dir}")
    log(f"  Output: Softplus → mm/day")

    # ---- Load data ----
    log("\n" + "=" * 60)
    log("LOADING DATA")
    log("=" * 60)

    data_module = SequenceDataModule(config, rank, world_size, warmup_days=args.warmup_days)

    # ---- Stage-specific date and basin logic ----
    if args.stage == 'pretrain':
        # Temporal split on TRAIN basins only
        # Eval basins are never seen during pretrain
        train_dates = build_date_list(config.data.train_years)
        val_dates = build_date_list(config.data.val_years)
        train_basin_nodes = data_module.train_gauged_nodes
        eval_basin_nodes = data_module.train_gauged_nodes  # validate on TRAIN basins (temporal holdout)

        log(f"\n--- PRETRAIN STAGE ---")
        log(f"Train dates: {len(train_dates)} (years {min(config.data.train_years)}-{max(config.data.train_years)})")
        log(f"Val dates: {len(val_dates)} (years {min(config.data.val_years)}-{max(config.data.val_years)})")
        log(f"Train basins: {len(train_basin_nodes)}")
        log(f"Val basins: {len(eval_basin_nodes)} (SAME as train — temporal validation)")
        log(f"Held-out basins (NEVER seen): {len(data_module.eval_gauged_nodes)}")

    else:  # finetune
        # Basin holdout: all years, eval on held-out basins
        all_years = unique_years(
            config.data.train_years,
            config.data.val_years,
            config.data.test_years,
        )
        train_dates = build_date_list(all_years)
        val_dates = build_date_list(all_years)
        train_basin_nodes = data_module.train_gauged_nodes
        eval_basin_nodes = data_module.eval_gauged_nodes  # held-out basins

        log(f"\n--- FINETUNE STAGE (PUB) ---")
        log(f"Train dates: {len(train_dates)} (all years)")
        log(f"Val dates: {len(val_dates)} (all years)")
        log(f"Train basins: {len(train_basin_nodes)}")
        log(f"Eval basins (held-out): {len(eval_basin_nodes)}")
        log(f"Train/eval overlap: 0 (guaranteed by basin split)")

    # Save split metadata
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if is_main_process():
        split_metadata_path = Path(checkpoint_dir) / "split_metadata.json"
        metadata = data_module.get_split_metadata()
        metadata['stage'] = args.stage
        with split_metadata_path.open('w') as f:
            json.dump(metadata, f, indent=2)
        log(f"Saved split metadata: {split_metadata_path}")

    # ---- Create trainer ----
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

    # ---- Load checkpoint ----
    start_epoch = 0

    if args.stage == 'finetune' and args.resume is None and not args.resume_best:
        # Finetune: load pretrain weights only (fresh optimizer)
        pretrain_ckpt = args.pretrain_checkpoint or 'checkpoints_v16/pretrain/best_model.pt'
        start_epoch = trainer.load_pretrain_weights(pretrain_ckpt)
    elif args.resume_best:
        resume_path = f"{checkpoint_dir}/best_model.pt"
        start_epoch = trainer.load_checkpoint(resume_path)
    elif args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)

    # Override LR after loading (in case checkpoint restored old LR)
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = args.lr
    trainer.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer, mode='max', factor=0.5, patience=5, verbose=is_main_process(),
    )
    log(f"  Learning rate set to: {args.lr}")

    # ---- Train ----
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
    log(f"V16 {args.stage} complete!")


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
