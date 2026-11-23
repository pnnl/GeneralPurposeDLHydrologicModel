#!/usr/bin/env python3
"""
Multi-GPU training v17 — PUB-oriented architectures.

Three model variants designed for Prediction in Ungauged Basins:
  - ea-lstm: Entity-Aware LSTM (static attributes modulate input gate)
  - decoupled-gnn: GNN on static attributes only (LSTM never mixed by GNN)
  - contrastive-pub: Contrastive pretrained basin encoder + LSTM

All variants use two-stage training (v13 showed direct basin holdout fails):
  Stage 1 (pretrain): Temporal split on TRAIN basins only
  Stage 2 (finetune): Basin holdout PUB evaluation

The contrastive-pub variant adds Stage 0 (contrastive pretraining of basin encoder).

Usage:
    # EA-LSTM
    sbatch submit_v17.sh ea-lstm pretrain
    sbatch submit_v17.sh ea-lstm finetune

    # Decoupled GNN
    sbatch submit_v17.sh decoupled-gnn pretrain
    sbatch submit_v17.sh decoupled-gnn finetune

    # Contrastive PUB
    sbatch submit_v17.sh contrastive-pub contrastive   # Stage 0
    sbatch submit_v17.sh contrastive-pub pretrain       # Stage 1
    sbatch submit_v17.sh contrastive-pub finetune       # Stage 2
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
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
from train_multi_gpu_v16 import V16Trainer
from config import get_config_v17
from model_pub import (
    StreamflowEALSTM,
    StreamflowDecoupledGNN,
    StreamflowContrastivePUB,
    BasinEncoder,
    InfoNCELoss,
    create_pub_model,
    count_parameters,
)


# =============================================================================
# V17 Trainer — PUB-oriented model support
# =============================================================================

class V17Trainer(V16Trainer):
    """
    Extends V16Trainer with support for EA-LSTM, Decoupled GNN, and Contrastive PUB.

    Overrides:
    - Model creation: uses create_pub_model() instead of StreamflowGNN
    - _forward_sequence: handles model-specific warmup/prediction flow
    """

    def __init__(
        self,
        config,
        data_module: SequenceDataModule,
        model_type: str = 'ea-lstm',
        seq_length: int = 90,
        warmup_days: int = 365,
        loss_type: str = 'kge',
        warmup_epochs: int = 0,
        sequences_per_epoch: int = 200,
        gauge_dropout_rate: float = 0.0,
        checkpoint_dir: str = 'checkpoints_v17',
        version: str = 'v17',
        neighbor_static: Optional[torch.Tensor] = None,
    ):
        # Store model type before parent __init__ (which creates the model)
        self.model_type = model_type
        self._neighbor_static = neighbor_static

        # Skip RollingLossTrainer.__init__ — we need to create a different model
        # Instead, do manual initialization matching RollingLossTrainer
        self.config = config
        self.seq_length = seq_length
        self.warmup_days = warmup_days
        self.loss_type = loss_type
        self.warmup_epochs = warmup_epochs
        self.sequences_per_epoch = sequences_per_epoch
        self.gauge_dropout_rate = gauge_dropout_rate

        if not (0.0 <= self.gauge_dropout_rate < 1.0):
            raise ValueError("gauge_dropout_rate must be in [0, 1)")

        self.rank, self.world_size, self.local_rank = setup_distributed()
        self.device = torch.device(f'cuda:{self.local_rank}')
        torch.cuda.set_device(self.device)

        if is_main_process():
            log(f"\nInitializing v17 trainer ({model_type}) on rank {self.rank}/{self.world_size}")
            log(f"  Device: {self.device}")
            log(f"  Model type: {model_type}")
            log(f"  Sequence length: {seq_length} days")
            log(f"  Warmup days: {warmup_days}")
            log(f"  Loss type: {loss_type}")
            log(f"  MSE warmup epochs: {warmup_epochs}")
            log(f"  Gauge dropout rate: {gauge_dropout_rate}")

        # Create PUB model
        self.model = create_pub_model(model_type, config.model).to(self.device)

        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,  # EA-LSTM may not use all params every step
                bucket_cap_mb=25,
                gradient_as_bucket_view=True,
            )
            dist.barrier()
            if is_main_process():
                log("  All ranks synchronized after DDP wrapping")

        if is_main_process():
            log(f"  Model parameters: {count_parameters(self.model):,}")

        # Forcing normalization
        self.forcing_mean = data_module.forcing_mean_tensor.to(self.device)
        self.forcing_std = data_module.forcing_std_tensor.to(self.device)

        # Edge feature normalization
        self.edge_feat_mean = data_module.edge_feat_mean
        self.edge_feat_std = data_module.edge_feat_std

        # Loss functions
        self.rolling_nse_loss = RollingNSELoss(min_valid_ratio=0.3)
        self.rolling_kge_loss = RollingKGELoss(min_valid_ratio=0.3)
        self.mse_loss = nn.MSELoss()

        # Optimizer
        try:
            self.optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=config.train.learning_rate,
                weight_decay=config.train.weight_decay,
                fused=True,
            )
        except TypeError:
            self.optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=config.train.learning_rate,
                weight_decay=config.train.weight_decay,
            )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=is_main_process(),
        )

        self.scaler = GradScaler()

        self.batch_size = 64
        self.gradient_accumulation_steps = 4
        self.max_subgraph_nodes = 4000

        self.data_split_summary = data_module.get_split_summary()
        self.history = {
            'train_loss': [], 'val_nse': [], 'val_kge': [],
            'best_epoch': 0, 'best_val_metric': -float('inf'),
        }

        # V16 additions
        self._checkpoint_dir = Path(checkpoint_dir)
        self._version = version

    def _forward_sequence(
        self,
        batch_data: Dict,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Day-by-day forward pass adapted for PUB model variants.

        EA-LSTM: needs x_static during warmup (for input gate modulation)
        Decoupled GNN: precomputes spatial embedding once, then standard LSTM
        Contrastive PUB: precomputes basin embedding once, then standard LSTM
        """
        all_forcings = batch_data['all_forcings'].to(device)
        x_static = batch_data['x_static'].to(device)
        up_ei = batch_data['upstream_edge_index'].to(device)
        up_ea = batch_data['upstream_edge_attr'].to(device)
        down_ei = batch_data['downstream_edge_index'].to(device)
        down_ea = batch_data['downstream_edge_attr'].to(device)
        seed_indices = batch_data['seed_indices']
        warmup_days = batch_data['warmup_days']

        total_days = all_forcings.shape[0]

        # Z-score normalize forcings
        all_forcings = (all_forcings - self.forcing_mean) / self.forcing_std

        raw_model = self.model.module if hasattr(self.model, 'module') else self.model

        # Model-specific setup (run once per sequence)
        if isinstance(raw_model, StreamflowEALSTM):
            raw_model.clear_static_gates()
        elif isinstance(raw_model, StreamflowDecoupledGNN):
            raw_model.clear_spatial_cache()
            with autocast():
                raw_model.precompute_spatial_embedding(
                    x_static, up_ei, up_ea, down_ei, down_ea
                )
        elif isinstance(raw_model, StreamflowContrastivePUB):
            raw_model.clear_embedding_cache()
            # Compute neighbor_static for this subgraph
            neighbor_static = self._get_subgraph_neighbor_static(batch_data, device)
            with autocast():
                raw_model.precompute_basin_embedding(x_static, neighbor_static)

        h_lstm = None
        predictions = []

        # === Warmup phase ===
        for t in range(warmup_days):
            x_t = torch.cat([all_forcings[t], x_static], dim=-1)
            with autocast():
                if isinstance(raw_model, StreamflowEALSTM):
                    _, h_lstm = raw_model.lstm_step(x_t, h_lstm, x_static=x_static)
                elif isinstance(raw_model, StreamflowDecoupledGNN):
                    _, h_lstm = raw_model.lstm_step(x_t, h_lstm)
                elif isinstance(raw_model, StreamflowContrastivePUB):
                    _, h_lstm = raw_model.lstm_step(x_t, h_lstm)
                else:
                    _, h_lstm = raw_model.lstm_step(x_t, h_lstm)
            h_lstm = (h_lstm[0].detach(), h_lstm[1].detach())

        # === Prediction phase ===
        for t in range(warmup_days, total_days):
            x_t = torch.cat([all_forcings[t], x_static], dim=-1)
            with autocast():
                pred, h_lstm = self.model(
                    x_t, up_ei, up_ea, down_ei, down_ea, h_lstm, x_static=x_static
                )
            predictions.append(pred[seed_indices].squeeze(-1))

        return torch.stack(predictions, dim=0).t()  # [n_seeds, seq_len]

    def _get_subgraph_neighbor_static(self, batch_data: Dict, device: torch.device) -> torch.Tensor:
        """
        Get mean-pooled neighbor static attributes for contrastive model.
        If precomputed neighbor_static is available, index into it.
        Otherwise, compute from the subgraph edges.
        """
        x_static = batch_data['x_static'].to(device)
        n_nodes = x_static.shape[0]

        if self._neighbor_static is not None:
            # We have global neighbor_static — need to extract subgraph portion
            # But subgraph indices map to original graph, and _neighbor_static is
            # indexed by original graph node indices. We need to recompute for subgraph.
            pass

        # Compute from subgraph edges: mean-pool neighbor attributes
        up_ei = batch_data['upstream_edge_index'].to(device)
        down_ei = batch_data['downstream_edge_index'].to(device)

        # Merge all edges (undirected for neighbor aggregation)
        if up_ei.shape[1] > 0 and down_ei.shape[1] > 0:
            all_edges = torch.cat([up_ei, down_ei], dim=1)
        elif up_ei.shape[1] > 0:
            all_edges = up_ei
        elif down_ei.shape[1] > 0:
            all_edges = down_ei
        else:
            return torch.zeros_like(x_static)

        # For each node, collect neighbor features and average
        neighbor_sum = torch.zeros_like(x_static)
        neighbor_count = torch.zeros(n_nodes, 1, device=device)

        src, dst = all_edges[0], all_edges[1]
        neighbor_sum.index_add_(0, dst, x_static[src])
        neighbor_count.index_add_(0, dst, torch.ones(src.shape[0], 1, device=device))

        # Also add reverse direction
        neighbor_sum.index_add_(0, src, x_static[dst])
        neighbor_count.index_add_(0, src, torch.ones(dst.shape[0], 1, device=device))

        neighbor_count = neighbor_count.clamp(min=1)
        return neighbor_sum / neighbor_count


# =============================================================================
# Contrastive pretraining (Stage 0)
# =============================================================================

def run_contrastive_pretraining(
    config,
    data_module: SequenceDataModule,
    checkpoint_dir: str,
    lr: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 2048,
    temperature: float = 0.07,
    similarity_threshold: float = 0.9,
):
    """
    Train BasinEncoder with InfoNCE loss on static attribute similarity.
    Single GPU — no DDP needed (small model, no sequence processing).
    """
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    if not is_main_process():
        # Only rank 0 does contrastive pretraining
        sync_if_distributed()
        return

    log("=" * 60)
    log("CONTRASTIVE PRETRAINING (Stage 0)")
    log("=" * 60)

    # Load precomputed basin pairs
    pairs_path = config.data.base_dir / "data" / "basin_pairs.npz"
    if not pairs_path.exists():
        log(f"ERROR: Basin pairs not found at {pairs_path}")
        log("Run: python precompute_basin_pairs.py")
        sys.exit(1)

    pairs_data = np.load(pairs_path)
    positive_pairs = pairs_data['positive_pairs']  # [N, 2]
    neighbor_static_np = pairs_data['neighbor_static']  # [n_nodes, n_static]

    log(f"  Positive pairs: {len(positive_pairs):,}")
    log(f"  Neighbor static: {neighbor_static_np.shape}")

    # Static features (z-scored, from data module)
    x_static = data_module.static_features.to(device)  # [n_nodes, n_static]
    neighbor_static = torch.tensor(neighbor_static_np, dtype=torch.float32, device=device)

    # Also z-score normalize neighbor_static using same stats as static_features
    neighbor_static = (neighbor_static - data_module.static_mean.to(device)) / data_module.static_std.to(device)

    n_nodes = x_static.shape[0]
    n_static = x_static.shape[1]

    # Create basin encoder
    encoder = BasinEncoder(
        static_dim=n_static,
        embed_dim=64,
        hidden_dim=128,
    ).to(device)

    log(f"  BasinEncoder parameters: {count_parameters(encoder):,}")

    loss_fn = InfoNCELoss(temperature=temperature)
    optimizer = optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Build positive pair lookup for batch-level masking
    pair_set = set()
    for i, j in positive_pairs:
        pair_set.add((int(i), int(j)))
        pair_set.add((int(j), int(i)))

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float('inf')

    for epoch in range(epochs):
        encoder.train()
        epoch_loss = 0.0
        n_batches = 0

        # Random batches of nodes
        indices = np.random.permutation(n_nodes)

        for start in range(0, n_nodes, batch_size):
            end = min(start + batch_size, n_nodes)
            batch_idx = indices[start:end]
            batch_idx_tensor = torch.tensor(batch_idx, dtype=torch.long, device=device)

            x_batch = x_static[batch_idx_tensor]
            nb_batch = neighbor_static[batch_idx_tensor]

            # Compute embeddings + projection
            embeddings = encoder(x_batch, nb_batch)
            projected = encoder.project(embeddings)
            projected = torch.nn.functional.normalize(projected, dim=-1)

            # Build positive mask for this batch
            bs = len(batch_idx)
            pos_mask = torch.zeros(bs, bs, dtype=torch.bool, device=device)
            idx_to_local = {int(batch_idx[i]): i for i in range(bs)}

            for local_i in range(bs):
                global_i = int(batch_idx[local_i])
                for local_j in range(local_i + 1, bs):
                    global_j = int(batch_idx[local_j])
                    if (global_i, global_j) in pair_set:
                        pos_mask[local_i, local_j] = True
                        pos_mask[local_j, local_i] = True

            # Skip if no positive pairs in batch
            if not pos_mask.any():
                continue

            loss = loss_fn(projected, pos_mask)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        log(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'loss': avg_loss,
                'n_static': n_static,
                'embed_dim': 64,
            }, ckpt_dir / 'best_encoder.pt')
            log(f"    NEW BEST! Saved: {ckpt_dir}/best_encoder.pt")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_dir / f'encoder_epoch_{epoch + 1}.pt')

    log(f"\nContrastive pretraining complete. Best loss: {best_loss:.4f}")

    sync_if_distributed()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train GNN v17 — PUB-oriented architectures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('model_type', type=str, choices=['ea-lstm', 'decoupled-gnn', 'contrastive-pub'],
                        help='Model variant')
    parser.add_argument('stage', type=str, choices=['contrastive', 'pretrain', 'finetune'],
                        help='Training stage')

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
    parser.add_argument('--encoder-checkpoint', type=str, default=None,
                        help='[contrastive-pub] Path to pretrained encoder checkpoint')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--resume-best', action='store_true')
    parser.add_argument('--contrastive-epochs', type=int, default=100)
    parser.add_argument('--contrastive-lr', type=float, default=1e-3)
    parser.add_argument('--contrastive-batch-size', type=int, default=2048)

    args = parser.parse_args()

    # Validate stage + model_type combinations
    if args.stage == 'contrastive' and args.model_type != 'contrastive-pub':
        parser.error("Contrastive stage only valid for contrastive-pub model type")

    # Stage-specific defaults
    if args.gauge_dropout_rate is None:
        args.gauge_dropout_rate = 0.0 if args.stage == 'pretrain' else 0.15
    if args.stage == 'contrastive':
        args.gauge_dropout_rate = 0.0

    if args.lr is None:
        args.lr = 5e-4 if args.stage == 'pretrain' else 1e-4
    if args.stage == 'contrastive':
        args.lr = args.contrastive_lr

    # Checkpoint directories
    base_ckpt = f'checkpoints_v17/{args.model_type}'
    if args.stage == 'contrastive':
        checkpoint_dir = f'{base_ckpt}/contrastive'
    elif args.stage == 'pretrain':
        checkpoint_dir = f'{base_ckpt}/pretrain'
    else:
        checkpoint_dir = f'{base_ckpt}/finetune'
    version = f'v17-{args.model_type}-{args.stage}'

    # Logging
    log("=" * 60)
    log(f"GNN STREAMFLOW V17 — PUB ARCHITECTURES")
    log(f"Model: {args.model_type.upper()}")
    log(f"Stage: {args.stage.upper()}")
    log("=" * 60)
    log(f"Start time: {datetime.now()}")

    rank, world_size, local_rank = setup_distributed()
    log(f"World size: {world_size}")

    # Config
    config = get_config_v17()
    if args.epochs:
        config.train.max_epochs = args.epochs

    config.data.split_mode = "basin"
    config.data.train_basin_fraction = args.train_basin_fraction
    config.data.basin_split_seed = args.basin_split_seed
    config.train.learning_rate = args.lr

    log(f"\nConfig:")
    log(f"  Model type: {args.model_type}")
    log(f"  LSTM: hidden={config.model.lstm_hidden_dim}, layers={config.model.lstm_num_layers}")
    log(f"  LR: {args.lr}")
    log(f"  Warmup days: {args.warmup_days}")
    log(f"  Seq length: {args.seq_length}")
    log(f"  Basin fraction: {args.train_basin_fraction}")
    log(f"  Loss: {args.loss}")
    log(f"  Gauge dropout: {args.gauge_dropout_rate}")

    # Load data
    log("\n" + "=" * 60)
    log("LOADING DATA")
    log("=" * 60)

    data_module = SequenceDataModule(config, rank, world_size, warmup_days=args.warmup_days)

    # Handle contrastive stage separately
    if args.stage == 'contrastive':
        run_contrastive_pretraining(
            config=config,
            data_module=data_module,
            checkpoint_dir=checkpoint_dir,
            lr=args.contrastive_lr,
            epochs=args.contrastive_epochs,
            batch_size=args.contrastive_batch_size,
        )
        log(f"\nEnd time: {datetime.now()}")
        return

    # Stage-specific date and basin logic (same as v16)
    if args.stage == 'pretrain':
        train_dates = build_date_list(config.data.train_years)
        val_dates = build_date_list(config.data.val_years)
        train_basin_nodes = data_module.train_gauged_nodes
        eval_basin_nodes = data_module.train_gauged_nodes  # temporal validation

        log(f"\n--- PRETRAIN STAGE ---")
        log(f"Train dates: {len(train_dates)}")
        log(f"Val dates: {len(val_dates)}")
        log(f"Train basins: {len(train_basin_nodes)}")
        log(f"Val basins: {len(eval_basin_nodes)} (SAME as train — temporal)")
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

    # Load neighbor_static for contrastive model
    neighbor_static = None
    if args.model_type == 'contrastive-pub':
        pairs_path = config.data.base_dir / "data" / "basin_pairs.npz"
        if pairs_path.exists():
            pairs_data = np.load(pairs_path)
            neighbor_static = torch.tensor(pairs_data['neighbor_static'], dtype=torch.float32)
            # Z-score normalize
            neighbor_static = (neighbor_static - data_module.static_mean) / data_module.static_std
            log(f"Loaded neighbor_static: {neighbor_static.shape}")

    # Save split metadata
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if is_main_process():
        metadata = data_module.get_split_metadata()
        metadata['stage'] = args.stage
        metadata['model_type'] = args.model_type
        with (Path(checkpoint_dir) / "split_metadata.json").open('w') as f:
            json.dump(metadata, f, indent=2)

    # Create trainer
    trainer = V17Trainer(
        config=config,
        data_module=data_module,
        model_type=args.model_type,
        seq_length=args.seq_length,
        warmup_days=args.warmup_days,
        loss_type=args.loss,
        warmup_epochs=args.warmup_epochs,
        sequences_per_epoch=args.sequences_per_epoch,
        gauge_dropout_rate=args.gauge_dropout_rate,
        checkpoint_dir=checkpoint_dir,
        version=version,
        neighbor_static=neighbor_static,
    )

    # Load checkpoint
    start_epoch = 0

    if args.model_type == 'contrastive-pub' and args.stage in ('pretrain', 'finetune'):
        # Load pretrained encoder into the model
        encoder_ckpt = args.encoder_checkpoint or f'checkpoints_v17/{args.model_type}/contrastive/best_encoder.pt'
        if Path(encoder_ckpt).exists():
            raw_model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
            ckpt = torch.load(encoder_ckpt, map_location=trainer.device, weights_only=False)
            raw_model.basin_encoder.load_state_dict(ckpt['encoder_state_dict'])
            if is_main_process():
                log(f"Loaded pretrained encoder from: {encoder_ckpt}")
                log(f"  Encoder epoch: {ckpt.get('epoch', 'N/A')}, loss: {ckpt.get('loss', 'N/A'):.4f}")
        else:
            if is_main_process():
                log(f"WARNING: No pretrained encoder found at {encoder_ckpt}")
                log("  Training contrastive-pub without pretrained encoder!")

    if args.stage == 'finetune' and args.resume is None and not args.resume_best:
        pretrain_ckpt = args.pretrain_checkpoint or f'checkpoints_v17/{args.model_type}/pretrain/best_model.pt'
        if Path(pretrain_ckpt).exists():
            start_epoch = trainer.load_pretrain_weights(pretrain_ckpt)
        else:
            if is_main_process():
                log(f"WARNING: No pretrain checkpoint at {pretrain_ckpt}, training from scratch")
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

    # Train
    trainer.fit(
        data_module,
        train_dates,
        val_dates,
        train_basin_nodes=train_basin_nodes,
        eval_basin_nodes=eval_basin_nodes,
        start_epoch=start_epoch,
        stage_label=f"{args.model_type}/{args.stage}",
    )

    log(f"\nEnd time: {datetime.now()}")
    log(f"V17 {args.model_type} {args.stage} complete!")


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
