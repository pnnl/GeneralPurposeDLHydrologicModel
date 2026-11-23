#!/usr/bin/env python3
"""
Core multi-GPU data loading, loss functions, and distributed training utilities
for river-network Graph Neural Network streamflow modeling.

Provides:
- RollingKGELoss: differentiable Kling-Gupta Efficiency loss in discharge space
- SequenceDataModule: shared-memory Zarr forcing loader, directed river graph
  builder, static attribute normalizer, and streamflow target cache
- Distributed Data Parallel (DDP) setup, synchronization, and checkpoint helpers
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json
from datetime import datetime, timedelta
import zarr
import warnings
import time
from multiprocessing import shared_memory
import argparse

warnings.filterwarnings('ignore')

# Global shared memory handle
_shm_handle = None
_shm_name = None

os.environ['PYTHONUNBUFFERED'] = '1'

import math

from config import Config, get_config_v15
from model_v15 import StreamflowGNN, count_parameters


# ============================================================================
# Rolling KGE Loss (discharge space)
# ============================================================================

class RollingKGELoss(nn.Module):
    """
    Rolling KGE loss for sequence-based training.
    KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
    """

    def __init__(self, min_valid_ratio: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.min_valid_ratio = min_valid_ratio
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len = pred.shape
        basin_losses = []
        basin_kge_values = []
        valid_basins = 0

        for b in range(batch_size):
            basin_pred = pred[b]
            basin_obs = obs[b]
            valid_mask = ~torch.isnan(basin_obs)
            n_valid = valid_mask.sum()

            if n_valid < seq_len * self.min_valid_ratio:
                continue

            valid_pred = basin_pred[valid_mask]
            valid_obs = basin_obs[valid_mask]

            pred_mean = valid_pred.mean()
            obs_mean = valid_obs.mean()
            pred_std = valid_pred.std()
            obs_std = valid_obs.std()

            pred_centered = valid_pred - pred_mean
            obs_centered = valid_obs - obs_mean
            cov = (pred_centered * obs_centered).mean()

            if pred_std > self.eps and obs_std > self.eps:
                r = cov / (pred_std * obs_std)
                r = torch.clamp(r, -1.0, 1.0)
            else:
                r = torch.tensor(0.0, device=pred.device)

            alpha = pred_std / (obs_std + self.eps)
            beta = pred_mean / (obs_mean + self.eps)

            basin_loss = torch.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
            basin_loss = torch.clamp(basin_loss, 0.0, 3.0)

            basin_losses.append(basin_loss)
            basin_kge_values.append(1 - basin_loss.item())
            valid_basins += 1

        if len(basin_losses) == 0:
            return pred.sum() * 0.0, {
                'n_basins': 0, 'median_kge': 0.0, 'mean_kge': 0.0
            }

        loss = torch.stack(basin_losses).mean()
        stats = {
            'n_basins': valid_basins,
            'median_kge': float(np.median(basin_kge_values)),
            'mean_kge': float(np.mean(basin_kge_values)),
        }
        return loss, stats


class RollingNSELoss(nn.Module):
    """Rolling NSE loss for sequence-based training."""

    def __init__(self, min_valid_ratio: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.min_valid_ratio = min_valid_ratio
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len = pred.shape
        basin_losses = []
        basin_nse_values = []
        valid_basins = 0

        for b in range(batch_size):
            basin_pred = pred[b]
            basin_obs = obs[b]
            valid_mask = ~torch.isnan(basin_obs)
            n_valid = valid_mask.sum()

            if n_valid < seq_len * self.min_valid_ratio:
                continue

            valid_pred = basin_pred[valid_mask]
            valid_obs = basin_obs[valid_mask]

            obs_mean = valid_obs.mean()
            ss_res = torch.sum((valid_obs - valid_pred) ** 2)
            ss_tot = torch.sum((valid_obs - obs_mean) ** 2)

            if ss_tot < self.eps:
                basin_loss = ss_res / (n_valid * self.eps)
            else:
                basin_loss = ss_res / (ss_tot + self.eps)

            basin_loss = torch.clamp(basin_loss, 0.0, 10.0)
            basin_losses.append(basin_loss)
            basin_nse_values.append(1 - basin_loss.item())
            valid_basins += 1

        if len(basin_losses) == 0:
            return pred.sum() * 0.0, {
                'n_basins': 0, 'median_nse': 0.0, 'mean_nse': 0.0
            }

        loss = torch.stack(basin_losses).mean()
        stats = {
            'n_basins': valid_basins,
            'median_nse': float(np.median(basin_nse_values)),
            'mean_nse': float(np.mean(basin_nse_values)),
        }
        return loss, stats


# ============================================================================
# Distributed Utilities
# ============================================================================

def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ.get('SLURM_LOCALID', 0))

        if 'MASTER_ADDR' not in os.environ:
            import subprocess
            result = subprocess.run(
                ['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']],
                capture_output=True, text=True
            )
            master = result.stdout.strip().split('\n')[0]
            os.environ['MASTER_ADDR'] = master
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1 and not dist.is_initialized():
        from datetime import timedelta
        try:
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank,
                timeout=timedelta(minutes=60),
                device_id=torch.device(f'cuda:{local_rank}'),
            )
        except Exception as e:
            print(f"[Rank {rank}] Failed to init process group: {e}", flush=True)
            print(f"[Rank {rank}] MASTER_ADDR={os.environ.get('MASTER_ADDR')}", flush=True)
            print(f"[Rank {rank}] MASTER_PORT={os.environ.get('MASTER_PORT')}", flush=True)
            raise

    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    return rank, world_size, local_rank


def is_main_process():
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def sync_if_distributed():
    if dist.is_initialized():
        dist.barrier()


def log(msg: str):
    if is_main_process():
        print(msg, flush=True)


def unique_years(*year_lists: List[int]) -> List[int]:
    """Return sorted unique years from one or more year lists."""
    years = set()
    for year_list in year_lists:
        years.update(int(year) for year in year_list)
    return sorted(years)


def build_date_list(years: List[int]) -> List[pd.Timestamp]:
    """Build a dense daily date list for the provided years."""
    dates = []
    for year in sorted(set(int(year) for year in years)):
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year}-12-31")
        dates.extend(pd.date_range(start, end, freq='D'))
    return dates


# ============================================================================
# Data Module — v15: directed edges + edge features (larger LSTM)
# ============================================================================

class SequenceDataModule:
    """
    Data module for v15 day-by-day LSTM training with directed edges and edge features.

    Features:
    - Builds directed (upstream/downstream) edges instead of bidirectional
    - Computes 6 edge features per directed edge
    - get_subgraph_directed_edges() returns separate up/down edge_index + edge_attr
    """

    # Gauge quality thresholds
    MIN_COVERAGE = 0.25
    MIN_MEAN_FLOW_CMS = 0.15

    EDGE_FEAT_DIM = 6  # Must match config.model.edge_feat_dim

    def __init__(self, config: Config, rank: int = 0, world_size: int = 1, warmup_days: int = 365):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.warmup_days = warmup_days
        self.split_mode = config.data.split_mode
        self.train_basin_fraction = float(config.data.train_basin_fraction)
        self.basin_split_seed = int(config.data.basin_split_seed)
        self.all_years = unique_years(
            config.data.train_years,
            config.data.val_years,
            config.data.test_years,
        )

        if self.split_mode not in {"temporal", "basin"}:
            raise ValueError(f"Unsupported split_mode: {self.split_mode}")
        if self.split_mode == "basin" and not (0.0 < self.train_basin_fraction < 1.0):
            raise ValueError("train_basin_fraction must be in (0, 1) for basin split mode")
        if not self.all_years:
            raise ValueError("At least one year must be configured")

        # Load graph
        log("Loading graph...")
        gpkg_path = config.data.base_dir / config.data.catchment_network
        gdf = gpd.read_file(gpkg_path, layer='catchments_network')
        gdf = gdf[gdf['COMID'].notna()].reset_index(drop=True)

        self.comid_to_idx = {int(comid): idx for idx, comid in enumerate(gdf['COMID'])}
        self.idx_to_comid = {idx: comid for comid, idx in self.comid_to_idx.items()}
        self.idx_to_hruid = {idx: int(hruid) for idx, hruid in enumerate(gdf['hruid'])}
        self.n_nodes = len(self.comid_to_idx)

        self.node_ids = [self.idx_to_comid[i] for i in range(self.n_nodes)]
        self.node_to_idx = self.comid_to_idx

        # Store node properties for edge feature computation
        self.node_uparea = {}   # idx → upstream area (km²)
        self.node_area = {}     # idx → local catchment area (m²)
        self.node_order = {}    # idx → stream order
        self.node_centroid = {} # idx → (lon, lat)

        centroids = gdf.geometry.centroid
        for idx in range(len(gdf)):
            row = gdf.iloc[idx]
            self.node_uparea[idx] = float(row['uparea']) if pd.notna(row['uparea']) else 0.0
            self.node_area[idx] = float(row['area']) if pd.notna(row['area']) else 0.0
            self.node_order[idx] = float(row['order']) if pd.notna(row['order']) else 1.0
            self.node_centroid[idx] = (centroids.iloc[idx].x, centroids.iloc[idx].y)

        log(f"  Nodes: {self.n_nodes}")

        # Store gdf reference for edge building (need NextDownID, up1-up4)
        self._gdf = gdf

        # Load static features (also extracts elevation for edge features)
        self._load_static_features(config)

        # Build directed edges with features (needs elevation, areas, centroids)
        self._build_directed_edges()

        # Load forcings
        self._load_forcings(config)

        # Load streamflow
        self._load_streamflow(config)

        # Build train/eval basin splits after gauges are known
        self._build_basin_splits()

        # Compute forcing normalization stats using the active training split
        self._compute_forcing_stats(config)

        # Build adjacency lists (undirected for K-hop, directed stored in _build_directed_edges)
        self._build_adjacency_lists()

        # Release gdf reference
        del self._gdf

    def _load_static_features(self, config: Config):
        """Load V2 static features, extract per-node catchment areas and elevations."""
        log("Loading static features (V2)...")

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

        log(f"  Using {len(feature_cols)} features: {feature_cols}")

        # Build feature matrix
        features = np.full((self.n_nodes, len(feature_cols)), np.nan, dtype=np.float32)
        for idx, hruid in self.idx_to_hruid.items():
            if hruid in attrs_df.index:
                features[idx] = attrs_df.loc[hruid, feature_cols].values

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
        # Override config: lstm_input_dim = forcing + static
        config.model.static_input_dim = self.n_static_features
        config.model.lstm_input_dim = len(config.data.forcing_vars) + self.n_static_features
        log(f"  Static features shape: {self.static_features.shape}")
        log(f"  lstm_input_dim set to: {config.model.lstm_input_dim} ({len(config.data.forcing_vars)} forcing + {self.n_static_features} static)")

    def _load_forcings(self, config: Config):
        """Load forcing data into POSIX shared memory across ranks."""
        log("Loading forcing data...")

        processed_dir = config.data.base_dir / config.data.forcings_processed_dir
        f16_zarr_path = processed_dir / "forcings_f16.zarr"
        f32_zarr_path = processed_dir / "forcings.zarr"

        if f16_zarr_path.exists():
            zarr_path = f16_zarr_path
            log(f"  Using float16 zarr: {zarr_path}")
            self.use_f16 = True
        else:
            zarr_path = f32_zarr_path
            log(f"  Using float32 zarr: {zarr_path}")
            self.use_f16 = False

        zarr_store = zarr.open(str(zarr_path), mode='r')

        all_dates = pd.to_datetime(zarr_store['dates'][:])
        catchment_ids = zarr_store['catchment_ids'][:]
        self.forcing_catchment_ids = catchment_ids

        min_year = min(self.all_years)
        max_year = max(self.all_years)

        # Need extra days before train start for warmup + buffer
        preload_days = self.warmup_days + 35
        start_date = pd.Timestamp(f"{min_year}-01-01") - pd.Timedelta(days=preload_days)
        end_date = pd.Timestamp(f"{max_year}-12-31")

        start_idx = 0
        end_idx = len(all_dates) - 1

        for i, d in enumerate(all_dates):
            if d >= start_date:
                start_idx = i
                break

        for i in range(len(all_dates) - 1, -1, -1):
            if all_dates[i] <= end_date:
                end_idx = i
                break

        self.forcing_dates = all_dates[start_idx:end_idx + 1]
        self.date_to_idx = {d: i for i, d in enumerate(self.forcing_dates)}
        self._zarr_start_idx = start_idx
        self._zarr_end_idx = end_idx

        log(f"  Full zarr date range: {all_dates[0].date()} to {all_dates[-1].date()}")
        log(f"  Loading subset: {self.forcing_dates[0].date()} to {self.forcing_dates[-1].date()} ({len(self.forcing_dates)} days)")

        self.hruid_to_forcing_idx = {int(cid): idx for idx, cid in enumerate(catchment_ids)}

        self.node_to_forcing_idx = {}
        for node_idx, hruid in self.idx_to_hruid.items():
            if hruid in self.hruid_to_forcing_idx:
                self.node_to_forcing_idx[node_idx] = self.hruid_to_forcing_idx[hruid]

        log(f"  Mapped {len(self.node_to_forcing_idx)} nodes to forcings")

        # Load into shared memory
        full_shape = zarr_store['data'].shape
        n_catchments = full_shape[0]
        n_days = end_idx - start_idx + 1
        n_vars = full_shape[2]
        data_shape = (n_catchments, n_days, n_vars)

        dtype = np.float16 if self.use_f16 else np.float32
        nbytes = int(np.prod(data_shape) * np.dtype(dtype).itemsize)

        log(f"  Full zarr shape: {full_shape}")
        log(f"  Subset shape: {data_shape}")
        log(f"  Memory needed: {nbytes / 1e9:.2f} GB")

        global _shm_handle, _shm_name
        job_id = os.environ.get('SLURM_JOB_ID', str(os.getpid()))
        shm_name = f'gnn_forcing_v15_{job_id}'

        if self.rank == 0:
            log(f"  [Rank 0] Loading data and creating shared memory '{shm_name}'...")

            try:
                old_shm = shared_memory.SharedMemory(name=shm_name)
                old_shm.close()
                old_shm.unlink()
            except:
                pass

            shm = shared_memory.SharedMemory(create=True, size=nbytes, name=shm_name)
            arr = np.ndarray(data_shape, dtype=dtype, buffer=shm.buf)

            start = time.time()
            arr[:] = zarr_store['data'][:, start_idx:end_idx + 1, :]
            log(f"  [Rank 0] Loaded from zarr in {time.time()-start:.1f}s")

            _shm_handle = shm
            _shm_name = shm_name

            log(f"  [Rank 0] Shared memory created, waiting for other ranks...")
            sync_if_distributed()
        else:
            sync_if_distributed()
            shm = shared_memory.SharedMemory(name=shm_name)
            _shm_handle = shm
            _shm_name = shm_name

        self.forcing_data = np.ndarray(data_shape, dtype=dtype, buffer=_shm_handle.buf)

        log(f"  Shared memory setup complete")
        log(f"  Shape: {self.forcing_data.shape}")

    def _build_basin_splits(self):
        """Create explicit train/eval basin partitions for PUB experiments."""
        self.all_gauged_nodes = list(self.gauged_nodes)

        if len(self.all_gauged_nodes) == 0:
            raise RuntimeError("No gauged basins available after streamflow quality filtering")

        if self.split_mode == "basin":
            if len(self.all_gauged_nodes) < 2:
                raise RuntimeError("Basin split requires at least two gauged basins")

            shuffled_nodes = np.array(self.all_gauged_nodes, dtype=np.int64)
            rng = np.random.RandomState(self.basin_split_seed)
            rng.shuffle(shuffled_nodes)

            n_total = len(shuffled_nodes)
            n_train = int(np.floor(n_total * self.train_basin_fraction))
            n_train = max(1, min(n_total - 1, n_train))

            self.train_gauged_nodes = sorted(shuffled_nodes[:n_train].tolist())
            self.eval_gauged_nodes = sorted(shuffled_nodes[n_train:].tolist())
        else:
            self.train_gauged_nodes = list(self.all_gauged_nodes)
            self.eval_gauged_nodes = list(self.all_gauged_nodes)

        self.split_summary = {
            'split_mode': self.split_mode,
            'basin_split_seed': self.basin_split_seed,
            'train_basin_fraction_requested': self.train_basin_fraction,
            'train_basin_fraction_actual': len(self.train_gauged_nodes) / len(self.all_gauged_nodes),
            'n_total_gauged_basins': len(self.all_gauged_nodes),
            'n_train_basins': len(self.train_gauged_nodes),
            'n_eval_basins': len(self.eval_gauged_nodes),
            'all_years': list(self.all_years),
        }

        overlap = len(set(self.train_gauged_nodes) & set(self.eval_gauged_nodes))
        if self.split_mode == "basin" and overlap != 0:
            raise RuntimeError("Train/eval basin split is not disjoint")

        log("Basin split summary:")
        log(f"  Mode: {self.split_mode}")
        log(f"  Total gauged basins: {len(self.all_gauged_nodes)}")
        log(f"  Train basins: {len(self.train_gauged_nodes)}")
        log(f"  Eval basins: {len(self.eval_gauged_nodes)}")
        if self.split_mode == "basin":
            log(f"  Train fraction: {self.split_summary['train_basin_fraction_actual']:.3f} "
                f"(requested {self.train_basin_fraction:.3f})")
            log(f"  Split seed: {self.basin_split_seed}")

    def get_split_summary(self) -> Dict:
        """Return lightweight metadata about the current basin split."""
        return dict(self.split_summary)

    def get_split_metadata(self) -> Dict:
        """Return reproducible basin split metadata for logging/export."""
        return {
            **self.get_split_summary(),
            'train_node_indices': [int(node_idx) for node_idx in self.train_gauged_nodes],
            'eval_node_indices': [int(node_idx) for node_idx in self.eval_gauged_nodes],
            'train_comids': [int(self.idx_to_comid[node_idx]) for node_idx in self.train_gauged_nodes],
            'eval_comids': [int(self.idx_to_comid[node_idx]) for node_idx in self.eval_gauged_nodes],
            'train_hruids': [int(self.idx_to_hruid[node_idx]) for node_idx in self.train_gauged_nodes],
            'eval_hruids': [int(self.idx_to_hruid[node_idx]) for node_idx in self.eval_gauged_nodes],
        }

    def _compute_forcing_stats(self, config: Config):
        """
        Compute per-variable mean and std for z-score normalization.
        Samples from training period catchments to be representative.
        """
        log("Computing forcing normalization statistics...")

        n_forcing_catchments = self.forcing_data.shape[0]

        if self.split_mode == "basin":
            train_forcing_indices = sorted({
                self.node_to_forcing_idx[node_idx]
                for node_idx in self.train_gauged_nodes
                if node_idx in self.node_to_forcing_idx
            })
            if train_forcing_indices:
                candidate_catchments = np.array(train_forcing_indices, dtype=np.int64)
                stats_years = self.all_years
                log(f"  Using train-basin forcings only for normalization ({len(candidate_catchments)} basins)")
            else:
                candidate_catchments = np.arange(n_forcing_catchments, dtype=np.int64)
                stats_years = self.all_years
                log("  WARNING: No forcing indices for train basins; falling back to all catchments")
        else:
            candidate_catchments = np.arange(n_forcing_catchments, dtype=np.int64)
            stats_years = sorted(set(config.data.train_years))
            log(f"  Using all catchments over temporal training years for normalization ({len(candidate_catchments)} catchments)")

        # Sample up to 10K catchments from the active training split
        sample_size = min(10000, len(candidate_catchments))
        rng = np.random.RandomState(42)
        sample_catchments = rng.choice(candidate_catchments, sample_size, replace=False)

        # Get training date indices
        train_start = pd.Timestamp(f"{min(stats_years)}-01-01")
        train_end = pd.Timestamp(f"{max(stats_years)}-12-31")

        train_start_idx = self.date_to_idx.get(train_start, 0)
        train_end_idx = self.date_to_idx.get(train_end, len(self.forcing_dates) - 1)

        # Sample data for stats — use float64 to avoid accumulator overflow
        sample_data = self.forcing_data[sample_catchments, train_start_idx:train_end_idx + 1, :]
        sample_data = sample_data.astype(np.float64)

        # Compute per-variable stats (ignore NaN/zero-padded)
        self.forcing_mean = np.nanmean(sample_data, axis=(0, 1)).astype(np.float32)  # [n_vars]
        self.forcing_std = np.nanstd(sample_data, axis=(0, 1)).astype(np.float32)    # [n_vars]
        self.forcing_std = np.maximum(self.forcing_std, 1e-8)      # avoid division by zero

        self.forcing_mean_tensor = torch.tensor(self.forcing_mean, dtype=torch.float32)
        self.forcing_std_tensor = torch.tensor(self.forcing_std, dtype=torch.float32)

        log(f"  Stats years: {min(stats_years)}-{max(stats_years)}")
        log(f"  Forcing means: {self.forcing_mean}")
        log(f"  Forcing stds:  {self.forcing_std}")

    def _load_streamflow(self, config: Config):
        """Load streamflow with quality filtering and area normalization."""
        log("Loading streamflow (quality-filtered, area-normalized)...")

        streamflow_dir = config.data.base_dir / config.data.streamflow_dir
        self.streamflow_cache = {}       # specific_discharge_mm_day
        self.gauged_nodes = []

        n_loaded = 0
        n_no_area = 0
        n_low_coverage = 0
        n_low_flow = 0

        for source in ['usgs', 'hydat', 'conagua']:
            source_dir = streamflow_dir / source
            if not source_dir.exists():
                continue
            for fpath in source_dir.glob("comid_*.parquet"):
                try:
                    comid = int(fpath.stem.split('_')[1])
                    node_idx = self.comid_to_idx.get(comid)
                    if node_idx is None:
                        continue

                    df = pd.read_parquet(fpath)
                    df = df.set_index('date')['flow_cms']
                    df = df[df >= 0]
                    if len(df) == 0:
                        continue

                    n_loaded += 1

                    date_span = (df.index.max() - df.index.min()).days + 1
                    coverage = len(df) / max(date_span, 1)
                    if coverage < self.MIN_COVERAGE:
                        n_low_coverage += 1
                        continue

                    mean_flow = df.groupby(df.index.year).mean().mean()
                    if mean_flow < self.MIN_MEAN_FLOW_CMS:
                        n_low_flow += 1
                        continue

                    area_m2 = self.node_areas.get(node_idx)
                    if area_m2 is None or area_m2 <= 0:
                        n_no_area += 1
                        continue

                    # Convert to specific discharge (mm/day)
                    q_mm_day = df * 86400.0 * 1000.0 / area_m2

                    self.streamflow_cache[node_idx] = q_mm_day
                    self.gauged_nodes.append(node_idx)

                except Exception:
                    continue

        self.gauged_nodes = sorted(set(self.gauged_nodes))

        log(f"  Gage quality filtering:")
        log(f"    Total files loaded: {n_loaded}")
        log(f"    Rejected - low coverage (<{self.MIN_COVERAGE*100:.0f}%): {n_low_coverage}")
        log(f"    Rejected - low mean flow (<{self.MIN_MEAN_FLOW_CMS} cms): {n_low_flow}")
        log(f"    Rejected - no catchment area: {n_no_area}")
        log(f"    Accepted gauged nodes: {len(self.gauged_nodes)}")

    @staticmethod
    def _haversine_km(lon1, lat1, lon2, lat2):
        """Haversine distance in km between two points in degrees."""
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        return R * 2 * math.asin(math.sqrt(a))

    def _build_directed_edges(self):
        """
        Build directed edges with 6 physical edge features.

        Upstream edges: src=upstream_node → dst=downstream_node  (flow accumulation)
        Downstream edges: src=downstream_node → dst=upstream_node  (backwater)

        Edge features (computed from src→dst perspective):
            0: upstream_area_ratio  = uparea_src / (uparea_dst + eps)
            1: elev_diff            = elev_src - elev_dst  (meters)
            2: distance_km          = haversine(centroid_src, centroid_dst)
            3: slope                = elev_diff / (distance_km * 1000 + eps)  (m/m)
            4: order_diff           = order_src - order_dst
            5: local_area_ratio     = area_src / (area_dst + eps)  (local catchment areas)
        """
        log("Building directed edges with features...")
        gdf = self._gdf
        eps = 1e-8

        # Default elevation for nodes without data
        mean_elev = np.mean(list(self.node_elevation.values())) if self.node_elevation else 0.0

        # Collect unique directed pairs: (upstream_node, downstream_node)
        # Use both NextDownID and up1-up4 for completeness, deduplicate
        seen_pairs = set()  # (upstream_idx, downstream_idx)
        directed_pairs = []  # list of (upstream_idx, downstream_idx)

        for idx in range(len(gdf)):
            row = gdf.iloc[idx]

            # NextDownID: idx is upstream, NextDownID is downstream
            next_down = row['NextDownID']
            if pd.notna(next_down) and int(next_down) != 0:
                down_idx = self.comid_to_idx.get(int(next_down))
                if down_idx is not None:
                    pair = (idx, down_idx)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        directed_pairs.append(pair)

            # up1-up4: up_col is upstream of idx (idx is downstream)
            for up_col in ['up1', 'up2', 'up3', 'up4']:
                up_id = row[up_col]
                if pd.notna(up_id) and int(up_id) != 0:
                    up_idx = self.comid_to_idx.get(int(up_id))
                    if up_idx is not None:
                        pair = (up_idx, idx)
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            directed_pairs.append(pair)

        log(f"  Unique directed connections: {len(directed_pairs)}")

        # Compute edge features for both directions
        upstream_edges = []    # (src=upstream, dst=downstream)
        upstream_feats = []
        downstream_edges = []  # (src=downstream, dst=upstream)
        downstream_feats = []

        for up_idx, down_idx in directed_pairs:
            # Node properties
            uparea_up = self.node_uparea.get(up_idx, 0.0)
            uparea_down = self.node_uparea.get(down_idx, 0.0)
            area_up = self.node_area.get(up_idx, 0.0)
            area_down = self.node_area.get(down_idx, 0.0)
            elev_up = self.node_elevation.get(up_idx, mean_elev)
            elev_down = self.node_elevation.get(down_idx, mean_elev)
            order_up = self.node_order.get(up_idx, 1.0)
            order_down = self.node_order.get(down_idx, 1.0)
            lon_up, lat_up = self.node_centroid[up_idx]
            lon_down, lat_down = self.node_centroid[down_idx]

            dist_km = self._haversine_km(lon_up, lat_up, lon_down, lat_down)
            elev_diff_up2down = elev_up - elev_down  # positive = downhill (natural flow)

            # Upstream edge features (src=upstream → dst=downstream)
            upstream_edges.append([up_idx, down_idx])
            upstream_feats.append([
                uparea_up / (uparea_down + eps),               # 0: area_ratio
                elev_diff_up2down,                             # 1: elev_diff (positive)
                dist_km,                                       # 2: distance_km
                elev_diff_up2down / (dist_km * 1000 + eps),   # 3: slope (m/m)
                order_up - order_down,                         # 4: order_diff (≤0)
                area_up / (area_down + eps),                   # 5: local_area_ratio
            ])

            # Downstream edge features (src=downstream → dst=upstream)
            downstream_edges.append([down_idx, up_idx])
            downstream_feats.append([
                uparea_down / (uparea_up + eps),               # 0: area_ratio (inverse)
                elev_down - elev_up,                           # 1: elev_diff (negative)
                dist_km,                                       # 2: distance_km (same)
                (elev_down - elev_up) / (dist_km * 1000 + eps),# 3: slope (negative)
                order_down - order_up,                         # 4: order_diff (≥0)
                area_down / (area_up + eps),                   # 5: local_area_ratio (inverse)
            ])

        # Convert to tensors
        if upstream_edges:
            self.upstream_edge_index = torch.tensor(upstream_edges, dtype=torch.long).t().contiguous()
            up_feat_raw = torch.tensor(upstream_feats, dtype=torch.float32)
        else:
            self.upstream_edge_index = torch.zeros((2, 0), dtype=torch.long)
            up_feat_raw = torch.zeros((0, self.EDGE_FEAT_DIM), dtype=torch.float32)

        if downstream_edges:
            self.downstream_edge_index = torch.tensor(downstream_edges, dtype=torch.long).t().contiguous()
            down_feat_raw = torch.tensor(downstream_feats, dtype=torch.float32)
        else:
            self.downstream_edge_index = torch.zeros((2, 0), dtype=torch.long)
            down_feat_raw = torch.zeros((0, self.EDGE_FEAT_DIM), dtype=torch.float32)

        # Z-score normalize edge features (computed across BOTH directions jointly)
        all_feats = torch.cat([up_feat_raw, down_feat_raw], dim=0)
        self.edge_feat_mean = all_feats.mean(dim=0)
        self.edge_feat_std = all_feats.std(dim=0) + 1e-8

        self.upstream_edge_attr = (up_feat_raw - self.edge_feat_mean) / self.edge_feat_std
        self.downstream_edge_attr = (down_feat_raw - self.edge_feat_mean) / self.edge_feat_std

        # Build directed adjacency lists for subgraph extraction
        # upstream_adj[src] = [(dst, edge_idx), ...] where src is upstream of dst
        self.upstream_adj = {}
        for i in range(len(upstream_edges)):
            src, dst = upstream_edges[i]
            self.upstream_adj.setdefault(src, []).append((dst, i))

        # downstream_adj[src] = [(dst, edge_idx), ...] where src is downstream of dst
        self.downstream_adj = {}
        for i in range(len(downstream_edges)):
            src, dst = downstream_edges[i]
            self.downstream_adj.setdefault(src, []).append((dst, i))

        log(f"  Upstream edges: {self.upstream_edge_index.shape[1]}")
        log(f"  Downstream edges: {self.downstream_edge_index.shape[1]}")
        log(f"  Edge feature means: {self.edge_feat_mean.numpy()}")
        log(f"  Edge feature stds:  {self.edge_feat_std.numpy()}")

    def _build_adjacency_lists(self):
        """Build undirected adjacency list for K-hop neighborhood expansion."""
        log("Building adjacency lists...")
        self.adj_list = {}

        # Build from directed edges (both directions give us undirected)
        for src, neighbors in self.upstream_adj.items():
            for dst, _ in neighbors:
                self.adj_list.setdefault(src, []).append(dst)
                self.adj_list.setdefault(dst, []).append(src)

        # Deduplicate
        for node in self.adj_list:
            self.adj_list[node] = list(set(self.adj_list[node]))

        log(f"  Built adjacency list for {len(self.adj_list)} nodes")

    def get_subgraph_directed_edges(
        self,
        subgraph_nodes: List[int],
        orig_to_sub: Dict[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract directed edges + features for a subgraph.

        Returns:
            up_edge_index: [2, n_up] — upstream→downstream edges (subgraph-local indices)
            up_edge_attr:  [n_up, edge_feat_dim] — normalized features
            down_edge_index: [2, n_down]
            down_edge_attr:  [n_down, edge_feat_dim]
        """
        subgraph_set = set(subgraph_nodes)
        up_edges, up_feats = [], []
        down_edges, down_feats = [], []

        for orig_idx in subgraph_nodes:
            # Upstream edges from this node
            if orig_idx in self.upstream_adj:
                for dst, feat_idx in self.upstream_adj[orig_idx]:
                    if dst in subgraph_set:
                        up_edges.append([orig_to_sub[orig_idx], orig_to_sub[dst]])
                        up_feats.append(self.upstream_edge_attr[feat_idx])

            # Downstream edges from this node
            if orig_idx in self.downstream_adj:
                for dst, feat_idx in self.downstream_adj[orig_idx]:
                    if dst in subgraph_set:
                        down_edges.append([orig_to_sub[orig_idx], orig_to_sub[dst]])
                        down_feats.append(self.downstream_edge_attr[feat_idx])

        if up_edges:
            up_ei = torch.tensor(up_edges, dtype=torch.long).t().contiguous()
            up_ea = torch.stack(up_feats)
        else:
            up_ei = torch.zeros((2, 0), dtype=torch.long)
            up_ea = torch.zeros((0, self.EDGE_FEAT_DIM), dtype=torch.float32)

        if down_edges:
            down_ei = torch.tensor(down_edges, dtype=torch.long).t().contiguous()
            down_ea = torch.stack(down_feats)
        else:
            down_ei = torch.zeros((2, 0), dtype=torch.long)
            down_ea = torch.zeros((0, self.EDGE_FEAT_DIM), dtype=torch.float32)

        return up_ei, up_ea, down_ei, down_ea

    def get_sequence_streamflow(
        self,
        node_idx: int,
        start_date: pd.Timestamp,
        seq_length: int,
    ) -> np.ndarray:
        """Get streamflow in specific discharge (mm/day). Returns [seq_length] with NaN for missing."""
        result = np.full(seq_length, np.nan, dtype=np.float32)

        if node_idx not in self.streamflow_cache:
            return result

        flow_series = self.streamflow_cache[node_idx]

        for t in range(seq_length):
            date = start_date + pd.Timedelta(days=t)
            if date in flow_series.index:
                value = flow_series[date]
                if pd.notna(value) and value >= 0:
                    result[t] = value

        return result

    def get_k_hop_neighbors(self, seed_nodes: List[int], k: int = 2, max_nodes: int = 2500) -> List[int]:
        """Get K-hop neighborhood for GNN message passing. Seed nodes always retained."""
        current = set(seed_nodes)
        all_nodes = set(seed_nodes)

        for _ in range(k):
            if len(all_nodes) >= max_nodes:
                break
            next_layer = set()
            for node in current:
                if node in self.adj_list:
                    next_layer.update(self.adj_list[node])
            all_nodes.update(next_layer)
            current = next_layer

        # Guarantee seed nodes are always included; fill remaining budget with neighbors
        seed_set = set(seed_nodes)
        non_seed = sorted(all_nodes - seed_set)
        remaining_budget = max(0, max_nodes - len(seed_set))
        return sorted(list(seed_set) + non_seed[:remaining_budget])

    def get_subgraph_edges(self, subgraph_nodes: List[int], orig_to_sub: Dict[int, int]) -> torch.Tensor:
        """Get undirected edge index for subgraph (backward compat, unused in v15)."""
        edge_list = []
        subgraph_set = set(subgraph_nodes)

        for orig_idx in subgraph_nodes:
            if orig_idx in self.adj_list:
                for neighbor in self.adj_list[orig_idx]:
                    if neighbor in subgraph_set:
                        edge_list.append([orig_to_sub[orig_idx], orig_to_sub[neighbor]])

        if edge_list:
            return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return torch.zeros((2, 0), dtype=torch.long)


# ============================================================================
# Trainer — v15: Larger LSTM + Directed GNN
# ============================================================================

class RollingLossTrainer:
    """
    Trainer v15 — Kratzert-style LSTM + DirectedRiverGNN with edge features.

    Key features:
    - Directed message passing: separate upstream/downstream GATConv layers
    - Edge features (6-dim): area ratio, elev diff, distance, slope, order diff, area ratio (local)
    - Rolling KGE loss with optional warmup schedule
    """

    def __init__(
        self,
        config: Config,
        data_module: SequenceDataModule,
        seq_length: int = 90,
        warmup_days: int = 365,
        loss_type: str = "kge",
        warmup_epochs: int = 0,
        sequences_per_epoch: int = 200,
        gauge_dropout_rate: float = 0.0,
    ):
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
            log(f"\nInitializing v15 trainer on rank {self.rank}/{self.world_size}")
            log(f"  Device: {self.device}")
            log(f"  Sequence length: {seq_length} days (prediction)")
            log(f"  Warmup days: {warmup_days} (LSTM-only, no GNN)")
            log(f"  Total days per sample: {warmup_days + seq_length}")
            log(f"  Sequences per epoch: {sequences_per_epoch}")
            log(f"  Loss type: {loss_type} (real-space mm/day)")
            log(f"  MSE warmup epochs: {warmup_epochs}")
            log(f"  Gauge dropout rate: {gauge_dropout_rate}")
            log(f"  Output: Softplus → mm/day")

        # Create model
        self.model = StreamflowGNN(config.model).to(self.device)

        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
                bucket_cap_mb=25,
                gradient_as_bucket_view=True,
            )
            dist.barrier()
            if is_main_process():
                log("  All ranks synchronized after DDP wrapping")

        if is_main_process():
            log(f"  Model parameters: {count_parameters(self.model):,}")

        # Forcing normalization (move to GPU)
        self.forcing_mean = data_module.forcing_mean_tensor.to(self.device)
        self.forcing_std = data_module.forcing_std_tensor.to(self.device)

        # Edge feature normalization stats (for checkpoint saving)
        self.edge_feat_mean = data_module.edge_feat_mean
        self.edge_feat_std = data_module.edge_feat_std

        # Loss functions
        self.rolling_nse_loss = RollingNSELoss(min_valid_ratio=0.3)
        self.rolling_kge_loss = RollingKGELoss(min_valid_ratio=0.3)
        self.mse_loss = nn.MSELoss()

        # Optimizer
        try:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.train.learning_rate,
                weight_decay=config.train.weight_decay,
                fused=True,
            )
            if is_main_process():
                log(f"  Using fused AdamW (lr={config.train.learning_rate})")
        except TypeError:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.train.learning_rate,
                weight_decay=config.train.weight_decay,
            )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=is_main_process(),
        )

        # Mixed precision
        self.scaler = GradScaler()

        # Training settings
        self.batch_size = 64
        self.gradient_accumulation_steps = 4
        self.max_subgraph_nodes = 4000  # increased for k=3 hop neighborhood

        self.data_split_summary = data_module.get_split_summary()
        self.history = {
            'train_loss': [], 'val_nse': [], 'val_kge': [],
            'best_epoch': 0, 'best_val_metric': -float('inf'),
        }

    def _apply_gauge_dropout(
        self,
        target_flows: torch.Tensor,
        rng: np.random.RandomState,
    ) -> Tuple[torch.Tensor, int]:
        """
        Mask a subset of supervised basins from the loss while keeping them in the graph.
        This simulates pseudo-ungauged basins during training.
        """
        n_basins = target_flows.shape[0]
        if self.gauge_dropout_rate <= 0.0 or n_basins <= 1:
            return target_flows, n_basins

        keep_mask = rng.rand(n_basins) >= self.gauge_dropout_rate
        if not keep_mask.any():
            keep_mask[rng.randint(0, n_basins)] = True

        masked_targets = target_flows.clone()
        keep_mask_tensor = torch.as_tensor(keep_mask, device=target_flows.device, dtype=torch.bool)
        masked_targets[~keep_mask_tensor] = float('nan')
        return masked_targets, int(keep_mask.sum())

    def _prepare_sequence_batch(
        self,
        data_module: SequenceDataModule,
        basin_nodes: List[int],
        start_date: pd.Timestamp,
    ) -> Optional[Tuple[Dict, torch.Tensor]]:
        """
        Prepare a batch for day-by-day LSTM processing.

        Returns:
            batch_data dict with:
                'all_forcings': [total_days, n_subgraph_nodes, n_vars]  — raw daily forcings
                'x_static': [n_subgraph_nodes, n_static]               — normalized static features
                'upstream_edge_index': [2, n_up]                        — upstream→downstream edges
                'upstream_edge_attr': [n_up, edge_feat_dim]             — edge features
                'downstream_edge_index': [2, n_down]                    — downstream→upstream edges
                'downstream_edge_attr': [n_down, edge_feat_dim]         — edge features
                'seed_indices': list of subgraph indices for gauged basins
                'warmup_days': int
            target_flows: [n_basins, seq_length] — observed specific discharge (mm/day)
        """
        total_days = self.warmup_days + self.seq_length

        # The prediction period starts at start_date; warmup starts before that
        warmup_start_date = start_date - pd.Timedelta(days=self.warmup_days)

        # Get subgraph
        subgraph_nodes = data_module.get_k_hop_neighbors(
            basin_nodes, k=3, max_nodes=self.max_subgraph_nodes
        )
        if len(subgraph_nodes) == 0:
            return None

        orig_to_sub = {orig: sub for sub, orig in enumerate(subgraph_nodes)}

        # Get day-by-day forcings for total_days
        n_vars = len(self.config.data.forcing_vars)
        n_sub = len(subgraph_nodes)

        # Check date range is valid
        first_date_idx = data_module.date_to_idx.get(warmup_start_date)
        if first_date_idx is None or first_date_idx < 0:
            return None

        last_date = warmup_start_date + pd.Timedelta(days=total_days - 1)
        last_date_idx = data_module.date_to_idx.get(last_date)
        if last_date_idx is None:
            return None

        # Bulk-read forcings
        all_forcings = np.zeros((total_days, n_sub, n_vars), dtype=np.float32)

        for sub_idx, orig_idx in enumerate(subgraph_nodes):
            if orig_idx in data_module.node_to_forcing_idx:
                forcing_idx = data_module.node_to_forcing_idx[orig_idx]
                all_forcings[:, sub_idx, :] = data_module.forcing_data[
                    forcing_idx, first_date_idx:first_date_idx + total_days, :
                ].astype(np.float32)

        # Target flows for prediction period only — raw mm/day
        target_flows = np.full((len(basin_nodes), self.seq_length), np.nan, dtype=np.float32)
        for i, node_idx in enumerate(basin_nodes):
            flows = data_module.get_sequence_streamflow(node_idx, start_date, self.seq_length)
            target_flows[i] = flows

        # Static features and directed edges with features
        x_static = data_module.static_features[subgraph_nodes]
        up_ei, up_ea, down_ei, down_ea = data_module.get_subgraph_directed_edges(
            subgraph_nodes, orig_to_sub
        )

        seed_indices = []
        retained_basin_indices = []
        for i, node_idx in enumerate(basin_nodes):
            if node_idx in orig_to_sub:
                seed_indices.append(orig_to_sub[node_idx])
                retained_basin_indices.append(i)

        # Filter targets to match retained seeds
        if len(retained_basin_indices) < len(basin_nodes):
            target_flows = target_flows[retained_basin_indices]

        return {
            'all_forcings': torch.from_numpy(all_forcings),     # [total_days, n_sub, n_vars]
            'x_static': x_static,                                # [n_sub, n_static]
            'upstream_edge_index': up_ei,                        # [2, n_up_edges]
            'upstream_edge_attr': up_ea,                         # [n_up_edges, edge_feat_dim]
            'downstream_edge_index': down_ei,                    # [2, n_down_edges]
            'downstream_edge_attr': down_ea,                     # [n_down_edges, edge_feat_dim]
            'seed_indices': seed_indices,
            'warmup_days': self.warmup_days,
            'n_nodes': n_sub,
        }, torch.from_numpy(target_flows)

    def _forward_sequence(
        self,
        batch_data: Dict,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Day-by-day forward pass with GNN skip during warmup.

        Warmup phase: LSTM-only (no GNN, no output head)
        Prediction phase: LSTM + DirectedGNN + output head (Softplus) → mm/day

        Returns: [n_seed_basins, seq_length] predictions in mm/day (real-space)
        """
        all_forcings = batch_data['all_forcings'].to(device)   # [total_days, n_nodes, n_vars]
        x_static = batch_data['x_static'].to(device)           # [n_nodes, n_static]
        up_ei = batch_data['upstream_edge_index'].to(device)
        up_ea = batch_data['upstream_edge_attr'].to(device)
        down_ei = batch_data['downstream_edge_index'].to(device)
        down_ea = batch_data['downstream_edge_attr'].to(device)
        seed_indices = batch_data['seed_indices']
        warmup_days = batch_data['warmup_days']

        total_days = all_forcings.shape[0]

        # Z-score normalize forcings
        all_forcings = (all_forcings - self.forcing_mean) / self.forcing_std

        # Get raw model (unwrap DDP if needed)
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model

        h_lstm = None  # LSTM carries state
        predictions = []

        # === Warmup phase: LSTM-only (no GNN) ===
        for t in range(warmup_days):
            x_t = torch.cat([all_forcings[t], x_static], dim=-1)
            with autocast():
                _, h_lstm = raw_model.lstm_step(x_t, h_lstm)
            # Detach each step during warmup — no gradients through warmup
            h_lstm = (h_lstm[0].detach(), h_lstm[1].detach())

        # === Prediction phase: LSTM + DirectedGNN + output (Softplus) → mm/day ===
        for t in range(warmup_days, total_days):
            x_t = torch.cat([all_forcings[t], x_static], dim=-1)
            with autocast():
                pred, h_lstm = self.model(x_t, up_ei, up_ea, down_ei, down_ea, h_lstm, x_static=x_static)
            # Model outputs mm/day directly via Softplus (non-negative, no overflow)
            predictions.append(pred[seed_indices].squeeze(-1))

        return torch.stack(predictions, dim=0).t()  # [n_seeds, seq_len] in mm/day

    def train_epoch(
        self,
        data_module: SequenceDataModule,
        train_dates: List[pd.Timestamp],
        train_basin_nodes: List[int],
        epoch: int,
    ) -> Tuple[float, Dict]:
        """Train for one epoch."""
        self.model.train()

        use_mse = epoch < self.warmup_epochs

        gauged_nodes = np.array(train_basin_nodes, dtype=np.int64)
        n_gauged = len(gauged_nodes)
        if n_gauged == 0:
            raise RuntimeError("No training basins available for train_epoch")

        # Prediction period start dates (warmup handled internally)
        max_date = max(train_dates) - pd.Timedelta(days=self.seq_length)
        valid_start_dates = [d for d in train_dates if d <= max_date]

        rng = np.random.RandomState(42 + epoch)
        rng.shuffle(valid_start_dates)

        n_sample = min(self.sequences_per_epoch, len(valid_start_dates))
        valid_start_dates = valid_start_dates[:n_sample]

        n_dates_truncated = (len(valid_start_dates) // self.world_size) * self.world_size
        valid_start_dates = valid_start_dates[:n_dates_truncated]

        my_dates = valid_start_dates[self.rank::self.world_size]

        n_batches_per_date = (n_gauged + self.batch_size - 1) // self.batch_size

        if is_main_process():
            log(f"  Sampling {len(valid_start_dates)} sequences ({len(my_dates)} per GPU)")
            log(f"  Batches per date: {n_batches_per_date}")
            if use_mse:
                log(f"  Mode: MSE warmup (epoch {epoch+1}/{self.warmup_epochs})")
            else:
                log(f"  Mode: Rolling {self.loss_type.upper()}")

        total_loss = 0.0
        n_batches = 0
        all_stats = {'n_basins': 0, 'nse_values': [], 'kge_values': [], 'supervised_basins': 0}

        self.optimizer.zero_grad()
        accumulated_steps = 0
        total_backward_calls = 0

        def get_dummy_loss():
            return sum(p.sum() * 0.0 for p in self.model.parameters() if p.requires_grad)

        pbar = tqdm(my_dates, desc="  Training", disable=not is_main_process())

        for start_date in pbar:
            rng.shuffle(gauged_nodes)

            for batch_start in range(0, len(gauged_nodes), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(gauged_nodes))
                basin_nodes = gauged_nodes[batch_start:batch_end].tolist()

                result = self._prepare_sequence_batch(data_module, basin_nodes, start_date)
                if result is None:
                    dummy_loss = get_dummy_loss()
                    scaled_dummy = dummy_loss / self.gradient_accumulation_steps
                    self.scaler.scale(scaled_dummy).backward()
                    total_backward_calls += 1
                    accumulated_steps += 1
                    if accumulated_steps >= self.gradient_accumulation_steps:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        accumulated_steps = 0
                    continue

                batch_data, target_flows = result

                try:
                    target_flows = target_flows.to(self.device)
                    target_flows, n_supervised = self._apply_gauge_dropout(target_flows, rng)
                    all_stats['supervised_basins'] += n_supervised

                    # v15: predictions in mm/day (Softplus in model)
                    predictions = self._forward_sequence(batch_data, self.device)

                    if use_mse:
                        valid_mask = ~torch.isnan(target_flows)
                        if valid_mask.sum() > 0:
                            loss = self.mse_loss(
                                predictions[valid_mask],
                                target_flows[valid_mask]
                            )
                        else:
                            loss = predictions.sum() * 0
                        stats = {}
                    else:
                        if self.loss_type == "nse":
                            loss, stats = self.rolling_nse_loss(predictions, target_flows)
                        else:
                            loss, stats = self.rolling_kge_loss(predictions, target_flows)

                    if torch.isnan(loss) or torch.isinf(loss) or loss.grad_fn is None:
                        loss = get_dummy_loss()
                    elif stats.get('n_basins', 0) > 0:
                        # Only record stats for valid (non-NaN) losses
                        all_stats['n_basins'] += stats['n_basins']
                        if 'median_nse' in stats:
                            all_stats['nse_values'].append(stats['median_nse'])
                        if 'median_kge' in stats:
                            all_stats['kge_values'].append(stats['median_kge'])

                except Exception as e:
                    if total_backward_calls % 100 == 0:
                        log(f"WARNING: Exception in forward pass on rank {self.rank}: {type(e).__name__}: {e}")
                    loss = get_dummy_loss()

                scaled_loss = loss / self.gradient_accumulation_steps
                self.scaler.scale(scaled_loss).backward()
                total_backward_calls += 1

                total_loss += loss.item()
                n_batches += 1
                accumulated_steps += 1

                if accumulated_steps >= self.gradient_accumulation_steps:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    accumulated_steps = 0

        pbar.close()

        # Handle remaining gradients (DDP sync)
        if self.world_size > 1:
            has_remaining = torch.tensor([1 if accumulated_steps > 0 else 0], device=self.device)
            dist.all_reduce(has_remaining, op=dist.ReduceOp.MAX)
            any_rank_has_remaining = has_remaining.item() > 0
        else:
            any_rank_has_remaining = accumulated_steps > 0

        if any_rank_has_remaining:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        avg_loss = total_loss / max(n_batches, 1)

        if self.world_size > 1:
            loss_tensor = torch.tensor([total_loss, n_batches], device=self.device, dtype=torch.float64)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor[0].item() / max(loss_tensor[1].item(), 1)

        epoch_stats = {
            'n_basins': all_stats['n_basins'],
            'median_nse': np.median(all_stats['nse_values']) if all_stats['nse_values'] else 0.0,
            'median_kge': np.median(all_stats['kge_values']) if all_stats['kge_values'] else 0.0,
            'supervised_basins': all_stats['supervised_basins'],
            'use_mse': use_mse,
        }

        return avg_loss, epoch_stats

    def validate(
        self,
        data_module: SequenceDataModule,
        val_dates: List[pd.Timestamp],
        eval_basin_nodes: List[int],
    ) -> Tuple[float, float, Dict]:
        """
        Validate using per-basin KGE/NSE on specific discharge (mm/day).
        """
        self.model.eval()

        gauged_nodes = np.array(eval_basin_nodes, dtype=np.int64)
        if len(gauged_nodes) == 0:
            return -float('inf'), -float('inf'), {'n_basins': 0}

        max_date = max(val_dates) - pd.Timedelta(days=self.seq_length)
        valid_start_dates = [d for d in val_dates if d <= max_date]

        rng = np.random.RandomState(999)
        rng.shuffle(valid_start_dates)
        valid_start_dates = valid_start_dates[:min(50, len(valid_start_dates))]

        my_dates = valid_start_dates[self.rank::self.world_size]

        basin_predictions = {}

        with torch.no_grad():
            pbar = tqdm(my_dates, desc="  Validating", disable=not is_main_process())

            for start_date in pbar:
                for batch_start in range(0, len(gauged_nodes), self.batch_size):
                    batch_end = min(batch_start + self.batch_size, len(gauged_nodes))
                    basin_nodes = gauged_nodes[batch_start:batch_end].tolist()

                    result = self._prepare_sequence_batch(data_module, basin_nodes, start_date)
                    if result is None:
                        continue

                    batch_data, target_flows = result
                    target_flows = target_flows.to(self.device)

                    # Predictions already in mm/day (Softplus in model)
                    predictions = self._forward_sequence(batch_data, self.device)

                    preds_np = predictions.float().cpu().numpy()
                    obs_np = target_flows.cpu().numpy()

                    for i, node_idx in enumerate(basin_nodes):
                        if node_idx not in basin_predictions:
                            basin_predictions[node_idx] = []

                        for t in range(self.seq_length):
                            if not np.isnan(obs_np[i, t]):
                                basin_predictions[node_idx].append((preds_np[i, t], obs_np[i, t]))

        pbar.close()

        # Compute per-basin NSE and KGE in real space (mm/day)
        basin_nse_values = []
        basin_kge_values = []
        min_samples = 30
        eps = 1e-8
        min_variance_threshold = 0.01

        for basin_id, pairs in basin_predictions.items():
            if len(pairs) < min_samples:
                continue

            preds = np.array([p[0] for p in pairs])
            obs = np.array([p[1] for p in pairs])

            obs_mean = np.mean(obs)
            ss_res = np.sum((obs - preds) ** 2)
            ss_tot = np.sum((obs - obs_mean) ** 2)

            if ss_tot <= min_variance_threshold:
                continue

            nse = 1 - ss_res / ss_tot
            basin_nse_values.append(nse)

            # KGE
            pred_mean = np.mean(preds)
            pred_std = np.std(preds)
            obs_std = np.std(obs)

            if pred_std > eps and obs_std > eps:
                r = np.corrcoef(preds, obs)[0, 1]
                if np.isnan(r):
                    r = 0.0
            else:
                r = 0.0

            alpha = pred_std / (obs_std + eps)
            beta = pred_mean / (obs_mean + eps)
            kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
            basin_kge_values.append(kge)

        # Aggregate across GPUs
        if self.world_size > 1:
            local_nse = torch.tensor(basin_nse_values, device=self.device, dtype=torch.float32)
            local_kge = torch.tensor(basin_kge_values, device=self.device, dtype=torch.float32)

            local_size = torch.tensor([len(basin_nse_values)], device=self.device, dtype=torch.int64)
            sizes = [torch.zeros(1, dtype=torch.int64, device=self.device) for _ in range(self.world_size)]
            dist.all_gather(sizes, local_size)

            max_size = max(s.item() for s in sizes)
            if max_size > 0:
                padded_nse = torch.full((max_size,), float('nan'), device=self.device)
                padded_kge = torch.full((max_size,), float('nan'), device=self.device)

                if len(basin_nse_values) > 0:
                    padded_nse[:len(basin_nse_values)] = local_nse
                    padded_kge[:len(basin_kge_values)] = local_kge

                all_nse = [torch.zeros(max_size, device=self.device) for _ in range(self.world_size)]
                all_kge = [torch.zeros(max_size, device=self.device) for _ in range(self.world_size)]
                dist.all_gather(all_nse, padded_nse)
                dist.all_gather(all_kge, padded_kge)

                combined_nse = torch.cat(all_nse)
                combined_kge = torch.cat(all_kge)
                valid_nse = ~torch.isnan(combined_nse)
                valid_kge = ~torch.isnan(combined_kge)

                basin_nse_values = combined_nse[valid_nse].cpu().numpy().tolist()
                basin_kge_values = combined_kge[valid_kge].cpu().numpy().tolist()

        if len(basin_nse_values) == 0:
            return -float('inf'), -float('inf'), {'n_basins': 0}

        median_nse = np.median(basin_nse_values)
        mean_nse = np.mean(basin_nse_values)
        median_kge = np.median(basin_kge_values)
        mean_kge = np.mean(basin_kge_values)

        stats = {
            'n_basins': len(basin_nse_values),
            'median_nse': median_nse,
            'mean_nse': mean_nse,
            'median_kge': median_kge,
            'mean_kge': mean_kge,
        }

        if is_main_process():
            log(f"    Per-basin stats ({stats['n_basins']} basins, specific discharge mm/day):")
            log(f"    NSE: median={median_nse:.4f}, mean={mean_nse:.4f}")
            log(f"    KGE: median={median_kge:.4f}, mean={mean_kge:.4f}")

        return median_nse, median_kge, stats

    def fit(
        self,
        data_module: SequenceDataModule,
        train_dates: List[pd.Timestamp],
        val_dates: List[pd.Timestamp],
        train_basin_nodes: Optional[List[int]] = None,
        eval_basin_nodes: Optional[List[int]] = None,
        start_epoch: int = 0,
    ):
        """Full training loop."""
        train_basin_nodes = list(train_basin_nodes or data_module.train_gauged_nodes)
        eval_basin_nodes = list(eval_basin_nodes or data_module.eval_gauged_nodes)
        overlap = len(set(train_basin_nodes) & set(eval_basin_nodes))

        log("\n" + "="*60)
        log(f"TRAINING v15 - Larger LSTM (512) + Directed GNN ({self.world_size} GPUs)")
        log("="*60)
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

        checkpoint_dir = Path("checkpoints_v15")
        checkpoint_dir.mkdir(exist_ok=True)

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

            # Validate (real-space metrics)
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
                        'version': 'v15',
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
                    'version': 'v15',
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

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load training state from checkpoint. Returns next epoch index."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        sync_if_distributed()
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location=self.device)

        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Restore forcing normalization stats from checkpoint
        if 'forcing_mean' in checkpoint:
            self.forcing_mean = checkpoint['forcing_mean'].to(self.device)
            self.forcing_std = checkpoint['forcing_std'].to(self.device)

        # Restore edge feature normalization stats
        if 'edge_feat_mean' in checkpoint:
            self.edge_feat_mean = checkpoint['edge_feat_mean']
            self.edge_feat_std = checkpoint['edge_feat_std']

        if 'history' in checkpoint and isinstance(checkpoint['history'], dict):
            self.history = checkpoint['history']

        if 'best_val_metric' not in self.history:
            self.history['best_val_metric'] = float(checkpoint.get('val_kge', checkpoint.get('val_nse', -float('inf'))))
        if 'best_epoch' not in self.history:
            self.history['best_epoch'] = int(checkpoint.get('epoch', 0))

        self.patience_counter = int(checkpoint.get('patience_counter', 0))
        start_epoch = int(checkpoint.get('epoch', -1)) + 1

        sync_if_distributed()
        if is_main_process():
            log(f"Loaded checkpoint: {ckpt_path}")
            log(f"  Version: {checkpoint.get('version', 'unknown')}")
            log(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
            log(f"  Resume from epoch: {start_epoch + 1}")
            log(f"  Best val metric: {self.history.get('best_val_metric', -float('inf')):.4f}")
            if 'data_split' in checkpoint:
                log(f"  Checkpoint split: {checkpoint['data_split']}")

        return start_epoch


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train GNN v15 - Directed GNN + edge features')
    parser.add_argument('--seq-length', type=int, default=90,
                        help='Prediction sequence length in days (default: 90)')
    parser.add_argument('--warmup-days', type=int, default=365,
                        help='LSTM warmup period in days (default: 365)')
    parser.add_argument('--split-mode', type=str, default='basin', choices=['temporal', 'basin'],
                        help='Split strategy: temporal years or basin holdout (default: basin)')
    parser.add_argument('--train-basin-fraction', type=float, default=0.8,
                        help='Fraction of gauged basins used for training when split-mode=basin (default: 0.8)')
    parser.add_argument('--basin-split-seed', type=int, default=42,
                        help='Random seed for basin holdout split (default: 42)')
    parser.add_argument('--loss', type=str, default='kge', choices=['nse', 'kge'],
                        help='Loss type (default: kge)')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='MSE warmup epochs before rolling loss (default: 0)')
    parser.add_argument('--gauge-dropout-rate', type=float, default=0.3,
                        help='Fraction of training gauges masked from the loss each batch (default: 0.3)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override max epochs')
    parser.add_argument('--sequences-per-epoch', type=int, default=200,
                        help='Number of sequence start dates per epoch (default: 200)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate (useful when resuming)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--resume-best', action='store_true',
                        help='Resume from checkpoints_v15/best_model.pt')
    args = parser.parse_args()

    if args.split_mode == 'basin' and not (0.0 < args.train_basin_fraction < 1.0):
        parser.error('--train-basin-fraction must be in (0, 1) when --split-mode basin')
    if not (0.0 <= args.gauge_dropout_rate < 1.0):
        parser.error('--gauge-dropout-rate must be in [0, 1)')

    log("="*60)
    log("GNN STREAMFLOW v15 - LARGER LSTM (512) + DIRECTED GNN")
    log("DirectedRiverGNN (separate up/down GAT), 6-dim edge features, Softplus output")
    log("="*60)
    log(f"Start time: {datetime.now()}")

    rank, world_size, local_rank = setup_distributed()
    log(f"World size: {world_size}")
    log(f"Rank {rank}, Local rank {local_rank}, Device cuda:{local_rank}")

    config = get_config_v15()
    if args.epochs:
        config.train.max_epochs = args.epochs
    config.data.split_mode = args.split_mode
    config.data.train_basin_fraction = args.train_basin_fraction
    config.data.basin_split_seed = args.basin_split_seed

    log(f"\nV15 Config:")
    log(f"  LSTM input: {config.model.lstm_input_dim} (will be overridden by actual data)")
    log(f"  LSTM: hidden={config.model.lstm_hidden_dim}, layers={config.model.lstm_num_layers}")
    log(f"  GNN: hidden={config.model.gnn_hidden_dim}, layers={config.model.gnn_num_layers}, type=DirectedGAT")
    log(f"  Edge features: {config.model.edge_feat_dim}-dim (area_ratio, elev_diff, dist, slope, order_diff, local_area_ratio)")
    log(f"  Learning rate: {config.train.learning_rate}")
    log(f"  Warmup days: {args.warmup_days}")
    log(f"  Sequence length: {args.seq_length}")
    log(f"  Split mode: {args.split_mode}")
    if args.split_mode == 'basin':
        log(f"  Train basin fraction: {args.train_basin_fraction}")
        log(f"  Basin split seed: {args.basin_split_seed}")
    log(f"  Loss type: {args.loss} (real-space)")
    log(f"  MSE warmup epochs: {args.warmup_epochs}")
    log(f"  Gauge dropout rate: {args.gauge_dropout_rate}")
    log(f"  Output: Softplus → mm/day")

    # Load data
    log("\n" + "="*60)
    log("LOADING DATA")
    log("="*60)

    data_module = SequenceDataModule(config, rank, world_size, warmup_days=args.warmup_days)

    if args.split_mode == 'basin':
        split_years = unique_years(
            config.data.train_years,
            config.data.val_years,
            config.data.test_years,
        )
        train_dates = build_date_list(split_years)
        val_dates = build_date_list(split_years)
    else:
        train_dates = build_date_list(config.data.train_years)
        val_dates = build_date_list(config.data.val_years)

    log(f"\nTotal train dates: {len(train_dates)}")
    log(f"Total val dates: {len(val_dates)}")
    log(f"Train basins: {len(data_module.train_gauged_nodes)}")
    log(f"Eval basins: {len(data_module.eval_gauged_nodes)}")

    checkpoint_dir = Path("checkpoints_v15")
    checkpoint_dir.mkdir(exist_ok=True)
    if is_main_process():
        split_metadata_path = checkpoint_dir / f"split_metadata_{args.split_mode}.json"
        with split_metadata_path.open('w') as f:
            json.dump(data_module.get_split_metadata(), f, indent=2)
        log(f"Saved split metadata: {split_metadata_path}")

    # Create trainer
    trainer = RollingLossTrainer(
        config=config,
        data_module=data_module,
        seq_length=args.seq_length,
        warmup_days=args.warmup_days,
        loss_type=args.loss,
        warmup_epochs=args.warmup_epochs,
        sequences_per_epoch=args.sequences_per_epoch,
        gauge_dropout_rate=args.gauge_dropout_rate,
    )

    resume_path = args.resume
    if args.resume_best and resume_path is None:
        resume_path = "checkpoints_v15/best_model.pt"
    elif args.resume_best and resume_path is not None and is_main_process():
        log("Both --resume and --resume-best provided; using --resume path.")

    start_epoch = 0
    if resume_path is not None:
        start_epoch = trainer.load_checkpoint(resume_path)

    # Override learning rate if specified
    if args.lr is not None:
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = args.lr
        # Reset scheduler with new LR
        trainer.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            trainer.optimizer, mode='max', factor=0.5, patience=5, verbose=True,
        )
        log(f"  Overriding learning rate → {args.lr}")

    # Train
    trainer.fit(
        data_module,
        train_dates,
        val_dates,
        train_basin_nodes=data_module.train_gauged_nodes,
        eval_basin_nodes=data_module.eval_gauged_nodes,
        start_epoch=start_epoch,
    )

    log(f"\nEnd time: {datetime.now()}")
    log("Training complete!")


def cleanup_shared_memory():
    """Clean up shared memory on exit."""
    global _shm_handle, _shm_name
    if _shm_handle is not None:
        try:
            _shm_handle.close()
            if is_main_process() and _shm_name:
                _shm_handle.unlink()
        except:
            pass


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
