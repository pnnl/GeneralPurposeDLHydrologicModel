#!/usr/bin/env python3
"""
Plot spatial KGE maps comparing v9 and v10 (or any v8+ checkpoint).

Auto-detects model version from checkpoint metadata and loads the correct
model class and config. Supports side-by-side comparison of two checkpoints.

Usage:
    # Compare v9 and v10 (default)
    python plot_spatial_kge.py

    # Single checkpoint
    python plot_spatial_kge.py --checkpoints checkpoints_v9/best_model.pt

    # Two checkpoints side-by-side
    python plot_spatial_kge.py --checkpoints checkpoints_v9/best_model.pt checkpoints_v10/best_model.pt

    # Faster run with date sampling
    python plot_spatial_kge.py --sample-rate 0.1

    # Validation only (skip test period)
    python plot_spatial_kge.py --val-only
"""

import argparse
import warnings
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
import zarr

from config import Config, get_config_v8, get_config_v9, get_config_v10

warnings.filterwarnings('ignore')


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize common wrapper prefixes from DDP/torch.compile checkpoints."""
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        normalized[new_key] = value
    return normalized


def detect_version(checkpoint: dict, checkpoint_path: str) -> str:
    """Detect model version from checkpoint metadata or path."""
    version = checkpoint.get('version')
    if version:
        return version
    path_str = str(checkpoint_path).lower()
    for v in ['v10', 'v9', 'v8', 'v11']:
        if v in path_str:
            return v
    return 'v9'


def load_model_for_version(version: str):
    """Return (StreamflowGNN_class, default_config_fn) for a given version."""
    if version == 'v8':
        from model_v8 import StreamflowGNN
        return StreamflowGNN, get_config_v8
    elif version in ('v9', 'v10'):
        from model_v9 import StreamflowGNN
        config_fn = get_config_v9 if version == 'v9' else get_config_v10
        return StreamflowGNN, config_fn
    else:
        raise ValueError(f"Unsupported version: {version}")


class InferenceDataModule:
    """Lightweight data module for inference (v8/v9/v10 compatible)."""

    MIN_COVERAGE = 0.25
    MIN_MEAN_FLOW_CMS = 0.15

    def __init__(self, config: Config, use_f16_zarr: bool = True):
        self.config = config
        self.use_f16_zarr = use_f16_zarr

    def setup(self):
        print("Loading data...")
        self._load_graph()
        self._load_forcings()
        self._load_streamflow()
        self._build_adj_list()
        print(f"  Ready: {self.num_nodes} catchments, {len(self.gauged_nodes)} gauged")

    def _load_graph(self):
        print("  Loading graph...")
        gpkg_path = self.config.data.base_dir / self.config.data.catchment_network
        self.gdf = gpd.read_file(gpkg_path, layer='catchments_network')
        self.gdf = self.gdf[self.gdf['COMID'].notna()].reset_index(drop=True)

        self.comid_to_idx = {int(comid): idx for idx, comid in enumerate(self.gdf['COMID'])}
        self.idx_to_comid = {idx: comid for comid, idx in self.comid_to_idx.items()}
        self.idx_to_hruid = {idx: int(hruid) for idx, hruid in enumerate(self.gdf['hruid'])}
        self.num_nodes = len(self.comid_to_idx)

        # Build edges (bidirectional — used for subgraph extraction)
        edges = []
        for idx, row in self.gdf.iterrows():
            if pd.notna(row['NextDownID']) and int(row['NextDownID']) != 0:
                down_idx = self.comid_to_idx.get(int(row['NextDownID']))
                if down_idx is not None:
                    edges.append([idx, down_idx])
                    edges.append([down_idx, idx])
            for up_col in ['up1', 'up2', 'up3', 'up4']:
                if pd.notna(row[up_col]) and int(row[up_col]) != 0:
                    up_idx = self.comid_to_idx.get(int(row[up_col]))
                    if up_idx is not None:
                        edges.append([up_idx, idx])
                        edges.append([idx, up_idx])

        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        self._edge_index_np = self.edge_index.numpy()

        # Load static features (prefer V2)
        v2_path = self.config.data.base_dir / "data" / "static" / "catchment_static_attributes_v2.parquet"
        v1_path = self.config.data.base_dir / "data" / "static" / "catchment_static_attributes.parquet"
        if v2_path.exists():
            static_path = v2_path
            print(f"    Using V2 static attributes")
        else:
            static_path = v1_path
            print(f"    WARNING: V2 not found, falling back to V1")

        attrs_df = pd.read_parquet(static_path).set_index('hruid')

        self.node_areas = {}
        if 'area_m2' in attrs_df.columns:
            for idx, hruid in self.idx_to_hruid.items():
                if hruid in attrs_df.index:
                    area = attrs_df.loc[hruid, 'area_m2']
                    if pd.notna(area) and area > 0:
                        self.node_areas[idx] = float(area)

        feature_cols = [c for c in attrs_df.columns
                       if attrs_df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
        non_constant_cols = []
        for col in feature_cols:
            col_std = attrs_df[col].std()
            if pd.notna(col_std) and col_std > 1e-10:
                non_constant_cols.append(col)
        feature_cols = non_constant_cols

        features = np.full((self.num_nodes, len(feature_cols)), np.nan, dtype=np.float32)
        for idx, hruid in self.idx_to_hruid.items():
            if hruid in attrs_df.index:
                features[idx] = attrs_df.loc[hruid, feature_cols].values

        self.static_features = torch.tensor(features, dtype=torch.float32)
        col_means = torch.nanmean(self.static_features, dim=0)
        for i in range(self.static_features.shape[1]):
            mask = torch.isnan(self.static_features[:, i])
            self.static_features[mask, i] = col_means[i] if not torch.isnan(col_means[i]) else 0.0
        self.static_features = (self.static_features - self.static_features.mean(dim=0)) / (self.static_features.std(dim=0) + 1e-8)

        self.n_static_features = len(feature_cols)
        print(f"    Nodes: {self.num_nodes}, Edges: {self.edge_index.shape[1]}, Static features: {self.n_static_features}")

    def _load_forcings(self):
        print("  Loading forcings...")
        processed_dir = self.config.data.base_dir / self.config.data.forcings_processed_dir

        f16_zarr_path = processed_dir / "forcings_f16.zarr"
        f32_zarr_path = processed_dir / "forcings.zarr"

        if self.use_f16_zarr and f16_zarr_path.exists():
            zarr_path = f16_zarr_path
        else:
            zarr_path = f32_zarr_path

        store = zarr.open(str(zarr_path), mode='r')
        self.forcing_dates = pd.to_datetime(store['dates'][:])
        self.forcing_catchment_ids = store['catchment_ids'][:]

        id_type = store.attrs.get('id_type', 'row_index')

        if id_type == 'hruid':
            self.hruid_to_forcing_idx = {int(cid): idx for idx, cid in enumerate(self.forcing_catchment_ids)}
        else:
            self.catchment_id_to_zarr_idx = {str(cid): idx for idx, cid in enumerate(self.forcing_catchment_ids)}
            orig_gpkg = self.config.data.base_dir / "data" / "delineations" / "conus_MERIT_v1_catch.nov25.gpkg"
            orig_gdf = gpd.read_file(orig_gpkg, columns=['hruid'])
            hruid_to_gpkg_row = {hruid: idx for idx, hruid in enumerate(orig_gdf['hruid'])}
            self.hruid_to_forcing_idx = {}
            for hruid, gpkg_row in hruid_to_gpkg_row.items():
                catchment_id = str(gpkg_row)
                zarr_idx = self.catchment_id_to_zarr_idx.get(catchment_id)
                if zarr_idx is not None:
                    self.hruid_to_forcing_idx[hruid] = zarr_idx

        self.node_to_forcing_idx = {}
        for node_idx, hruid in self.idx_to_hruid.items():
            if hruid in self.hruid_to_forcing_idx:
                self.node_to_forcing_idx[node_idx] = self.hruid_to_forcing_idx[hruid]

        print(f"    Mapped {len(self.node_to_forcing_idx)} nodes to forcings")

        print(f"    Loading forcings into memory...")
        self.forcing_data = store['data'][:]
        self.date_to_forcing_idx = {pd.Timestamp(d): i for i, d in enumerate(self.forcing_dates)}
        print(f"    Shape: {self.forcing_data.shape}, dtype: {self.forcing_data.dtype}")

    def _load_streamflow(self):
        print("  Loading streamflow (quality-filtered, area-normalized to mm/day)...")
        streamflow_dir = self.config.data.base_dir / self.config.data.streamflow_dir

        self.streamflow_cache = {}
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

                    q_mm_day = df * 86400.0 * 1000.0 / area_m2

                    self.streamflow_cache[node_idx] = q_mm_day
                    self.gauged_nodes.append(node_idx)
                except Exception:
                    continue

        self.gauged_nodes = sorted(set(self.gauged_nodes))

        print(f"    Accepted gauged nodes: {len(self.gauged_nodes)} (from {n_loaded} loaded)")
        print(f"    Rejected: {n_low_coverage} low coverage, {n_low_flow} low flow, {n_no_area} no area")

    def _build_adj_list(self):
        self.adj_list = defaultdict(set)
        edge_np = self._edge_index_np
        for i in range(edge_np.shape[1]):
            src, dst = int(edge_np[0, i]), int(edge_np[1, i])
            self.adj_list[src].add(dst)
            self.adj_list[dst].add(src)

        self._node_edges = defaultdict(list)
        for edge_idx in range(edge_np.shape[1]):
            src = int(edge_np[0, edge_idx])
            self._node_edges[src].append(edge_idx)

    def get_forcings(self, node_indices: List[int], date: pd.Timestamp,
                     lookback: int) -> np.ndarray:
        n_vars = self.forcing_data.shape[2]
        result = np.zeros((len(node_indices), lookback, n_vars), dtype=np.float32)

        end_date = pd.Timestamp(date)
        if end_date not in self.date_to_forcing_idx:
            return result

        end_date_idx = self.date_to_forcing_idx[end_date]
        start_date_idx = max(0, end_date_idx - lookback + 1)
        actual_lookback = end_date_idx - start_date_idx + 1

        for i, node_idx in enumerate(node_indices):
            forcing_idx = self.node_to_forcing_idx.get(node_idx)
            if forcing_idx is None:
                continue

            data = np.asarray(self.forcing_data[forcing_idx, start_date_idx:end_date_idx+1, :], dtype=np.float32)
            pad_size = lookback - actual_lookback
            if pad_size > 0:
                result[i, pad_size:, :] = data
            else:
                result[i, :, :] = data

        return result

    def get_subgraph_edges(self, subgraph_nodes: List[int],
                           orig_to_sub: Dict[int, int]) -> torch.Tensor:
        subgraph_set = set(subgraph_nodes)
        sub_edges = []

        for node in subgraph_nodes:
            if node in self._node_edges:
                for edge_idx in self._node_edges[node]:
                    dst = int(self._edge_index_np[1, edge_idx])
                    if dst in subgraph_set:
                        sub_edges.append([orig_to_sub[node], orig_to_sub[dst]])

        if len(sub_edges) == 0:
            sub_edges = [[i, i] for i in range(len(subgraph_nodes))]

        return torch.tensor(sub_edges, dtype=torch.long).t().contiguous()


def compute_per_basin_kge(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute KGE for each basin. Predictions are in mm/day (specific discharge)."""
    columns = ['comid', 'hruid', 'kge', 'nse', 'r', 'alpha', 'beta', 'n_samples', 'mean_obs', 'mean_pred']
    if predictions.empty:
        return pd.DataFrame(columns=columns)

    results = []
    min_samples = 30
    eps = 1e-6

    for comid, group in predictions.groupby('comid'):
        valid = group.dropna(subset=['observed_mm_day', 'predicted_mm_day'])
        if len(valid) < min_samples:
            continue

        obs = valid['observed_mm_day'].values
        pred = valid['predicted_mm_day'].values

        obs_mean = obs.mean()
        pred_mean = pred.mean()
        obs_std = obs.std()
        pred_std = pred.std()

        if obs_std < eps or pred_std < eps:
            continue

        r = np.corrcoef(obs, pred)[0, 1]
        if np.isnan(r):
            continue

        alpha = pred_std / (obs_std + eps)
        beta = pred_mean / (obs_mean + eps)
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

        # Also compute NSE
        ss_res = np.sum((obs - pred) ** 2)
        ss_tot = np.sum((obs - obs_mean) ** 2)
        nse = 1 - ss_res / ss_tot if ss_tot > eps else np.nan

        results.append({
            'comid': comid,
            'hruid': valid['hruid'].iloc[0],
            'kge': kge,
            'nse': nse,
            'r': r,
            'alpha': alpha,
            'beta': beta,
            'n_samples': len(valid),
            'mean_obs': obs_mean,
            'mean_pred': pred_mean,
        })

    if not results:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(results, columns=columns)


def run_predictions(model, data_module, dates: List[pd.Timestamp],
                    device: torch.device, warmup_days: int,
                    forcing_mean: torch.Tensor, forcing_std: torch.Tensor,
                    batch_size: int = 512, max_subgraph_nodes: int = 2048) -> pd.DataFrame:
    """
    Run day-by-day predictions with streaming LSTM state.

    Loops batch-first: for each node batch, warms up once then carries h_lstm
    forward across all prediction dates.
    """
    target_nodes = data_module.gauged_nodes
    dates = sorted(dates)
    results = []

    first_date = dates[0]
    warmup_start = first_date - pd.Timedelta(days=warmup_days)

    all_days = pd.date_range(warmup_start, dates[-1], freq='D')
    pred_date_set = set(dates)

    n_batches = (len(target_nodes) + batch_size - 1) // batch_size
    print(f"  Total days: {len(all_days)} ({warmup_days} warmup + {len(dates)} prediction)")
    print(f"  Node batches: {n_batches}")

    with torch.no_grad():
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(target_nodes))
            seed_nodes = target_nodes[batch_start:batch_end]

            # Build 2-hop subgraph
            current = set(seed_nodes)
            all_nodes = set(seed_nodes)
            for _ in range(2):
                if len(all_nodes) >= max_subgraph_nodes:
                    break
                next_layer = set()
                for node in current:
                    if node in data_module.adj_list:
                        next_layer.update(data_module.adj_list[node])
                all_nodes.update(next_layer)
                current = next_layer

            neighbor_nodes = all_nodes - set(seed_nodes)
            max_neighbors = max_subgraph_nodes - len(seed_nodes)
            subgraph_nodes = list(seed_nodes) + sorted(list(neighbor_nodes))[:max_neighbors]
            orig_to_sub = {orig: sub for sub, orig in enumerate(subgraph_nodes)}

            edge_index = data_module.get_subgraph_edges(subgraph_nodes, orig_to_sub).to(device)
            x_static = data_module.static_features[subgraph_nodes].to(device)

            forcing_data = data_module.get_forcings(subgraph_nodes, dates[-1], len(all_days))

            h_lstm = None
            for t, day in enumerate(tqdm(all_days, desc=f"Batch {batch_idx+1}/{n_batches}",
                                         leave=False)):
                x_forcing = torch.from_numpy(forcing_data[:, t, :]).to(device)
                x_forcing = (x_forcing - forcing_mean) / forcing_std
                x_t = torch.cat([x_forcing, x_static], dim=-1)

                with autocast():
                    pred, h_lstm = model(x_t, edge_index, h_lstm)

                if t % 30 == 0:
                    h_lstm = (h_lstm[0].detach(), h_lstm[1].detach())

                if day in pred_date_set:
                    pred_np = pred.float().cpu().numpy().flatten()
                    for orig_idx in seed_nodes:
                        sub_idx = orig_to_sub[orig_idx]
                        comid = data_module.idx_to_comid[orig_idx]
                        hruid = data_module.idx_to_hruid[orig_idx]
                        pred_flow = float(pred_np[sub_idx])

                        obs_flow = np.nan
                        if orig_idx in data_module.streamflow_cache:
                            flow_series = data_module.streamflow_cache[orig_idx]
                            if day in flow_series.index:
                                obs_flow = float(flow_series[day])

                        results.append({
                            'date': day,
                            'comid': comid,
                            'hruid': hruid,
                            'predicted_mm_day': pred_flow,
                            'observed_mm_day': obs_flow,
                        })

    return pd.DataFrame(results)


def load_checkpoint_and_model(checkpoint_path: str, device: torch.device):
    """Load checkpoint, auto-detect version, return (model, config, forcing_mean, forcing_std, version)."""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    version = detect_version(checkpoint, checkpoint_path)
    print(f"  Detected version: {version}")

    config = checkpoint.get('config')
    ModelClass, default_config_fn = load_model_for_version(version)

    if config is None:
        config = default_config_fn()
        print(f"  No config in checkpoint; using default {default_config_fn.__name__}()")

    forcing_mean = checkpoint.get('forcing_mean')
    forcing_std = checkpoint.get('forcing_std')
    if forcing_mean is None or forcing_std is None:
        raise RuntimeError(f"Checkpoint missing forcing_mean/forcing_std")
    forcing_mean = forcing_mean.to(device)
    forcing_std = forcing_std.to(device)

    print(f"  Target units: {checkpoint.get('target_units', 'specific_discharge_mm_day')}")
    print(f"  Best epoch: {checkpoint.get('epoch', 'N/A')}")

    model = ModelClass(config.model)
    model_state_dict = checkpoint['model_state_dict']
    try:
        model.load_state_dict(model_state_dict)
    except RuntimeError:
        normalized = _normalize_state_dict_keys(model_state_dict)
        model.load_state_dict(normalized)
        print("  Loaded after normalizing state_dict key prefixes")

    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    return model, config, forcing_mean, forcing_std, version


def _plot_metric_on_ax(ax, gdf, kge_df, metric, title, cmap, norm):
    """Plot a single metric on an axis using catchment centroids."""
    if kge_df.empty:
        ax.set_title(f'{title}\nNo data', fontsize=10)
        ax.set_xlim(-130, -60)
        ax.set_ylim(22, 55)
        ax.grid(True, alpha=0.3)
        return

    kge_df = kge_df.copy()
    kge_df['COMID'] = kge_df['comid'].astype(float)
    merged = gdf.merge(kge_df[['COMID', metric]], on='COMID', how='inner')

    merged['centroid'] = merged.geometry.centroid
    merged_points = gpd.GeoDataFrame(merged, geometry='centroid')
    merged_points.plot(column=metric, ax=ax, cmap=cmap, norm=norm,
                       markersize=8, edgecolor='black', linewidth=0.15)

    med = kge_df[metric].median()
    n = len(kge_df)

    if metric == 'kge':
        n_good = (kge_df['kge'] > 0.5).sum()
        pct_good = 100 * n_good / n
        subtitle = f'Median: {med:.3f}, N={n}, KGE>0.5: {pct_good:.0f}%'
    else:
        subtitle = f'Median: {med:.3f}, N={n}'

    ax.set_title(f'{title}\n{subtitle}', fontsize=9)
    ax.set_xlim(-130, -60)
    ax.set_ylim(22, 55)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('')
    ax.set_ylabel('')


def plot_comparison(kge_results: Dict[str, Dict[str, pd.DataFrame]],
                    gdf: gpd.GeoDataFrame, output_path: str):
    """
    Plot spatial KGE maps with subcomponents (r, alpha, beta).

    kge_results: {version_label: {'val': kge_df, 'test': kge_df}}

    Layout: rows = metrics (KGE, r, alpha, beta), cols = model versions/periods
    """
    # Build list of (label, kge_df) panels
    panels = []
    for version_label, period_data in kge_results.items():
        for period_name, kge_df in period_data.items():
            if not kge_df.empty:
                period_full = {'val': 'Val 2015-2019', 'test': 'Test 2020-2024',
                               'mosart': 'Year 2000'}
                label = f'{version_label}\n{period_full.get(period_name, period_name)}'
                panels.append((label, kge_df))

    if not panels:
        print("No KGE results to plot.")
        return

    metrics = ['kge', 'r', 'alpha', 'beta']
    metric_labels = {
        'kge': 'KGE',
        'r': 'Correlation (r)',
        'alpha': 'Variability Ratio (alpha)',
        'beta': 'Bias Ratio (beta)',
    }

    n_cols = len(panels)
    n_rows = len(metrics)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # Different norms per metric
    norms = {
        'kge': mcolors.Normalize(vmin=-1, vmax=1),
        'r': mcolors.Normalize(vmin=-0.5, vmax=1),
        'alpha': mcolors.Normalize(vmin=0, vmax=3),
        'beta': mcolors.Normalize(vmin=0, vmax=3),
    }
    cmaps = {
        'kge': plt.cm.viridis,
        'r': plt.cm.viridis,
        'alpha': plt.cm.plasma,
        'beta': plt.cm.plasma,
    }

    for col, (label, kge_df) in enumerate(panels):
        for row, metric in enumerate(metrics):
            ax = axes[row, col]
            title = f'{label}' if row == 0 else ''
            _plot_metric_on_ax(ax, gdf, kge_df, metric, title, cmaps[metric], norms[metric])

            # Row labels on the left
            if col == 0:
                ax.set_ylabel(metric_labels[metric], fontsize=11, fontweight='bold')

    plt.suptitle('Spatial KGE Comparison: GNN v9 vs MOSART',
                 fontsize=14, fontweight='bold')

    # Use gridspec to reserve space at bottom for colorbars
    plt.tight_layout(rect=[0, 0.15, 1, 0.97])

    # Place 4 colorbars at the bottom with proper spacing
    cbar_height = 0.013
    cbar_gap = 0.03
    cbar_left = 0.12
    cbar_width = 0.76

    for i, metric in enumerate(metrics):
        sm = plt.cm.ScalarMappable(cmap=cmaps[metric], norm=norms[metric])
        sm.set_array([])
        y_pos = 0.12 - i * cbar_gap
        cax = fig.add_axes([cbar_left, y_pos, cbar_width, cbar_height])
        cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.set_label(metric_labels[metric], fontsize=9, labelpad=2)
        cbar.ax.tick_params(labelsize=7)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved plot to: {output_path}")
    plt.close()


def plot_separate_metrics(kge_results: Dict[str, Dict[str, pd.DataFrame]],
                          gdf: gpd.GeoDataFrame, output_path: str):
    """
    Save separate spatial plot for each metric (KGE, r, alpha, beta).
    Each figure has V9 vs MOSART side by side with its own colorbar at bottom.
    """
    # Build panels
    panels = []
    for version_label, period_data in kge_results.items():
        for period_name, kge_df in period_data.items():
            if not kge_df.empty:
                period_full = {'val': 'Val 2015-2019', 'test': 'Test 2020-2024'}
                label = f'{version_label}\n{period_full.get(period_name, period_name)}'
                panels.append((label, kge_df))

    metrics = ['kge', 'r', 'alpha', 'beta']
    metric_labels = {
        'kge': 'KGE',
        'r': 'Correlation (r)',
        'alpha': 'Variability Ratio (alpha)',
        'beta': 'Bias Ratio (beta)',
    }
    norms = {
        'kge': mcolors.Normalize(vmin=-1, vmax=1),
        'r': mcolors.Normalize(vmin=-0.5, vmax=1),
        'alpha': mcolors.Normalize(vmin=0, vmax=3),
        'beta': mcolors.Normalize(vmin=0, vmax=3),
    }
    cmaps = {
        'kge': plt.cm.viridis,
        'r': plt.cm.viridis,
        'alpha': plt.cm.plasma,
        'beta': plt.cm.plasma,
    }

    stem = Path(output_path).stem
    out_dir = Path(output_path).parent

    for metric in metrics:
        n_cols = len(panels)
        fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5))
        if n_cols == 1:
            axes = [axes]

        for col, (label, kge_df) in enumerate(panels):
            _plot_metric_on_ax(axes[col], gdf, kge_df, metric, label,
                               cmaps[metric], norms[metric])

        fig.suptitle(f'{metric_labels[metric]}: GNN v9 vs MOSART',
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])

        # Colorbar at bottom
        sm = plt.cm.ScalarMappable(cmap=cmaps[metric], norm=norms[metric])
        sm.set_array([])
        cax = fig.add_axes([0.15, 0.03, 0.7, 0.02])
        cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.set_label(metric_labels[metric], fontsize=10)

        fpath = out_dir / f'{stem}_{metric}.png'
        plt.savefig(fpath, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {fpath}")
        plt.close()


def plot_cdfs(kge_results: Dict[str, Dict[str, pd.DataFrame]], output_path: str):
    """
    Plot CDF comparison of KGE and its subcomponents (r, alpha, beta).
    One subplot per metric, overlaid lines for each model/period.
    """
    # Build list of (label, kge_df) series
    series = []
    colors = {'GNN V9': '#1f77b4', 'MOSART': '#d62728'}
    linestyles = {'val': '-', 'test': '--', 'mosart': ':'}

    for version_label, period_data in kge_results.items():
        for period_name, kge_df in period_data.items():
            if kge_df.empty:
                continue
            period_full = {'val': 'Val 2015-2019', 'test': 'Test 2020-2024'}
            label = f'{version_label} ({period_full.get(period_name, period_name)})'
            color = colors.get(version_label, '#333333')
            ls = linestyles.get(period_name, '-')
            series.append((label, kge_df, color, ls))

    metrics = ['kge', 'r', 'alpha', 'beta']
    metric_labels = {
        'kge': 'KGE',
        'r': 'Correlation (r)',
        'alpha': 'Variability Ratio (alpha)',
        'beta': 'Bias Ratio (beta)',
    }
    # x-axis ranges per metric
    xlims = {
        'kge': (-2, 1),
        'r': (-1, 1),
        'alpha': (0, 4),
        'beta': (0, 4),
    }
    # Reference lines (ideal values)
    ideal = {'kge': 1, 'r': 1, 'alpha': 1, 'beta': 1}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for label, kge_df, color, ls in series:
            vals = kge_df[metric].dropna().clip(xlims[metric][0], xlims[metric][1])
            sorted_vals = np.sort(vals.values)
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            ax.plot(sorted_vals, cdf, color=color, linestyle=ls, linewidth=2,
                    label=f'{label} (med={np.median(sorted_vals):.2f})')

        ax.axvline(ideal[metric], color='gray', linestyle=':', linewidth=1, alpha=0.7,
                   label=f'Ideal = {ideal[metric]}')
        if metric == 'kge':
            ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.axvline(0.5, color='green', linestyle='--', linewidth=0.8, alpha=0.5,
                       label='KGE = 0.5')

        ax.set_xlabel(metric_labels[metric], fontsize=11)
        ax.set_ylabel('CDF', fontsize=11)
        ax.set_title(metric_labels[metric], fontsize=12, fontweight='bold')
        ax.set_xlim(xlims[metric])
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.suptitle('CDF Comparison: GNN v9 vs MOSART (Validation 2015-2019)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    cdf_path = str(Path(output_path).with_name(
        Path(output_path).stem + '_cdf' + Path(output_path).suffix))
    plt.savefig(cdf_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved CDF plot to: {cdf_path}")
    plt.close()


def print_summary(kge_results: Dict[str, Dict[str, pd.DataFrame]]):
    """Print summary statistics table."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for version_label, period_data in kge_results.items():
        for period_name, kge_df in period_data.items():
            period_full = {'val': 'Validation', 'test': 'Test'}[period_name]
            print(f"\n{version_label} - {period_full}:")
            print(f"  N basins:      {len(kge_df)}")
            if kge_df.empty:
                print("  No valid KGE values.")
                continue
            print(f"  Median KGE:    {kge_df['kge'].median():.4f}")
            print(f"  Mean KGE:      {kge_df['kge'].mean():.4f}")
            print(f"  Median NSE:    {kge_df['nse'].median():.4f}")
            print(f"  Median r:      {kge_df['r'].median():.4f}")
            print(f"  Median alpha:  {kge_df['alpha'].median():.4f}")
            print(f"  Median beta:   {kge_df['beta'].median():.4f}")
            print(f"  KGE > 0:       {(kge_df['kge'] > 0).sum()} ({100*(kge_df['kge'] > 0).mean():.1f}%)")
            print(f"  KGE > 0.5:     {(kge_df['kge'] > 0.5).sum()} ({100*(kge_df['kge'] > 0.5).mean():.1f}%)")
            print(f"  KGE > 0.7:     {(kge_df['kge'] > 0.7).sum()} ({100*(kge_df['kge'] > 0.7).mean():.1f}%)")


def plot_from_parquets():
    """
    Plot V9 vs MOSART comparison from pre-computed parquet files.
    No inference needed — just loads results and plots.
    """
    parser = argparse.ArgumentParser(description='Plot spatial KGE: GNN v9 vs MOSART')
    parser.add_argument('--v9-parquet', type=str, default='spatial_kge_comparison.parquet',
                        help='V9 KGE results parquet')
    parser.add_argument('--mosart-parquet', type=str, default='mosart_kge_results.parquet',
                        help='MOSART KGE results parquet')
    parser.add_argument('--output', type=str, default='spatial_kge_comparison.png',
                        help='Output image file')
    parser.add_argument('--network-gpkg', type=str,
                        default='data/catchment_network/catchment_network.gpkg',
                        help='Catchment network GeoPackage')
    parser.add_argument('--period', type=str, default='validation',
                        choices=['validation', 'test', 'both'],
                        help='Which V9 period to show (default: validation)')
    args = parser.parse_args()

    # Load catchment geometries
    print("Loading catchment geometries...")
    gdf = gpd.read_file(args.network_gpkg, layer='catchments_network')
    gdf = gdf[gdf['COMID'].notna()].reset_index(drop=True)
    print(f"  {len(gdf)} catchments")

    # Load V9 results
    print("Loading V9 results...")
    v9_all = pd.read_parquet(args.v9_parquet)
    v9_all = v9_all[v9_all['version'] == 'V9']

    kge_results = {}

    if args.period in ('validation', 'both'):
        v9_val = v9_all[v9_all['period'] == 'validation'].copy()
        # Clip extreme KGE values for plotting
        for col in ['kge', 'r']:
            v9_val[col] = v9_val[col].clip(-2, 2)
        for col in ['alpha', 'beta']:
            v9_val[col] = v9_val[col].clip(0, 5)
        kge_results['GNN V9'] = {'val': v9_val}
        print(f"  V9 validation: {len(v9_val)} basins, median KGE={v9_val['kge'].median():.3f}")

    if args.period in ('test', 'both'):
        v9_test = v9_all[v9_all['period'] == 'test'].copy()
        for col in ['kge', 'r']:
            v9_test[col] = v9_test[col].clip(-2, 2)
        for col in ['alpha', 'beta']:
            v9_test[col] = v9_test[col].clip(0, 5)
        if 'GNN V9' not in kge_results:
            kge_results['GNN V9'] = {}
        kge_results['GNN V9']['test'] = v9_test
        print(f"  V9 test: {len(v9_test)} basins, median KGE={v9_test['kge'].median():.3f}")

    # Load MOSART results
    print("Loading MOSART results...")
    mosart = pd.read_parquet(args.mosart_parquet)
    # Clip extremes for plotting
    for col in ['kge', 'r']:
        mosart[col] = mosart[col].clip(-2, 2)
    for col in ['alpha', 'beta']:
        mosart[col] = mosart[col].clip(0, 5)
    kge_results['MOSART'] = {'val': mosart}
    print(f"  MOSART: {len(mosart)} basins, median KGE={mosart['kge'].median():.3f}")

    # Print summary
    print_summary_from_results(kge_results)

    # Plot
    print("\nCreating spatial comparison map...")
    plot_comparison(kge_results, gdf, args.output)

    print("\nCreating separate metric plots...")
    plot_separate_metrics(kge_results, gdf, args.output)

    print("\nCreating CDF plots...")
    plot_cdfs(kge_results, args.output)


def print_summary_from_results(kge_results):
    """Print summary statistics for all results."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    period_labels = {'val': 'Validation', 'test': 'Test', 'mosart': 'MOSART (Year 2000)'}

    for version_label, period_data in kge_results.items():
        for period_name, kge_df in period_data.items():
            print(f"\n{version_label} - {period_labels.get(period_name, period_name)}:")
            print(f"  N basins:      {len(kge_df)}")
            if kge_df.empty:
                continue
            print(f"  Median KGE:    {kge_df['kge'].median():.4f}")
            print(f"  Mean KGE:      {kge_df['kge'].mean():.4f}")
            if 'nse' in kge_df.columns:
                print(f"  Median NSE:    {kge_df['nse'].median():.4f}")
            print(f"  Median r:      {kge_df['r'].median():.4f}")
            print(f"  Median alpha:  {kge_df['alpha'].median():.4f}")
            print(f"  Median beta:   {kge_df['beta'].median():.4f}")
            print(f"  KGE > 0:       {(kge_df['kge'] > 0).sum()} ({100*(kge_df['kge'] > 0).mean():.1f}%)")
            print(f"  KGE > 0.5:     {(kge_df['kge'] > 0.5).sum()} ({100*(kge_df['kge'] > 0.5).mean():.1f}%)")
            print(f"  KGE > 0.7:     {(kge_df['kge'] > 0.7).sum()} ({100*(kge_df['kge'] > 0.7).mean():.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Plot spatial KGE maps (v9/v10 comparison)')
    parser.add_argument('--checkpoints', nargs='+',
                        default=['checkpoints_v9/best_model.pt', 'checkpoints_v10/best_model.pt'],
                        help='Checkpoint path(s). Default: v9 and v10 best models.')
    parser.add_argument('--output', type=str, default='spatial_kge_comparison.png',
                        help='Output image file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--sample-rate', type=float, default=1.0,
                        help='Sample rate for dates (0-1) for faster testing')
    parser.add_argument('--warmup-days', type=int, default=365,
                        help='LSTM warmup days (default: 365)')
    parser.add_argument('--val-only', action='store_true',
                        help='Only run validation period (skip test)')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Node batch size for inference (default: 512)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load all checkpoints
    models = []
    for ckpt_path in args.checkpoints:
        if not Path(ckpt_path).exists():
            print(f"WARNING: Checkpoint not found: {ckpt_path}, skipping")
            continue
        model, config, forcing_mean, forcing_std, version = load_checkpoint_and_model(ckpt_path, device)
        models.append({
            'model': model,
            'config': config,
            'forcing_mean': forcing_mean,
            'forcing_std': forcing_std,
            'version': version,
            'path': ckpt_path,
        })

    if not models:
        print("ERROR: No valid checkpoints found.")
        return

    # Data module is shared (same graph, forcings, streamflow)
    # Use config from first model
    data_module = InferenceDataModule(models[0]['config'])
    data_module.setup()

    # Build date lists
    config = models[0]['config']

    def get_dates_for_years(years: List[int], sample_rate: float = 1.0) -> List[pd.Timestamp]:
        all_dates = []
        for year in years:
            start = pd.Timestamp(f"{year}-01-01")
            end = pd.Timestamp(f"{year}-12-31")
            year_dates = [d for d in pd.date_range(start, end, freq='D')
                         if d in data_module.date_to_forcing_idx]
            all_dates.extend(year_dates)
        if sample_rate < 1.0:
            np.random.seed(42)
            n_sample = max(1, int(len(all_dates) * sample_rate))
            sampled = np.random.choice(all_dates, n_sample, replace=False)
            all_dates = sorted(pd.Timestamp(d) for d in sampled)
        return all_dates

    val_dates = get_dates_for_years(config.data.val_years, args.sample_rate)
    test_dates = get_dates_for_years(config.data.test_years, args.sample_rate) if not args.val_only else []

    print(f"\nVal dates: {len(val_dates)}, Test dates: {len(test_dates)}")

    # Run predictions for each model
    kge_results = {}
    all_kge_dfs = []

    for m in models:
        label = m['version'].upper()
        print(f"\n{'='*60}")
        print(f"Running predictions for {label}")
        print(f"{'='*60}")

        period_kge = {}

        # Validation
        print(f"\n  --- {label} Validation ---")
        val_preds = run_predictions(
            m['model'], data_module, val_dates, device, args.warmup_days,
            m['forcing_mean'], m['forcing_std'], batch_size=args.batch_size)
        val_kge = compute_per_basin_kge(val_preds)
        period_kge['val'] = val_kge
        if not val_kge.empty:
            print(f"  {label} Val median KGE: {val_kge['kge'].median():.4f}")
            val_kge_save = val_kge.copy()
            val_kge_save['period'] = 'validation'
            val_kge_save['version'] = label
            all_kge_dfs.append(val_kge_save)

        # Test
        if test_dates:
            print(f"\n  --- {label} Test ---")
            test_preds = run_predictions(
                m['model'], data_module, test_dates, device, args.warmup_days,
                m['forcing_mean'], m['forcing_std'], batch_size=args.batch_size)
            test_kge = compute_per_basin_kge(test_preds)
            period_kge['test'] = test_kge
            if not test_kge.empty:
                print(f"  {label} Test median KGE: {test_kge['kge'].median():.4f}")
                test_kge_save = test_kge.copy()
                test_kge_save['period'] = 'test'
                test_kge_save['version'] = label
                all_kge_dfs.append(test_kge_save)

        kge_results[label] = period_kge

    # Summary
    print_summary(kge_results)

    # Save all KGE values
    if all_kge_dfs:
        all_kge = pd.concat(all_kge_dfs, ignore_index=True)
        kge_parquet = Path(args.output).with_suffix('.parquet')
        all_kge.to_parquet(kge_parquet, index=False)
        print(f"\nKGE values saved to: {kge_parquet}")

    # Plot
    print("\nCreating spatial comparison map...")
    plot_comparison(kge_results, data_module.gdf, args.output)


if __name__ == '__main__':
    plot_from_parquets()
