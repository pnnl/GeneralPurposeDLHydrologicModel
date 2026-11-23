#!/usr/bin/env python3
"""
Plot spatial uncertainty quantification maps for the MDN model across the 965 held-out basins.

Generates a three-panel map over North America:
  1. High-flow coverage (peak_cov99): fraction of top-2% peak-flow days bracketed by P99
  2. Calibration        (cov95):      empirical P95 coverage (nominal target = 0.95)
  3. Sharpness          (crps_norm):  CRPS normalized by mean observed flow

Maps evaluation node indices to catchment COMIDs via split_metadata.json and joins
catchment centroids from catchment_network.gpkg.

Outputs:
  plots/d2_spatial_uq.png

Usage:
  python code/figures/plot_spatial_d2_uq.py
  python code/figures/plot_spatial_d2_uq.py --csv outputs/eval_d2_finetune_heldout_test.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature

PLOTS = Path('plots'); PLOTS.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.linewidth': 0.6, 'axes.edgecolor': '#333',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

PROJ = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=37.5,
                            standard_parallels=(29.5, 45.5))
DATA_PROJ = ccrs.PlateCarree()


def base_map(ax):
    ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='#999', zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='#333', zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#666', zorder=2)
    ax.set_extent([-125, -66, 24, 50], crs=DATA_PROJ)


def draw_panel(ax, mm, col, vmin, vmax, cmap, panel_title, cb_label,
               target=None, target_label=None, lower_is_better=False):
    base_map(ax)
    c = mm[col].clip(vmin, vmax)
    sc = ax.scatter(mm['lon'], mm['lat'], c=c,
                    cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax),
                    s=14, alpha=0.92, edgecolor='none',
                    transform=DATA_PROJ, zorder=5)
    ticks = [vmin, vmax] if target is None else [vmin, target, vmax]
    cb = plt.colorbar(sc, ax=ax, shrink=0.78, pad=0.02, aspect=22, ticks=ticks)
    if target is not None:
        cb.ax.axhline(target, color='red', linewidth=1.6)
    cb.set_label(cb_label, fontsize=9)

    med = float(mm[col].median())
    extra = ''
    if target is not None:
        n_under = int((mm[col] < target).sum())
        n_over = int((mm[col] > target).sum())
        extra = f',  target = {target:.2f}  ({n_under} under, {n_over} over)'
    elif lower_is_better:
        extra = '  (lower = sharper)'
    ax.set_title(f'{panel_title}\nmedian = {med:.3f},  n = {len(mm)}{extra}',
                 fontsize=11, fontweight='bold', pad=6)


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=None,
                    help='per-basin D2 eval CSV (default: frozen, else unfrozen)')
    ap.add_argument('--meta', default=str(
        root / 'weights' / 'checkpoints_v24' / 'finetune' / 'split_metadata.json'))
    ap.add_argument('--gpkg', default=str(
        root / 'data' / 'catchment_network' / 'catchment_network.gpkg'))
    args = ap.parse_args()

    csv = args.csv
    if csv is None:
        for cand in (root / 'outputs' / 'eval_d2_finetune_frozen_heldout_test.csv',
                     root / 'outputs' / 'eval_d2_finetune_heldout_test.csv'):
            if cand.exists():
                csv = cand
                break
    if csv is None or not Path(csv).exists():
        raise SystemExit("No eval CSV found (frozen/unfrozen). Pass --csv.")
    tag = Path(csv).stem
    print(f'Loading D2 eval: {csv}')
    df = pd.read_csv(csv).drop_duplicates(subset='node', keep='first')

    # node_idx -> comid via the saved split (eval + train, same-order pairs)
    meta = json.load(open(args.meta))
    node_to_comid = {int(n): int(c) for n, c in
                     zip(meta['eval_node_indices'], meta['eval_comids'])}
    node_to_comid.update({int(n): int(c) for n, c in
                          zip(meta['train_node_indices'], meta['train_comids'])})
    df['comid'] = df['node'].map(node_to_comid)
    df = df.dropna(subset=['comid'])
    df['comid'] = df['comid'].astype(int)
    print(f'  basins with comid: {len(df)}')

    print('Loading catchment centroids...')
    gdf = gpd.read_file(args.gpkg, layer='catchments_network')
    gdf = gdf[gdf['COMID'].notna()].copy()
    gdf['COMID'] = gdf['COMID'].astype(float).astype(int)
    # centroids in a projected CRS to silence the geographic-centroid warning,
    # then back to lon/lat for plotting.
    cents = gdf.geometry.to_crs(5070).centroid.to_crs(4326)
    cdf = pd.DataFrame({'comid': gdf['COMID'].values,
                        'lon': cents.x.values, 'lat': cents.y.values})

    have_crps = 'crps_norm' in df.columns
    join_cols = ['lon', 'lat', 'peak_cov99', 'cov95'] + (['crps_norm'] if have_crps else [])
    mm = df.merge(cdf, on='comid', how='left').dropna(subset=join_cols)
    print(f'  basins with centroid + metrics: {len(mm)}')

    ncol = 3 if have_crps else 2
    fig, axes = plt.subplots(1, ncol, figsize=(10 * ncol, 6),
                             subplot_kw={'projection': PROJ}, facecolor='white')
    axes = np.atleast_1d(axes)

    # Panel 1 — extreme bracketing (the flagship application of the UQ).
    # Bright = tail catches the floods; dark = tail misses them.
    draw_panel(
        axes[0], mm, col='peak_cov99', vmin=0.5, vmax=1.0, cmap='viridis',
        panel_title='P99 extreme bracketing  (top-2% flood days)',
        cb_label='fraction of peak days bracketed by P99',
    )

    # Panel 2 — P95 calibration. Diverging around the 0.95 target.
    draw_panel(
        axes[1], mm, col='cov95', vmin=0.85, vmax=1.0, cmap='cividis',
        panel_title='P95 coverage  (calibration)',
        cb_label='empirical coverage', target=0.95,
    )

    # Panel 3 — normalized CRPS (sharpness/skill). Reversed map: dark = better.
    if have_crps:
        vmax_crps = float(np.nanquantile(mm['crps_norm'], 0.95))
        draw_panel(
            axes[2], mm, col='crps_norm', vmin=0.0, vmax=vmax_crps,
            cmap='magma_r',
            panel_title='Normalized CRPS  (sharpness)',
            cb_label='CRPS / mean obs flow', lower_is_better=True,
        )

    fig.suptitle(
        f'D2 MDN — spatial uncertainty quantification on PUB held-out basins ({len(mm)})\n{tag}',
        fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    out = PLOTS / 'd2_spatial_uq.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'\nwrote {out}')
    print(f'  P99 peak-cov median = {mm.peak_cov99.median():.3f}  '
          f'P95 cov median = {mm.cov95.median():.3f}'
          + (f'  nCRPS median = {mm.crps_norm.median():.3f}' if have_crps else ''))


if __name__ == '__main__':
    main()
