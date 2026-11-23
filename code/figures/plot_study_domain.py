#!/usr/bin/env python3
"""Figure 1: MERIT river network + gauges used in this study.

All quality-filtered gauges (training + held-out) are drawn as a single
set. There is no v20 / v24 split.

Usage, from the bundle root:
    python code/figures/plot_study_domain.py
    python code/figures/plot_study_domain.py --dpi 200
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyogrio
from matplotlib.lines import Line2D
from shapely.geometry import LineString

# Wong (2011) blue and vermillion: distinguishable under common color-vision deficiencies.
GAUGE_COLOR = "#D55E00"
DATA_CRS = ccrs.PlateCarree()

# Keep cartopy on the already-cached Natural Earth tiles.
os.environ.setdefault("CARTOPY_DATA_DIR", str(Path.home() / ".local/share/cartopy"))

ROOT = Path(__file__).resolve().parents[2]
NETWORK = ROOT / "data" / "catchment_network" / "catchment_network.gpkg"
SPLIT = ROOT / "weights" / "checkpoints_v24" / "finetune" / "split_metadata.json"
OUT_DEFAULT = ROOT / "outputs" / "figures" / "study_domain_network_gauges.png"

# Contributing network is about 25–53°N. The old 14–60°N frame added empty
# Mexico and northern Canada; southern Canada and northern Mexico stay in.
EXTENT = [-126.4, -65.5, 23.4, 54.2]
PROJ = ccrs.AlbersEqualArea(
    central_longitude=-96, central_latitude=37.5, standard_parallels=(29.5, 45.5)
)


def _basemap(ax):
    ax.set_extent(EXTENT, crs=DATA_CRS)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#f4f7f8", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f7f4ef", edgecolor="none", zorder=1)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#f4f7f8", edgecolor="none", zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.35, edgecolor="#8a8a8a", zorder=4)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.4, edgecolor="#666666", zorder=4)
    try:
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.18, edgecolor="#d0d0d0", zorder=3)
    except Exception:
        pass
    ax.set_xticks([])
    ax.set_yticks([])
    if "geo" in ax.spines:
        ax.spines["geo"].set_linewidth(0.4)
        ax.spines["geo"].set_edgecolor("#4a4a4a")


def load_catchments():
    print("Loading catchments (COMID, order, geometry)...")
    catch = pyogrio.read_dataframe(
        NETWORK, layer="catchments_network", columns=["COMID", "order"]
    )
    catch = catch[catch["COMID"].notna()].copy()
    catch["COMID"] = catch["COMID"].astype(np.int64)
    print(f"  {len(catch):,} catchments with COMID")
    print("  Computing centroids...")
    pts = catch.geometry.representative_point()
    catch["lon"] = pts.x
    catch["lat"] = pts.y
    return catch


def load_edges(catch):
    print("Loading river-network edges...")
    edges = pyogrio.read_dataframe(NETWORK, layer="network_edges")
    print(f"  {len(edges):,} edges")
    order_map = catch.set_index("COMID")["order"]
    edges["order"] = edges["from_catchment"].map(order_map).fillna(1.0)
    return edges


def load_gauges(catch):
    meta = json.loads(SPLIT.read_text())
    comids = np.array(meta["train_comids"] + meta["eval_comids"], dtype=np.int64)
    comids = np.unique(comids)
    print(f"  study gauges from split metadata: {len(comids):,}")
    g = catch[catch["COMID"].isin(comids)][["COMID", "lon", "lat", "order"]].copy()
    print(f"  matched to network centroids: {len(g):,}")
    return g


def plot_network(ax, edges):
    """Draw MERIT links; thicker / darker lines for higher stream order."""
    styles = [
        (edges["order"] <= 2, dict(color="#c5d0d8", linewidth=0.12, alpha=0.45)),
        ((edges["order"] > 2) & (edges["order"] <= 4), dict(color="#7f9aaa", linewidth=0.28, alpha=0.70)),
        ((edges["order"] > 4) & (edges["order"] <= 6), dict(color="#3d5a6c", linewidth=0.55, alpha=0.85)),
        (edges["order"] > 6, dict(color="#1b3340", linewidth=0.95, alpha=0.95)),
    ]
    for mask, kw in styles:
        sub = edges.loc[mask]
        if len(sub) == 0:
            continue
        print(f"  plotting {len(sub):,} reaches (order rule {kw['linewidth']})")
        # aspect=None keeps the Albers equal-area scaling. The layer is
        # geographic, and geopandas would otherwise apply a plate-carrée stretch.
        sub.plot(ax=ax, transform=DATA_CRS, zorder=5, aspect=None, autolim=False, **kw)


def _projected_ratio(extent):
    """Width/height of the Albers window. Both axes are metres."""
    x0, x1, y0, y1 = extent
    line = LineString([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])
    minx, miny, maxx, maxy = PROJ.project_geometry(line, DATA_CRS).bounds
    return (maxx - minx) / (maxy - miny)


def _lock_extent(ax):
    ax.set_extent(EXTENT, crs=DATA_CRS)
    ax.set_aspect("equal", adjustable="box")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    catch = load_catchments()
    gauges = load_gauges(catch)
    # drop heavy polygons before loading edges
    catch_pts = catch[["COMID", "order", "lon", "lat"]].copy()
    del catch
    edges = load_edges(gpd.GeoDataFrame(catch_pts, geometry=gpd.points_from_xy(catch_pts.lon, catch_pts.lat)))

    import paper_mpl_font
    paper_mpl_font.use()
    plt.rcParams.update({
        "font.family": "Latin Modern Roman",
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    # Each panel matches the projected domain, inside the 5.5 in text block.
    ratio = _projected_ratio(EXTENT)
    text_w = 5.5
    side, col_gap = 0.02, 0.14
    panel_w = (text_w - 2 * side - col_gap) / 2
    panel_h = panel_w / ratio
    title_in, legend_in, top, bot = 0.26, 0.30, 0.03, 0.02
    fig_h = bot + legend_in + panel_h + title_in + top
    print(f"  projected width/height {ratio:.3f}; figure {text_w:.2f} x {fig_h:.2f} in")
    fig = plt.figure(figsize=(text_w, fig_h))
    y_panel = bot + legend_in
    axes = []
    for j in range(2):
        x = side + j * (panel_w + col_gap)
        axes.append(fig.add_axes(
            (x / text_w, y_panel / fig_h, panel_w / text_w, panel_h / fig_h),
            projection=PROJ,
        ))

    print("Panel (a): river network")
    _basemap(axes[0])
    plot_network(axes[0], edges)
    _lock_extent(axes[0])
    axes[0].set_title("(a)  River network", fontsize=8, pad=3)

    print("Panel (b): study gauges")
    _basemap(axes[1])
    hi = edges[edges["order"] >= 4]
    if len(hi):
        hi.plot(
            ax=axes[1], transform=DATA_CRS, color="#c5d0d6",
            linewidth=0.22, alpha=0.7, zorder=5, aspect=None, autolim=False,
        )
    axes[1].scatter(
        gauges["lon"], gauges["lat"],
        s=2.4, c=GAUGE_COLOR, alpha=0.72, linewidths=0, zorder=6,
        transform=DATA_CRS,
    )
    _lock_extent(axes[1])
    axes[1].set_title("(b)  Streamflow gauges", fontsize=8, pad=3)
    handle = Line2D(
        [0], [0], marker="o", linestyle="none",
        markerfacecolor=GAUGE_COLOR, markeredgecolor="none", markersize=4.0,
        label=f"Gauges (n = {len(gauges):,})",
    )
    fig.legend(
        handles=[handle], loc="lower center",
        bbox_to_anchor=(0.5, bot / fig_h),
        frameon=False, fontsize=6.5, ncol=1,
        handletextpad=0.35, borderaxespad=0.0,
    )

    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    pdf = args.out.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    print(f"Wrote {args.out}")
    print(f"Wrote {pdf} ({pdf.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
