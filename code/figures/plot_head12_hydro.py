#!/usr/bin/env python3
"""Paper figures for gauged forecasts, ungauged evaluations, and ensemble spread.

Reads the full-period val/test CSVs. Does not subsample random 180-day windows.

Outputs (PNG + PDF) under outputs/figures/eval_head12/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
from matplotlib.colors import (
    BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm,
)

# Wong (2011) vermillion–blue diverging scale. Blue is the high end, so
# greater skill, higher KGE, and higher coverage stay blue. The pair remains
# distinct for the common color-vision deficiencies.
CB_DIV = LinearSegmentedColormap.from_list(
    "cb_div",
    ["#D55E00", "#F4E4D4", "#F7F7F5", "#D3E4F2", "#0072B2"],
    N=256,
)
# Single-hue ramp for scores with no sign change. Darker blue is greater skill.
CB_SKILL = LinearSegmentedColormap.from_list(
    "cb_skill",
    ["#DEEBF7", "#9ECAE1", "#4292C6", "#2171B5", "#08519C"],
    N=256,
)
# Ten steps of 0.1 on a shared 0–1 scale. A smooth ramp leaves 0.6 and 0.8
# in nearly the same dark blue. The last steps drop in lightness so those
# two bins stay apart.
SKILL_STEP_COLORS = [
    "#F7FBFF", "#DEEBF7", "#C6DBEF", "#9ECAE1", "#6BAED6",
    "#4292C6", "#2171B5", "#08519C", "#08306B", "#041C3A",
]
CB_SKILL_STEP = ListedColormap(SKILL_STEP_COLORS, name="cb_skill_step")
SKILL_BOUNDS = np.linspace(0.0, 1.0, 11)
SKILL_NORM = BoundaryNorm(SKILL_BOUNDS, ncolors=CB_SKILL_STEP.N, clip=True)
SKILL_TICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "outputs" / "eval_hydro"
NETWORK = ROOT / "data" / "catchment_network" / "catchment_network.gpkg"
STATIC = ROOT / "data" / "static" / "catchment_static_attributes.parquet"
CENT_CACHE = EVAL / "catchment_centroids.parquet"

KGE_BENCH = 1.0 - np.sqrt(2.0)  # Knoben et al. 2019 mean-flow benchmark
PROJ = ccrs.AlbersEqualArea(
    central_longitude=-96, central_latitude=37.5, standard_parallels=(29.5, 45.5)
)
DATA_CRS = ccrs.PlateCarree()
# Fallback only. Maps use the bounding box of evaluated gauges.
EXTENT = [-126.5, -65.5, 25.5, 54.0]
MAP_EXTENT = None

HEADS = {
    "head1": dict(
        key="head1",
        label="Gauged forecasts",
        short="Gauged forecasts",
        tag="Gauged",
        color="#2C5F8A",
        files={
            "val": "head1_gauged_val_per_basin.csv",
            "test": "head1_gauged_test_per_basin.csv",
        },
        size=9,
    ),
    "head2": dict(
        key="head2",
        label="Ungauged evaluations",
        short="Ungauged evaluations",
        tag="Ungauged",
        color="#B85A32",
        files={
            "val": "head2_pub_val_per_basin.csv",
            "test": "head2_pub_test_per_basin.csv",
        },
        size=18,
    ),
    "head3": dict(
        key="head3",
        label="Ensemble evaluations",
        short="Ensemble evaluations",
        tag="Ensemble",
        color="#2A7A5B",
        files={
            "val": "head3_mdn_val_per_basin.csv",
            "test": "head3_mdn_test_per_basin.csv",
        },
        size=18,
    ),
}
PERIODS = {
    "val": "Validation 2015-2019",
    "test": "Test 2020-2024",
}

import paper_mpl_font
paper_mpl_font.use()
plt.rcParams.update(
    {
        "font.family": "Latin Modern Roman",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.45,
        "axes.edgecolor": "#333333",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

ROW_LABEL = {
    "head1": "Gauged prediction",
    "head2": "Ungauged prediction",
}
YEAR_LABEL = {"val": "2015–2019", "test": "2020–2024"}


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    fig.savefig(
        path.with_suffix(".pdf"), dpi=300, bbox_inches="tight",
        facecolor="white", pad_inches=0.02,
    )
    plt.close(fig)
    print(f"  wrote {path}  and  {path.with_suffix('.pdf')}")


def load_centroids():
    if CENT_CACHE.exists():
        print(f"centroids from cache {CENT_CACHE}")
        return pd.read_parquet(CENT_CACHE)
    print(f"computing centroids from {NETWORK} ...")
    catch = pyogrio.read_dataframe(
        NETWORK, layer="catchments_network", columns=["COMID", "hruid"]
    )
    catch = catch[catch["COMID"].notna()].copy()
    catch["comid"] = catch["COMID"].astype(np.int64)
    catch["hruid"] = pd.to_numeric(catch["hruid"], errors="coerce")
    pts = catch.geometry.representative_point()
    out = pd.DataFrame(
        {
            "comid": catch["comid"].to_numpy(),
            "hruid": catch["hruid"].to_numpy(),
            "lon": pts.x.to_numpy(),
            "lat": pts.y.to_numpy(),
        }
    )
    if STATIC.exists():
        st = pd.read_parquet(STATIC, columns=["hruid", "aridity", "uparea", "frac_snow", "p_mean"])
        st["hruid"] = pd.to_numeric(st["hruid"], errors="coerce")
        out = out.merge(st, on="hruid", how="left")
    CENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CENT_CACHE, index=False)
    print(f"  cached {len(out):,} centroids")
    return out


def load_metrics(eval_dir: Path):
    frames = {}
    for hid, spec in HEADS.items():
        for period, fname in spec["files"].items():
            path = eval_dir / fname
            if not path.exists():
                raise FileNotFoundError(path)
            df = pd.read_csv(path)
            df["comid"] = df["comid"].astype(np.int64)
            df["head"] = hid
            df["period"] = period
            frames[(hid, period)] = df
            print(f"  {hid} {period}: n={len(df)}  median KGE={df.kge.median():.3f}")
    return frames


def attach_xy(df, cents):
    keep = ["comid", "lon", "lat"]
    extra = [c for c in ("aridity", "uparea", "frac_snow") if c in cents.columns]
    return df.merge(cents[keep + extra], on="comid", how="left")


def set_map_extent(frames, cents, pad_lon=1.6, pad_lat=1.0):
    """Crop maps to the evaluated gauges, plus a small pad."""
    global MAP_EXTENT
    lons, lats = [], []
    for df in frames.values():
        m = attach_xy(df, cents)
        lons.append(m["lon"].to_numpy())
        lats.append(m["lat"].to_numpy())
    lon = np.concatenate(lons)
    lat = np.concatenate(lats)
    ok = np.isfinite(lon) & np.isfinite(lat)
    lon, lat = lon[ok], lat[ok]
    MAP_EXTENT = [
        float(lon.min() - pad_lon),
        float(lon.max() + pad_lon),
        float(lat.min() - pad_lat),
        float(lat.max() + pad_lat),
    ]
    print(f"  map extent lon {MAP_EXTENT[0]:.1f}..{MAP_EXTENT[1]:.1f}  "
          f"lat {MAP_EXTENT[2]:.1f}..{MAP_EXTENT[3]:.1f}")
    return MAP_EXTENT


def _basemap(ax):
    ax.set_extent(MAP_EXTENT or EXTENT, crs=DATA_CRS)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#f7f7f5", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f3f1ec", zorder=1)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#f7f7f5", zorder=2)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.18, edgecolor="#d0d0d0", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.35, edgecolor="#6a6a6a", zorder=4)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.35, edgecolor="#666666", zorder=4)
    ax.set_xticks([])
    ax.set_yticks([])
    if "geo" in ax.spines:
        ax.spines["geo"].set_linewidth(0.35)
        ax.spines["geo"].set_edgecolor("#4a4a4a")


def _mark(hid):
    """Point area. Gauged maps are dense, so those markers stay smaller."""
    return 2.4 if hid == "head1" else 8.0


def _scatter(ax, df, values, cmap, norm, size, vmin_clip=None, vmax_clip=None):
    v = values.to_numpy(dtype=float)
    if vmin_clip is not None or vmax_clip is not None:
        v = np.clip(v, vmin_clip if vmin_clip is not None else -np.inf,
                    vmax_clip if vmax_clip is not None else np.inf)
    m = np.isfinite(v) & np.isfinite(df["lon"]) & np.isfinite(df["lat"])
    # Vector markers. rasterized=True embedded them at ~100 dpi in the PDF.
    sc = ax.scatter(
        df.loc[m, "lon"],
        df.loc[m, "lat"],
        c=v[m],
        cmap=cmap,
        norm=norm,
        s=size,
        alpha=0.88,
        linewidths=0,
        edgecolor="none",
        transform=DATA_CRS,
        zorder=5,
    )
    return sc, int(m.sum())


def _map_wh_ratio(extent=None):
    """Width/height of the projected extent. Albers x and y are both metres."""
    from shapely.geometry import LineString

    x0, x1, y0, y1 = extent or MAP_EXTENT or EXTENT
    line = LineString([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])
    minx, miny, maxx, maxy = PROJ.project_geometry(line, DATA_CRS).bounds
    return (maxx - minx) / (maxy - miny)


def _finish_map(ax):
    """Keep the Albers window. Equal aspect stops the axes box from stretching it."""
    ax.set_extent(MAP_EXTENT or EXTENT, crs=DATA_CRS)
    ax.set_aspect("equal", adjustable="box")


def _paper_maps(nrow, ncol, cbar="shared"):
    """Journal-width map grid whose panels match the projected aspect.

    Color scales are separate axes under the maps, so a colorbar cannot
    resize the GeoAxes and stretch the coastline.
    cbar: 'shared' (one bar), 'column' (one bar per column), 'panel'.
    """
    ratio = _map_wh_ratio()
    text_w = 5.5
    side = 0.02
    col_gap = 0.10 if ncol >= 3 else 0.16
    panel_w = (text_w - 2 * side - col_gap * (ncol - 1)) / ncol
    panel_h = panel_w / ratio
    title_in = 0.38
    # Room under the strip for tick labels, and for a colorbar name when set.
    under = 0.24 if cbar == "panel" else 0.48
    row_gap = 0.10
    if cbar == "panel":
        fig_h = 0.04 + nrow * (title_in + panel_h + under) + (nrow - 1) * row_gap
    else:
        fig_h = (
            0.04
            + nrow * (title_in + panel_h)
            + max(nrow - 1, 0) * row_gap
            + under
        )
    fig = plt.figure(figsize=(text_w, fig_h))
    axes = np.empty((nrow, ncol), dtype=object)

    def x_in(j):
        return side + j * (panel_w + col_gap)

    cursor = fig_h - 0.02
    bottoms = []
    for i in range(nrow):
        cursor -= title_in + panel_h
        bottoms.append(cursor)
        if cbar == "panel":
            cursor -= under
        if i < nrow - 1:
            cursor -= row_gap
    for i in range(nrow):
        for j in range(ncol):
            rect = (
                x_in(j) / text_w,
                bottoms[i] / fig_h,
                panel_w / text_w,
                panel_h / fig_h,
            )
            axes[i, j] = fig.add_axes(rect, projection=PROJ)

    bar_h = 0.07
    pad = 0.045
    # Pull per-panel bars in so the end tick labels stay inside the column
    # instead of colliding with the next scale.
    inset = 0.0 if cbar == "shared" else 0.13

    def add_cax(x, y, w):
        return fig.add_axes((x / text_w, y / fig_h, w / text_w, bar_h / fig_h))

    if cbar == "panel":
        caxes = np.empty((nrow, ncol), dtype=object)
        for i in range(nrow):
            for j in range(ncol):
                caxes[i, j] = add_cax(
                    x_in(j) + inset, bottoms[i] - pad - bar_h, panel_w - 2 * inset
                )
    elif cbar == "column":
        yb = bottoms[-1] - pad - bar_h
        caxes = [
            add_cax(x_in(j) + inset, yb, panel_w - 2 * inset) for j in range(ncol)
        ]
    else:
        yb = bottoms[-1] - pad - bar_h
        width = panel_w * ncol + col_gap * (ncol - 1)
        caxes = add_cax(x_in(0), yb, width)
    return fig, axes, caxes


def _cbar(fig, ax, mappable, label, ticks=None, ticklabels=None, shrink=0.78, cax=None):
    if cax is None:
        cb = fig.colorbar(
            mappable, ax=ax, orientation="horizontal",
            fraction=0.046, pad=0.06, aspect=28, shrink=shrink,
        )
    else:
        cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=6.5, width=0.4, length=2.0, pad=1.2)
    cb.outline.set_linewidth(0.35)
    if label:
        cb.set_label(label, fontsize=7, labelpad=3)
    if ticks is not None:
        cb.set_ticks(ticks)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)
    return cb


def _panel_title(ax, line1, line2):
    ax.set_title(f"{line1}\n{line2}", fontsize=7.2, pad=2)


def fig_kge_spatial(frames, cents, outdir, hids=("head1", "head2")):
    """KGE maps for the gauged and ungauged predictions, both periods."""
    hids = list(hids)
    nrow = len(hids)
    fig, axes, cax = _paper_maps(nrow, 2, cbar="shared")
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=KGE_BENCH, vmax=1.0)
    last = None
    for i, hid in enumerate(hids):
        for j, period in enumerate(("val", "test")):
            ax = axes[i, j]
            _basemap(ax)
            df = attach_xy(frames[(hid, period)], cents)
            sc, n = _scatter(ax, df, df["kge"], CB_DIV, norm, _mark(hid), -1, 1)
            _finish_map(ax)
            last = sc
            med = float(np.nanmedian(df["kge"]))
            _panel_title(ax, f"{ROW_LABEL[hid]}, {YEAR_LABEL[period]}",
                         f"median {med:.2f},  n = {n:,}")
    _cbar(
        fig, axes, last, "KGE",
        ticks=[-1.0, KGE_BENCH, 0.0, 0.5, 1.0],
        ticklabels=["−1", "−0.41", "0", "0.5", "1"],
        cax=cax,
    )
    _save(fig, outdir / "head12_kge_spatial.png")


def fig_component_spatial(frames, cents, outdir, period="test"):
    """KGE components for the gauged and ungauged predictions.

    Correlation has no sign change, so it uses the sequential blue ramp.
    The two ratios are centered on 1, where orange is below one and blue
    is above one.
    """
    specs = [
        ("r", r"Correlation $r$", CB_SKILL,
         Normalize(vmin=0.0, vmax=1.0), 0, 1, [0.0, 0.5, 1.0]),
        ("alpha", r"Variability $\alpha$", CB_DIV,
         TwoSlopeNorm(vmin=0.4, vcenter=1.0, vmax=1.6), 0.4, 1.6, [0.4, 1.0, 1.6]),
        ("beta", r"Bias ratio $\beta$", CB_DIV,
         TwoSlopeNorm(vmin=0.4, vcenter=1.0, vmax=1.6), 0.4, 1.6, [0.4, 1.0, 1.6]),
    ]
    hids = ["head1", "head2"]
    fig, axes, caxes = _paper_maps(len(hids), 3, cbar="column")
    last = [None, None, None]
    for i, hid in enumerate(hids):
        df = attach_xy(frames[(hid, period)], cents)
        for j, (col, label, cmap, norm, lo, hi, ticks) in enumerate(specs):
            ax = axes[i, j]
            _basemap(ax)
            sc, n = _scatter(ax, df, df[col], cmap, norm, _mark(hid), lo, hi)
            _finish_map(ax)
            last[j] = sc
            med = float(np.nanmedian(df[col]))
            _panel_title(
                ax, f"{ROW_LABEL[hid]}, {YEAR_LABEL[period]}",
                f"median {med:.2f},  n = {n:,}",
            )
    for j, (_, label, *rest) in enumerate(specs):
        _cbar(fig, axes[:, j], last[j], label, ticks=rest[-1], cax=caxes[j])
    _save(fig, outdir / f"head12_kge_parts_spatial_{period}.png")


def fig_signature_spatial(frames, cents, outdir, period="test"):
    """FHV and seasonal timing maps. Orange is negative; blue is positive."""
    specs = [
        ("fhv", "FHV bias (%)", CB_DIV, TwoSlopeNorm(vmin=-80, vcenter=0, vmax=80), -80, 80, "{:.0f}"),
        ("com_bias_days", "Timing bias (days)", CB_DIV,
         TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30), -30, 30, "{:.1f}"),
    ]
    hids = ["head1", "head2"]
    fig, axes, caxes = _paper_maps(len(hids), 2, cbar="column")
    last = [None, None]
    for i, hid in enumerate(hids):
        df = attach_xy(frames[(hid, period)], cents)
        for j, (col, label, cmap, norm, lo, hi, fmt) in enumerate(specs):
            ax = axes[i, j]
            _basemap(ax)
            sc, n = _scatter(ax, df, df[col], cmap, norm, _mark(hid), lo, hi)
            _finish_map(ax)
            last[j] = sc
            med = float(np.nanmedian(df[col]))
            _panel_title(
                ax, f"{ROW_LABEL[hid]}, {YEAR_LABEL[period]}",
                f"median {fmt.format(med)},  n = {n:,}",
            )
    for j, (_, label, *_) in enumerate(specs):
        _cbar(fig, axes[:, j], last[j], label, cax=caxes[j])
    _save(fig, outdir / f"head12_signature_spatial_{period}.png")


def _cdf_xy(values, clip=None):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if clip is not None:
        v = np.clip(v, clip[0], clip[1])
    if len(v) == 0:
        return np.array([np.nan]), np.array([np.nan])
    x = np.sort(v)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def _legend_handles():
    from matplotlib.lines import Line2D
    h = []
    for hid, spec in HEADS.items():
        h.append(Line2D([0], [0], color=spec["color"], lw=1.8, label=f"{spec['short']}, test"))
        h.append(Line2D([0], [0], color=spec["color"], lw=1.15, ls="--", label=f"{spec['short']}, val"))
    return h


def fig_metric_cdfs(frames, outdir):
    from matplotlib.lines import Line2D
    panels = [
        ("kge", "KGE", (-1.0, 1.0), True),
        ("r", r"Correlation $r$", (0.0, 1.0), False),
        ("alpha", r"Variability $\alpha$", (0.0, 2.0), False),
        ("beta", r"Bias ratio $\beta$", (0.0, 2.0), False),
    ]
    colors = {"head1": "#0072B2", "head2": "#D55E00"}
    names = {"head1": "Gauged", "head2": "Ungauged"}
    years = {"test": "2020–2024", "val": "2015–2019"}
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.6), constrained_layout=True)
    for ax, (col, label, clip, show_bench) in zip(axes.ravel(), panels):
        lines = []
        for hid in ("head1", "head2"):
            for period, ls, lw in (("test", "-", 1.5), ("val", "--", 1.1)):
                x, y = _cdf_xy(frames[(hid, period)][col], clip)
                med = float(np.nanmedian(np.clip(frames[(hid, period)][col], *clip)))
                ax.plot(x, y, color=colors[hid], ls=ls, lw=lw)
                lines.append(f"{names[hid]}, {years[period]}  {med:.2f}")
        if show_bench:
            ax.axvline(KGE_BENCH, color="#777777", ls=":", lw=0.8)
        if col in ("alpha", "beta"):
            ax.axvline(1.0, color="#bbbbbb", ls=":", lw=0.7)
        if col == "kge":
            ax.axvline(0.0, color="#bbbbbb", ls=":", lw=0.7)
        ax.set_xlim(*clip)
        ax.set_ylim(0, 1)
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylabel("Fraction of basins", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, lw=0.4)
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top",
                ha="left", fontsize=6.2, linespacing=1.25,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))
    handles = [
        Line2D([0], [0], color=colors["head1"], lw=1.5, label="Gauged prediction"),
        Line2D([0], [0], color=colors["head2"], lw=1.5, label="Ungauged prediction"),
        Line2D([0], [0], color="#555555", lw=1.5, ls="-", label="2020–2024"),
        Line2D([0], [0], color="#555555", lw=1.1, ls="--", label="2015–2019"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, -0.04))
    _save(fig, outdir / "head12_skill_cdfs.png")


def fig_signature_cdfs(frames, outdir):
    panels = [
        ("fhv", "High-flow bias FHV (%)", (-100, 100)),
        ("q95_bias", "Q95 (high) relative bias", (-1.0, 1.0)),
        ("q50_bias", "Q50 (median) relative bias", (-1.0, 1.0)),
        ("q5_bias", "Q5 (low) relative bias", (-1.5, 1.5)),
        ("monthly_r", "Monthly-mean correlation", (0.0, 1.0)),
        ("com_bias_days", "Center-of-mass bias (days)", (-40, 40)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.4), constrained_layout=True)
    for ax, (col, label, clip) in zip(axes.ravel(), panels):
        for hid, spec in HEADS.items():
            for period, ls, lw, alpha in (("test", "-", 1.8, 1.0), ("val", "--", 1.15, 0.75)):
                x, y = _cdf_xy(frames[(hid, period)][col], clip)
                med = float(np.nanmedian(np.clip(frames[(hid, period)][col], *clip)))
                lab = f"{spec['short']} {period}  (med={med:.2f})"
                ax.plot(x, y, color=spec["color"], ls=ls, lw=lw, alpha=alpha, label=lab)
        if col != "monthly_r":
            ax.axvline(0.0, color="#bbbbbb", ls=":", lw=0.7)
        ax.set_xlim(*clip)
        ax.set_ylim(0, 1)
        ax.set_xlabel(label)
        ax.set_ylabel("Fraction of basins")
        ax.grid(True, alpha=0.25, lw=0.4)
        ax.text(0.03, 0.97, "\n".join(
            f"{HEADS[h]['tag']} {p} {float(np.nanmedian(np.clip(frames[(h, p)][col], *clip))):.2f}"
            for h in HEADS for p in ("test", "val")
        ), transform=ax.transAxes, va="top", ha="left", fontsize=6.4, linespacing=1.25,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5))
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Hydrologic-signature agreement with observations", fontsize=11)
    _save(fig, outdir / "head12_signature_cdfs.png")


def fig_component_scatter(frames, outdir, period="test"):
    hids = list(HEADS)
    fig, axes = plt.subplots(1, len(hids), figsize=(4.6 * len(hids), 4.4),
                             constrained_layout=True, sharex=True, sharey=True)
    for ax, hid in zip(axes, hids):
        spec = HEADS[hid]
        df = frames[(hid, period)]
        m = np.isfinite(df["r"]) & np.isfinite(df["alpha"]) & np.isfinite(df["kge"])
        hb = ax.hexbin(
            df.loc[m, "r"], df.loc[m, "alpha"].clip(0, 2),
            C=df.loc[m, "kge"].clip(-1, 1),
            reduce_C_function=np.median,
            gridsize=35, mincnt=2, cmap="RdYlBu", vmin=-0.2, vmax=0.9,
            extent=(0, 1, 0, 2),
        )
        ax.axhline(1.0, color="#666666", ls=":", lw=0.7)
        ax.axvline(1.0, color="#666666", ls=":", lw=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 2)
        ax.set_xlabel(r"Correlation $r$")
        ax.set_ylabel(r"Variability $\alpha$")
        ax.set_title(f"{spec['label']}\n{PERIODS[period]}   n = {int(m.sum())}", fontsize=9)
        _cbar(fig, ax, hb, "median KGE in bin", shrink=0.78)
    fig.suptitle("Where KGE is lost: timing versus variability", fontsize=11)
    _save(fig, outdir / f"head12_r_alpha_kge_{period}.png")


def fig_kge_vs_attributes(frames, cents, outdir, period="test"):
    if "aridity" not in cents.columns:
        print("  skip attribute scatter (no aridity on centroids)")
        return
    hids = list(HEADS)
    fig, axes = plt.subplots(1, len(hids), figsize=(4.6 * len(hids), 4.3),
                             constrained_layout=True, sharey=True)
    for ax, hid in zip(axes, hids):
        spec = HEADS[hid]
        df = attach_xy(frames[(hid, period)], cents)
        m = np.isfinite(df["kge"]) & np.isfinite(df["aridity"])
        arid = df.loc[m, "aridity"].clip(upper=4.0)
        ax.scatter(
            arid, df.loc[m, "kge"].clip(-1, 1),
            s=spec["size"], c=spec["color"], alpha=0.35, edgecolor="none", rasterized=True,
        )
        x = arid.to_numpy()
        y = df.loc[m, "kge"].clip(-1, 1).to_numpy()
        bins = np.linspace(0.05, 3.5, 12)
        xc, yc = [], []
        for a, b in zip(bins[:-1], bins[1:]):
            sel = (x >= a) & (x < b)
            if sel.sum() >= 8:
                xc.append(0.5 * (a + b))
                yc.append(np.median(y[sel]))
        ax.plot(xc, yc, color="black", lw=1.6)
        ax.axhline(KGE_BENCH, color="#777777", ls=":", lw=0.8)
        ax.set_xlabel("Aridity (PET / P)")
        ax.set_ylabel("KGE")
        ax.set_xlim(0, 4.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"{spec['label']}   n = {int(m.sum())}", fontsize=9)
        ax.grid(True, alpha=0.25, lw=0.4)
    fig.suptitle(f"KGE versus catchment aridity  |  {PERIODS[period]}", fontsize=11)
    _save(fig, outdir / f"head12_kge_vs_aridity_{period}.png")


def fig_single_kge_maps(frames, cents, outdir):
    """Standalone maps at the same size as the existing v24 KGE figure."""
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=KGE_BENCH, vmax=1.0)
    for hid in HEADS:
        spec = HEADS[hid]
        for period in ("val", "test"):
            df = attach_xy(frames[(hid, period)], cents)
            fig, ax = plt.subplots(figsize=(10.4, 5.2), subplot_kw={"projection": PROJ})
            _basemap(ax)
            sc, n = _scatter(ax, df, df["kge"], "RdYlBu", norm, spec["size"], -1, 1)
            med = float(np.nanmedian(df["kge"]))
            n_bad = int((df["kge"] < KGE_BENCH).sum())
            ax.set_title(
                f"{spec['label']}  |  {PERIODS[period]}\n"
                f"median KGE = {med:.3f},  n = {n},  "
                f"{n_bad} basins below mean-flow benchmark",
                fontsize=11, fontweight="bold", pad=8,
            )
            _cbar(
                fig, ax, sc, "KGE",
                ticks=[-1.0, KGE_BENCH, 0.0, 0.5, 1.0],
                ticklabels=["-1.0", f"{KGE_BENCH:.2f}\n(mean-flow\nbenchmark)", "0.0", "0.5", "1.0"],
            )
            fig.tight_layout()
            _save(fig, outdir / f"{hid}_kge_spatial_{period}.png")


def _load_ens_metrics(period):
    path = EVAL / f"head3_ens_metrics_{period}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def fig_ensemble_spatial(frames, cents, outdir, period="test"):
    """Spatial maps of CRPSS, nCRPS, reliability, and extreme coverage."""
    if ("head3", period) not in frames:
        print("  skip ensemble maps (no head3 frame)")
        return
    df = attach_xy(frames[("head3", period)], cents)
    extra = _load_ens_metrics(period)
    if extra is not None:
        keep = [c for c in extra.columns if c == "comid" or c not in df.columns]
        df = df.merge(extra[keep], on="comid", how="left")
    size = HEADS["head3"]["size"]

    # One 0–1 scale on every panel, in steps of 0.1. nCRPS is reversed so
    # darker blue remains the better score (lower nCRPS). Basin-mean CRPSS
    # below zero is clipped to the lightest step. The raw minimum is about
    # -22, and a negative limit would print a bound the color scale does not use.
    skill_specs = [
        ("crpss_cl", "Basin-mean CRPSS", CB_SKILL_STEP, SKILL_NORM, SKILL_TICKS),
        ("crpss_pt", "Ungauged CRPSS", CB_SKILL_STEP, SKILL_NORM, SKILL_TICKS),
        ("ncrps", "Normalized CRPS", CB_SKILL_STEP.reversed(), SKILL_NORM, SKILL_TICKS),
    ]
    rel_specs = [
        ("cov90", "P90 coverage", CB_DIV,
         TwoSlopeNorm(vmin=0.55, vcenter=0.90, vmax=1.0)),
        ("cov95", "P95 coverage", CB_DIV,
         TwoSlopeNorm(vmin=0.65, vcenter=0.95, vmax=1.0)),
        # Five default ticks (0.80–1.00) collide on this narrow bar.
        ("cov99", "P99 coverage", CB_DIV,
         TwoSlopeNorm(vmin=0.80, vcenter=0.99, vmax=1.0),
         [0.8, 0.9, 1.0]),
        ("peak95", "P95 bracketing", CB_DIV,
         TwoSlopeNorm(vmin=0.0, vcenter=0.67, vmax=1.0)),
        ("peak99", "P99 bracketing", CB_DIV,
         TwoSlopeNorm(vmin=0.2, vcenter=0.86, vmax=1.0)),
        ("pit", "Median PIT", CB_DIV,
         TwoSlopeNorm(vmin=0.15, vcenter=0.50, vmax=0.85)),
    ]
    # fall back to columns already on the hydro CSV if merge names differ
    rename = {"crps_norm": "ncrps", "peak_cov95": "peak95",
              "peak_cov99": "peak99", "median_pit": "pit"}
    for old, new in rename.items():
        if new not in df.columns and old in df.columns:
            df[new] = df[old]

    def _panel(path, specs, nrow, ncol):
        fig, axes, caxes = _paper_maps(nrow, ncol, cbar="panel")
        for ax, cax, spec in zip(np.ravel(axes), np.ravel(caxes), specs):
            col, label, cmap, norm = spec[:4]
            ticks = spec[4] if len(spec) > 4 else None
            if col not in df.columns:
                ax.set_visible(False)
                cax.set_visible(False)
                continue
            _basemap(ax)
            sc, n = _scatter(ax, df, df[col], cmap, norm, 8.5)
            _finish_map(ax)
            med = float(np.nanmedian(df[col]))
            _panel_title(ax, label, f"median {med:.2f},  n = {n:,}")
            _cbar(fig, ax, sc, "", ticks=ticks, cax=cax)
        _save(fig, path)

    _panel(outdir / f"ensemble_crpss_spatial_{period}.png", skill_specs, 1, 3)
    _panel(outdir / f"ensemble_reliability_spatial_{period}.png", rel_specs, 2, 3)
    combo = skill_specs + rel_specs[:3]
    _panel(outdir / f"ensemble_spread_spatial_{period}.png", combo, 2, 3)


def fig_ensemble_cdfs(frames, outdir):
    if ("head3", "test") not in frames:
        return
    merged = {}
    for period in ("test", "val"):
        df = frames[("head3", period)].copy()
        extra = _load_ens_metrics(period)
        if extra is not None:
            keep = [c for c in extra.columns if c == "comid" or c not in df.columns]
            df = df.merge(extra[keep], on="comid", how="left")
        if "ncrps" not in df.columns and "crps_norm" in df.columns:
            df["ncrps"] = df["crps_norm"]
        if "peak95" not in df.columns and "peak_cov95" in df.columns:
            df["peak95"] = df["peak_cov95"]
        if "peak99" not in df.columns and "peak_cov99" in df.columns:
            df["peak99"] = df["peak_cov99"]
        merged[period] = df
    panels = [
        ("crpss_cl", "Basin-mean CRPSS", (-0.3, 0.85), 0.0),
        ("crpss_pt", "Ungauged CRPSS", (0.0, 0.5), None),
        ("ncrps", "Normalized CRPS", (0.0, 1.2), None),
        ("cov95", "P95 coverage", (0.5, 1.0), 0.95),
        ("peak95", "P95 bracketing", (0.0, 1.0), None),
        ("peak99", "P99 bracketing", (0.0, 1.0), None),
    ]
    from matplotlib.lines import Line2D
    years = {"test": "2020–2024", "val": "2015–2019"}
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.85), constrained_layout=True)
    for ax, (col, label, clip, target) in zip(axes.ravel(), panels):
        lines = []
        for period, ls, lw in (("test", "-", 1.5), ("val", "--", 1.05)):
            if col not in merged[period].columns:
                continue
            x, y = _cdf_xy(merged[period][col], clip)
            med = float(np.nanmedian(np.clip(merged[period][col], *clip)))
            ax.plot(x, y, color="#0072B2", ls=ls, lw=lw)
            lines.append(f"{years[period]}  {med:.2f}")
        if target is not None:
            ax.axvline(target, color="#777777", ls=":", lw=0.8)
        ax.set_xlim(*clip)
        ax.set_ylim(0, 1)
        ax.set_xlabel(label, fontsize=7.5)
        ax.set_ylabel("Fraction of basins", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.grid(True, alpha=0.25, lw=0.4)
        # CDFs leave the upper left open, so the medians sit there.
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top",
                ha="left", fontsize=6, linespacing=1.2,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.8))
    fig.legend(
        handles=[
            Line2D([0], [0], color="#0072B2", lw=1.5, label="2020–2024"),
            Line2D([0], [0], color="#0072B2", lw=1.05, ls="--", label="2015–2019"),
        ],
        loc="lower center", ncol=2, frameon=False, fontsize=7,
        bbox_to_anchor=(0.5, -0.05),
    )
    _save(fig, outdir / "ensemble_spread_cdfs.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, default=EVAL)
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "outputs" / "figures" / "eval_head12",
    )
    ap.add_argument(
        "--only",
        default="all",
        help="all, or a comma list: kge, signature, ensemble",
    )
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",")} if args.only != "all" else None
    def want(name):
        return only is None or name in only
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Loading per-basin metrics")
    frames = load_metrics(args.eval_dir)
    cents = load_centroids()
    set_map_extent(frames, cents)

    if want("kge"):
        print("\nSpatial KGE (2x2)")
        fig_kge_spatial(frames, cents, args.outdir)
    if want("single"):
        print("Standalone KGE maps")
        fig_single_kge_maps(frames, cents, args.outdir)
    if want("components"):
        print("KGE-component maps (test)")
        fig_component_spatial(frames, cents, args.outdir, "test")
    if want("signature"):
        print("Signature maps (test)")
        fig_signature_spatial(frames, cents, args.outdir, "test")
    if want("cdf"):
        print("Skill CDFs")
        fig_metric_cdfs(frames, args.outdir)
        print("Signature CDFs")
        fig_signature_cdfs(frames, args.outdir)
        print("r-alpha hexbin")
        fig_component_scatter(frames, args.outdir, "test")
        print("KGE vs aridity")
        fig_kge_vs_attributes(frames, cents, args.outdir, "test")
        print("Ensemble CDFs")
        fig_ensemble_cdfs(frames, args.outdir)
    if want("ensemble"):
        print("Ensemble spatial maps")
        fig_ensemble_spatial(frames, cents, args.outdir, "test")
    print("done", args.outdir)


if __name__ == "__main__":
    main()
