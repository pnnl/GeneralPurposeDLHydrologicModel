#!/usr/bin/env python3
"""Hydroclimate- and regulation-stratified predictive performance on held-out basins.

Joins per-basin evaluation metrics with static catchment attributes to summarize
medians, interquartile ranges, and basin counts across aridity, snow-fraction,
drainage-area, and regulation groups.

Inputs (relative to the repository root):
    outputs/eval_hydro/head3_mdn_{test,val}_per_basin.csv
    outputs/eval_hydro/head3_ens_metrics_{test,val}.csv
    data/static/catchment_static_attributes.parquet
    data/gagesii/conterm_bas_classif.txt (optional, for regulation stratification)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_mpl_font
paper_mpl_font.use()

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# Metric columns to display and the reference line drawn for each.
METRICS = [
    ("fhv", "FHV bias (%)", 0.0),
    ("crpss_pt", "CRPSS vs ungauged prediction", 0.0),
    ("peak_cov99", "P99 upper-quantile coverage", 0.99),
]
# Display clipping (stats are always computed on raw values).
YCLIP = {"kge": (-1.0, 1.0), "fhv": (-120.0, 120.0),
         "crpss_pt": (-1.0, 1.0), "peak_cov99": (0.0, 1.0)}

PERIOD_LABEL = {"test": "2020-2024", "val": "2015-2019"}

# Muted, colorblind-aware fills. Each class keeps one color in every panel.
GROUP_COLORS = {
    "Humid (<1)": "#1F6A8A",
    "Transitional (1-2)": "#C4A15A",
    "Arid (>2)": "#C4623D",
    "Rain (<0.1)": "#3C7EA6",
    "Mixed (0.1-0.4)": "#7A6894",
    "Snow (>=0.4)": "#8EADC0",
    # Short tick labels used by the text-width SI figure.
    "Humid": "#1F6A8A",
    "Transitional": "#C4A15A",
    "Arid": "#C4623D",
    "Rain": "#3C7EA6",
    "Mixed": "#7A6894",
    "Snow": "#8EADC0",
    "<100": "#A9C6D4",
    "100-1,000": "#6FA3B8",
    "1,000-10,000": "#2F6D8A",
    ">10,000": "#16384A",
    "100–1k": "#6FA3B8",
    "1–10k": "#2F6D8A",
    ">10k": "#16384A",
    "Natural": "#2E6B4F",
    "Managed": "#A15A49",
}


def _edge(color):
    rgb = matplotlib.colors.to_rgb(color)
    return tuple(max(0.0, c * 0.62) for c in rgb)


def load_period(period: str) -> pd.DataFrame:
    """Per-basin metrics for one period joined to static attributes on comid==hruid."""
    mdn = pd.read_csv(ROOT / "outputs" / "eval_hydro" / f"head3_mdn_{period}_per_basin.csv")
    ens = pd.read_csv(ROOT / "outputs" / "eval_hydro" / f"head3_ens_metrics_{period}.csv")
    df = mdn.merge(ens[["comid", "crpss_pt", "crpss_cl"]], on="comid", how="left")
    stat = pd.read_parquet(ROOT / "data" / "static" / "catchment_static_attributes.parquet",
                           columns=["hruid", "aridity", "frac_snow", "uparea"])
    stat = stat.rename(columns={"hruid": "comid"})
    n_before = len(df)
    df = df.merge(stat, on="comid", how="left")
    n_attr = df["aridity"].notna().sum()
    print(f"[{period}] basins={n_before}  with static attrs={n_attr}  "
          f"dropped(no attr)={n_before - n_attr}")
    return df


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Aridity (Hamon PET/P). <1 energy-limited (humid); >2 water-limited (arid).
    df["aridity_grp"] = pd.cut(
        df["aridity"], bins=[-np.inf, 1.0, 2.0, np.inf],
        labels=["Humid (<1)", "Transitional (1-2)", "Arid (>2)"])
    # Snow fraction (fraction of wet days with T<0).
    df["snow_grp"] = pd.cut(
        df["frac_snow"], bins=[-np.inf, 0.1, 0.4, np.inf],
        labels=["Rain (<0.1)", "Mixed (0.1-0.4)", "Snow (>=0.4)"])
    # Contributing drainage area (uparea, km^2).
    df["area_grp"] = pd.cut(
        df["uparea"], bins=[-np.inf, 100, 1000, 10000, np.inf],
        labels=["<100", "100-1,000", "1,000-10,000", ">10,000"])
    return df


def summarise(df, grp_col, stratifier, period, records):
    for grp, g in df.groupby(grp_col, observed=True):
        for col, _lbl, _ref in METRICS:
            v = pd.to_numeric(g[col], errors="coerce").dropna()
            if len(v) == 0:
                continue
            records.append(dict(
                stratifier=stratifier, period=PERIOD_LABEL[period], group=str(grp),
                metric=col, n=int(len(v)),
                median=float(np.median(v)),
                q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75))))


def add_management(df: pd.DataFrame) -> pd.DataFrame:
    """GAGES-II reference status for USGS spatial holdouts. Other gauges stay unclassified."""
    df = df.copy()
    idx = pd.read_csv(ROOT / "data" / "streamflow" / "streamflow_index.csv")
    usgs = idx[idx["source"] == "USGS"].copy()
    usgs["gage_id"] = usgs["gage_id"].astype(str).str.strip().str.zfill(8)
    usgs = usgs.drop_duplicates("comid")[["comid", "gage_id"]]
    df = df.merge(usgs, on="comid", how="left")
    gages = ROOT / "data" / "gagesii" / "conterm_bas_classif.txt"
    classif = pd.read_csv(gages, encoding="latin-1")
    classif["gage_id"] = classif["STAID"].astype(str).str.zfill(8)
    df = df.merge(classif[["gage_id", "CLASS"]], on="gage_id", how="left")
    label = df["CLASS"].map({"Ref": "Natural", "Non-ref": "Managed"})
    df["mgmt_grp"] = pd.Categorical(label, categories=["Natural", "Managed"])
    return df


def _bounds(df):
    """1st and 99th percentile of each metric over the plotted basins."""
    bounds = {}
    for col, _lbl, ref in METRICS:
        v = pd.to_numeric(df[col], errors="coerce").dropna()
        lo, hi = np.percentile(v, [1, 99])
        if ref is not None:
            lo, hi = min(lo, float(ref)), max(hi, float(ref))
        bounds[col] = (float(lo), float(hi))
    return bounds


def _trimmed(values, bounds):
    """Values inside the shared percentile window, used only for the violin shape."""
    v = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    lo, hi = bounds
    kept = v[(v >= lo) & (v <= hi)]
    return kept if len(kept) >= 2 else v


def draw_row(axes_row, df, grp_col, row_label, kind="box", bounds=None,
             title_size=9, tick_size=7, ylab_size=9, ylab_weight="bold",
             tick_rotation=0):
    """One stratifier -> a row of metric violins or boxplots across its groups."""
    groups = [g for g in df[grp_col].cat.categories
              if (df[grp_col] == g).any()]
    for ax, (col, lbl, ref) in zip(axes_row, METRICS):
        shown, ns, iqr = [], [], []
        window = None if bounds is None else bounds[col]
        for grp in groups:
            v = pd.to_numeric(df.loc[df[grp_col] == grp, col], errors="coerce").dropna()
            shown.append(_trimmed(v, window) if window else v.to_numpy())
            ns.append(len(v))
            iqr.append(np.percentile(v, [25, 50, 75]) if len(v) else (np.nan, np.nan, np.nan))
        positions = list(range(1, len(groups) + 1))
        colors = [GROUP_COLORS.get(str(g), "#6FA3B8") for g in groups]
        if kind == "violin":
            parts = ax.violinplot(
                shown, positions=positions, widths=0.72,
                showmeans=False, showmedians=False, showextrema=False)
            for body, color in zip(parts["bodies"], colors):
                body.set_facecolor(color)
                body.set_edgecolor(_edge(color))
                body.set_alpha(0.92)
            lo, hi = window
            for pos, (q1, med, q3) in zip(positions, iqr):
                ax.plot([pos, pos], [max(q1, lo), min(q3, hi)], color="black",
                        lw=4.0, solid_capstyle="butt", zorder=3)
                ax.scatter([pos], [med], s=16, color="white",
                           edgecolor="black", linewidth=0.6, zorder=4)
            pad = 0.04 * (hi - lo) if hi > lo else 0.05
            ax.set_ylim(lo - pad, hi + pad)
        else:
            lo, hi = YCLIP[col]
            clipped = [np.clip(a, lo, hi) for a in shown]
            bp = ax.boxplot(clipped, positions=positions, showfliers=False,
                            patch_artist=True, widths=0.6)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_edgecolor(_edge(color))
                patch.set_alpha(0.92)
            for med in bp["medians"]:
                med.set_color("black")
            ax.set_ylim(*YCLIP[col])
        ax.set_xticks(positions)
        # Upright class names are one line so neighboring labels do not cross.
        tick_labels = (
            [f"{g} (n={n})" for g, n in zip(groups, ns)]
            if tick_rotation else
            [f"{g}\n(n={n})" for g, n in zip(groups, ns)]
        )
        ax.set_xticklabels(
            tick_labels,
            fontsize=tick_size,
            rotation=tick_rotation,
            ha="center",
            va="top" if tick_rotation else "center",
        )
        ax.tick_params(labelsize=tick_size)
        if ref is not None:
            ax.axhline(ref, color="#6E3B3B", ls="--", lw=0.9, alpha=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col == "kge":
            ax.axhline(0.0, color="gray", ls=":", lw=0.8, alpha=0.6)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_title(lbl, fontsize=title_size)
    axes_row[0].set_ylabel(row_label, fontsize=ylab_size, fontweight=ylab_weight)


def main():
    records = []
    test = add_bins(load_period("test"))
    val = add_bins(load_period("val"))

    for df, period in [(test, "test"), (val, "val")]:
        summarise(df, "aridity_grp", "aridity", period, records)
        summarise(df, "snow_grp", "snow", period, records)
        summarise(df, "area_grp", "drainage_area", period, records)

    summary = pd.DataFrame.from_records(records)
    summary = summary[["stratifier", "period", "group", "metric",
                       "median", "q25", "q75", "n"]]
    summary.to_csv(TAB_DIR / "fig2_stratified_summary.csv", index=False)
    print(f"wrote {TAB_DIR/'fig2_stratified_summary.csv'}  ({len(summary)} rows)")

    # ---- Main figure: 2020-2024 holdouts, four violin rows ----
    # Included at 0.92 of the 5.5 in text block, so type on this 11.2 in
    # canvas is scaled by 0.452. 16.6 pt here prints at the same size as
    # the 7.5 pt titles in Figure 7, which is drawn at the text width.
    test = add_management(test)
    bounds = _bounds(test)
    fig, axes = plt.subplots(4, 3, figsize=(11.2, 13.6))
    kw = dict(title_size=16.6, tick_size=14.5, ylab_size=16.6,
              ylab_weight="regular", tick_rotation=0)
    draw_row(axes[0], test, "aridity_grp", "Aridity", kind="violin", bounds=bounds, **kw)
    draw_row(axes[1], test, "snow_grp", "Snow fraction", kind="violin", bounds=bounds, **kw)
    # Four full area labels do not fit across a text-width panel. The
    # caption gives the bounds in km^2.
    area = test.copy()
    area["area_grp"] = area["area_grp"].cat.rename_categories({
        "100-1,000": "100–1k",
        "1,000-10,000": "1–10k",
        ">10,000": ">10k",
    })
    draw_row(axes[2], area, "area_grp", "Drainage area\n(km$^2$)", kind="violin", bounds=bounds, **kw)
    draw_row(axes[3], test, "mgmt_grp", "Natural /\nmanaged", kind="violin", bounds=bounds, **kw)
    # The long titles do not fit on one line at this printed size.
    titles = [
        "FHV bias (%)",
        "CRPSS vs\nungauged prediction",
        "P99 upper-quantile\ncoverage",
    ]
    for row in axes:
        for ax, title in zip(row, titles):
            ax.set_title(title, fontsize=16.6, pad=4)
            # Class names are wider than the column, so the tick line is
            # smaller than the axis title. The sample size stays on its own line.
            ax.tick_params(axis="x", labelsize=8.6)
            for lab in ax.get_xticklabels():
                lab.set_fontsize(8.6)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.96, bottom=0.045,
                        wspace=0.42, hspace=0.48)
    fig.savefig(FIG_DIR / "hydroclimate_stratified_performance.pdf")
    plt.close(fig)
    print(f"wrote {FIG_DIR/'hydroclimate_stratified_performance.pdf'}")

    render_si(val)
    print(f"wrote {FIG_DIR/'si_hydroclimate_stratified_extra.pdf'}")


def render_si(val=None):
    """2015-2019 aridity and snow boxes at the journal text width."""
    plt.rcParams.update({
        "font.family": "Latin Modern Roman",
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.45,
    })
    if val is None:
        val = add_bins(load_period("val"))
    # Full class names do not fit across a text-width panel, so the ticks
    # use the short name and the sample size. The class bounds are the
    # same ones named in the main-text figure.
    short = {
        "Humid (<1)": "Humid",
        "Transitional (1-2)": "Transitional",
        "Arid (>2)": "Arid",
        "Rain (<0.1)": "Rain",
        "Mixed (0.1-0.4)": "Mixed",
        "Snow (>=0.4)": "Snow",
    }
    val = val.copy()
    for col in ("aridity_grp", "snow_grp"):
        val[col] = val[col].cat.rename_categories(short)
    # Three class names do not fit side by side at the journal text width.
    # Each name stays on its own line, with the sample size under it, and
    # the block is turned upright. Titles wrap so the right-hand panel
    # is not clipped.
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 6.0))
    kw = dict(title_size=7.5, tick_size=6.5, ylab_size=8, ylab_weight="regular",
              tick_rotation=0)
    draw_row(axes[0], val, "aridity_grp", "Aridity", **kw)
    draw_row(axes[1], val, "snow_grp", "Snow fraction", **kw)
    titles = [
        "FHV bias (%)",
        "CRPSS vs\nungauged prediction",
        "P99 upper-quantile\ncoverage",
    ]
    for row in axes:
        for ax, title in zip(row, titles):
            ax.set_title(title, fontsize=7.5, pad=3)
            labels = [t.get_text() for t in ax.get_xticklabels()]
            ax.set_xticklabels(
                labels, fontsize=6.5, rotation=90,
                ha="right", va="center", rotation_mode="anchor",
            )
            ax.tick_params(axis="x", pad=2)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.90, bottom=0.16,
                        wspace=0.48, hspace=0.78)
    fig.savefig(FIG_DIR / "si_hydroclimate_stratified_extra.pdf")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", choices=["all", "si"])
    args = ap.parse_args()
    if args.only == "si":
        render_si()
    else:
        main()
