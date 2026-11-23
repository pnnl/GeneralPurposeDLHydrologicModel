#!/usr/bin/env python3
"""CDFs of how often the ungauged prediction falls inside the mixture."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_mpl_font  # noqa: E402
import plot_head12_hydro as maps  # noqa: E402
paper_mpl_font.use()

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "outputs" / "standalone"
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SHARED = "#0072B2"
TASK = "#D55E00"


def ecdf(values):
    x = np.sort(np.asarray(values, float))
    x = x[np.isfinite(x)]
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def add_ecdf(ax, values, color, ls, label):
    x, y = ecdf(values)
    ax.plot(x, y, color=color, ls=ls, lw=1.4, label=label)


def style(ax, xlabel):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction of catchments")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)


def load_pair(period):
    shared = pd.read_csv(EXP / f"within_mixture_{period}_shared.csv")
    task = pd.read_csv(EXP / f"within_mixture_{period}_standalone.csv")
    return shared.merge(task, on="comid", suffixes=("_shared", "_task"))


def inclusion_figure(pair, name):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    panels = [
        (axes[0], "frac_in_50", 0.50, "Fraction of days inside the 25th-75th percentiles", "(a)"),
        (axes[1], "frac_in_80", 0.80, "Fraction of days inside the 10th-90th percentiles", "(b)"),
    ]
    notes = {}
    for ax, column, nominal, xlabel, tag in panels:
        shared = pair[f"{column}_shared"]
        task = pair[f"{column}_task"]
        add_ecdf(ax, shared, SHARED, "-",
                 f"Shared (median {np.median(shared):.2f})")
        add_ecdf(ax, task, TASK, "--",
                 f"Task-specific (median {np.median(task):.2f})")
        ax.axvline(nominal, color="0.55", lw=0.7, zorder=0)
        style(ax, xlabel)
        ax.legend(frameon=False, loc="upper left")
        ax.text(0.98, 0.04, tag, transform=ax.transAxes, ha="right")
        notes[column] = {
            "n": int(len(pair)),
            "shared_median": float(np.median(shared)),
            "task_median": float(np.median(task)),
            "shared_above_nominal": float(np.mean(shared >= nominal)),
            "task_above_nominal": float(np.mean(task >= nominal)),
        }
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return notes


def inclusion_figure_closeness(pair, name):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    panels = [
        (axes[0], "mae_to_median_norm", "Mean |ungauged - mixture median| / mean flow", "(a)"),
        (axes[1], "mae_to_mean_norm", "Mean |ungauged - mixture mean| / mean flow", "(b)"),
    ]
    notes = {}
    for ax, column, xlabel, tag in panels:
        shared = pair[f"{column}_shared"]
        task = pair[f"{column}_task"]
        finite = np.isfinite(shared) & np.isfinite(task)
        shared = shared[finite]
        task = task[finite]
        add_ecdf(ax, shared, SHARED, "-", f"Shared (median {np.median(shared):.2f})")
        add_ecdf(ax, task, TASK, "--", f"Task-specific (median {np.median(task):.2f})")
        upper = float(np.nanquantile(np.concatenate([shared, task]), 0.98))
        ax.set_xlim(0, max(0.5, upper))
        ax.set_ylim(0, 1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Fraction of catchments")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="lower right")
        ax.text(0.04, 0.96, tag, transform=ax.transAxes, va="top")
        notes[column] = {
            "n": int(finite.sum()),
            "shared_median": float(np.median(shared)),
            "task_median": float(np.median(task)),
        }
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return notes


def cdf_grid(pair, panels, name, ncol=2):
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 2.7 * nrow))
    axes = np.atleast_1d(axes).ravel()
    notes = {}
    for ax, (column, xlabel, nominal, tag) in zip(axes, panels):
        shared = pair[f"{column}_shared"].to_numpy(float)
        task = pair[f"{column}_task"].to_numpy(float)
        finite = np.isfinite(shared) & np.isfinite(task)
        shared, task = shared[finite], task[finite]
        shared_median = float(np.median(shared))
        task_median = float(np.median(task))
        if nominal is None:
            lo = float(np.nanquantile(np.concatenate([shared, task]), 0.01))
            hi = float(np.nanquantile(np.concatenate([shared, task]), 0.99))
            shared = np.clip(shared, lo, hi)
            task = np.clip(task, lo, hi)
            ax.set_xlim(lo, hi)
            ax.axvline(0.0, color="0.55", lw=0.7, zorder=0)
        elif isinstance(nominal, tuple) and nominal[0] == "positive":
            hi = float(nominal[1])
            shared = np.minimum(shared, hi)
            task = np.minimum(task, hi)
            ax.set_xlim(0, hi)
        else:
            ax.set_xlim(0, 1)
            ax.axvline(nominal, color="0.55", lw=0.7, zorder=0)
        add_ecdf(ax, shared, SHARED, "-", f"Shared (median {shared_median:.2f})")
        add_ecdf(ax, task, TASK, "--", f"Task-specific (median {task_median:.2f})")
        ax.set_ylim(0, 1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Fraction of catchments")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # The width panel is open at the upper right. "best" puts the
        # legend on the axis label there.
        loc = "upper right" if column == "mae_to_median_over_width" else "best"
        ax.legend(frameon=False, loc=loc, fontsize=8.5)
        ax.text(0.98, 0.04, tag, transform=ax.transAxes, ha="right")
        notes[column] = {
            "n": int(finite.sum()),
            "shared_median": shared_median,
            "task_median": task_median,
        }
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return notes


def _pooled_limit(pair, columns, q=0.95):
    values = np.concatenate([
        pair[f"{column}_shared"].to_numpy(float) for column in columns
    ] + [
        pair[f"{column}_task"].to_numpy(float) for column in columns
    ])
    values = values[np.isfinite(values)]
    return float(np.quantile(np.abs(values), q))


def spatial_figure(pair, cents, rows, filename, cbar_label, norm, cmap, clip, ticks=None):
    located = maps.attach_xy(pair, cents)
    lon = located["lon"].to_numpy(float)
    lat = located["lat"].to_numpy(float)
    ok = np.isfinite(lon) & np.isfinite(lat)
    maps.MAP_EXTENT = [
        float(lon[ok].min() - 1.6), float(lon[ok].max() + 1.6),
        float(lat[ok].min() - 1.0), float(lat[ok].max() + 1.0),
    ]
    fig, axes, cax = maps._paper_maps(len(rows), 2, cbar="shared")
    if len(rows) == 1:
        axes = np.atleast_2d(axes)
    last = None
    for i, (column, row_title) in enumerate(rows):
        for j, (suffix, model) in enumerate((("_shared", "Shared"), ("_task", "Task-specific"))):
            ax = axes[i, j]
            maps._basemap(ax)
            values = located[f"{column}{suffix}"]
            sc, n = maps._scatter(
                ax, located, values, cmap, norm, 8.0, clip[0], clip[1])
            maps._finish_map(ax)
            last = sc
            med = float(np.nanmedian(values))
            maps._panel_title(ax, f"{model}, {row_title}", f"median {med:.2f},  n = {n:,}")
    colorbar = fig.colorbar(last, cax=cax, orientation="horizontal")
    colorbar.ax.tick_params(labelsize=6.5, width=0.4, length=2.0, pad=1.2)
    colorbar.outline.set_linewidth(0.35)
    colorbar.set_label(cbar_label, fontsize=7, labelpad=3)
    if ticks is not None:
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels([f"{tick:g}" for tick in ticks])
        colorbar.ax.minorticks_off()
    maps._save(fig, OUT / f"{filename}.png")


def scatter_figure(pair):
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.plot([0, 1], [0, 1], color="0.35", lw=0.8, zorder=0)
    ax.scatter(pair["frac_in_80_task"], pair["frac_in_80_shared"],
               s=8, c=SHARED, alpha=0.45, linewidths=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Task-specific fraction inside 10th-90th")
    ax.set_ylabel("Shared fraction inside 10th-90th")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "taskspecific_within_mixture_scatter.pdf", bbox_inches="tight")
    fig.savefig(OUT / "taskspecific_within_mixture_scatter.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def median_block(pair, column):
    return {
        "n": int(len(pair)),
        "shared": float(pair[f"{column}_shared"].median()),
        "task": float(pair[f"{column}_task"].median()),
    }


def main():
    plt.rcParams.update({
        "font.family": "Latin Modern Roman",
        # Included at the 5.5 in text block from a 7.2 in canvas, so 10 pt
        # here prints at the Figure 7 size.
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    test = load_pair("2020-2024")
    span = load_pair("2015-2024")
    cents = maps.load_centroids()
    cdf_panels = [
        ("frac_in_50", "Fraction of days inside the 25th-75th percentiles", 0.50, "(a)"),
        ("frac_in_80", "Fraction of days inside the 10th-90th percentiles", 0.80, "(b)"),
        ("mae_to_median_norm", "Mean |ungauged - mixture median| / mean flow", ("positive", 1.5), "(c)"),
        ("mae_to_mean_norm", "Mean |ungauged - mixture mean| / mean flow", ("positive", 2.0), "(d)"),
        ("mae_to_median_over_width", "Mean |ungauged - median| / (90th-10th width)", ("positive", 0.6), "(e)"),
    ]
    stats = {
        "test_cdf": cdf_grid(test, cdf_panels, "taskspecific_within_mixture_cdf"),
        "span_cdf": cdf_grid(span, cdf_panels, "si_taskspecific_within_mixture_cdf_2015_2024", ncol=2),
        "test_width_norm": median_block(test, "width_norm"),
        "span_width_norm": median_block(span, "width_norm"),
    }
    fraction_norm = Normalize(vmin=0.25, vmax=1.0)
    distance_norm = LogNorm(vmin=0.03, vmax=2.0)
    width_norm = LogNorm(vmin=0.05, vmax=0.40)
    viridis = plt.get_cmap("viridis")
    for label, pair, tag in (("2020-2024", test, "2020_2024"), ("2015-2024", span, "2015_2024")):
        spatial_figure(
            pair, cents,
            [("frac_in_50", "inside 25th-75th"), ("frac_in_80", "inside 10th-90th")],
            f"taskspecific_within_mixture_maps_inclusion_{tag}",
            "Fraction of days", fraction_norm, viridis, (0.25, 1.0),
            ticks=[0.25, 0.5, 0.75, 1.0])
        spatial_figure(
            pair, cents,
            [("mae_to_median_norm", "distance to median"),
             ("mae_to_mean_norm", "distance to mean")],
            f"taskspecific_within_mixture_maps_distance_{tag}",
            "Mean absolute difference / mean flow",
            distance_norm, viridis, (0.03, 2.0),
            ticks=[0.03, 0.1, 0.3, 1, 2])
        spatial_figure(
            pair, cents,
            [("mae_to_median_over_width", "distance / interval width")],
            f"taskspecific_within_mixture_maps_width_{tag}",
            "Mean |ungauged - median| / (90th-10th width)",
            width_norm, viridis, (0.05, 0.40),
            ticks=[0.05, 0.1, 0.2, 0.4])
        stats[f"{tag}_distance_color_limit"] = [0.03, 2.0]
        stats[f"{tag}_width_color_limit"] = [0.05, 0.40]
    (OUT / "within_mixture_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
