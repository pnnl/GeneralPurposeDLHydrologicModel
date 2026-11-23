#!/usr/bin/env python3
"""Horizontal violins of agreement for the shared path and separately trained models.

The x-axis is the agreement metric. Blue is the shared path and orange is
the separately trained model. Medians use the unclipped scores. Distance
violins are drawn from values at or below the right-hand axis limit.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_mpl_font
paper_mpl_font.use()

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "outputs" / "standalone"
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SHARED = "#0072B2"
TASK = "#D55E00"


def load():
    shared = pd.read_csv(EXP / "within_mixture_2020-2024_shared.csv")
    task = pd.read_csv(EXP / "within_mixture_2020-2024_standalone.csv")
    mixture = shared.merge(task, on="comid", suffixes=("_sh", "_ta"))
    pretrain = pd.read_csv(EXP / "pretrain_pub_consistency_2020_2024.csv")
    return mixture, pretrain


def draw(ax, shared, task, xlim, clip_hi):
    shared = np.asarray(shared, float)
    task = np.asarray(task, float)
    ok = np.isfinite(shared) & np.isfinite(task)
    shared, task = shared[ok], task[ok]
    if clip_hi is None:
        shown = (shared, task)
    else:
        shown = (np.minimum(shared, clip_hi), np.minimum(task, clip_hi))
    parts = ax.violinplot(
        shown, positions=[1, 0], vert=False, widths=0.72,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for body, color in zip(parts["bodies"], (SHARED, TASK)):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.82)
        body.set_linewidth(0.4)
    for values, ypos in ((shared, 1), (task, 0)):
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        q1 = min(max(q1, xlim[0]), xlim[1])
        q3 = min(max(q3, xlim[0]), xlim[1])
        med_draw = min(max(med, xlim[0]), xlim[1])
        ax.hlines(ypos, q1, q3, color="white", lw=2.0, zorder=3)
        ax.vlines(med_draw, ypos - 0.12, ypos + 0.12, color="0.1", lw=1.05, zorder=4)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.55, 1.55)
    ax.set_yticks([0, 1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=0.5, labelsize=9.5)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.6)


def main():
    plt.rcParams.update({
        "font.family": "Latin Modern Roman",
        # The PDF is 7.15 in wide and is included at the 5.5 in text
        # block, so these sizes print at the Figure 7 type.
        "font.size": 10,
        "axes.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    mixture, pretrain = load()
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.35), facecolor="white")
    axes = axes.ravel()
    specs = [
        (axes[0], pretrain.mae_norm_finetuned, pretrain.mae_norm_standalone,
         (0, 1.5), 1.5, "(a)  Checkpoint distance / mean flow", "Distance"),
        (axes[1], mixture.frac_in_50_sh, mixture.frac_in_50_ta,
         (0, 1), None, "(b)  Days inside the 25th–75th", "Fraction of days"),
        (axes[2], mixture.frac_in_80_sh, mixture.frac_in_80_ta,
         (0, 1), None, "(c)  Days inside the 10th–90th", "Fraction of days"),
        (axes[3], mixture.mae_to_median_norm_sh, mixture.mae_to_median_norm_ta,
         (0, 1.5), 1.5, "(d)  Distance to median / mean flow", "Distance"),
    ]
    for ax, shared, task, xlim, clip, title, xlabel in specs:
        draw(ax, shared, task, xlim, clip)
        ax.set_title(title, loc="left", fontsize=10, pad=3, color="0.1")
        ax.set_xlabel(xlabel)
        ax.set_yticklabels(["Separate", "Shared"])
    axes[1].axvline(0.50, color="0.55", lw=0.6, ls=(0, (1.4, 1.4)), zorder=1)
    axes[2].axvline(0.80, color="0.55", lw=0.6, ls=(0, (1.4, 1.4)), zorder=1)
    fig.tight_layout(w_pad=1.15, h_pad=1.6)
    png = OUT / "consistency_violin.png"
    pdf = OUT / "consistency_violin.pdf"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(png)


if __name__ == "__main__":
    main()
