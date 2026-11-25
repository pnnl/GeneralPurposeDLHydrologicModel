#!/usr/bin/env python3
"""Agreement of fine-tuned and standalone PUB with the time-series checkpoint.

All three series are daily ungauged predictions on the 2020-2024 holdouts.
A day counts when the observation and all three predictions are finite and
non-negative. "Near" means the absolute difference is at most 25% of that
basin's mean observed flow.
"""

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
FINETUNE = "#0072B2"
STANDALONE = "#D55E00"
NEAR = 0.25
MIN_DAYS = 30


def load(path):
    frame = pd.read_parquet(path, columns=["comid", "date", "obs_mm_day", "pred_mm_day"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def score(reference, other, obs_name="obs_mm_day"):
    rows = []
    for comid, group in reference.groupby("comid", sort=False):
        obs = group[obs_name].to_numpy(float)
        base = group["pred_ref"].to_numpy(float)
        pred = group["pred_other"].to_numpy(float)
        ok = np.isfinite(obs) & (obs >= 0) & np.isfinite(base) & (base >= 0)
        ok &= np.isfinite(pred) & (pred >= 0)
        if int(ok.sum()) < MIN_DAYS:
            continue
        obs, base, pred = obs[ok], base[ok], pred[ok]
        mean_obs = float(obs.mean())
        if mean_obs <= 1e-8:
            continue
        gap = np.abs(pred - base) / mean_obs
        rows.append({
            "comid": int(comid),
            "n_days": int(ok.sum()),
            "frac_near": float(np.mean(gap <= NEAR)),
            "mae_norm": float(gap.mean()),
        })
    return pd.DataFrame(rows)


def ecdf(values):
    x = np.sort(np.asarray(values, float))
    x = x[np.isfinite(x)]
    return x, np.arange(1, len(x) + 1) / len(x)


def consistency_maps(pair):
    """Maps of the normalized distance to the time-series checkpoint."""
    located = maps.attach_xy(pair, maps.load_centroids())
    lon = located["lon"].to_numpy(float)
    lat = located["lat"].to_numpy(float)
    ok = np.isfinite(lon) & np.isfinite(lat)
    maps.MAP_EXTENT = [
        float(lon[ok].min() - 1.6), float(lon[ok].max() + 1.6),
        float(lat[ok].min() - 1.0), float(lat[ok].max() + 1.0),
    ]
    rows = [
        ("mae_norm", "distance to time-series checkpoint",
         LogNorm(vmin=0.08, vmax=2.0), (0.08, 2.0),
         [0.08, 0.2, 0.5, 1, 2], "Mean absolute difference / mean flow"),
    ]
    viridis = plt.get_cmap("viridis")
    fig, axes, caxes = maps._paper_maps(len(rows), 2, cbar="panel")
    columns = (("finetuned", "Fine-tuned"), ("standalone", "Standalone"))
    for i, (column, row_title, norm, clip, ticks, cbar_label) in enumerate(rows):
        last = None
        for j, (suffix, model) in enumerate(columns):
            ax = axes[i, j]
            maps._basemap(ax)
            values = located[f"{column}_{suffix}"]
            sc, n = maps._scatter(ax, located, values, viridis, norm, 8.0, clip[0], clip[1])
            maps._finish_map(ax)
            last = sc
            med = float(np.nanmedian(values))
            maps._panel_title(ax, f"{model}, {row_title}", f"median {med:.2f},  n = {n:,}")
        caxes[i, 0].set_visible(False)
        colorbar = fig.colorbar(last, cax=caxes[i, 1], orientation="horizontal")
        colorbar.ax.tick_params(labelsize=6.5, width=0.4, length=2.0, pad=1.2)
        colorbar.outline.set_linewidth(0.35)
        colorbar.set_label(cbar_label, fontsize=7, labelpad=3)
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels([f"{tick:g}" for tick in ticks])
        colorbar.ax.minorticks_off()
    maps._save(fig, OUT / "taskspecific_pretrain_pub_consistency_maps.png")


def main():
    temporal = load(EXP / "point_pretrain_holdout_test.parquet")
    finetuned = load(EXP / "point_paper_h2_reeval_test.parquet")
    standalone = load(EXP / "point_test.parquet")
    base = temporal.rename(columns={"pred_mm_day": "pred_ref"})[
        ["comid", "date", "obs_mm_day", "pred_ref"]]
    fine = base.merge(
        finetuned.rename(columns={"pred_mm_day": "pred_other"})[["comid", "date", "pred_other"]],
        on=["comid", "date"])
    alone = base.merge(
        standalone.rename(columns={"pred_mm_day": "pred_other"})[["comid", "date", "pred_other"]],
        on=["comid", "date"])
    fine_scores = score(fine, None).rename(columns={
        "frac_near": "frac_near_finetuned", "mae_norm": "mae_norm_finetuned",
        "n_days": "n_days_finetuned"})
    alone_scores = score(alone, None).rename(columns={
        "frac_near": "frac_near_standalone", "mae_norm": "mae_norm_standalone",
        "n_days": "n_days_standalone"})
    pair = fine_scores.merge(alone_scores, on="comid")
    pair.to_csv(EXP / "pretrain_pub_consistency_2020_2024.csv", index=False)

    plt.rcParams.update({
        # Included at the 5.5 in text block from a 7.2 in canvas, so 10 pt
        # here prints at the Figure 7 size.
        "font.family": "Latin Modern Roman", "font.size": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 8.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.15))
    panels = [
        (ax, "mae_norm", 0, 1.5,
         "Mean |difference| / mean flow", ""),
    ]
    notes = {"n": int(len(pair)), "near_threshold": NEAR}
    for ax, column, lo, hi, xlabel, tag in panels:
        for key, color, ls, label in (
            ("finetuned", FINETUNE, "-", "Fine-tuned"),
            ("standalone", STANDALONE, "--", "Standalone"),
        ):
            values = pair[f"{column}_{key}"].to_numpy(float)
            shown = np.clip(values, lo, hi)
            x, y = ecdf(shown)
            ax.plot(x, y, color=color, ls=ls, lw=1.4,
                    label=f"{label} (median {np.median(values):.2f})")
            notes[f"{column}_{key}"] = float(np.median(values))
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Fraction of catchments")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="best")
        ax.text(0.98, 0.04, tag, transform=ax.transAxes, ha="right")
    fig.tight_layout()
    fig.savefig(OUT / "taskspecific_pretrain_pub_consistency_cdf.pdf", bbox_inches="tight")
    fig.savefig(OUT / "taskspecific_pretrain_pub_consistency_cdf.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    (OUT / "pretrain_pub_consistency_stats.json").write_text(json.dumps(notes, indent=2) + "\n")
    print(json.dumps(notes, indent=2))
    consistency_maps(pair)


if __name__ == "__main__":
    main()
