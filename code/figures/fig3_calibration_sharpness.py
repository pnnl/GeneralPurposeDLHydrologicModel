#!/usr/bin/env python3
"""Manuscript Figure 7 - predictive calibration and sharpness.

Recomputes per-day probability-integral-transform (PIT), two-sided central-interval
reliability, sharpness, and peak-conditioned upper-quantile coverage from the stored
100-member ensemble - no model run, no GPU (CPU pass over the zarr).

Source ensemble: outputs/predictions_downscaled/ens100_wrftgw_2015_2024.zarr
    ensemble [n_catch, n_days, 100] f16 cms, comid, hruid, dates(int64 ns)
    This zarr is not included in the bundle.
Observations: outputs/predictions_v24_finetune_test.parquet (2020-2024),
              outputs/predictions_v24_val.parquet (2015-2019)  -> obs_cms by (comid, date)
Holdout basins: weights/checkpoints_d2/finetune_frozen/split_metadata.json -> eval_comids (965)

Panels:
    (a) real PIT histogram, 20 bins over [0,1], each basin weighted equally, both
        periods + uniform expectation + basin-bootstrap band
    (b) central-interval reliability - empirical vs nominal for 50/80/90/98% intervals
    (c) sharpness - median normalized central-interval width vs nominal coverage
    (d) peak-conditioned reliability - P95 and P99 upper-quantile coverage for all days
        and the highest 20/10/5/2/1% of observed flows

A descriptive KS statistic against the uniform is annotated (no pooled p-value, which
would be invalid under within-basin temporal dependence).

Outputs (under outputs/):
    figures/predictive_calibration_sharpness.pdf   (vector, main text)
    tables/fig3_pit_bins.csv
    tables/fig3_interval_reliability.csv
    tables/fig3_peak_coverage.csv
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import zarr
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

ZARR = ROOT / "outputs" / "predictions_downscaled" / "ens100_wrftgw_2015_2024.zarr"
SPLIT = ROOT / "weights" / "checkpoints_d2" / "finetune_frozen" / "split_metadata.json"
OBS = {"2020-2024": ROOT / "outputs" / "predictions_v24_finetune_test.parquet",
       "2015-2019": ROOT / "outputs" / "predictions_v24_val.parquet"}
PERIOD_YEARS = {"2020-2024": range(2020, 2025), "2015-2019": range(2015, 2020)}

MIN_N = 30
NBINS = 20
NOMINAL = [0.50, 0.80, 0.90, 0.98]      # central-interval nominal coverages
PEAK_FRACS = [1.0, 0.20, 0.10, 0.05, 0.02, 0.01]
N_BOOT = 500
SEED = 20240831

# Same muted palette as the stratified figure. The 2015-2019 gold is
# darkened so a thin line remains readable on white.
C_TEST = "#1F6A8A"
C_VAL = "#C4A15A"
C_P99 = "#C4623D"
C_REF = "#6E3B3B"
PERIOD_COLOR = {"2020-2024": C_TEST, "2015-2019": C_VAL}


def _edge(color):
    rgb = matplotlib.colors.to_rgb(color)
    return tuple(max(0.0, c * 0.55) for c in rgb)


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#E2E2E2", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    # Matplotlib point sizes print about a third larger than the manuscript's
    # 10 pt Computer Modern, so 7.5 pt here matches the narrative.
    ax.tick_params(length=2.8, width=0.6, labelsize=7.2, color="#444444")
    ax.xaxis.label.set_size(7.5)
    ax.yaxis.label.set_size(7.5)
    ax.title.set_size(7.5)


def _period_line(ax, x, y, period, marker, filled=True):
    color = PERIOD_COLOR[period]
    ax.plot(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float),
        marker=marker, color=color, lw=1.5,
        ms=4.6 if filled else 5.6,
        markerfacecolor=color if filled else "white",
        markeredgecolor=_edge(color) if filled else color,
        markeredgewidth=0.5 if filled else 1.0,
        label=period, zorder=3 if filled else 4,
    )


def render_figure(results, out_path):
    """Draw the four calibration panels from the aggregated period dicts."""
    plt.rcParams.update({
        "font.family": "Latin Modern Roman",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
    })
    # Final width is the 5.5 in AGU text block, so type is set at print size.
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.55))
    bin_centers = (np.arange(NBINS) + 0.5) / NBINS
    w = 1.0 / NBINS

    ax = axes[0, 0]
    Rt = results["2020-2024"]
    Rv = results["2015-2019"]
    band = ax.bar(
        bin_centers, Rt["ci_hi"] - Rt["ci_lo"], bottom=Rt["ci_lo"],
        width=w * 0.94, color=C_TEST, alpha=0.22, linewidth=0, zorder=1,
    )
    bars = ax.bar(
        bin_centers, Rt["pooled"], width=w * 0.62, color=C_TEST,
        edgecolor=_edge(C_TEST), linewidth=0.35, zorder=2,
    )
    (step,) = ax.step(
        bin_centers, Rv["pooled"], where="mid", color=C_VAL, lw=1.5, zorder=3,
    )
    uni = ax.axhline(1.0 / NBINS, color=C_REF, ls="--", lw=0.8, zorder=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.36)
    ax.set_xlabel("Probability integral transform")
    ax.set_ylabel("Basin-balanced frequency")
    ax.set_title("(a)  PIT histogram")
    ax.text(
        0.50, 0.62,
        f"KS = {Rt['ks']:.2f} (n={Rt['nb']}) and {Rv['ks']:.2f} (n={Rv['nb']})",
        transform=ax.transAxes, ha="center", va="center", fontsize=6.4, color="#333333",
    )
    ax.legend(
        [bars, band, step, uni],
        ["2020–2024", "95% bootstrap", "2015–2019", "Uniform"],
        ncol=2, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.50, 0.99),
        handlelength=1.4, columnspacing=0.6, labelspacing=0.15,
    )
    _style_ax(ax)

    ax = axes[0, 1]
    ax.plot([0.20, 1.05], [0.20, 1.05], color=C_REF, ls="--", lw=0.8,
            label="1:1", zorder=1)
    for period, marker, filled in (
        ("2020-2024", "o", True), ("2015-2019", "D", False),
    ):
        R = results[period]
        emp = [float(R["cov"][c].median()) for c in NOMINAL]
        _period_line(ax, NOMINAL, emp, period, marker, filled=filled)
    ax.set_xlim(0.20, 1.05)
    ax.set_ylim(0.20, 1.05)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Basin-median coverage")
    ax.set_title("(b)  Central-interval reliability")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=6.5, loc="upper left", borderpad=0.1, handlelength=1.4)
    _style_ax(ax)

    ax = axes[1, 0]
    for period, marker, filled in (
        ("2020-2024", "o", True), ("2015-2019", "D", False),
    ):
        R = results[period]
        widths = [float(R["width"][c].median()) for c in NOMINAL]
        _period_line(ax, NOMINAL, widths, period, marker, filled=filled)
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0, 2.15)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Median width / mean flow")
    ax.set_title("(c)  Sharpness")
    ax.legend(fontsize=6.5, loc="upper left", borderpad=0.15, handlelength=1.4)
    _style_ax(ax)

    ax = axes[1, 1]
    labels = ["All", "20%", "10%", "5%", "2%", "1%"]
    x = np.arange(len(PEAK_FRACS))
    R = results["2020-2024"]
    for lvl, color, marker, name in (
        (95, C_TEST, "o", "P95"),
        (99, C_P99, "s", "P99"),
    ):
        y = [float(R["peak"][(f, lvl)].median()) for f in PEAK_FRACS]
        ax.plot(
            x, y, marker=marker, color=color, lw=1.5, ms=4.6,
            markerfacecolor=color, markeredgecolor=_edge(color),
            markeredgewidth=0.5, label=name, zorder=3,
        )
        ax.axhline(lvl / 100.0, color=color, ls="--", lw=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.3, 5.25)
    ax.set_ylim(0.50, 1.06)
    ax.set_xlabel("Highest flow fraction, 2020–2024")
    ax.set_ylabel("Basin-median coverage")
    ax.set_title("(d)  Peak-conditioned reliability")
    ax.text(5.15, 0.962, "0.95", color=C_TEST, fontsize=6.0, ha="right", va="bottom")
    ax.text(5.15, 1.002, "0.99", color=C_P99, fontsize=6.0, ha="right", va="bottom")
    ax.legend(fontsize=6.5, loc="lower left", borderpad=0.15, handlelength=1.4)
    _style_ax(ax)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_from_tables():
    """Redraw the figure from the saved calibration tables. No ensemble read."""
    pit = pd.read_csv(TAB_DIR / "fig3_pit_bins.csv")
    rel = pd.read_csv(TAB_DIR / "fig3_interval_reliability.csv")
    peak = pd.read_csv(TAB_DIR / "fig3_peak_coverage.csv")
    results = {}
    for period in ("2020-2024", "2015-2019"):
        p = pit[pit["period"] == period].sort_values("bin_center")
        r = rel[rel["period"] == period]
        cov = pd.DataFrame({
            float(row.nominal): [row.empirical_median] for row in r.itertuples()
        })
        width = pd.DataFrame({
            float(row.nominal): [row.median_norm_width] for row in r.itertuples()
        })
        pk = peak[peak["period"] == period]
        peak_df = pd.DataFrame({
            (float(row.obs_top_frac), int(row.level)): [row.coverage_median]
            for row in pk.itertuples()
        })
        results[period] = dict(
            nb=int(p["n_basins"].iloc[0]),
            pooled=p["pooled"].to_numpy(),
            ci_lo=p["ci_lo"].to_numpy(),
            ci_hi=p["ci_hi"].to_numpy(),
            ks=float(p["ks_stat"].iloc[0]),
            cov=cov, width=width, peak=peak_df,
        )
    render_figure(results, FIG_DIR / "predictive_calibration_sharpness.pdf")


def load_obs_matrix(path, eval_comids, period_dates):
    """(n_basin, n_day) obs_cms aligned to eval_comids x period_dates order."""
    df = pd.read_parquet(path, columns=["date", "comid", "obs_cms"])
    df = df[df["comid"].isin(set(eval_comids))].copy()
    df["date"] = pd.to_datetime(df["date"])
    piv = df.pivot_table(index="comid", columns="date", values="obs_cms",
                         aggfunc="first")
    piv = piv.reindex(index=eval_comids, columns=period_dates)
    return piv.to_numpy(dtype=np.float64)


def per_basin_stats(ens, obs):
    """Compute per-basin calibration stats for one basin.

    ens: (D, M) float32 members;  obs: (D,) cms. Returns None if < MIN_N valid days.
    """
    m = np.isfinite(obs) & np.all(np.isfinite(ens), axis=1) & (obs >= 0)
    if m.sum() < MIN_N:
        return None
    e = ens[m].astype(np.float32)
    o = obs[m]
    n = len(o)

    # PIT (empirical): fraction of members <= obs, per day
    pit = np.mean(e <= o[:, None], axis=1)
    pit_hist, _ = np.histogram(pit, bins=NBINS, range=(0.0, 1.0))
    pit_hist = pit_hist.astype(np.float64) / n          # sums to 1 per basin

    # Two-sided central intervals
    cov, width_norm = {}, {}
    omean = o.mean() + 1e-8
    for c in NOMINAL:
        lo = np.percentile(e, (1.0 - c) / 2.0 * 100.0, axis=1)
        hi = np.percentile(e, (1.0 + c) / 2.0 * 100.0, axis=1)
        cov[c] = float(np.mean((o >= lo) & (o <= hi)))
        width_norm[c] = float(np.median(hi - lo) / omean)

    # Peak-conditioned one-sided upper-quantile coverage
    q95 = np.percentile(e, 95, axis=1)
    q99 = np.percentile(e, 99, axis=1)
    order = np.argsort(o)
    peak = {}
    for f in PEAK_FRACS:
        k = max(1, int(round(f * n)))
        sel = order[-k:]
        peak[(f, 95)] = float(np.mean(o[sel] <= q95[sel]))
        peak[(f, 99)] = float(np.mean(o[sel] <= q99[sel]))
    return dict(pit_hist=pit_hist, cov=cov, width_norm=width_norm, peak=peak)


def compute_period(z, ens_col, rows, eval_comids, period, zdates):
    years = PERIOD_YEARS[period]
    day_idx = np.where(np.isin(zdates.year, list(years)))[0]
    assert np.all(np.diff(day_idx) == 1), "period days not contiguous"
    sl = slice(int(day_idx[0]), int(day_idx[-1]) + 1)
    period_dates = pd.DatetimeIndex(zdates[sl])
    print(f"[{period}] days={len(period_dates)} "
          f"{period_dates[0].date()}..{period_dates[-1].date()}")

    obs = load_obs_matrix(OBS[period], eval_comids, period_dates)
    n_obs_basins = np.isfinite(obs).any(axis=1).sum()
    print(f"[{period}] obs basins with any valid day: {n_obs_basins}/{len(eval_comids)}")

    # Orthogonal read: fancy rows x contiguous day slice x all members
    print(f"[{period}] reading ensemble block {len(rows)} x {sl.stop-sl.start} x "
          f"{z[ens_col].shape[2]} ...")
    block = z[ens_col].oindex[np.asarray(rows), sl, :]      # (B, D, M) f16
    print(f"[{period}] block shape {block.shape} dtype {block.dtype}")

    per_hist, cov_rows, width_rows, peak_rows = [], [], [], []
    for i in range(block.shape[0]):
        st = per_basin_stats(block[i], obs[i])
        if st is None:
            continue
        per_hist.append(st["pit_hist"])
        cov_rows.append(st["cov"])
        width_rows.append(st["width_norm"])
        peak_rows.append(st["peak"])
    per_hist = np.asarray(per_hist)                          # (nb, NBINS)
    nb = len(per_hist)
    print(f"[{period}] basins passing MIN_N: {nb}")

    # PIT: equal-weight pooled histogram + basin-bootstrap band
    pooled = per_hist.mean(axis=0)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, nb, size=(N_BOOT, nb))
    boot = per_hist[idx].mean(axis=1)                        # (N_BOOT, NBINS)
    ci_lo = np.percentile(boot, 2.5, axis=0)
    ci_hi = np.percentile(boot, 97.5, axis=0)
    # descriptive KS vs uniform on the pooled CDF at bin edges
    cdf = np.cumsum(pooled)
    unif = np.linspace(1.0 / NBINS, 1.0, NBINS)
    ks = float(np.max(np.abs(cdf - unif)))

    cov_df = pd.DataFrame(cov_rows)
    width_df = pd.DataFrame(width_rows)
    peak_df = pd.DataFrame(peak_rows)
    return dict(period=period, nb=nb, pooled=pooled, ci_lo=ci_lo, ci_hi=ci_hi,
                ks=ks, cov=cov_df, width=width_df, peak=peak_df)


def main():
    z = zarr.open(str(ZARR), mode="r")
    zdates = pd.DatetimeIndex(pd.to_datetime(np.asarray(z["dates"][:])))
    zcomid = np.asarray(z["comid"][:]).astype(np.int64)
    comid_to_row = {int(c): i for i, c in enumerate(zcomid)}
    ens_col = "ensemble"

    eval_comids = [int(c) for c in json.load(open(SPLIT))["eval_comids"]]
    rows = [comid_to_row[c] for c in eval_comids if c in comid_to_row]
    keep = [c for c in eval_comids if c in comid_to_row]
    print(f"holdout comids: {len(eval_comids)}  present in zarr: {len(keep)}")
    eval_comids = keep

    results = {p: compute_period(z, ens_col, rows, eval_comids, p, zdates)
               for p in ["2020-2024", "2015-2019"]}

    # ---- Tables ----
    bin_centers = (np.arange(NBINS) + 0.5) / NBINS
    pit_rows = []
    for p, R in results.items():
        for b in range(NBINS):
            pit_rows.append(dict(period=p, bin_center=bin_centers[b],
                                 pooled=R["pooled"][b], ci_lo=R["ci_lo"][b],
                                 ci_hi=R["ci_hi"][b], uniform=1.0 / NBINS,
                                 ks_stat=R["ks"], n_basins=R["nb"]))
    pd.DataFrame(pit_rows).to_csv(TAB_DIR / "fig3_pit_bins.csv", index=False)

    rel_rows = []
    for p, R in results.items():
        for c in NOMINAL:
            rel_rows.append(dict(period=p, nominal=c,
                                 empirical_median=float(R["cov"][c].median()),
                                 empirical_q25=float(R["cov"][c].quantile(.25)),
                                 empirical_q75=float(R["cov"][c].quantile(.75)),
                                 median_norm_width=float(R["width"][c].median()),
                                 n_basins=R["nb"]))
    pd.DataFrame(rel_rows).to_csv(TAB_DIR / "fig3_interval_reliability.csv", index=False)

    peak_rows = []
    for p, R in results.items():
        for f in PEAK_FRACS:
            for lvl in (95, 99):
                peak_rows.append(dict(period=p, obs_top_frac=f, level=lvl,
                                      coverage_median=float(R["peak"][(f, lvl)].median()),
                                      n_basins=R["nb"]))
    pd.DataFrame(peak_rows).to_csv(TAB_DIR / "fig3_peak_coverage.csv", index=False)
    print("wrote fig3 tables")
    render_figure(results, FIG_DIR / "predictive_calibration_sharpness.pdf")


if __name__ == "__main__":
    import sys
    if "--from-tables" in sys.argv:
        plot_from_tables()
    else:
        main()
