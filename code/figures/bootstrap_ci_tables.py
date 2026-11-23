#!/usr/bin/env python3
"""Basin-bootstrap 95% confidence intervals for the headline per-basin medians.

Pure post-processing over the already-computed per-basin CSVs. Resamples basins with
replacement (default 5000 iterations, fixed seed) and reports the 2.5/97.5 percentiles
of the resampled median for each metric. These intervals quantify basin-sampling
variability ONLY - they are not a significance test of the model against another model,
and they do not account for temporal autocorrelation within a basin.

Products (all on the PUB spatial holdouts unless noted):
    H1  supervised basins, in-sample   eval_hydro/head1_gauged_{test,val}_per_basin.csv
    H2  PUB point estimate             eval_hydro/head2_pub_{test,val}_per_basin.csv
    H3  PUB MDN (deployed)             eval_hydro/head3_mdn_{test,val}_per_basin.csv
                                       eval_hydro/head3_ens_metrics_{test,val}.csv (CRPSS)
    Paired dKGE (H3 - H2) on the matched holdout basins.

Outputs (under outputs/tables/):
    tables/three_head_bootstrap_ci.csv
    tables/three_head_bootstrap_ci.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAB_DIR = ROOT / "outputs" / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 5000
SEED = 20240831


def boot_median_ci(values: np.ndarray, n_boot=N_BOOT, seed=SEED):
    """Point median + basin-bootstrap 95% CI of the median."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return dict(point=np.nan, ci_lo=np.nan, ci_hi=np.nan, n=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    meds = np.median(v[idx], axis=1)
    return dict(point=float(np.median(v)),
                ci_lo=float(np.percentile(meds, 2.5)),
                ci_hi=float(np.percentile(meds, 97.5)),
                n=int(n))


def boot_paired_median_ci(diff: np.ndarray, n_boot=N_BOOT, seed=SEED):
    """Median of a paired difference with a basin-resampling CI (basins kept paired)."""
    return boot_median_ci(diff, n_boot=n_boot, seed=seed + 1)


# metric column -> display label; present in the H3 MDN per-basin CSV
H3_METRICS = [
    ("kge", "KGE"), ("nse", "NSE"), ("fhv", "FHV bias (%)"),
    ("cov95", "P95 upper-quantile coverage"),
    ("cov99", "P99 upper-quantile coverage"),
    ("peak_cov99", "P99 peak-day coverage"),
    ("crps_norm", "Normalized CRPS"),
    ("median_pit", "Median PIT"),
]
CRPSS_METRICS = [("crpss_pt", "CRPSS vs point"), ("crpss_cl", "CRPSS vs climatology")]


def rows_for_period(period: str, plabel: str):
    out = []
    # H1 supervised (KGE only - headline)
    h1 = pd.read_csv(ROOT / "outputs" / "eval_hydro" / f"head1_gauged_{period}_per_basin.csv")
    out.append(dict(product="H1 supervised (in-sample)", period=plabel,
                    metric="KGE", **boot_median_ci(h1["kge"].to_numpy())))
    # H2 PUB point (KGE)
    h2 = pd.read_csv(ROOT / "outputs" / "eval_hydro" / f"head2_pub_{period}_per_basin.csv")
    out.append(dict(product="H2 PUB point", period=plabel,
                    metric="KGE", **boot_median_ci(h2["kge"].to_numpy())))
    # H3 PUB MDN (deployed) - full metric set
    h3 = pd.read_csv(ROOT / "outputs" / "eval_hydro" / f"head3_mdn_{period}_per_basin.csv")
    ens = pd.read_csv(ROOT / "outputs" / "eval_hydro" / f"head3_ens_metrics_{period}.csv")
    for col, lbl in H3_METRICS:
        if col in h3.columns:
            out.append(dict(product="H3 PUB MDN", period=plabel, metric=lbl,
                            **boot_median_ci(h3[col].to_numpy())))
    for col, lbl in CRPSS_METRICS:
        if col in ens.columns:
            out.append(dict(product="H3 PUB MDN", period=plabel, metric=lbl,
                            **boot_median_ci(ens[col].to_numpy())))
    # Paired dKGE (H3 - H2) on matched holdout basins
    m = h2[["comid", "kge"]].merge(h3[["comid", "kge"]], on="comid",
                                   suffixes=("_h2", "_h3"))
    diff = (m["kge_h3"] - m["kge_h2"]).to_numpy()
    out.append(dict(product="Paired dKGE (H3 - H2)", period=plabel,
                    metric="median dKGE", **boot_paired_median_ci(diff)))
    return out


def main():
    records = []
    for period, plabel in [("test", "2020-2024"), ("val", "2015-2019")]:
        records += rows_for_period(period, plabel)
    df = pd.DataFrame.from_records(records)
    df = df[["product", "period", "metric", "point", "ci_lo", "ci_hi", "n"]]
    df.to_csv(TAB_DIR / "three_head_bootstrap_ci.csv", index=False)
    print(f"wrote {TAB_DIR/'three_head_bootstrap_ci.csv'}  ({len(df)} rows)")

    # Markdown rendering
    lines = [
        "# Basin-bootstrap 95% confidence intervals (per-basin medians)",
        "",
        f"Resampling: {N_BOOT} basin bootstrap iterations, seed {SEED}. CIs are the "
        "2.5/97.5 percentiles of the resampled median.",
        "**Scope:** these intervals quantify basin-sampling variability only; they are "
        "not a model-vs-model significance test and ignore within-basin temporal "
        "autocorrelation.",
        "",
        "| Product | Period | Metric | Median | 95% CI | n |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        ci = (f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]"
              if np.isfinite(r.ci_lo) else "-")
        pt = f"{r.point:.3f}" if np.isfinite(r.point) else "-"
        lines.append(f"| {r['product']} | {r['period']} | {r['metric']} | "
                     f"{pt} | {ci} | {int(r['n'])} |")
    (TAB_DIR / "three_head_bootstrap_ci.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {TAB_DIR/'three_head_bootstrap_ci.md'}")


if __name__ == "__main__":
    main()
