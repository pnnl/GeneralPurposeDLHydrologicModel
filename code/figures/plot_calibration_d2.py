#!/usr/bin/env python
"""
Calibration figures for the D2 MDN on the PUB held-out basins (the UQ headline).

Reads the per-basin eval CSV from evaluate_d2.py (columns cov90/cov95/cov99 are
the empirical coverages = mean(PIT <= level) per basin) and produces:

  plots/d2_reliability_diagram.png   — nominal vs observed coverage (the money figure)
  plots/d2_pit_histogram.png         — distribution of per-basin coverage at each level
  plots/d2_crps_cdf.png              — CDF of normalized CRPS across basins (sharpness)

A perfectly calibrated model sits on the 1:1 line in the reliability diagram.

Usage:
  python plot_calibration_d2.py                                  # frozen by default
  python plot_calibration_d2.py --csv eval_d2_finetune_heldout_test.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# the PIT levels we recorded per basin, and the nominal coverage each targets
LEVELS = [('cov90', 0.90), ('cov95', 0.95), ('cov99', 0.99)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=None,
                    help='per-basin eval CSV (default: frozen, else unfrozen)')
    ap.add_argument('--outdir', default='plots')
    args = ap.parse_args()

    csv = args.csv
    if csv is None:
        for cand in ('eval_d2_finetune_frozen_heldout_test.csv',
                     'eval_d2_finetune_heldout_test.csv'):
            if Path(cand).exists():
                csv = cand
                break
    if csv is None or not Path(csv).exists():
        raise SystemExit(f"No eval CSV found (looked for frozen/unfrozen). Pass --csv.")
    df = pd.read_csv(csv)
    tag = Path(csv).stem
    Path(args.outdir).mkdir(exist_ok=True)
    print(f"Calibration from {csv}: {len(df)} basins")

    # ---- 1. Reliability diagram: nominal vs mean observed coverage -----------
    nominal = [lvl for _, lvl in LEVELS]
    obs_mean = [df[col].mean() for col, _ in LEVELS]
    obs_med = [df[col].median() for col, _ in LEVELS]
    # IQR band across basins
    lo = [df[col].quantile(0.25) for col, _ in LEVELS]
    hi = [df[col].quantile(0.75) for col, _ in LEVELS]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0.85, 1.0], [0.85, 1.0], 'k--', lw=1, label='perfect calibration')
    # center the IQR band on the median (median is always within [Q25,Q75], so
    # the asymmetric error bars are non-negative; the mean can fall outside it).
    ax.errorbar(nominal, obs_med,
                yerr=[np.array(obs_med) - np.array(lo), np.array(hi) - np.array(obs_med)],
                fmt='o-', color='C0', capsize=4, label='MDN (median ± IQR across basins)')
    ax.plot(nominal, obs_mean, 's:', color='C1', alpha=0.8, label='MDN (mean)')
    for x, y in zip(nominal, obs_med):
        ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points', xytext=(6, -10), fontsize=8)
    ax.set_xlabel('Nominal coverage (PIT level)')
    ax.set_ylabel('Observed coverage  =  mean( PIT ≤ level )')
    ax.set_title(f'D2 MDN reliability — PUB held-out\n{tag}')
    ax.set_xlim(0.86, 1.0); ax.set_ylim(0.80, 1.0)
    ax.grid(alpha=0.3); ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout(); fig.savefig(f'{args.outdir}/d2_reliability_diagram.png', dpi=150)
    print(f"  wrote {args.outdir}/d2_reliability_diagram.png")

    # ---- 2. Per-basin coverage distributions (calibration spread) ------------
    fig, axes = plt.subplots(1, len(LEVELS), figsize=(4 * len(LEVELS), 3.5), sharey=True)
    for ax, (col, lvl) in zip(axes, LEVELS):
        ax.hist(df[col].dropna(), bins=30, color='C0', alpha=0.75, edgecolor='white')
        ax.axvline(lvl, color='k', ls='--', lw=1.2, label=f'nominal {lvl:.2f}')
        ax.axvline(df[col].median(), color='C3', ls='-', lw=1.2,
                   label=f'median {df[col].median():.3f}')
        ax.set_title(f'P{int(lvl*100)} coverage')
        ax.set_xlabel('per-basin observed coverage')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('# basins')
    fig.suptitle(f'D2 MDN per-basin coverage — PUB held-out ({len(df)} basins)')
    fig.tight_layout(); fig.savefig(f'{args.outdir}/d2_pit_histogram.png', dpi=150)
    print(f"  wrote {args.outdir}/d2_pit_histogram.png")

    # ---- 3. Sharpness: CDF of normalized CRPS across basins ------------------
    if 'crps_norm' in df.columns:
        x = np.sort(df['crps_norm'].dropna().clip(upper=5).values)
        y = np.arange(1, len(x) + 1) / len(x)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot(x, y, color='C2', lw=2)
        med = df['crps_norm'].median()
        ax.axvline(med, color='C3', ls='--', label=f'median nCRPS = {med:.3f}')
        ax.set_xlabel('normalized CRPS  (CRPS / mean obs flow)')
        ax.set_ylabel('cumulative fraction of basins')
        ax.set_title(f'D2 MDN sharpness — PUB held-out\n{tag}')
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(f'{args.outdir}/d2_crps_cdf.png', dpi=150)
        print(f"  wrote {args.outdir}/d2_crps_cdf.png")
    else:
        print("  (no crps_norm column yet — re-run evaluate_d2.py to add CRPS)")


if __name__ == '__main__':
    main()
