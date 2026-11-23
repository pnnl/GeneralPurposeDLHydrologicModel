#!/usr/bin/env python
"""
Build a unified evaluation summary table on the 2020-2024 test period across:

  1. Gauged deterministic    : temporally pretrained model evaluated on training basins
  2. Ungauged deterministic  : spatially fine-tuned model evaluated on 965 held-out basins
  3. Ungauged probabilistic  : MDN model evaluated on 965 held-out basins

Reads deterministic inference parquets for tasks 1-2 and the summary JSON from
evaluate_d2.py for task 3.

Outputs:
  three_head_table.csv
  three_head_table.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PEAK_FRAC = 0.02  # top-2% of observed days = "peaks" (matches evaluate_d2.py)
KGE_BENCHMARK = -0.414  # mean-flow KGE floor


def kge_components(obs, sim):
    if len(obs) < 30 or np.std(obs) == 0 or np.std(sim) == 0:
        return np.nan, np.nan, np.nan, np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    kge = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge, r, alpha, beta


def peak_bias(obs, sim):
    """Relative bias on the top-2% observed-flow days."""
    if len(obs) < 50:
        return np.nan
    thr = np.quantile(obs, 1.0 - PEAK_FRAC)
    m = obs >= thr
    if m.sum() < 5 or obs[m].sum() == 0:
        return np.nan
    return (sim[m].sum() - obs[m].sum()) / obs[m].sum()


def per_basin_metrics(parquet_path, keep_comids):
    df = pd.read_parquet(parquet_path)
    # column auto-detection (predicted/observed naming varies across inference scripts)
    pred_col = next(c for c in ('pred_cms', 'predicted_flow_cms', 'sim', 'pred', 'flow_pred')
                    if c in df.columns)
    obs_col = next(c for c in ('obs_cms', 'observed_flow_cms', 'obs', 'flow_obs', 'flow_cms')
                   if c in df.columns)
    df = df[df['comid'].isin(keep_comids)]
    df = df.dropna(subset=[obs_col, pred_col])
    df = df[df[obs_col] >= 0]  # filter missing/negative obs
    rows = []
    for comid, g in df.groupby('comid', sort=False):
        obs = g[obs_col].to_numpy(float)
        sim = g[pred_col].to_numpy(float)
        kge, r, alpha, beta = kge_components(obs, sim)
        rows.append({'comid': comid, 'kge': kge, 'r': r, 'alpha': alpha,
                     'beta': beta, 'peak_bias': peak_bias(obs, sim),
                     'n_valid': len(obs)})
    out = pd.DataFrame(rows).dropna(subset=['kge'])
    return out


def winsorized_mean(x, lo=0.05, hi=0.95):
    """Mean after clipping to [lo, hi] percentiles — robust to small-basin
    ratio explosion (a 0.74-cms basin can produce peak_bias=+4382 that wrecks
    a plain mean; the median is robust but discards magnitude info, so we
    report a winsorized mean as the magnitude-aware robust statistic)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return float('nan')
    a, b = np.quantile(x, [lo, hi])
    return float(np.clip(x, a, b).mean())


def summarize(df, label):
    return {
        'head': label,
        'n_basins': len(df),
        'median_kge': float(df['kge'].median()),
        'median_r': float(df['r'].median()),
        'median_alpha': float(df['alpha'].median()),
        'median_beta': float(df['beta'].median()),
        'median_peak_bias': float(df['peak_bias'].median()),
        'wmean_peak_bias': winsorized_mean(df['peak_bias']),
        'frac_kge_gt_0.5': float((df['kge'] > 0.5).mean()),
        'frac_kge_gt_0': float((df['kge'] > 0).mean()),
    }


def main():
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(root / "weights" / "checkpoints_v24" / "finetune" / "split_metadata.json"))
    heldout = set(int(c) for c in meta['eval_comids'])    # 965 PUB basins
    train_basins = set(int(c) for c in meta['train_comids'])  # 8,677 in-sample

    rows = []

    # Head 1 — v24 pretrain on TRAIN basins (gauged, in-sample)
    p1 = out_dir / 'predictions_v24_pretrain_test.parquet'
    if p1.exists():
        m1 = per_basin_metrics(p1, train_basins)
        m1.to_csv(out_dir / 'three_head_h1_gauged_per_basin.csv', index=False)
        rows.append(summarize(m1, 'Head 1 — gauged (v24 pretrain, train basins)'))
    else:
        print(f"WAIT: {p1} not present yet")

    # Head 2 — v24 finetune on PUB held-out basins
    p2 = out_dir / 'predictions_v24_finetune_test.parquet'
    if p2.exists():
        m2 = per_basin_metrics(p2, heldout)
        m2.to_csv(out_dir / 'three_head_h2_pub_per_basin.csv', index=False)
        rows.append(summarize(m2, 'Head 2 — PUB (v24 finetune, 965 held-out)'))
    else:
        print(f"WAIT: {p2} not present yet")

    # Head 3 — D2 MDN (frozen) on PUB held-out. Read the dedup'd per-basin CSV and
    # map node_idx -> comid via the saved split (eval_node_indices ↔ eval_comids,
    # same order) so it shares a basin key with the v24 heads.
    node_to_comid = {int(n): int(c) for n, c in
                     zip(meta['eval_node_indices'], meta['eval_comids'])}
    node_to_comid.update({int(n): int(c) for n, c in
                          zip(meta['train_node_indices'], meta['train_comids'])})

    m3 = None
    # prefer the frozen CSV; fall back to whatever D2 heldout CSV exists
    d2_csv = None
    for cand in (out_dir / 'eval_d2_finetune_frozen_heldout_test.csv',
                 out_dir / 'eval_d2_finetune_heldout_test.csv'):
        if Path(cand).exists():
            d2_csv = cand
            break
    if d2_csv:
        d3 = pd.read_csv(d2_csv).drop_duplicates(subset='node', keep='first')
        d3['comid'] = d3['node'].map(node_to_comid)
        d3 = d3.dropna(subset=['comid'])
        d3['comid'] = d3['comid'].astype(int)
        m3 = d3  # has kge,r,alpha,beta,peak_bias,cov90/95/99,peak_cov95/99
        m3.to_csv(out_dir / 'three_head_h3_mdn_per_basin.csv', index=False)
        s = summarize(m3, f'Head 3 — MDN (D2 frozen, {d2_csv})')
        s.update({
            'coverage_p90': float(m3['cov90'].median()),
            'coverage_p95': float(m3['cov95'].median()),
            'coverage_p99': float(m3['cov99'].median()),
            'peak_coverage_p95': float(m3['peak_cov95'].median()),
            'peak_coverage_p99': float(m3['peak_cov99'].median()),
        })
        # CRPS is only present once the CRPS-enabled eval has been re-run; the
        # older heldout CSVs predate it, so guard on the column.
        if 'crps' in m3.columns:
            s['median_crps'] = float(m3['crps'].median())
        if 'crps_norm' in m3.columns:
            s['median_crps_norm'] = float(m3['crps_norm'].median())
        rows.append(s)
    else:
        print("WAIT: no D2 heldout per-basin CSV present yet")

    if not rows:
        print("No inputs ready. Re-run after jobs finish.")
        sys.exit(0)

    # --- MATCHED block: Head 2 vs Head 3 on the IDENTICAL held-out basins ---
    # (Head 1 is in-sample/train basins by design and is not part of this set.)
    matched_lines = []
    if p2.exists() and m3 is not None:
        common = sorted(set(m2['comid']) & set(m3['comid']))
        h2c = m2[m2['comid'].isin(common)]
        h3c = m3[m3['comid'].isin(common)]
        matched_lines = [
            "",
            f"## Matched PUB comparison — Head 2 vs Head 3 on the SAME {len(common)} held-out basins",
            "",
            "| Head | median KGE | r | α | β | peak bias (top-2%) |",
            "|---|---|---|---|---|---|",
            f"| Head 2 — v24 mean | {h2c['kge'].median():.3f} | {h2c['r'].median():.3f} | "
            f"{h2c['alpha'].median():.3f} | {h2c['beta'].median():.3f} | {h2c['peak_bias'].median()*100:+.1f}% |",
            f"| Head 3 — MDN mean | {h3c['kge'].median():.3f} | {h3c['r'].median():.3f} | "
            f"{h3c['alpha'].median():.3f} | {h3c['beta'].median():.3f} | {h3c['peak_bias'].median()*100:+.1f}% |",
            "",
            f"Paired ΔKGE (H3−H2) median: {(h3c.set_index('comid')['kge'] - h2c.set_index('comid')['kge']).median():+.4f}",
        ]

    tbl = pd.DataFrame(rows)
    tbl.to_csv(out_dir / 'three_head_table.csv', index=False)

    # Markdown
    def f(x, p='{:.3f}'):
        return p.format(x) if isinstance(x, (int, float)) and not pd.isna(x) else '—'
    lines = [
        "# Three-head results — TEST period 2020–2024 (matched protocol)",
        "",
        "| Head | n basins | median KGE | r | α | β | peak bias med (top-2%) | peak bias wmean | KGE>0.5 | P90 cov | P95 cov | P95 peak-cov | P99 peak-cov | nCRPS |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['head']} | {r['n_basins']} | {f(r['median_kge'])} | "
            f"{f(r['median_r'])} | {f(r['median_alpha'])} | {f(r['median_beta'])} | "
            f"{f(r['median_peak_bias']*100,'{:+.1f}%')} | {f(r['wmean_peak_bias']*100,'{:+.1f}%')} | "
            f"{f(r['frac_kge_gt_0.5']*100,'{:.0f}%')} | "
            f"{f(r.get('coverage_p90',float('nan')))} | {f(r.get('coverage_p95',float('nan')))} | "
            f"{f(r.get('peak_coverage_p95',float('nan')))} | {f(r.get('peak_coverage_p99',float('nan')))} | "
            f"{f(r.get('median_crps_norm',float('nan')))} |"
        )
    lines += matched_lines
    lines += [
        "",
        "Notes: Head 1 = in-sample (train basins, temporal generalization); Heads 2/3 = "
        "PUB held-out (spatial generalization). Coverage/peak-coverage and nCRPS columns are "
        "MDN-only (point heads emit no distribution). nCRPS = per-basin CRPS normalized by mean "
        "observed flow (relative sharpness/skill, lower is better); blank until the CRPS-enabled "
        "eval is re-run. Peak bias is on the top-2% observed-flow days. "
        "'med' = per-basin median (robust); 'wmean' = winsorized mean clipped to [5,95] "
        "percentiles (magnitude-aware, immune to small-basin ratio explosion — a plain mean "
        "is dominated by tiny basins whose peak ratio can reach +4000%). H1 vs H2 peak bias is "
        "NOT controlled (different checkpoint AND disjoint basins); the controlled peak-bias "
        "comparison is the matched H2-vs-H3 block below (same basins, finetune stage).",
    ]
    (out_dir / 'three_head_table.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f"\nWrote {out_dir / 'three_head_table.csv'} and {out_dir / 'three_head_table.md'}")


if __name__ == '__main__':
    main()
