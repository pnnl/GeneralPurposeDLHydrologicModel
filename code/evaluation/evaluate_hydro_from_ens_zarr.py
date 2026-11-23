#!/usr/bin/env python3
"""Score the 100-member MDN zarr on every day of validation and test.

Reads the full-graph ensemble zarr by its native chunks (once), keeps only
the holdout gauges used for the ungauged product, then scores KGE parts,
signatures, and ensemble PIT/CRPS on the full val and test records.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hydro_signatures import (  # noqa: E402
    ensemble_prob_metrics,
    hydro_signatures,
    summarize,
)

OBS_FILES = {
    "val": "predictions_v24_val.parquet",
    "test": "predictions_v24_finetune_test.parquet",
}
PERIODS = {
    "val": (pd.Timestamp("2015-01-01"), pd.Timestamp("2019-12-31")),
    "test": (pd.Timestamp("2020-01-01"), pd.Timestamp("2024-12-31")),
}


def load_obs_parquet(path: Path, keep_comids: set):
    df = pd.read_parquet(path, columns=["date", "comid", "obs_cms"])
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["comid"].isin(keep_comids)]
    df = df.dropna(subset=["obs_cms"])
    df = df[df["obs_cms"] >= 0]
    return df


def gather_holdout(z, rows):
    """Pull holdout rows out of the native (catch, day, member) chunks."""
    mean_z = z["mean"]
    ens_z = z["ensemble"]
    n_nodes, n_days = mean_z.shape
    n_mem = ens_z.shape[-1]
    def _c0(chunks, axis):
        c = chunks[axis]
        return int(c[0] if isinstance(c, (tuple, list)) else c)

    cblock = _c0(mean_z.chunks, 0)
    tblock = _c0(mean_z.chunks, 1)
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows)
    rows_s = rows[order]
    mean_h = np.empty((len(rows), n_days), dtype=np.float32)
    ens_h = np.empty((len(rows), n_days, n_mem), dtype=np.float16)

    n_c = (n_nodes + cblock - 1) // cblock
    n_t = (n_days + tblock - 1) // tblock
    for ic in tqdm(range(n_c), desc="gather catch-blocks"):
        c0 = ic * cblock
        c1 = min(c0 + cblock, n_nodes)
        lo = np.searchsorted(rows_s, c0, side="left")
        hi = np.searchsorted(rows_s, c1, side="left")
        if lo == hi:
            continue
        local = rows_s[lo:hi] - c0
        inv = order[lo:hi]
        for it in range(n_t):
            t0 = it * tblock
            t1 = min(t0 + tblock, n_days)
            mean_h[inv, t0:t1] = np.asarray(mean_z[c0:c1, t0:t1], dtype=np.float32)[local]
            ens_h[inv, t0:t1, :] = np.asarray(ens_z[c0:c1, t0:t1, :])[local]
    return mean_h, ens_h


def align_period(obs_dates, obs_vals, period_dates):
    per = np.asarray(period_dates, dtype="datetime64[ns]")
    od = np.asarray(obs_dates, dtype="datetime64[ns]")
    pos = np.searchsorted(per, od)
    ok = pos < len(per)
    pos_ok = np.clip(pos, 0, len(per) - 1)
    ok &= per[pos_ok] == od
    return pos[ok], np.asarray(obs_vals, dtype=np.float64)[ok]


def score_period(mean_h, ens_h, hold_comids, dates, obs, start, end, label):
    mask = (dates >= start) & (dates <= end)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return pd.DataFrame()
    i0, i1 = int(idx[0]), int(idx[-1]) + 1
    period_dates = dates[i0:i1]
    comid_to_i = {int(c): i for i, c in enumerate(hold_comids)}
    rows = []
    obs_p = obs[(obs["date"] >= start) & (obs["date"] <= end)]
    for comid, g in tqdm(obs_p.groupby("comid", sort=False), desc=label):
        j = comid_to_i.get(int(comid))
        if j is None:
            continue
        pos, obs_v = align_period(g["date"].to_numpy(), g["obs_cms"].to_numpy(), period_dates)
        if len(pos) < 30:
            continue
        pred = np.asarray(mean_h[j, i0:i1], dtype=np.float64)[pos]
        ens = np.asarray(ens_h[j, i0:i1, :], dtype=np.float64)[pos]
        sig = hydro_signatures(obs_v, pred, dates=period_dates[pos])
        if not np.isfinite(sig.get("kge", np.nan)):
            continue
        sig.update(ensemble_prob_metrics(obs_v, ens))
        sig["comid"] = int(comid)
        rows.append(sig)
    return pd.DataFrame(rows)


def _merge_table(outdir: Path, summaries: dict):
    existing = outdir / "hydro_summary.json"
    all_summ = {}
    if existing.exists():
        try:
            all_summ = json.loads(existing.read_text())
        except Exception:
            all_summ = {}
    all_summ.update(summaries)
    with open(existing, "w") as f:
        json.dump(all_summ, f, indent=2)
    rows = [{"product": name, **s} for name, s in all_summ.items()]
    if rows:
        pd.DataFrame(rows).to_csv(outdir / "hydro_three_head_table.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--gnn-root", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    z = zarr.open(args.zarr, mode="r")
    comid_arr = np.asarray(z["comid"][:])
    dates = pd.to_datetime(np.asarray(z["dates"][:]))
    print(f"zarr {args.zarr}: ensemble {z['ensemble'].shape} chunks={z['ensemble'].chunks} "
          f"dates {dates.min().date()}..{dates.max().date()}")

    split = json.loads((args.gnn_root / "weights" / "checkpoints_v24" / "finetune" / "split_metadata.json").read_text())
    hold = [int(x) for x in split["eval_comids"]]
    comid_to_row = {int(c): i for i, c in enumerate(comid_arr)}
    hold = [c for c in hold if c in comid_to_row]
    rows = [comid_to_row[c] for c in hold]
    print(f"holdout gauges on graph: {len(hold)}")

    mean_h, ens_h = gather_holdout(z, rows)
    print(f"gathered mean {mean_h.shape}  ensemble {ens_h.shape}")

    summaries = {}
    for name, (a, b) in PERIODS.items():
        obs_path = args.gnn_root / "outputs" / OBS_FILES[name]
        obs = load_obs_parquet(obs_path, set(hold))
        print(f"{name} obs rows={len(obs)} basins={obs.comid.nunique()} from {obs_path.name}")
        df = score_period(mean_h, ens_h, hold, dates, obs, a, b, name)
        df.to_csv(args.outdir / f"head3_mdn_{name}_per_basin.csv", index=False)
        summaries[f"head3_mdn_{name}"] = summarize(df)
        summaries[f"head3_mdn_{name}"]["n_rows"] = int(len(df))
        summaries[f"head3_mdn_{name}"]["protocol"] = "full_period"
        s = summaries[f"head3_mdn_{name}"]
        print(f"  {name}: n={len(df)} KGE={s.get('median_kge')} "
              f"r={s.get('median_r')} cov95={s.get('median_cov95')}")
    with open(args.outdir / "head3_mdn_fullperiod_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    _merge_table(args.outdir, summaries)
    print("wrote", args.outdir)


if __name__ == "__main__":
    main()
