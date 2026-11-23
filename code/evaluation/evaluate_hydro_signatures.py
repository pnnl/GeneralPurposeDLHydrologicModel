#!/usr/bin/env python3
"""Score KGE parts and hydrologic signatures vs observations.

Three products, test period 2020-2024:
  head1_gauged   : v24 pretrain  on TRAIN basins
  head2_pub      : v24 finetune  on HOLDOUT basins
  head3_mdn      : D2 frozen mixture mean on HOLDOUT basins
                   (optional daily parquet; otherwise skip head 3)

Usage, from the bundle root:
  python code/evaluation/evaluate_hydro_signatures.py --gnn-root .
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hydro_signatures import hydro_signatures, summarize  # noqa: E402

MIN_N = 30


def _load_split(split_json: Path):
    meta = json.loads(split_json.read_text())
    train = set(int(x) for x in meta["train_comids"])
    hold = set(int(x) for x in meta["eval_comids"])
    return train, hold


def score_parquet(path: Path, keep_comids: set, pred_col="pred_cms", obs_col="obs_cms"):
    df = pd.read_parquet(path, columns=["date", "comid", pred_col, obs_col])
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["comid"].isin(keep_comids)]
    df = df.dropna(subset=[obs_col, pred_col])
    df = df[df[obs_col] >= 0]
    rows = []
    for comid, g in df.groupby("comid", sort=False):
        if len(g) < MIN_N:
            continue
        m = hydro_signatures(
            g[obs_col].to_numpy(),
            g[pred_col].to_numpy(),
            dates=g["date"].to_numpy(),
        )
        if not np.isfinite(m.get("kge", np.nan)):
            continue
        m["comid"] = int(comid)
        rows.append(m)
    return pd.DataFrame(rows)


def score_daily_mdn(path: Path):
    df = pd.read_parquet(path)
    pred_col = next(c for c in ("pred_cms", "mean_cms", "pred", "sim") if c in df.columns)
    obs_col = next(c for c in ("obs_cms", "obs", "flow_cms") if c in df.columns)
    id_col = "comid" if "comid" in df.columns else "node"
    has_date = "date" in df.columns
    rows = []
    for key, g in df.groupby(id_col, sort=False):
        if len(g) < MIN_N:
            continue
        dates = g["date"].to_numpy() if has_date else None
        m = hydro_signatures(g[obs_col].to_numpy(), g[pred_col].to_numpy(), dates=dates)
        if not np.isfinite(m.get("kge", np.nan)):
            continue
        m["comid"] = int(key)
        rows.append(m)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn-root", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--mdn-daily", type=Path, default=None,
                    help="Optional daily parquet for the frozen MDN (head 3)")
    args = ap.parse_args()
    root = args.gnn_root
    outdir = args.outdir or (root / "outputs" / "eval_hydro")
    outdir.mkdir(parents=True, exist_ok=True)

    split = root / "weights" / "checkpoints_v24" / "finetune" / "split_metadata.json"
    train, hold = _load_split(split)
    print(f"split: {len(train)} train / {len(hold)} holdout comids")

    jobs = [
        ("head1_gauged_test", root / "outputs" / "predictions_v24_pretrain_test.parquet", train),
        ("head2_pub_test", root / "outputs" / "predictions_v24_finetune_test.parquet", hold),
        ("head1_gauged_val", root / "outputs" / "predictions_v24_pretrain_trainbasins_val.parquet", train),
        ("head2_pub_val", root / "outputs" / "predictions_v24_val.parquet", hold),
    ]
    summaries = {}
    for name, path, keep in jobs:
        if not path.exists():
            print(f"MISSING {path}")
            continue
        print(f"scoring {name} from {path.name} ...")
        df = score_parquet(path, keep)
        df.to_csv(outdir / f"{name}_per_basin.csv", index=False)
        summaries[name] = summarize(df)
        summaries[name]["n_rows"] = int(len(df))
        print(f"  {len(df)} basins  median KGE={summaries[name].get('median_kge'):.3f}  "
              f"r={summaries[name].get('median_r'):.3f}  "
              f"alpha={summaries[name].get('median_alpha'):.3f}  "
              f"beta={summaries[name].get('median_beta'):.3f}")

    mdn_daily = args.mdn_daily
    if mdn_daily is None:
        cand = root / "outputs" / "eval_d2_finetune_frozen_heldout_test_daily.parquet"
        if cand.exists():
            mdn_daily = cand
    if mdn_daily is not None and Path(mdn_daily).exists():
        print(f"scoring head3_mdn from {mdn_daily} ...")
        df = score_daily_mdn(Path(mdn_daily))
        df.to_csv(outdir / "head3_mdn_per_basin.csv", index=False)
        summaries["head3_mdn"] = summarize(df)
        print(f"  {len(df)} basins  median KGE={summaries['head3_mdn'].get('median_kge'):.3f}")
    else:
        print("head3_mdn daily parquet not found; GPU eval will write it")

    with open(outdir / "hydro_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    # wide table for the paper
    rows = []
    for name, s in summaries.items():
        row = {"product": name}
        row.update(s)
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(outdir / "hydro_three_head_table.csv", index=False)
    print(f"wrote {outdir}")


if __name__ == "__main__":
    main()
