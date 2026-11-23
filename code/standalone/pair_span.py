#!/usr/bin/env python3
"""Paired catchment medians for one continuous evaluation span."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

here = Path(__file__).resolve().parent
bundle = here.parents[1]
sys.path.insert(0, str(bundle / "code" / "evaluation"))
from build_3head_table import per_basin_metrics  # noqa: E402


def holdouts():
    metadata = json.loads((bundle / "weights" / "checkpoints_v24" / "finetune" /
                           "split_metadata.json").read_text())
    return set(int(x) for x in metadata["eval_comids"])


def write_report(path, report):
    path = Path(path)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def pair_point(args):
    kept = holdouts()
    standalone = per_basin_metrics(args.standalone, kept)
    shared = per_basin_metrics(args.shared, kept)
    paired = standalone.merge(shared, on="comid", suffixes=("_standalone", "_shared"))
    if paired.empty:
        raise RuntimeError("No paired point catchments")
    paired.to_csv(args.paired_csv, index=False)
    report = {
        "protocol": "inference_v24_v25.py then build_3head_table.per_basin_metrics",
        "years": args.years,
        "holdout_basins": len(kept),
        "standalone_valid_basins": len(standalone),
        "shared_valid_basins": len(shared),
        "paired_basins": len(paired),
        "paired_median_kge_standalone": float(paired["kge_standalone"].median()),
        "paired_median_kge_shared": float(paired["kge_shared"].median()),
        "median_paired_delta_kge": float(
            (paired["kge_standalone"] - paired["kge_shared"]).median()),
        "paired_median_peak_bias_standalone": float(paired["peak_bias_standalone"].median()),
        "paired_median_peak_bias_shared": float(paired["peak_bias_shared"].median()),
    }
    write_report(args.output, report)


def pair_mdn(args):
    standalone = pd.read_csv(args.standalone).drop_duplicates("node")
    shared = pd.read_csv(args.shared).drop_duplicates("node")
    paired = standalone.merge(shared, on="node", suffixes=("_standalone", "_shared"))
    if paired.empty:
        raise RuntimeError("No paired mixture catchments")
    paired.to_csv(args.paired_csv, index=False)

    def med(column):
        return float(paired[column].median())

    report = {
        "protocol": "evaluate_d2.py on one continuous year span, held-out basins",
        "years": args.years,
        "paired_basins": len(paired),
        "paired_median_kge_standalone": med("kge_standalone"),
        "paired_median_kge_shared": med("kge_shared"),
        "median_paired_delta_kge": float(
            (paired.kge_standalone - paired.kge_shared).median()),
        "paired_median_crps_norm_standalone": med("crps_norm_standalone"),
        "paired_median_crps_norm_shared": med("crps_norm_shared"),
        "median_paired_delta_crps_norm": float(
            (paired.crps_norm_standalone - paired.crps_norm_shared).median()),
        "paired_median_cov95_standalone": med("cov95_standalone"),
        "paired_median_cov95_shared": med("cov95_shared"),
        "paired_median_peak_cov99_standalone": med("peak_cov99_standalone"),
        "paired_median_peak_cov99_shared": med("peak_cov99_shared"),
    }
    write_report(args.output, report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("point", "mdn"))
    parser.add_argument("--standalone", required=True)
    parser.add_argument("--shared", required=True)
    parser.add_argument("--years", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--paired-csv", required=True)
    args = parser.parse_args()
    if args.kind == "point":
        pair_point(args)
    else:
        pair_mdn(args)


if __name__ == "__main__":
    main()
