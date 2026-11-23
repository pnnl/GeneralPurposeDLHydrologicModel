#!/usr/bin/env python3
"""Compute paired deterministic performance metrics on held-out basins."""

import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
bundle = here.parents[1]
out = bundle / "outputs" / "standalone"
sys.path.insert(0, str(bundle / "code" / "evaluation"))
from build_3head_table import per_basin_metrics

metadata = json.loads((bundle / "weights" / "checkpoints_v24" / "finetune" /
                       "split_metadata.json").read_text())
heldout = set(int(x) for x in metadata["eval_comids"])
out.mkdir(parents=True, exist_ok=True)
new = per_basin_metrics(out / "point_test.parquet", heldout)
paper = per_basin_metrics(out / "point_paper_h2_reeval_test.parquet", heldout)
new.to_csv(out / "point_standalone_per_basin.csv", index=False)
paper.to_csv(out / "point_paper_h2_per_basin.csv", index=False)
paired = new.merge(paper, on="comid", suffixes=("_standalone", "_paper"))
if paired.empty:
    raise RuntimeError("No paired point basins")
report = {
    "protocol": "same v24 test inference and build_3head_table.per_basin_metrics for both checkpoints",
    "years": "2020-2024", "holdout_basins": len(heldout),
    "standalone_valid_basins": len(new), "paper_valid_basins": len(paper),
    "paired_basins": len(paired),
    "paired_median_kge_standalone": float(paired["kge_standalone"].median()),
    "paired_median_kge_paper": float(paired["kge_paper"].median()),
    "median_paired_delta_kge": float((paired["kge_standalone"] - paired["kge_paper"]).median()),
    "paired_median_peak_bias_standalone": float(paired["peak_bias_standalone"].median()),
    "paired_median_peak_bias_paper": float(paired["peak_bias_paper"].median()),
}
(out / "point_comparison.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
