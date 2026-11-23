#!/usr/bin/env python3
"""Export held-out evaluation catchment COMIDs from split_metadata.json to CSV."""

import csv
import json
from pathlib import Path

here = Path(__file__).resolve().parent
bundle = here.parents[1]
meta = json.loads((bundle / "weights" / "checkpoints_v24" / "finetune" /
                   "split_metadata.json").read_text())
out = bundle / "outputs" / "standalone" / "heldout_comids.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["comid"])
    writer.writerows([[int(comid)] for comid in meta["eval_comids"]])
print(f"Wrote {len(meta['eval_comids'])} holdout COMIDs to {out}")
