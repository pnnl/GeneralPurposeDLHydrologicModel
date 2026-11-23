#!/usr/bin/env python3
"""Save daily mixture means for the two hydrograph catchments."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
OUT = ROOT / "outputs" / "standalone"
sys.path.insert(0, str(ROOT / "code" / "training"))
sys.path.insert(0, str(ROOT / "code" / "evaluation"))

from config import get_config_v24  # noqa: E402
from evaluate_d2 import D2Evaluator, tile_period_starts  # noqa: E402
from train_multi_gpu_v15 import setup_distributed  # noqa: E402
from train_multi_gpu_v20 import SequenceDataModuleV20  # noqa: E402


class DailyEvaluator(D2Evaluator):
    def daily(self, eval_dates, nodes):
        self.model.eval()
        gauged = np.array(nodes, dtype=np.int64)
        starts = tile_period_starts(eval_dates, self.seq_length)
        records = []
        import torch
        from model_d2 import mixture_mean
        with torch.no_grad():
            for node in gauged.tolist():
                for start_date in starts:
                    result = self._prepare_sequence_batch([node], start_date)
                    if result is None:
                        continue
                    batch_data, target_flows = result
                    target_flows = target_flows.to(self.device)
                    pi, mu, sigma = self._forward_sequence(batch_data, self.device)
                    mean = mixture_mean(pi, mu, sigma)
                    obs = target_flows.float().cpu().numpy()
                    pred = mean.float().cpu().numpy()
                    if obs.shape[0] != 1:
                        raise RuntimeError(
                            f"Node {node} kept {obs.shape[0]} rows on {start_date.date()}")
                    seq_dates = pd.date_range(start_date, periods=obs.shape[1], freq="D")
                    for t, date in enumerate(seq_dates):
                        if date < eval_dates[0] or date > eval_dates[-1]:
                            continue
                        if not np.isfinite(obs[0, t]):
                            continue
                        records.append({
                            "node": int(node),
                            "date": date,
                            "obs_mm_day": float(obs[0, t]),
                            "pred_mm_day": float(pred[0, t]),
                        })
        frame = pd.DataFrame(records)
        if frame.empty:
            raise RuntimeError("No daily mixture rows")
        frame = frame.sort_values(["node", "date"]).drop_duplicates(
            ["node", "date"], keep="first")
        return frame


def main():
    spec = json.loads((EXP / "hydrograph_catchments.json").read_text())
    nodes = [int(row["node"]) for row in spec["catchments"]]
    dates = list(pd.date_range(spec["start"], spec["end"], freq="D"))
    checkpoints = {
        "shared": ROOT / "weights/checkpoints_d2/finetune_frozen/best_model.pt",
        "standalone": OUT / "checkpoints" / "mdn" / "best_model.pt",
    }
    setup_distributed()
    config = get_config_v24()
    config.data.split_mode = "basin"
    data = SequenceDataModuleV20(config, rank=0, world_size=1, warmup_days=365)
    frames = []
    for label, path in checkpoints.items():
        evaluator = DailyEvaluator(
            config=config, data_module=data, init_checkpoint=str(path),
            stage="finetune", seq_length=180, warmup_days=365,
            sequences_per_epoch=1, gauge_dropout_rate=0.0,
            checkpoint_dir=str(OUT / "_hydro_eval_tmp"), n_components=3,
        )
        frame = evaluator.daily(dates, nodes)
        frame["model"] = label
        frames.append(frame)
        del evaluator
    out = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "hydrograph_mixture_wy2022.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} rows={len(out)}")


if __name__ == "__main__":
    main()
