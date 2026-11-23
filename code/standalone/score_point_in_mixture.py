#!/usr/bin/env python3
"""How often the ungauged prediction falls inside the mixture.

For each holdout catchment-day, compare the deterministic ungauged
prediction with the mixture quantiles on that day. Both series are
specific discharge in mm/day. A day counts only when the observation and
the ungauged prediction are finite and non-negative and the two series
share that date.

Outputs one row per catchment:
  frac_in_50  fraction of days inside the mixture 25th-75th percentiles
  frac_in_80  fraction of days inside the mixture 10th-90th percentiles
  median_abs_pit  median |F(ungauged) - 0.5|
  width_norm  mean (q90 - q10) / mean observed flow
  mae_to_median_norm  mean |ungauged - mixture median| / mean observed flow
  mae_to_mean_norm    mean |ungauged - mixture mean| / mean observed flow
  bias_to_median_norm mean (ungauged - mixture median) / mean observed flow
  mae_to_median_over_width  mean |ungauged - median| / (q90 - q10)
"""

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
OUT = ROOT / "outputs" / "standalone"
sys.path.insert(0, str(ROOT / "code" / "training"))
sys.path.insert(0, str(ROOT / "code" / "evaluation"))

from config import get_config_v24  # noqa: E402
from evaluate_d2 import D2Evaluator, tile_period_starts  # noqa: E402
from model_d2 import mixture_mean, mixture_pit, mixture_quantiles  # noqa: E402
from train_multi_gpu_v15 import (  # noqa: E402
    build_date_list, cleanup_shared_memory, is_main_process, log,
    setup_distributed,
)
from train_multi_gpu_v20 import SequenceDataModuleV20  # noqa: E402

LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90)
MIN_DAYS = 30


def node_to_comid():
    meta = json.loads(
        (ROOT / "weights/checkpoints_v24/finetune/split_metadata.json").read_text()
    )
    mapping = {
        int(n): int(c)
        for n, c in zip(meta["eval_node_indices"], meta["eval_comids"])
    }
    mapping.update({
        int(n): int(c)
        for n, c in zip(meta["train_node_indices"], meta["train_comids"])
    })
    return mapping


def load_point(path):
    frame = pd.read_parquet(path, columns=["comid", "date", "pred_mm_day"])
    frame["date"] = pd.to_datetime(frame["date"]).values.astype("datetime64[D]")
    frame = frame.dropna(subset=["pred_mm_day"])
    frame = frame[frame["pred_mm_day"] >= 0]
    tables = {}
    for comid, group in frame.groupby("comid", sort=False):
        group = group.drop_duplicates("date", keep="first").sort_values("date")
        tables[int(comid)] = (
            group["date"].to_numpy(dtype="datetime64[D]"),
            group["pred_mm_day"].to_numpy(np.float32),
        )
    return tables


def lookup_point(table, dates):
    """Return predictions aligned to dates. Missing dates are NaN."""
    pred = np.full(len(dates), np.nan, dtype=np.float32)
    if table is None or len(dates) == 0:
        return pred
    known, values = table
    idx = np.searchsorted(known, dates)
    ok = idx < len(known)
    matched = np.zeros(len(dates), dtype=bool)
    matched[ok] = known[idx[ok]] == dates[ok]
    pred[matched] = values[idx[matched]]
    return pred


class InclusionEvaluator(D2Evaluator):
    def score(self, eval_dates, nodes, point_tables, mapping, band_nodes, band_start, band_end):
        self.model.eval()
        gauged = [int(n) for n in nodes]
        index = {node: i for i, node in enumerate(gauged)}
        n_basins = len(gauged)
        t0 = pd.Timestamp(min(eval_dates)).normalize()
        t1 = pd.Timestamp(max(eval_dates)).normalize()
        n_span = (t1 - t0).days + 1
        seen = np.zeros((n_basins, n_span), dtype=bool)
        n_days = np.zeros(n_basins, dtype=np.int32)
        in50 = np.zeros(n_basins, dtype=np.int32)
        in80 = np.zeros(n_basins, dtype=np.int32)
        below = np.zeros(n_basins, dtype=np.int32)
        above = np.zeros(n_basins, dtype=np.int32)
        abs_pit = np.full((n_basins, n_span), np.nan, dtype=np.float32)
        width_sum = np.zeros(n_basins, dtype=np.float64)
        obs_sum = np.zeros(n_basins, dtype=np.float64)
        abs_median_sum = np.zeros(n_basins, dtype=np.float64)
        abs_mean_sum = np.zeros(n_basins, dtype=np.float64)
        signed_median_sum = np.zeros(n_basins, dtype=np.float64)
        rel_width_sum = np.zeros(n_basins, dtype=np.float64)
        rel_width_n = np.zeros(n_basins, dtype=np.int32)
        band_rows = []
        band_nodes = set(int(n) for n in band_nodes)
        starts = tile_period_starts(eval_dates, self.seq_length)
        logged_window = False

        with torch.no_grad():
            for start_date in starts:
                for batch_start in range(0, len(gauged), self.batch_size):
                    requested = gauged[batch_start:batch_start + self.batch_size]
                    subgraph = self.data_module.get_k_hop_neighbors(
                        requested, k=3, max_nodes=self.max_subgraph_nodes)
                    present = set(int(n) for n in subgraph)
                    kept = [node for node in requested if node in present]
                    if not kept:
                        continue
                    result = self._prepare_sequence_batch(kept, start_date)
                    if result is None:
                        continue
                    batch_data, target_flows = result
                    if target_flows.shape[0] != len(kept):
                        raise RuntimeError(
                            f"{start_date.date()} kept {len(kept)} nodes but "
                            f"the batch returned {target_flows.shape[0]}")
                    target_flows = target_flows.to(self.device)
                    pi, mu, sigma = self._forward_sequence(batch_data, self.device)
                    quant = mixture_quantiles(pi, mu, sigma, levels=list(LEVELS))
                    center = mixture_mean(pi, mu, sigma)
                    obs = target_flows.float().cpu().numpy()
                    quant = quant.float().cpu().numpy()
                    center = center.float().cpu().numpy()
                    seq_dates = pd.date_range(start_date, periods=obs.shape[1], freq="D")
                    date64 = seq_dates.values.astype("datetime64[D]")
                    point = np.vstack([
                        lookup_point(point_tables.get(mapping.get(node)), date64)
                        for node in kept
                    ])
                    point_t = torch.as_tensor(point, device=self.device)
                    pit = mixture_pit(point_t, pi, mu, sigma).float().cpu().numpy()
                    if not logged_window:
                        finite_obs = np.isfinite(obs).sum()
                        matched = np.isfinite(point).sum()
                        log(f"  first window {start_date.date()}: "
                            f"matched ungauged days {int(matched)} / "
                            f"finite observations {int(finite_obs)}")
                        if finite_obs > 0 and matched == 0:
                            raise RuntimeError("Ungauged predictions did not match mixture dates")
                        logged_window = True
                    self._accumulate(
                        kept, index, seq_dates, date64, obs, point, quant, center, pit,
                        t0, t1, seen, n_days, in50, in80, below, above,
                        abs_pit, width_sum, obs_sum, abs_median_sum, abs_mean_sum,
                        signed_median_sum, rel_width_sum, rel_width_n, band_nodes,
                        band_start, band_end, band_rows)
        rows = []
        for node, i in index.items():
            n = int(n_days[i])
            if n < MIN_DAYS:
                continue
            rows.append({
                "node": node,
                "comid": int(mapping[node]),
                "n_days": n,
                "frac_in_50": float(in50[i] / n),
                "frac_in_80": float(in80[i] / n),
                "frac_below_10": float(below[i] / n),
                "frac_above_90": float(above[i] / n),
                "median_abs_pit": float(np.nanmedian(abs_pit[i])),
                "width_norm": float(width_sum[i] / obs_sum[i]) if obs_sum[i] > 0 else np.nan,
                "mae_to_median_norm": (
                    float(abs_median_sum[i] / obs_sum[i]) if obs_sum[i] > 0 else np.nan),
                "mae_to_mean_norm": (
                    float(abs_mean_sum[i] / obs_sum[i]) if obs_sum[i] > 0 else np.nan),
                "bias_to_median_norm": (
                    float(signed_median_sum[i] / obs_sum[i]) if obs_sum[i] > 0 else np.nan),
                "mae_to_median_over_width": (
                    float(rel_width_sum[i] / rel_width_n[i]) if rel_width_n[i] else np.nan),
            })
        return pd.DataFrame(rows), pd.DataFrame(band_rows)

    def _accumulate(self, kept, index, seq_dates, date64, obs, point, quant, center, pit,
                    t0, t1, seen, n_days, in50, in80, below, above, abs_pit,
                    width_sum, obs_sum, abs_median_sum, abs_mean_sum, signed_median_sum,
                    rel_width_sum, rel_width_n, band_nodes, band_start, band_end, band_rows):
        for i, node in enumerate(kept):
            basin = index[node]
            for t, date in enumerate(seq_dates):
                if date < t0 or date > t1:
                    continue
                offset = (date - t0).days
                if seen[basin, offset]:
                    continue
                observed = obs[i, t]
                predicted = point[i, t]
                if not np.isfinite(observed) or observed < 0:
                    continue
                if not np.isfinite(predicted):
                    continue
                q10, q25, q50, q75, q90 = quant[i, t]
                center_flow = center[i, t]
                if not np.isfinite([q10, q25, q50, q75, q90, center_flow]).all():
                    continue
                seen[basin, offset] = True
                n_days[basin] += 1
                in50[basin] += int(q25 <= predicted <= q75)
                inside80 = bool(q10 <= predicted <= q90)
                in80[basin] += int(inside80)
                below[basin] += int(predicted < q10)
                above[basin] += int(predicted > q90)
                gap_median = float(predicted - q50)
                abs_median_sum[basin] += abs(gap_median)
                signed_median_sum[basin] += gap_median
                abs_mean_sum[basin] += abs(float(predicted - center_flow))
                width = float(q90 - q10)
                if width > 1e-8:
                    rel_width_sum[basin] += abs(gap_median) / width
                    rel_width_n[basin] += 1
                value = pit[i, t]
                if np.isfinite(value):
                    abs_pit[basin, offset] = abs(float(value) - 0.5)
                width_sum[basin] += width
                obs_sum[basin] += float(observed)
                if node in band_nodes and band_start <= date <= band_end:
                    band_rows.append({
                        "node": int(node),
                        "date": date,
                        "q10_mm_day": float(q10),
                        "q90_mm_day": float(q90),
                        "obs_mm_day": float(observed),
                    })


def evaluate_one(config, data, checkpoint, years, point_path, mapping, band_nodes,
                 band_start, band_end, collect_bands):
    n_comp = torch.load(checkpoint, map_location="cpu", weights_only=False).get(
        "n_components", 3)
    evaluator = InclusionEvaluator(
        config=config, data_module=data, init_checkpoint=str(checkpoint),
        stage="finetune", seq_length=180, warmup_days=365,
        sequences_per_epoch=1, gauge_dropout_rate=0.0,
        checkpoint_dir=str(OUT / "_within_eval_tmp"), n_components=n_comp,
    )
    dates = build_date_list(years)
    nodes = data.eval_gauged_nodes
    log(f"Scoring {Path(checkpoint).parent.name} on {years[0]}-{years[-1]} "
        f"({len(nodes)} basins, {len(dates)} days)")
    frame, bands = evaluator.score(
        dates, nodes, load_point(point_path), mapping, band_nodes,
        band_start, band_end if collect_bands else band_start - pd.Timedelta(days=1),
    )
    del evaluator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return frame, bands


def main():
    setup_distributed()
    if not is_main_process():
        return
    mapping = node_to_comid()
    spec = json.loads((EXP / "hydrograph_catchments.json").read_text())
    band_nodes = [int(row["node"]) for row in spec["catchments"]]
    band_start = pd.Timestamp(spec["start"])
    band_end = pd.Timestamp(spec["end"])
    config = get_config_v24()
    config.data.split_mode = "basin"
    data = SequenceDataModuleV20(config, rank=0, world_size=1, warmup_days=365)
    shared_ckpt = ROOT / "weights/checkpoints_d2/finetune_frozen/best_model.pt"
    task_ckpt = OUT / "checkpoints" / "mdn" / "best_model.pt"
    runs = [
        ("2020-2024", list(config.data.test_years), "shared", shared_ckpt,
         OUT / "point_paper_h2_reeval_test.parquet", True),
        ("2020-2024", list(config.data.test_years), "standalone", task_ckpt,
         OUT / "point_test.parquet", True),
        ("2015-2024", list(range(2015, 2025)), "shared", shared_ckpt,
         OUT / "point_2015_2024_shared.parquet", False),
        ("2015-2024", list(range(2015, 2025)), "standalone", task_ckpt,
         OUT / "point_2015_2024_standalone.parquet", False),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    band_frames = []
    for label, years, model, checkpoint, point_path, collect_bands in runs:
        frame, bands = evaluate_one(
            config, data, checkpoint, years, point_path, mapping, band_nodes,
            band_start, band_end, collect_bands)
        out = OUT / f"within_mixture_{label}_{model}.csv"
        frame.to_csv(out, index=False)
        log(f"  wrote {out} basins={len(frame)} "
            f"median in80={frame['frac_in_80'].median():.3f} "
            f"median in50={frame['frac_in_50'].median():.3f} "
            f"mae/median={frame['mae_to_median_norm'].median():.3f} "
            f"mae/mean={frame['mae_to_mean_norm'].median():.3f}")
        if collect_bands and not bands.empty:
            bands["model"] = model
            band_frames.append(bands)
    if band_frames:
        bands = pd.concat(band_frames, ignore_index=True)
        bands = bands.sort_values(["model", "node", "date"]).drop_duplicates(
            ["model", "node", "date"], keep="first")
        band_path = OUT / "within_mixture_bands_wy2022.csv"
        bands.to_csv(band_path, index=False)
        log(f"  wrote {band_path} rows={len(bands)}")
    log("DONE within-mixture scoring")
    cleanup_shared_memory()


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_shared_memory()
