"""
Evaluate a trained Mixture Density Network (MDN) checkpoint on the test or validation period.

Computes three groups of per-basin metrics:
  1. Point accuracy (closed-form mixture mean):
        per-basin KGE (and r, alpha, beta), NSE, and top-2% peak flow bias
  2. Probabilistic calibration:
        empirical PIT coverage at nominal P90 / P95 / P99 levels
  3. High-flow tail coverage:
        fraction of top-2% observed peak-flow days bracketed by the P95 / P99
        mixture quantiles, plus Continuous Ranked Probability Score (CRPS).

Usage (from the repository root):
    torchrun --standalone --nproc_per_node=8 code/evaluation/evaluate_d2.py \
        --checkpoint weights/checkpoints_d2/finetune_frozen/best_model.pt \
        --basin-set heldout --years test --output outputs/eval_d2_finetune_frozen_heldout_test.csv

  --basin-set train    : training basins (temporal generalization)
  --basin-set heldout  : held-out ungauged basins (spatial generalization)
  --years     test     : 2020-2024 (default) | val : 2015-2019
"""

import argparse
import json
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BUNDLE / "code" / "training"))

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm

from train_multi_gpu_v15 import (
    setup_distributed, is_main_process, sync_if_distributed, log,
    build_date_list, cleanup_shared_memory,
)
from train_multi_gpu_v20 import SequenceDataModuleV20
from train_multi_gpu_d2 import D2Trainer, PEAK_FRAC
from model_d2 import (
    mixture_mean, mixture_pit, mixture_quantiles, mixture_crps,
    QUANTILES, QUANTILE_LABELS,
)
from config import get_config_v24

EPS = 1e-8


def tile_period_starts(eval_dates, seq_length):
    """Successive windows that cover every day of the eval period.

    Same tiling as inference_d2_allcatch.py. A final window is aligned to
    the last eval date so the tail is not dropped. Callers must keep
    unique (basin, date) rows when windows overlap.
    """
    t_start = pd.Timestamp(min(eval_dates))
    t_end = pd.Timestamp(max(eval_dates))
    step = pd.Timedelta(days=int(seq_length))
    last_full = t_end - pd.Timedelta(days=int(seq_length) - 1)
    starts = []
    cursor = t_start
    while cursor <= last_full:
        starts.append(cursor)
        cursor += step
    if starts and starts[-1] < last_full:
        starts.append(last_full)
    elif not starts and last_full >= t_start:
        starts.append(t_start)
    return starts


class D2Evaluator(D2Trainer):
    """Adds a full-period evaluation pass that keeps per-basin time series."""

    def evaluate(self, eval_dates, eval_basin_nodes):
        self.model.eval()
        gauged = np.array(eval_basin_nodes, dtype=np.int64)
        starts = tile_period_starts(eval_dates, self.seq_length)
        log(f"  tiled starts: {len(starts)} windows covering "
            f"{min(eval_dates).date()} .. {max(eval_dates).date()} "
            f"(not random {self.seq_length}-day samples)")
        my_dates = starts[self.rank::self.world_size]

        # per-basin collections
        b_obs, b_mean, b_pit, b_q, b_crps, b_dates = {}, {}, {}, {}, {}, {}
        # indices of the P95 / P99 levels in the QUANTILES list
        i95 = QUANTILES.index(0.95)
        i99 = QUANTILES.index(0.99)

        with torch.no_grad():
            for start_date in tqdm(my_dates, desc='  Evaluating',
                                   disable=not is_main_process()):
                for bs in range(0, len(gauged), self.batch_size):
                    nodes = gauged[bs:min(bs + self.batch_size, len(gauged))].tolist()
                    result = self._prepare_sequence_batch(nodes, start_date)
                    if result is None:
                        continue
                    batch_data, target_flows = result
                    target_flows = target_flows.to(self.device)
                    pi, mu, sigma = self._forward_sequence(batch_data, self.device)

                    mean = mixture_mean(pi, mu, sigma)                 # [seeds, seq]
                    pit = mixture_pit(target_flows, pi, mu, sigma)     # [seeds, seq]
                    quant = mixture_quantiles(pi, mu, sigma)           # [seeds, seq, Q]
                    crps = mixture_crps(target_flows, pi, mu, sigma)   # [seeds, seq]

                    obs_np = target_flows.cpu().numpy()
                    mean_np = mean.float().cpu().numpy()
                    pit_np = pit.float().cpu().numpy()
                    crps_np = crps.float().cpu().numpy()
                    q95 = quant[..., i95].float().cpu().numpy()
                    q99 = quant[..., i99].float().cpu().numpy()
                    seq_dates = pd.date_range(start_date, periods=obs_np.shape[1], freq="D")
                    for i, nd in enumerate(nodes):
                        m = np.isfinite(obs_np[i])
                        if m.any():
                            b_obs.setdefault(nd, []).append(obs_np[i][m])
                            b_mean.setdefault(nd, []).append(mean_np[i][m])
                            b_pit.setdefault(nd, []).append(pit_np[i][m])
                            b_crps.setdefault(nd, []).append(crps_np[i][m])
                            b_q.setdefault(nd, []).append(
                                np.stack([q95[i][m], q99[i][m]], axis=-1))
                            b_dates.setdefault(nd, []).append(seq_dates[m].to_numpy())

        rows = []
        for nd in b_obs:
            obs = np.concatenate(b_obs[nd])
            pred = np.concatenate(b_mean[nd])
            pit = np.concatenate(b_pit[nd])
            crps = np.concatenate(b_crps[nd])
            q = np.concatenate(b_q[nd], axis=0)                        # [N, 2] -> P95,P99
            dates = np.concatenate(b_dates[nd]) if nd in b_dates else None
            if dates is not None and len(dates) == len(obs):
                _, keep = np.unique(dates, return_index=True)
                keep = np.sort(keep)
                obs, pred, pit, crps, q = (
                    obs[keep], pred[keep], pit[keep], crps[keep], q[keep])
            if len(obs) < 30 or np.std(obs) < 1e-4:
                continue
            ps, osd = np.std(pred), np.std(obs)
            r = np.corrcoef(pred, obs)[0, 1] if ps > EPS else 0.0
            if not np.isfinite(r):
                r = 0.0
            alpha = ps / (osd + EPS)
            beta = np.mean(pred) / (np.mean(obs) + EPS)
            kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
            nse = 1 - np.sum((pred - obs) ** 2) / (np.sum((obs - obs.mean()) ** 2) + EPS)

            n_peak = max(1, int(round(PEAK_FRAC * len(obs))))
            top = np.argsort(obs)[-n_peak:]
            ot, pt = obs[top].mean(), pred[top].mean()
            peak_bias = (pt - ot) / ot if ot > EPS else np.nan
            # does the upper tail BRACKET the peak days?
            peak_cov95 = float(np.mean(obs[top] <= q[top, 0]))
            peak_cov99 = float(np.mean(obs[top] <= q[top, 1]))

            rows.append(dict(
                node=int(nd), n=int(len(obs)),
                kge=float(kge), r=float(r), alpha=float(alpha), beta=float(beta),
                nse=float(nse), peak_bias=float(peak_bias),
                cov90=float(np.mean(pit <= 0.90)),
                cov95=float(np.mean(pit <= 0.95)),
                cov99=float(np.mean(pit <= 0.99)),
                peak_cov95=peak_cov95, peak_cov99=peak_cov99,
                crps=float(np.mean(crps)),
                # CRPS scales with flow magnitude; normalize by mean obs so it is
                # comparable across basins (a relative skill score, like nRMSE).
                crps_norm=float(np.mean(crps) / (obs.mean() + EPS)),
                obs_mean=float(obs.mean()),
            ))

        # gather rows across ranks to rank 0
        all_rows = [None] * self.world_size if self.world_size > 1 else [rows]
        if self.world_size > 1:
            dist.all_gather_object(all_rows, rows)
        flat = [r for chunk in all_rows for r in chunk]
        return flat


def main():
    ap = argparse.ArgumentParser(description='Evaluate D2 MDN checkpoint on test period')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--basin-set', choices=['train', 'heldout'], default=None,
                    help='train = in-sample basins; heldout = PUB held-out basins. '
                         'Default: heldout if ckpt stage==finetune else train.')
    ap.add_argument('--years', choices=['test', 'val'], default='test')
    ap.add_argument('--start-year', type=int, default=None,
                    help='First year of one continuous span. Set with --end-year.')
    ap.add_argument('--end-year', type=int, default=None,
                    help='Last year of one continuous span. Set with --start-year.')
    ap.add_argument('--seq-length', type=int, default=180)
    ap.add_argument('--warmup-days', type=int, default=365)
    ap.add_argument('-o', '--output', default=None)
    args = ap.parse_args()

    if not Path(args.checkpoint).exists():
        ap.error(f"checkpoint not found: {args.checkpoint}")

    rank, world_size, local_rank = setup_distributed()

    # read stage from checkpoint to choose a sensible default basin set
    ckpt_meta = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    ckpt_stage = ckpt_meta.get('stage', 'pretrain')
    n_comp = ckpt_meta.get('n_components', 3)
    del ckpt_meta
    basin_set = args.basin_set or ('heldout' if ckpt_stage == 'finetune' else 'train')

    log("=" * 60)
    log(f"D2 EVALUATION  |  ckpt: {args.checkpoint}")
    log(f"  ckpt stage: {ckpt_stage} | components: {n_comp}")
    log(f"  basin set: {basin_set} | years: {args.years}")
    log("=" * 60)

    config = get_config_v24()
    config.data.split_mode = 'basin'
    data_module = SequenceDataModuleV20(config, rank=rank, world_size=world_size,
                                        warmup_days=args.warmup_days)

    if (args.start_year is None) != (args.end_year is None):
        ap.error('--start-year and --end-year must be set together')
    if args.start_year is not None:
        if args.end_year < args.start_year:
            ap.error('--end-year is before --start-year')
        years = list(range(args.start_year, args.end_year + 1))
    else:
        years = config.data.test_years if args.years == 'test' else config.data.val_years
    eval_dates = build_date_list(years)
    if basin_set == 'heldout':
        eval_nodes = data_module.eval_gauged_nodes
    else:
        eval_nodes = data_module.train_gauged_nodes
    log(f"  eval basins: {len(eval_nodes)} | eval days: {len(eval_dates)} "
        f"({years[0]}-{years[-1]})")

    # build evaluator: stage='finetune' so the trunk is in its joint config; we
    # only ever call model.eval(). init_checkpoint = the checkpoint to evaluate.
    evaluator = D2Evaluator(
        config=config, data_module=data_module, init_checkpoint=args.checkpoint,
        stage='finetune', seq_length=args.seq_length, warmup_days=args.warmup_days,
        sequences_per_epoch=1, gauge_dropout_rate=0.0,
        checkpoint_dir=str(BUNDLE / 'outputs' / 'checkpoints_d2' / '_eval_tmp'),
        n_components=n_comp)

    sync_if_distributed()
    rows = evaluator.evaluate(eval_dates, eval_nodes)

    if is_main_process():
        df = pd.DataFrame(rows)
        # all_gather can return the same basin from multiple ranks; keep one row
        # per basin so n_basins and medians reflect unique basins, not GPU count.
        if 'node' in df.columns:
            df = df.drop_duplicates(subset='node', keep='first').reset_index(drop=True)
        # name output by the checkpoint's dir (e.g. finetune_frozen vs finetune)
        # so frozen/unfrozen evals do NOT overwrite each other.
        ckpt_tag = Path(args.checkpoint).parent.name
        out = args.output or str(
            BUNDLE / "outputs" / f"eval_d2_{ckpt_tag}_{basin_set}_{args.years}.csv")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)

        def med(c):
            return float(np.nanmedian(df[c])) if len(df) else float('nan')

        summary = {
            'checkpoint': args.checkpoint, 'stage': ckpt_stage,
            'basin_set': basin_set, 'years': f"{years[0]}-{years[-1]}",
            'n_basins': int(len(df)),
            'median_kge': med('kge'), 'mean_kge': float(np.nanmean(df['kge'])),
            'median_nse': med('nse'),
            'median_r': med('r'), 'median_alpha': med('alpha'),
            'median_beta': med('beta'),
            'median_peak_bias': med('peak_bias'),
            'median_crps': med('crps'), 'median_crps_norm': med('crps_norm'),
            'coverage_p90': med('cov90'), 'coverage_p95': med('cov95'),
            'coverage_p99': med('cov99'),
            'peak_coverage_p95': med('peak_cov95'),
            'peak_coverage_p99': med('peak_cov99'),
            'kge_gt_0': int((df['kge'] > 0).sum()),
            'kge_gt_0.5': int((df['kge'] > 0.5).sum()),
        }
        with open(Path(out).with_suffix('.summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        log("\n" + "=" * 60)
        log("RESULTS")
        log("=" * 60)
        log(f"  basins evaluated : {summary['n_basins']}")
        log(f"  POINT  median KGE : {summary['median_kge']:.4f}   "
            f"(r={summary['median_r']:.3f}  alpha={summary['median_alpha']:.3f}  "
            f"beta={summary['median_beta']:.3f})")
        log(f"         median NSE : {summary['median_nse']:.4f}")
        log(f"         peak bias  : {summary['median_peak_bias']*100:+.1f}% (top-2%)")
        log(f"  CALIB  coverage   : P90={summary['coverage_p90']:.3f} "
            f"P95={summary['coverage_p95']:.3f} P99={summary['coverage_p99']:.3f}")
        log(f"  SHARP  CRPS       : {summary['median_crps']:.4f} cms "
            f"(normalized {summary['median_crps_norm']:.3f})")
        log(f"  EXTREME peak cov  : P95 brackets {summary['peak_coverage_p95']*100:.0f}% "
            f"/ P99 brackets {summary['peak_coverage_p99']*100:.0f}% of peak days")
        log(f"  KGE>0: {summary['kge_gt_0']}  |  KGE>0.5: {summary['kge_gt_0.5']}")
        log(f"\n  wrote {out}  +  {Path(out).with_suffix('.summary.json')}")

    cleanup_shared_memory()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    import atexit
    atexit.register(cleanup_shared_memory)
    try:
        main()
    finally:
        cleanup_shared_memory()
