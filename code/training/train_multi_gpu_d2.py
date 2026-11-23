#!/usr/bin/env python3
"""
D2 training: v24 trunk + LogNormal Mixture Density head, trained jointly.

Three stages (run in order, each warm-starts from the previous):

    A. warmup   : freeze trunk, train MDN head only on NLL (cheap, hours)
                  -> head catches up to the trunk before we disturb it
    B. pretrain : unfreeze trunk, NLL, TEMPORAL split on TRAIN basins
                  -> trunk learns peak-relevant features (the part v26/v27/A1
                     could not reach with a frozen / lightly-tuned trunk)
    C. finetune : unfreeze trunk, NLL + gauge dropout, PUB basin holdout
                  -> distributional skill that transfers to ungauged basins

Validation reports NLL (checkpoint metric) AND, on the eval basins:
    median KGE (mixture mean)  |  peak bias (top 2% obs)  |  P90/P95 coverage (PIT)

Usage, from the bundle root:
    sbatch code/training/submit_d2.sh warmup
    sbatch code/training/submit_d2.sh pretrain
    sbatch code/training/submit_d2.sh finetune

Warmup starts from weights/checkpoints_v24/finetune/best_model.pt.
New checkpoints are written under outputs/checkpoints_d2/.
"""

import argparse
import json
import os
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

warnings.filterwarnings('ignore')
os.environ['PYTHONUNBUFFERED'] = '1'

from train_multi_gpu_v15 import (
    setup_distributed, is_main_process, sync_if_distributed, log,
    unique_years, build_date_list, cleanup_shared_memory,
)
from train_multi_gpu_v20 import SequenceDataModuleV20
from train_multi_gpu_v26 import normalize_state_dict
from model_v15 import StreamflowGNN
from model_d2 import (
    StreamflowGNNMDN, mixture_nll, mixture_mean, mixture_pit, QUANTILES,
)
from config import get_config_v24

PEAK_FRAC = 0.02


class D2Trainer:
    def __init__(self, config, data_module, init_checkpoint, stage,
                 seq_length, warmup_days, sequences_per_epoch,
                 gauge_dropout_rate, checkpoint_dir, n_components=3,
                 head_lr=1e-3, trunk_lr=1e-5, freeze_trunk=False):
        self.config = config
        self.data_module = data_module
        self.stage = stage
        self.seq_length = seq_length
        self.warmup_days = warmup_days
        self.sequences_per_epoch = sequences_per_epoch
        self.gauge_dropout_rate = gauge_dropout_rate
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.rank = data_module.rank
        self.world_size = data_module.world_size
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.device = torch.device(f"cuda:{self.local_rank}")

        # --- build base trunk + load init weights ---
        log("Building base model (v24 architecture)...")
        base_model = StreamflowGNN(config.model).to(self.device)
        ckpt = torch.load(init_checkpoint, map_location=self.device, weights_only=False)
        # init checkpoint can be a v24 trunk (model_state_dict) OR a prior D2 stage
        state = normalize_state_dict(ckpt['model_state_dict'])
        # if it's a D2 checkpoint, the trunk weights are under base.* — strip if present
        if any(k.startswith('base.') for k in state):
            trunk_state = {k[len('base.'):]: v for k, v in state.items()
                           if k.startswith('base.')}
            base_model.load_state_dict(trunk_state, strict=True)
            self._init_has_head = True
        else:
            base_model.load_state_dict(state, strict=True)
            self._init_has_head = False
        if is_main_process():
            log(f"  Loaded init from {init_checkpoint} "
                f"(epoch {ckpt.get('epoch', '?')}, has_head={self._init_has_head})")

        self.model = StreamflowGNNMDN(base_model, n_components=n_components).to(self.device)

        # load MDN head too if init was a D2 checkpoint
        if self._init_has_head:
            head_state = {k[len('mdn_head.'):]: v for k, v in state.items()
                          if k.startswith('mdn_head.')}
            if head_state:
                self.model.mdn_head.load_state_dict(head_state, strict=True)
                if is_main_process():
                    log("  Loaded MDN head from init checkpoint")

        # --- freeze policy: warmup always freezes; other stages freeze only if asked ---
        self.trunk_frozen = (stage == 'warmup') or freeze_trunk
        if self.trunk_frozen:
            self.model.freeze_trunk()
        else:
            self.model.unfreeze_trunk()

        n_total = sum(p.numel() for p in self.model.parameters())
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if is_main_process():
            log(f"  Stage: {stage}  |  Total params: {n_total:,}  |  Trainable: {n_train:,}")

        # DDP — find_unused only needed when trunk frozen (warmup)
        if self.world_size > 1:
            self.model = DDP(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank,
                find_unused_parameters=self.trunk_frozen,
                bucket_cap_mb=25, gradient_as_bucket_view=True,
            )
            dist.barrier()

        self.forcing_mean = data_module.forcing_mean_tensor.to(self.device)
        self.forcing_std = data_module.forcing_std_tensor.to(self.device)

        # 2-group optimizer for joint stages (low trunk LR protects the median);
        # warmup stage has only head params.
        raw = self._raw()
        head_params = [p for p in raw.mdn_head.parameters() if p.requires_grad]
        trunk_params = [p for p in raw.base.parameters() if p.requires_grad]
        if trunk_params:
            self.optimizer = optim.AdamW(
                [{'params': head_params, 'lr': head_lr},
                 {'params': trunk_params, 'lr': trunk_lr}], weight_decay=1e-5)
            if is_main_process():
                log(f"  Optimizer: head_lr={head_lr}, trunk_lr={trunk_lr} "
                    f"(head {sum(p.numel() for p in head_params):,}, "
                    f"trunk {sum(p.numel() for p in trunk_params):,})")
        else:
            self.optimizer = optim.AdamW(head_params, lr=head_lr, weight_decay=1e-5)
            if is_main_process():
                log(f"  Optimizer: head-only lr={head_lr}")

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5)
        self.scaler = GradScaler()

        self.batch_size = 64
        self.gradient_accumulation_steps = 4
        self.max_subgraph_nodes = 4000  # used by borrowed _prepare_sequence_batch (k=3 hop)
        self.history = {'best_val_loss': float('inf'), 'best_epoch': 0}

    def _raw(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _prepare_sequence_batch(self, basin_nodes, start_date):
        from train_multi_gpu_v15 import RollingLossTrainer
        return RollingLossTrainer._prepare_sequence_batch.__get__(self)(
            self.data_module, basin_nodes, start_date)

    def _forward_sequence(self, batch_data, device):
        """Day-by-day forward -> (pi, mu, sigma) each [n_seeds, seq_len, K].

        Warmup phase always no_grad (truncated BPTT — detached LSTM state).
        Prediction phase builds a graph through whatever has requires_grad
        (head only in stage A; trunk+head in B/C).
        """
        all_forcings = batch_data['all_forcings'].to(device)
        x_static = batch_data['x_static'].to(device)
        up_ei = batch_data['upstream_edge_index'].to(device)
        up_ea = batch_data['upstream_edge_attr'].to(device)
        down_ei = batch_data['downstream_edge_index'].to(device)
        down_ea = batch_data['downstream_edge_attr'].to(device)
        seed_indices = batch_data['seed_indices']
        warmup_days = batch_data['warmup_days']
        total_days = all_forcings.shape[0]

        all_forcings = (all_forcings - self.forcing_mean) / self.forcing_std
        raw_model = self._raw()

        h_lstm = None
        pis, mus, sigmas = [], [], []

        # Warmup: LSTM-only, detached (no BPTT into warmup)
        with torch.no_grad():
            for t in range(warmup_days):
                x_t = torch.cat([all_forcings[t], x_static], dim=-1)
                with autocast():
                    _, h_lstm = raw_model.lstm_step(x_t, h_lstm)
                h_lstm = (h_lstm[0].detach(), h_lstm[1].detach())

        for t in range(warmup_days, total_days):
            x_t = torch.cat([all_forcings[t], x_static], dim=-1)
            with autocast():
                _, (pi, mu, sigma), h_lstm = raw_model(
                    x_t, up_ei, up_ea, down_ei, down_ea, h_lstm, x_static=x_static)
            pis.append(pi[seed_indices])
            mus.append(mu[seed_indices])
            sigmas.append(sigma[seed_indices])

        stack = lambda L: torch.stack(L, dim=0).permute(1, 0, 2)  # [seeds, seq, K]
        return stack(pis), stack(mus), stack(sigmas)

    def _apply_gauge_dropout(self, target_flows, rng):
        n = target_flows.shape[0]
        if self.gauge_dropout_rate <= 0.0 or n <= 1:
            return target_flows
        keep = rng.rand(n) >= self.gauge_dropout_rate
        if not keep.any():
            keep[rng.randint(0, n)] = True
        masked = target_flows.clone()
        keep_t = torch.as_tensor(keep, device=target_flows.device, dtype=torch.bool)
        masked[~keep_t] = float('nan')
        return masked

    def _dummy(self):
        return sum(p.sum() * 0 for p in self.model.parameters() if p.requires_grad)

    def train_epoch(self, train_dates, train_basin_nodes, epoch):
        self.model.train()
        gauged = np.array(train_basin_nodes, dtype=np.int64)
        rng = np.random.RandomState(epoch * 7 + 11)
        rng.shuffle(gauged)
        max_date = max(train_dates) - pd.Timedelta(days=self.seq_length)
        starts = [d for d in train_dates if d <= max_date]
        rng.shuffle(starts)
        per_rank = self.sequences_per_epoch // self.world_size
        my_dates = starts[self.rank * per_rank:(self.rank + 1) * per_rank]
        if is_main_process():
            log(f"  Sampling {self.sequences_per_epoch} sequences ({per_rank} per GPU)")

        total_loss, n_batches, accumulated = 0.0, 0, 0
        self.optimizer.zero_grad()
        pbar = tqdm(my_dates, desc='  Training', disable=not is_main_process())
        for start_date in pbar:
            for bs in range(0, len(gauged), self.batch_size):
                basin_nodes = gauged[bs:min(bs + self.batch_size, len(gauged))].tolist()
                result = self._prepare_sequence_batch(basin_nodes, start_date)
                if result is None:
                    self.scaler.scale(self._dummy()).backward()
                    accumulated += 1
                    if accumulated >= self.gradient_accumulation_steps:
                        self._step()
                        accumulated = 0
                    continue
                batch_data, target_flows = result
                target_flows = self._apply_gauge_dropout(target_flows.to(self.device), rng)
                try:
                    pi, mu, sigma = self._forward_sequence(batch_data, self.device)
                    loss, _ = mixture_nll(target_flows, pi, mu, sigma)
                    if torch.isnan(loss) or torch.isinf(loss) or loss.grad_fn is None:
                        loss = self._dummy()
                except Exception as e:
                    if is_main_process() and n_batches % 50 == 0:
                        log(f"  WARN forward exception {type(e).__name__}: {e}")
                    loss = self._dummy()
                self.scaler.scale(loss / self.gradient_accumulation_steps).backward()
                accumulated += 1
                total_loss += float(loss.detach().cpu()); n_batches += 1
                if accumulated >= self.gradient_accumulation_steps:
                    self._step()
                    accumulated = 0
        pbar.close()

        # flush remaining grads (DDP-synced condition)
        if self.world_size > 1:
            has_rem = torch.tensor([1 if accumulated > 0 else 0], device=self.device)
            dist.all_reduce(has_rem, op=dist.ReduceOp.MAX)
            if has_rem.item() > 0:
                self._step()
        elif accumulated > 0:
            self._step()

        avg = total_loss / max(n_batches, 1)
        if self.world_size > 1:
            t = torch.tensor([total_loss, n_batches], device=self.device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            avg = t[0].item() / max(t[1].item(), 1)
        return avg

    def _step(self):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def validate(self, val_dates, val_basin_nodes):
        """Returns NLL (checkpoint metric) and logs KGE/peak/coverage diagnostics."""
        self.model.eval()
        gauged = np.array(val_basin_nodes, dtype=np.int64)
        max_date = max(val_dates) - pd.Timedelta(days=self.seq_length)
        starts = [d for d in val_dates if d <= max_date]
        rng = np.random.RandomState(999)
        rng.shuffle(starts)
        starts = starts[:min(50, len(starts))]
        my_dates = starts[self.rank::self.world_size]

        total_nll, n_nll = 0.0, 0
        # per-basin collections for diagnostics
        basin_obs, basin_mean, basin_pit = {}, {}, {}

        with torch.no_grad():
            for start_date in tqdm(my_dates, desc='  Validating',
                                   disable=not is_main_process()):
                for bs in range(0, len(gauged), self.batch_size):
                    nodes = gauged[bs:min(bs + self.batch_size, len(gauged))].tolist()
                    result = self._prepare_sequence_batch(nodes, start_date)
                    if result is None:
                        continue
                    batch_data, target_flows = result
                    target_flows = target_flows.to(self.device)
                    pi, mu, sigma = self._forward_sequence(batch_data, self.device)
                    loss, st = mixture_nll(target_flows, pi, mu, sigma)
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        total_nll += float(loss.cpu()); n_nll += 1
                    # diagnostics
                    mean = mixture_mean(pi, mu, sigma)               # [seeds, seq]
                    pit = mixture_pit(target_flows, pi, mu, sigma)   # [seeds, seq]
                    obs_np = target_flows.cpu().numpy()
                    mean_np = mean.float().cpu().numpy()
                    pit_np = pit.float().cpu().numpy()
                    for i, nd in enumerate(nodes):
                        m = np.isfinite(obs_np[i])
                        if m.any():
                            basin_obs.setdefault(nd, []).append(obs_np[i][m])
                            basin_mean.setdefault(nd, []).append(mean_np[i][m])
                            basin_pit.setdefault(nd, []).append(pit_np[i][m])

        avg_nll = total_nll / max(n_nll, 1)
        if self.world_size > 1:
            t = torch.tensor([total_nll, n_nll], device=self.device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            avg_nll = t[0].item() / max(t[1].item(), 1)

        self._log_diagnostics(basin_obs, basin_mean, basin_pit)
        return avg_nll

    def _log_diagnostics(self, basin_obs, basin_mean, basin_pit):
        """Compute per-basin KGE / peak bias / P90,P95 coverage and gather to rank 0."""
        eps = 1e-8
        kges, peak_bias, cov90, cov95 = [], [], [], []
        for nd in basin_obs:
            obs = np.concatenate(basin_obs[nd])
            pred = np.concatenate(basin_mean[nd])
            pit = np.concatenate(basin_pit[nd])
            if len(obs) < 30 or np.std(obs) < 1e-4:
                continue
            ps, os_ = np.std(pred), np.std(obs)
            r = np.corrcoef(pred, obs)[0, 1] if ps > eps else 0.0
            if not np.isfinite(r):
                r = 0.0
            alpha = ps / (os_ + eps)
            beta = np.mean(pred) / (np.mean(obs) + eps)
            kges.append(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
            n_peak = max(1, int(round(PEAK_FRAC * len(obs))))
            top = np.argsort(obs)[-n_peak:]
            ot, pt = obs[top].mean(), pred[top].mean()
            if ot > eps:
                peak_bias.append((pt - ot) / ot)
            # coverage: P(obs <= P90) should be 0.90  -> mean(pit <= 0.90)
            cov90.append(float(np.mean(pit <= 0.90)))
            cov95.append(float(np.mean(pit <= 0.95)))

        def _gather_med(vals):
            if self.world_size <= 1:
                return float(np.median(vals)) if vals else float('nan')
            local = torch.tensor(vals, device=self.device, dtype=torch.float32)
            sz = torch.tensor([len(vals)], device=self.device, dtype=torch.int64)
            sizes = [torch.zeros(1, dtype=torch.int64, device=self.device)
                     for _ in range(self.world_size)]
            dist.all_gather(sizes, sz)
            mx = max(s.item() for s in sizes)
            if mx == 0:
                return float('nan')
            pad = torch.full((mx,), float('nan'), device=self.device)
            if vals:
                pad[:len(vals)] = local
            allp = [torch.zeros(mx, device=self.device) for _ in range(self.world_size)]
            dist.all_gather(allp, pad)
            c = torch.cat(allp)
            c = c[~torch.isnan(c)]
            return float(c.median().cpu()) if c.numel() else float('nan')

        mk, mp, c90, c95 = (_gather_med(kges), _gather_med(peak_bias),
                            _gather_med(cov90), _gather_med(cov95))
        if is_main_process():
            log(f"    KGE(mix mean) median={mk:.4f}  |  "
                f"PEAK bias (top {PEAK_FRAC*100:.0f}%) median={mp:+.4f} ({mp*100:+.1f}%)")
            log(f"    Coverage  P90={c90:.3f} (target 0.90)  P95={c95:.3f} (target 0.95)")

    def save_checkpoint(self, epoch, val_loss, is_best):
        if not is_main_process():
            return
        raw = self._raw()
        state = {
            'epoch': epoch,
            'model_state_dict': raw.state_dict(),
            'mdn_head_state_dict': raw.mdn_head.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'stage': self.stage,
            'n_components': raw.K,
            'forcing_mean': self.forcing_mean.cpu(),
            'forcing_std': self.forcing_std.cpu(),
            'edge_feat_mean': self.data_module.edge_feat_mean,
            'edge_feat_std': self.data_module.edge_feat_std,
            'quantiles': QUANTILES,
        }
        torch.save(state, self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")
        log(f"  Saved: checkpoint_epoch_{epoch}.pt")
        if is_best:
            torch.save(state, self.checkpoint_dir / "best_model.pt")
            log(f"  Saved: best_model.pt")


def main():
    p = argparse.ArgumentParser(description='D2 — MDN head joint training')
    p.add_argument('stage', choices=['warmup', 'pretrain', 'finetune'])
    p.add_argument('--seq-length', type=int, default=180)
    p.add_argument('--warmup-days', type=int, default=365)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--sequences-per-epoch', type=int, default=200)
    p.add_argument('--n-components', type=int, default=3)
    p.add_argument('--head-lr', type=float, default=1e-3)
    p.add_argument('--trunk-lr', type=float, default=1e-5)
    p.add_argument('--gauge-dropout-rate', type=float, default=None)
    p.add_argument('--init-checkpoint', type=str, default=None,
                   help='Override init. Defaults: warmup<-v24 finetune, '
                        'pretrain<-d2 warmup best, finetune<-d2 pretrain best')
    p.add_argument('--freeze-trunk', action='store_true',
                   help='Keep trunk frozen during finetune (head-only PUB). '
                        'Defaults init<-d2 warmup best, ckpt dir<-outputs/checkpoints_d2/finetune_frozen')
    p.add_argument('--ckpt-dir', type=str, default=None,
                   help='Override checkpoint output directory')
    args = p.parse_args()

    # stage-specific defaults. Included paper weights stay under weights/.
    # A new run writes under outputs/ and does not replace those files.
    bundle = Path(__file__).resolve().parents[2]
    defaults = {
        'warmup': (
            str(bundle / 'weights' / 'checkpoints_v24' / 'finetune' / 'best_model.pt'),
            str(bundle / 'outputs' / 'checkpoints_d2' / 'warmup'),
            10, 0.0),
        'pretrain': (
            str(bundle / 'outputs' / 'checkpoints_d2' / 'warmup' / 'best_model.pt'),
            str(bundle / 'outputs' / 'checkpoints_d2' / 'pretrain'),
            30, 0.0),
        'finetune': (
            str(bundle / 'outputs' / 'checkpoints_d2' / 'pretrain' / 'best_model.pt'),
            str(bundle / 'outputs' / 'checkpoints_d2' / 'finetune'),
            30, 0.15),
    }
    init_default, ckpt_dir, ep_default, gd_default = defaults[args.stage]
    # frozen finetune: head-only on the v24-PUB trunk -> warm-start from warmup-best
    # (co-adapted to that exact frozen trunk), separate dir so it doesn't clobber
    # the unfrozen finetune run.
    if args.stage == 'finetune' and args.freeze_trunk:
        init_default = str(bundle / 'outputs' / 'checkpoints_d2' / 'warmup' / 'best_model.pt')
        ckpt_dir = str(bundle / 'outputs' / 'checkpoints_d2' / 'finetune_frozen')
    init_checkpoint = args.init_checkpoint or init_default
    ckpt_dir = args.ckpt_dir or ckpt_dir
    epochs = args.epochs or ep_default
    gauge_dropout = args.gauge_dropout_rate if args.gauge_dropout_rate is not None else gd_default

    log("=" * 60)
    log(f"D2 — MDN head  |  STAGE: {args.stage.upper()}")
    log("=" * 60)
    log(f"Start: {datetime.now()}")
    log(f"  init: {init_checkpoint}")
    log(f"  ckpt dir: {ckpt_dir}  | epochs: {epochs} | gauge_dropout: {gauge_dropout}")
    log(f"  components: {args.n_components} | head_lr: {args.head_lr} | trunk_lr: {args.trunk_lr}")

    if not Path(init_checkpoint).exists():
        p.error(f"init checkpoint not found: {init_checkpoint}\n"
                f"(run the previous stage first, or pass --init-checkpoint)")

    rank, world_size, local_rank = setup_distributed()
    log(f"World size: {world_size}")

    config = get_config_v24()
    config.train.max_epochs = epochs
    config.data.split_mode = 'basin'

    data_module = SequenceDataModuleV20(config, rank=rank, world_size=world_size,
                                        warmup_days=args.warmup_days)

    # --- splits per stage ---
    if args.stage in ('warmup', 'pretrain'):
        # temporal split on TRAIN basins
        train_basin_nodes = data_module.train_gauged_nodes
        eval_basin_nodes = data_module.train_gauged_nodes
        train_dates = build_date_list(config.data.train_years)
        val_dates = build_date_list(config.data.val_years)
        log(f"  [{args.stage}] temporal split on TRAIN basins: "
            f"{len(train_basin_nodes)} basins")
    else:
        # PUB basin holdout, all years
        all_years = unique_years(config.data.train_years, config.data.val_years,
                                 config.data.test_years)
        train_basin_nodes = data_module.train_gauged_nodes
        eval_basin_nodes = data_module.eval_gauged_nodes
        train_dates = build_date_list(all_years)
        val_dates = build_date_list(config.data.val_years)
        log(f"  [finetune] PUB holdout: train {len(train_basin_nodes)} / "
            f"eval(held-out) {len(eval_basin_nodes)}")

    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    if is_main_process():
        meta = data_module.get_split_metadata()
        meta.update({'model': 'd2-mdn', 'stage': args.stage,
                     'n_components': args.n_components, 'init': init_checkpoint,
                     'quantiles': QUANTILES})
        with (Path(ckpt_dir) / "split_metadata.json").open('w') as f:
            json.dump(meta, f, indent=2, default=str)

    trainer = D2Trainer(
        config=config, data_module=data_module, init_checkpoint=init_checkpoint,
        stage=args.stage, seq_length=args.seq_length, warmup_days=args.warmup_days,
        sequences_per_epoch=args.sequences_per_epoch, gauge_dropout_rate=gauge_dropout,
        checkpoint_dir=ckpt_dir, n_components=args.n_components,
        head_lr=args.head_lr, trunk_lr=args.trunk_lr,
        freeze_trunk=args.freeze_trunk)

    sync_if_distributed()
    log("\nStarting training loop...")
    for epoch in range(epochs):
        if is_main_process():
            log(f"\nEpoch {epoch+1}/{epochs}")
        train_loss = trainer.train_epoch(train_dates, train_basin_nodes, epoch)
        if is_main_process():
            log(f"  Train NLL: {train_loss:.6f}")
        val_loss = trainer.validate(val_dates, eval_basin_nodes)
        if is_main_process():
            log(f"  Val NLL: {val_loss:.6f}")
        is_best = val_loss < trainer.history['best_val_loss']
        if is_best:
            trainer.history['best_val_loss'] = val_loss
            trainer.history['best_epoch'] = epoch
            if is_main_process():
                log(f"  NEW BEST! Val NLL: {val_loss:.6f}")
        trainer.save_checkpoint(epoch + 1, val_loss, is_best)
        trainer.scheduler.step(val_loss)
        sync_if_distributed()

    if is_main_process():
        log(f"\nDone. Best epoch {trainer.history['best_epoch']+1}, "
            f"val NLL {trainer.history['best_val_loss']:.6f}")
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
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()
