#!/usr/bin/env python3
"""Train independent point and probabilistic encoders on the paper split."""

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parents[1]
TRAINING = BUNDLE / "code" / "training"
sys.path.insert(1, str(TRAINING))

from config import get_config_v24
from model_v15 import StreamflowGNN
from train_multi_gpu_d2 import D2Trainer
from train_multi_gpu_v15 import (
    build_date_list, cleanup_shared_memory, is_main_process, log,
    setup_distributed, sync_if_distributed, unique_years,
)
from train_multi_gpu_v16 import V16Trainer
from train_multi_gpu_v20 import SequenceDataModuleV20

PAPER_SPLIT = BUNDLE / "weights" / "checkpoints_v24" / "finetune" / "split_metadata.json"


def check_split(data_module):
    reference = json.loads(PAPER_SPLIT.read_text())
    current = data_module.get_split_metadata()
    for key in ("train_comids", "eval_comids", "train_node_indices", "eval_node_indices"):
        if current[key] != reference[key]:
            raise RuntimeError(f"Source data no longer reproduce the paper split: {key}")
    if (len(current["train_comids"]), len(current["eval_comids"])) != (8677, 965):
        raise RuntimeError("Unexpected gauge split counts")
    return current


def seed_model(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inspect_checkpoint(path):
    """Reject a damaged MDN checkpoint before any recovery run uses it."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "finetune" or checkpoint.get("n_components") != 3:
        raise ValueError(f"Unexpected MDN checkpoint type: {path}")
    epoch = checkpoint.get("epoch")
    val_loss = checkpoint.get("val_loss")
    if not isinstance(epoch, int) or epoch < 1 or not math.isfinite(float(val_loss)):
        raise ValueError(f"Invalid checkpoint epoch or validation loss: {path}")
    for section in ("model_state_dict", "mdn_head_state_dict"):
        tensors = checkpoint.get(section)
        if not isinstance(tensors, dict) or not tensors:
            raise ValueError(f"Missing {section}: {path}")
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
                raise ValueError(f"Nonfinite {section}.{name}: {path}")
    for section in ("forcing_mean", "forcing_std", "edge_feat_mean", "edge_feat_std"):
        tensor = torch.as_tensor(checkpoint[section])
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite {section}: {path}")
    return {
        "path": str(path), "epoch": epoch, "val_loss": float(val_loss),
        "model_tensors": len(checkpoint["model_state_dict"]),
        "optimizer_state_saved": "optimizer_state_dict" in checkpoint,
        "scaler_state_saved": "scaler_state_dict" in checkpoint,
        "scheduler_state_saved": "scheduler_state_dict" in checkpoint,
    }


class SyncedD2Trainer(D2Trainer):
    """Average gradients because the inherited sequence loop bypasses DDP.forward."""

    def _skip_global_overflow(self, reason):
        skipped = getattr(self, "global_skipped_steps", 0) + 1
        self.global_skipped_steps = skipped
        old_scale = self.scaler.get_scale()
        self.scaler.update(new_scale=max(old_scale / 2, torch.finfo(torch.float32).tiny))
        self.optimizer.zero_grad(set_to_none=True)
        if is_main_process() and (skipped <= 5 or skipped % 50 == 0):
            log(f"  WARN synchronized optimizer skip ({reason}); "
                f"count={skipped}, AMP scale {old_scale:g} -> {self.scaler.get_scale():g}")
        return False

    def _step(self):
        self.scaler.unscale_(self.optimizer)
        params = [p for p in self._raw().parameters() if p.requires_grad]
        grads = []
        for param in params:
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            grads.append(param.grad)
        flat = torch.cat([grad.reshape(-1) for grad in grads])
        # GradScaler's local found_inf decision cannot safely govern a manually
        # synchronized optimizer: one rank could skip while the others step.
        overflow = (~torch.isfinite(flat).all()).to(torch.int32)
        if self.world_size > 1:
            dist.all_reduce(overflow, op=dist.ReduceOp.MAX)
        if overflow.item():
            return self._skip_global_overflow("nonfinite local gradient")
        if self.world_size > 1:
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.div_(self.world_size)
            if not torch.isfinite(flat).all():
                return self._skip_global_overflow("nonfinite reduced gradient")
            offset = 0
            for grad in grads:
                n = grad.numel()
                grad.copy_(flat[offset:offset + n].view_as(grad))
                offset += n
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(grad_norm):
            return self._skip_global_overflow("nonfinite gradient norm")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("point", "mdn"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--resume-checkpoint", type=Path,
                        help="Resume MDN weights from a validated standalone checkpoint")
    parser.add_argument("--output-dir", type=Path,
                        help="Separate output directory for a recovery run")
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--trunk-lr", type=float, default=5e-4)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.head_lr <= 0 or args.trunk_lr <= 0:
        parser.error("learning rates must be positive")
    if args.resume_checkpoint and args.model != "mdn":
        parser.error("--resume-checkpoint is only supported for mdn")
    if args.resume_checkpoint and not args.output_dir:
        parser.error("--output-dir is required for recovery to preserve prior checkpoints")

    rank, world_size, _ = setup_distributed()
    output = (args.output_dir or BUNDLE / "outputs" / "standalone" / "checkpoints" / args.model).resolve()
    resume_info = best_info = None
    if args.resume_checkpoint:
        resume_path = args.resume_checkpoint.resolve()
        if output == resume_path.parent:
            raise ValueError("Recovery output must differ from the source checkpoint directory")
        resume_info = inspect_checkpoint(resume_path)
        best_path = resume_path.parent / "best_model.pt"
        best_info = inspect_checkpoint(best_path)
        if (best_info["epoch"] > resume_info["epoch"] or
                best_info["val_loss"] > resume_info["val_loss"]):
            raise ValueError("Source best checkpoint is inconsistent with recovery epoch")
        if args.epochs <= resume_info["epoch"]:
            parser.error("--epochs must exceed the resumed epoch")
        if output.exists() and any(output.glob("checkpoint_epoch_*.pt")):
            raise RuntimeError(f"Recovery output already contains training checkpoints: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = get_config_v24()
    config.data.split_mode = "basin"
    config.data.train_basin_fraction = 0.9
    config.data.basin_split_seed = 42
    config.train.max_epochs = args.epochs
    config.train.patience = 15
    config.train.learning_rate = 5e-4

    log(f"Standalone {args.model}: seed={args.seed}, ranks={world_size}, epochs={args.epochs}")
    log(f"Source data: {config.data.base_dir}")
    data = SequenceDataModuleV20(config, rank=rank, world_size=world_size, warmup_days=365)
    metadata = check_split(data)
    if is_main_process():
        metadata.update({
            "experiment": f"standalone_{args.model}",
            "model_seed": args.seed,
            "source_training_dir": str(TRAINING),
            "paper_split": str(PAPER_SPLIT),
        })
        (output / "split_metadata.json").write_text(json.dumps(metadata, indent=2))
    sync_if_distributed()
    seed_model(args.seed)

    all_years = unique_years(config.data.train_years, config.data.val_years,
                             config.data.test_years)
    train_dates = build_date_list(all_years)
    train_nodes = data.train_gauged_nodes
    eval_nodes = data.eval_gauged_nodes

    if args.model == "point":
        # Direct spatial training, with no point checkpoint loaded.
        val_dates = build_date_list(all_years)
        trainer = V16Trainer(
            config=config, data_module=data, seq_length=180, warmup_days=365,
            loss_type="kge", warmup_epochs=0, sequences_per_epoch=200,
            gauge_dropout_rate=0.15, checkpoint_dir=str(output),
            version="standalone-pub-point",
        )
        log("Point: all-year train and all-year spatial validation, as paper Head 2")
        trainer.fit(
            data, train_dates, val_dates, train_basin_nodes=train_nodes,
            eval_basin_nodes=eval_nodes, start_epoch=0,
            stage_label="standalone-pub",
        )
    else:
        # The original MDN trainer expects an initial checkpoint. This checkpoint
        # contains a fresh random encoder, with no weights from the paper models.
        init_path = args.resume_checkpoint.resolve() if resume_info else output / "random_init.pt"
        if is_main_process() and not resume_info:
            base = StreamflowGNN(config.model)
            torch.save({
                "epoch": -1,
                "model_state_dict": base.state_dict(),
                "initialization": "fresh random; no paper weights",
            }, init_path)
            del base
        if is_main_process() and resume_info:
            shutil.copy2(best_path, output / "best_model.pt")
            log(f"Recovery from epoch {resume_info['epoch']} (val NLL "
                f"{resume_info['val_loss']:.6f}); prior best epoch "
                f"{best_info['epoch']} (NLL {best_info['val_loss']:.6f})")
            log("Legacy checkpoints have no optimizer, scaler or scheduler state; "
                "those states restart. This is a weight recovery, not an exact resume.")
        sync_if_distributed()
        val_dates = build_date_list(config.data.val_years)
        trainer = SyncedD2Trainer(
            config=config, data_module=data, init_checkpoint=str(init_path),
            stage="finetune", seq_length=180, warmup_days=365,
            sequences_per_epoch=200, gauge_dropout_rate=0.15,
            checkpoint_dir=str(output), n_components=3,
            head_lr=args.head_lr, trunk_lr=args.trunk_lr, freeze_trunk=False,
        )
        log("MDN: all-year train and 2015-2019 spatial validation, as paper Head 3")
        best = best_info["val_loss"] if best_info else math.inf
        bad_epochs = resume_info["epoch"] - best_info["epoch"] if resume_info else 0
        start_epoch = resume_info["epoch"] if resume_info else 0
        if best_info:
            trainer.history["best_val_loss"] = best
            trainer.history["best_epoch"] = best_info["epoch"] - 1
        for epoch in range(start_epoch, args.epochs):
            log(f"Epoch {epoch + 1}/{args.epochs}")
            skipped_before = getattr(trainer, "global_skipped_steps", 0)
            loss = trainer.train_epoch(train_dates, train_nodes, epoch)
            skipped = getattr(trainer, "global_skipped_steps", 0) - skipped_before
            val = trainer.validate(val_dates, eval_nodes)
            if not math.isfinite(loss) or not math.isfinite(val):
                raise RuntimeError(
                    f"Nonfinite MDN loss at epoch {epoch + 1}: train={loss}, val={val}")
            improved = val < best
            if improved:
                best = val
                bad_epochs = 0
                trainer.history["best_val_loss"] = val
                trainer.history["best_epoch"] = epoch
            else:
                bad_epochs += 1
            log(f"  train NLL={loss:.6f}, val NLL={val:.6f}, "
                f"best={best:.6f}, patience={bad_epochs}/15, "
                f"globally skipped AMP steps={skipped}")
            trainer.save_checkpoint(epoch + 1, val, improved)
            trainer.scheduler.step(val)
            sync_if_distributed()
            if bad_epochs >= 15:
                log(f"Early stopping at epoch {epoch + 1}")
                break

    sync_if_distributed()
    if is_main_process() and not (output / "best_model.pt").exists():
        raise RuntimeError(f"No best checkpoint written to {output}")
    log(f"Finished standalone {args.model}; best checkpoint: {output / 'best_model.pt'}")


if __name__ == "__main__":
    import atexit
    atexit.register(cleanup_shared_memory)
    try:
        main()
    finally:
        cleanup_shared_memory()
        if dist.is_initialized():
            dist.destroy_process_group()
