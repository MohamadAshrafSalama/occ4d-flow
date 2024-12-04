"""
Distributed training with DDP, AMP, gradient checkpointing, and EMA.
"""

import argparse
import copy
import math
import os
import sys
import time
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.default import Occ4DFlowConfig, ModelConfig
from config.nuscenes import get_nuscenes_config
from src.model import Occ4DFlow
from src.data.nuscenes_occ import NuScenesOccupancyDataset
from src.losses.occupancy_losses import OccupancyLoss
from src.losses.temporal_losses import TemporalLoss
from src.losses.physics_losses import PhysicsLoss
from src.losses.uncertainty_losses import UncertaintyLoss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train occ4d-flow")
    p.add_argument("--data-root", type=str, default="/data/nuscenes")
    p.add_argument("--output-dir", type=str, default="checkpoints/run")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--grad-clip", type=float, default=35.0)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--grad-checkpoint", action="store_true", default=True)
    p.add_argument("--past-frames", type=int, default=5)
    p.add_argument("--future-frames", type=int, default=5)
    p.add_argument("--flow-steps", type=int, default=10)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--val-interval", type=int, default=1)
    p.add_argument("--save-interval", type=int, default=2)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    p.add_argument("--wandb-project", type=str, default="occ4d-flow")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def setup_distributed(local_rank: int) -> Tuple[int, int]:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


def is_main(rank: int) -> bool:
    return rank == 0


def build_model(cfg: Occ4DFlowConfig, use_grad_checkpoint: bool) -> Occ4DFlow:
    model = Occ4DFlow(cfg.model)
    if use_grad_checkpoint:
        model.flow_head.unet.use_grad_checkpoint = True
    return model


class EMAModel:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())
        for k in self.shadow:
            self.shadow[k] = self.shadow[k].float()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        for k in self.shadow:
            if self.shadow[k].dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(
                    model_state[k].float(), alpha=1.0 - self.decay
                )

    def apply_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


def get_cosine_lr(
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    base_lr: float,
    min_lr: float,
) -> float:
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate that handles variable-length pillar lists."""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k in ("past_voxels", "past_coords", "past_num_points"):
            T = len(batch[0][k])
            out[k] = [
                torch.stack([sample[k][t] for sample in batch], dim=0)
                for t in range(T)
            ]
        elif k in ("future_occ", "future_poses", "past_timestamps", "current_pose"):
            out[k] = torch.stack([sample[k] for sample in batch], dim=0)
        else:
            out[k] = [sample[k] for sample in batch]
    return out


def compute_total_loss(
    model_out: Dict,
    batch: Dict,
    occ_criterion: OccupancyLoss,
    temporal_criterion: TemporalLoss,
    physics_criterion: PhysicsLoss,
    unc_criterion: UncertaintyLoss,
    loss_cfg,
) -> Dict[str, torch.Tensor]:
    future_occ_gt = batch["future_occ"]
    T = future_occ_gt.shape[1]

    blended_list = model_out["blended"]
    disc_out_list = model_out["disc_out"]

    total_occ = future_occ_gt.new_zeros(1)
    total_unc = future_occ_gt.new_zeros(1)

    for t in range(T):
        gt_t = future_occ_gt[:, t].unsqueeze(1)
        blend_t = blended_list[t]["blend"]
        disc_t = disc_out_list[t]

        occ_losses = occ_criterion(
            disc_t["logits"],
            disc_t["occ_prob"],
            gt_t,
        )
        total_occ = total_occ + occ_losses["occupancy_total"]

        if "logvar" in disc_t:
            unc_losses = unc_criterion(
                disc_t["occ_prob"], gt_t, disc_t["logvar"]
            )
            total_unc = total_unc + unc_losses["uncertainty_total"]

    total_occ = total_occ / T
    total_unc = total_unc / T

    occ_seq = [blended_list[t]["blend"] for t in range(T)]
    temp_losses = temporal_criterion(occ_seq)

    phys_losses = physics_criterion(occ_seq)

    cfm_loss = model_out.get("cfm_loss", future_occ_gt.new_zeros(1))
    tv_loss = model_out.get("tv_loss", future_occ_gt.new_zeros(1))

    total = (
        loss_cfg.focal_weight * total_occ
        + loss_cfg.uncertainty_nll_weight * total_unc
        + loss_cfg.temporal_consistency_weight * temp_losses["temporal_total"]
        + loss_cfg.ray_freespace_weight * phys_losses["physics_total"]
        + loss_cfg.flow_matching_weight * cfm_loss
        + loss_cfg.tv_weight * tv_loss
    )

    return {
        "total": total,
        "occupancy": total_occ,
        "uncertainty": total_unc,
        "temporal": temp_losses["temporal_total"],
        "physics": phys_losses["physics_total"],
        "cfm": cfm_loss,
        "tv": tv_loss,
    }


def save_checkpoint(
    path: str,
    model: nn.Module,
    ema: EMAModel,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    best_miou: float,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "ema_state": ema.shadow,
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_miou": best_miou,
    }, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model_state"]
    if hasattr(model, "module"):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt["epoch"], ckpt.get("best_miou", 0.0)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    ema: EMAModel,
    occ_criterion: OccupancyLoss,
    temporal_criterion: TemporalLoss,
    physics_criterion: PhysicsLoss,
    unc_criterion: UncertaintyLoss,
    cfg: Occ4DFlowConfig,
    epoch: int,
    rank: int,
    wandb_run=None,
) -> Dict[str, float]:
    model.train()
    total_losses = {k: 0.0 for k in ["total", "occupancy", "uncertainty", "temporal", "physics", "cfm", "tv"]}
    n_steps = 0

    for step, batch in enumerate(loader):
        device = next(model.parameters()).device

        past_voxels = [v.to(device) for v in batch["past_voxels"]]
        past_coords = [c.to(device) for c in batch["past_coords"]]
        past_timestamps = batch["past_timestamps"].to(device)
        future_poses = batch["future_poses"].to(device)
        future_occ_gt = batch["future_occ"].to(device)

        lr = get_cosine_lr(epoch, cfg.train.epochs, cfg.train.warmup_epochs, cfg.train.lr, cfg.train.min_lr)
        set_lr(optimizer, lr)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.train.amp):
            out = model(
                past_voxels=past_voxels,
                past_coords=past_coords,
                past_timestamps=past_timestamps,
                future_poses=future_poses,
                future_occ_gt=future_occ_gt,
            )
            losses = compute_total_loss(
                out, batch, occ_criterion, temporal_criterion,
                physics_criterion, unc_criterion, cfg.loss,
            )

        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % cfg.train.ema_update_every == 0:
            raw_model = model.module if hasattr(model, "module") else model
            ema.update(raw_model)

        for k in total_losses:
            total_losses[k] += losses[k].item()
        n_steps += 1

        if is_main(rank) and step % cfg.train.log_interval == 0:
            step_losses = {k: v / n_steps for k, v in total_losses.items()}
            msg = (
                f"Epoch {epoch} step {step}/{len(loader)} "
                f"loss={step_losses['total']:.4f} "
                f"occ={step_losses['occupancy']:.4f} "
                f"cfm={step_losses['cfm']:.4f} "
                f"lr={lr:.6f}"
            )
            print(msg)

            if wandb_run is not None:
                wandb_run.log({f"train/{k}": v for k, v in step_losses.items()}, step=epoch * len(loader) + step)

    return {k: v / max(n_steps, 1) for k, v in total_losses.items()}


def main() -> None:
    from typing import Tuple
    args = parse_args()

    rank, world_size = setup_distributed(args.local_rank)
    device = torch.device(f"cuda:{args.local_rank}")

    torch.manual_seed(args.seed + rank)

    cfg = get_nuscenes_config(args.data_root)
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.lr = args.lr
    cfg.train.warmup_epochs = args.warmup_epochs
    cfg.train.amp = args.amp
    cfg.train.grad_checkpoint = args.grad_checkpoint
    cfg.train.ema_decay = args.ema_decay

    train_dataset = NuScenesOccupancyDataset(
        data_root=args.data_root,
        split="train",
        past_frames=args.past_frames,
        future_frames=args.future_frames,
        augment=True,
    )
    val_dataset = NuScenesOccupancyDataset(
        data_root=args.data_root,
        split="val",
        past_frames=args.past_frames,
        future_frames=args.future_frames,
        augment=False,
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=cfg.data.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=cfg.data.num_workers > 0,
    )

    model = build_model(cfg, args.grad_checkpoint).to(device)

    if dist.get_world_size() > 1:
        if cfg.train.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=cfg.train.find_unused_parameters)

    raw_model = model.module if hasattr(model, "module") else model
    ema = EMAModel(raw_model, decay=args.ema_decay)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=cfg.train.betas,
        eps=cfg.train.eps,
    )
    scaler = GradScaler(enabled=args.amp)

    occ_criterion = OccupancyLoss(
        focal_alpha=cfg.loss.focal_alpha,
        focal_gamma=cfg.loss.focal_gamma,
        lovasz_weight=cfg.loss.lovasz_weight,
        dice_weight=cfg.loss.dice_weight,
        sparsity_weight=cfg.loss.sparsity_weight,
        sparsity_target=cfg.loss.sparsity_target,
    )
    temporal_criterion = TemporalLoss(
        consistency_weight=cfg.loss.temporal_consistency_weight,
        smoothness_weight=cfg.loss.velocity_smoothness_weight,
    )
    physics_criterion = PhysicsLoss(
        ray_freespace_weight=cfg.loss.ray_freespace_weight,
        mass_conservation_weight=cfg.loss.mass_conservation_weight,
    )
    unc_criterion = UncertaintyLoss(nll_weight=cfg.loss.uncertainty_nll_weight)

    start_epoch = 0
    best_miou = 0.0

    if args.resume:
        start_epoch, best_miou = load_checkpoint(args.resume, model, optimizer, scaler)
        if is_main(rank):
            print(f"Resumed from {args.resume}, epoch {start_epoch}, best mIoU {best_miou:.4f}")

    wandb_run = None
    if is_main(rank) and not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=cfg.experiment_name,
                config=vars(args),
            )
        except Exception:
            pass

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        train_losses = train_epoch(
            model, train_loader, optimizer, scaler, ema,
            occ_criterion, temporal_criterion, physics_criterion, unc_criterion,
            cfg, epoch, rank, wandb_run,
        )

        if is_main(rank):
            print(f"Epoch {epoch} train loss: {train_losses['total']:.4f}")

            if epoch % args.save_interval == 0:
                ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch:03d}.pth")
                save_checkpoint(ckpt_path, model, ema, optimizer, scaler, epoch, best_miou)

    if is_main(rank):
        final_path = os.path.join(args.output_dir, "final.pth")
        save_checkpoint(final_path, model, ema, optimizer, scaler, args.epochs, best_miou)
        print(f"Training complete. Final checkpoint saved to {final_path}")

    if wandb_run is not None:
        wandb_run.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    from typing import Tuple
    main()
