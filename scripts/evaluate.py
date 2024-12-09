"""
Evaluation script computing mIoU, VPQ, and ray-based precision/recall.
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.nuscenes import get_nuscenes_config
from src.model import Occ4DFlow
from src.data.nuscenes_occ import NuScenesOccupancyDataset
from scripts.train import collate_fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate occ4d-flow")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-root", type=str, default="/data/nuscenes")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--past-frames", type=int, default=5)
    p.add_argument("--future-frames", type=int, default=5)
    p.add_argument("--flow-steps", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-json", type=str, default=None)
    return p.parse_args()


class MeanIoUMeter:
    """Running mIoU computation for binary occupancy."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_bin = (pred >= self.threshold).bool()
        target_bin = target.bool()

        self.tp += (pred_bin & target_bin).sum().item()
        self.fp += (pred_bin & ~target_bin).sum().item()
        self.fn += (~pred_bin & target_bin).sum().item()
        self.tn += (~pred_bin & ~target_bin).sum().item()

    def compute(self) -> Dict[str, float]:
        iou_occ = self.tp / (self.tp + self.fp + self.fn + 1e-10)
        iou_free = self.tn / (self.tn + self.fp + self.fn + 1e-10)
        miou = (iou_occ + iou_free) / 2.0
        precision = self.tp / (self.tp + self.fp + 1e-10)
        recall = self.tp / (self.tp + self.fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        return {
            "iou_occ": iou_occ,
            "iou_free": iou_free,
            "miou": miou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def reset(self) -> None:
        self.tp = self.fp = self.fn = self.tn = 0


class VPQMeter:
    """
    Video Panoptic Quality approximation for occupancy sequences.

    Since we treat occupancy as binary, VPQ simplifies to measuring
    temporal consistency of the IoU across predicted future steps.
    We report mean IoU per future timestep and the decay rate.
    """

    def __init__(self, num_future: int, threshold: float = 0.5) -> None:
        self.num_future = num_future
        self.threshold = threshold
        self.iou_per_step: List[List[float]] = [[] for _ in range(num_future)]

    def update(self, pred_list: List[torch.Tensor], target: torch.Tensor) -> None:
        T = min(len(pred_list), self.num_future)
        for t in range(T):
            pred_t = (pred_list[t] >= self.threshold).float()
            gt_t = target[:, t].unsqueeze(1)

            B = pred_t.shape[0]
            for b in range(B):
                p = pred_t[b].bool().reshape(-1)
                g = gt_t[b].bool().reshape(-1)
                tp = (p & g).sum().item()
                fp = (p & ~g).sum().item()
                fn = (~p & g).sum().item()
                iou = tp / (tp + fp + fn + 1e-10)
                self.iou_per_step[t].append(iou)

    def compute(self) -> Dict[str, float]:
        mean_iou_per_step = [
            float(np.mean(self.iou_per_step[t])) if self.iou_per_step[t] else 0.0
            for t in range(self.num_future)
        ]
        vpq = float(np.mean(mean_iou_per_step))

        first_iou = mean_iou_per_step[0] if mean_iou_per_step else 0.0
        last_iou = mean_iou_per_step[-1] if mean_iou_per_step else 0.0
        decay = first_iou - last_iou

        result = {"vpq": vpq, "iou_decay": decay}
        for t, v in enumerate(mean_iou_per_step):
            result[f"iou_t{t + 1}"] = v
        return result


class RayMetricMeter:
    """
    Ray-based precision / recall along sensor rays.

    Casts uniform rays from the sensor origin. For each ray:
      - Find the first predicted occupied voxel (predicted hit)
      - Find the first GT occupied voxel (GT hit)
    Compute precision/recall based on whether hits are at similar depths.
    """

    def __init__(
        self,
        num_rays: int = 512,
        depth_tolerance_m: float = 1.0,
        voxel_size: float = 0.5,
        grid_size: Tuple[int, int, int] = (200, 200, 16),
        threshold: float = 0.5,
    ) -> None:
        self.num_rays = num_rays
        self.depth_tolerance = depth_tolerance_m / voxel_size
        self.threshold = threshold
        self.grid_size = grid_size

        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        self.ray_dx = np.cos(angles)
        self.ray_dy = np.sin(angles)

        self.tp = 0
        self.fp = 0
        self.fn = 0

    def _first_hit(self, occ_slice: np.ndarray, dx: float, dy: float) -> Optional[int]:
        """Find depth index of first occupancy along a 2D ray in the BEV slice."""
        X, Y = occ_slice.shape
        cx, cy = X // 2, Y // 2
        max_steps = int(np.sqrt(X ** 2 + Y ** 2)) + 1

        for step in range(max_steps):
            xi = int(cx + step * dx)
            yi = int(cy + step * dy)
            if not (0 <= xi < X and 0 <= yi < Y):
                break
            if occ_slice[xi, yi] >= self.threshold:
                return step
        return None

    def update(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        pred_np = pred.squeeze(1).cpu().numpy()
        gt_np = gt.cpu().numpy()

        B = pred_np.shape[0]
        for b in range(B):
            pred_bev = pred_np[b].max(axis=-1)
            gt_bev = gt_np[b].max(axis=-1)

            for r in range(self.num_rays):
                dx = self.ray_dx[r]
                dy = self.ray_dy[r]

                pred_depth = self._first_hit(pred_bev, dx, dy)
                gt_depth = self._first_hit(gt_bev, dx, dy)

                if gt_depth is None:
                    if pred_depth is None:
                        pass
                    else:
                        self.fp += 1
                else:
                    if pred_depth is None:
                        self.fn += 1
                    elif abs(pred_depth - gt_depth) <= self.depth_tolerance:
                        self.tp += 1
                    else:
                        self.fp += 1
                        self.fn += 1

    def compute(self) -> Dict[str, float]:
        precision = self.tp / (self.tp + self.fp + 1e-10)
        recall = self.tp / (self.tp + self.fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        return {"ray_precision": precision, "ray_recall": recall, "ray_f1": f1}


def load_model(checkpoint_path: str, cfg, device: torch.device) -> Occ4DFlow:
    model = Occ4DFlow(cfg.model).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt.get("ema_state", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def evaluate(
    model: Occ4DFlow,
    loader: DataLoader,
    cfg,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    miou_meter = MeanIoUMeter(threshold=args.threshold)
    vpq_meter = VPQMeter(num_future=args.future_frames, threshold=args.threshold)
    ray_meter = RayMetricMeter(threshold=args.threshold)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            past_voxels = [v.to(device) for v in batch["past_voxels"]]
            past_coords = [c.to(device) for c in batch["past_coords"]]
            past_timestamps = batch["past_timestamps"].to(device)
            future_poses = batch["future_poses"].to(device)
            future_occ_gt = batch["future_occ"].to(device)

            blended = model.predict(
                past_voxels=past_voxels,
                past_coords=past_coords,
                past_timestamps=past_timestamps,
                future_poses=future_poses,
                num_flow_steps=args.flow_steps,
            )

            for t, pred_t in enumerate(blended):
                gt_t = future_occ_gt[:, t].unsqueeze(1)
                miou_meter.update(pred_t, gt_t)

            vpq_meter.update(blended, future_occ_gt)

            if blended:
                ray_meter.update(blended[0], future_occ_gt[:, 0].unsqueeze(1))

    results = {}
    results.update(miou_meter.compute())
    results.update(vpq_meter.compute())
    results.update(ray_meter.compute())
    return results


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = get_nuscenes_config(args.data_root)

    dataset = NuScenesOccupancyDataset(
        data_root=args.data_root,
        split=args.split,
        past_frames=args.past_frames,
        future_frames=args.future_frames,
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = load_model(args.checkpoint, cfg, device)
    print(f"Model parameters: {model.count_parameters():,}")

    results = evaluate(model, loader, cfg, args, device)

    print("\n=== Evaluation Results ===")
    for k, v in sorted(results.items()):
        print(f"  {k:30s}: {v:.4f}")

    if args.output_json:
        import json
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
