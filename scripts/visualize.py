"""
3D voxel grid visualization using Open3D and matplotlib.
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.nuscenes import get_nuscenes_config
from src.model import Occ4DFlow
from src.data.nuscenes_occ import NuScenesOccupancyDataset
from scripts.train import collate_fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize occ4d-flow predictions")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-root", type=str, default="/data/nuscenes")
    p.add_argument("--scene-token", type=str, default=None)
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="vis")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--flow-steps", type=int, default=20)
    p.add_argument("--use-open3d", action="store_true")
    p.add_argument("--voxel-size", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def load_model(checkpoint_path: str, cfg, device: torch.device) -> Occ4DFlow:
    model = Occ4DFlow(cfg.model).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt.get("ema_state", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def voxel_to_pointcloud(
    occ_grid: np.ndarray,
    threshold: float = 0.5,
    voxel_size: float = 0.5,
    grid_origin: Tuple[float, float, float] = (-50.0, -50.0, -5.0),
) -> np.ndarray:
    """
    Convert binary voxel grid to a colored point cloud.

    Args:
        occ_grid: (X, Y, Z) binary or probability array
        threshold: occupancy threshold
        voxel_size: meters per voxel for coordinate recovery
        grid_origin: metric origin of the grid corner

    Returns:
        points: (N, 3) occupied voxel centers in metric space
    """
    occupied = np.argwhere(occ_grid >= threshold)
    if len(occupied) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    ox, oy, oz = grid_origin
    points = occupied.astype(np.float32) * voxel_size + np.array([ox, oy, oz]) + voxel_size / 2.0
    return points


def colorize_by_height(points: np.ndarray) -> np.ndarray:
    """
    Assign RGB colors to points by their Z coordinate.

    Returns (N, 3) float32 in [0, 1].
    """
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    z = points[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
    cmap = cm.get_cmap("plasma")
    colors = cmap(z_norm)[:, :3].astype(np.float32)
    return colors


def render_bev_matplotlib(
    occ_pred: np.ndarray,
    occ_gt: Optional[np.ndarray],
    output_path: str,
    title: str = "",
    threshold: float = 0.5,
) -> None:
    """
    Render a BEV (top-down) view of occupancy using matplotlib.

    Takes the max over the Z axis to collapse to 2D.
    """
    pred_bev = occ_pred.max(axis=-1)
    pred_binary = (pred_bev >= threshold).astype(np.float32)

    if occ_gt is not None:
        gt_bev = occ_gt.max(axis=-1)
        gt_binary = (gt_bev >= threshold).astype(np.float32)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(pred_binary.T, origin="lower", cmap="inferno", vmin=0, vmax=1)
        axes[0].set_title("Predicted occupancy (BEV)")
        axes[0].axis("off")
        axes[1].imshow(gt_binary.T, origin="lower", cmap="inferno", vmin=0, vmax=1)
        axes[1].set_title("Ground truth (BEV)")
        axes[1].axis("off")
    else:
        fig, axes = plt.subplots(1, 1, figsize=(7, 7))
        axes.imshow(pred_binary.T, origin="lower", cmap="inferno", vmin=0, vmax=1)
        axes.set_title("Predicted occupancy (BEV)")
        axes.axis("off")
        axes = [axes]

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_open3d(
    pred_points: np.ndarray,
    pred_colors: np.ndarray,
    gt_points: Optional[np.ndarray],
    window_title: str = "occ4d-flow",
    save_path: Optional[str] = None,
) -> None:
    """
    Render voxel point clouds with Open3D.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("open3d not installed, skipping 3D render")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_title, width=1280, height=720)

    if len(pred_points) > 0:
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(pred_points)
        pred_pcd.colors = o3d.utility.Vector3dVector(pred_colors)
        vis.add_geometry(pred_pcd)

    if gt_points is not None and len(gt_points) > 0:
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(gt_points)
        gt_pcd.colors = o3d.utility.Vector3dVector(
            np.tile([0.2, 0.8, 0.2], (len(gt_points), 1)).astype(np.float32)
        )
        vis.add_geometry(gt_pcd)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.05])
    opt.point_size = 3.0

    if save_path:
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path)
        vis.destroy_window()
    else:
        vis.run()
        vis.destroy_window()


def visualize_sample(
    model: Occ4DFlow,
    batch: dict,
    args: argparse.Namespace,
    device: torch.device,
    sample_label: str = "sample",
) -> None:
    past_voxels = [v.to(device) for v in batch["past_voxels"]]
    past_coords = [c.to(device) for c in batch["past_coords"]]
    past_timestamps = batch["past_timestamps"].to(device)
    future_poses = batch["future_poses"].to(device)
    future_occ_gt = batch["future_occ"]

    predictions = model.predict(
        past_voxels=past_voxels,
        past_coords=past_coords,
        past_timestamps=past_timestamps,
        future_poses=future_poses,
        num_flow_steps=args.flow_steps,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for t, pred_t in enumerate(predictions):
        pred_np = pred_t[0, 0].cpu().numpy()

        gt_np = None
        if future_occ_gt is not None and t < future_occ_gt.shape[1]:
            gt_np = future_occ_gt[0, t].numpy()

        bev_path = os.path.join(args.output_dir, f"{sample_label}_t{t + 1:02d}_bev.png")
        render_bev_matplotlib(
            pred_np, gt_np, bev_path,
            title=f"t+{t + 1} prediction",
            threshold=args.threshold,
        )
        print(f"Saved {bev_path}")

        if args.use_open3d:
            pred_pts = voxel_to_pointcloud(pred_np, args.threshold, args.voxel_size)
            pred_cols = colorize_by_height(pred_pts)
            gt_pts = voxel_to_pointcloud(gt_np, args.threshold, args.voxel_size) if gt_np is not None else None

            o3d_path = os.path.join(args.output_dir, f"{sample_label}_t{t + 1:02d}_3d.png")
            render_open3d(
                pred_pts, pred_cols, gt_pts,
                window_title=f"occ4d-flow t+{t + 1}",
                save_path=o3d_path,
            )
            print(f"Saved {o3d_path}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = get_nuscenes_config(args.data_root)
    model = load_model(args.checkpoint, cfg, device)

    dataset = NuScenesOccupancyDataset(
        data_root=args.data_root,
        split="val",
        past_frames=5,
        future_frames=5,
        augment=False,
    )

    idx = args.sample_idx
    sample = dataset[idx]

    keys_to_stack = ["past_timestamps", "current_pose", "future_poses", "future_occ"]
    batch = {}
    for k in sample:
        if k in keys_to_stack:
            batch[k] = sample[k].unsqueeze(0)
        elif k in ("past_voxels", "past_coords", "past_num_points"):
            batch[k] = [t.unsqueeze(0) for t in sample[k]]
        else:
            batch[k] = [sample[k]]

    label = f"sample{idx:04d}"
    if args.scene_token:
        label = args.scene_token[:12]

    visualize_sample(model, batch, args, device, sample_label=label)
    print(f"Visualization complete. Output in {args.output_dir}/")


if __name__ == "__main__":
    main()
