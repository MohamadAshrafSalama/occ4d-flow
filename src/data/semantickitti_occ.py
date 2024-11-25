import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.voxel_utils import (
    points_to_voxel_grid,
    build_occupancy_grid,
    apply_se3_to_points,
)


class SemanticKITTIOccupancyDataset(Dataset):
    """
    SemanticKITTI occupancy dataset loader.

    Loads sequences from the SemanticKITTI dataset and produces
    past/future occupancy frames for training.

    Voxel grid: 256 x 256 x 32 at 0.2m resolution.
    Coverage: x in [-25.6, 25.6], y in [-25.6, 25.6], z in [-3.2, 3.2] meters.

    Expected data layout:
        data_root/
            sequences/
                00/
                    velodyne/
                        000000.bin
                    poses.txt
                    calib.txt
                01/
                ...
    """

    POINT_CLOUD_RANGE = (-25.6, -25.6, -3.2, 25.6, 25.6, 3.2)
    GRID_SIZE = (256, 256, 32)
    VOXEL_SIZE = 0.2
    MAX_POINTS_PER_PILLAR = 20
    MAX_PILLARS = 40000

    TRAIN_SEQUENCES = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
    VAL_SEQUENCES = ["08"]
    TEST_SEQUENCES = ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        past_frames: int = 5,
        future_frames: int = 5,
        augment: bool = True,
        aug_flip_x: bool = True,
        aug_rotate_range: float = 0.3927,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.augment = augment and (split == "train")
        self.aug_flip_x = aug_flip_x
        self.aug_rotate_range = aug_rotate_range

        if split == "train":
            self.sequences = self.TRAIN_SEQUENCES
        elif split == "val":
            self.sequences = self.VAL_SEQUENCES
        else:
            self.sequences = self.TEST_SEQUENCES

        self.samples = self._index_samples()

    def _load_poses(self, seq_path: str) -> np.ndarray:
        """Load Nx4x4 pose array from poses.txt."""
        pose_file = os.path.join(seq_path, "poses.txt")
        poses = []
        with open(pose_file, "r") as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                pose = np.array(vals, dtype=np.float32).reshape(3, 4)
                pose = np.vstack([pose, [0, 0, 0, 1]])
                poses.append(pose)
        return np.stack(poses, axis=0)

    def _load_calib(self, seq_path: str) -> np.ndarray:
        """Load Tr (velodyne to camera) calibration matrix."""
        calib_file = os.path.join(seq_path, "calib.txt")
        with open(calib_file, "r") as f:
            for line in f:
                if line.startswith("Tr:"):
                    vals = list(map(float, line.split(":")[1].strip().split()))
                    tr = np.array(vals, dtype=np.float32).reshape(3, 4)
                    return np.vstack([tr, [0, 0, 0, 1]])
        return np.eye(4, dtype=np.float32)

    def _index_samples(self) -> List[Dict[str, Any]]:
        """Build the list of valid (sequence, frame_index) tuples."""
        samples = []
        for seq in self.sequences:
            seq_path = os.path.join(self.data_root, "sequences", seq)
            velodyne_path = os.path.join(seq_path, "velodyne")

            if not os.path.isdir(velodyne_path):
                n_frames = 100
                for i in range(self.past_frames - 1, n_frames - self.future_frames):
                    samples.append({
                        "seq": seq,
                        "frame_idx": i,
                        "seq_path": seq_path,
                        "stub": True,
                    })
                continue

            bin_files = sorted([f for f in os.listdir(velodyne_path) if f.endswith(".bin")])
            n_frames = len(bin_files)

            try:
                poses = self._load_poses(seq_path)
                calib = self._load_calib(seq_path)
            except Exception:
                poses = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))
                calib = np.eye(4, dtype=np.float32)

            for i in range(self.past_frames - 1, n_frames - self.future_frames):
                samples.append({
                    "seq": seq,
                    "frame_idx": i,
                    "seq_path": seq_path,
                    "n_frames": n_frames,
                    "poses": poses,
                    "calib": calib,
                    "stub": False,
                })

        return samples

    def _load_lidar_frame(self, seq_path: str, frame_idx: int) -> np.ndarray:
        """Load a single LiDAR frame as (N, 4) array."""
        velodyne_path = os.path.join(seq_path, "velodyne")
        fname = f"{frame_idx:06d}.bin"
        fpath = os.path.join(velodyne_path, fname)
        points = np.fromfile(fpath, dtype=np.float32).reshape(-1, 4)
        return points

    def _transform_points(
        self,
        points: np.ndarray,
        src_pose: np.ndarray,
        dst_pose: np.ndarray,
        calib: np.ndarray,
    ) -> np.ndarray:
        """Transform from src sensor frame to dst sensor frame."""
        src_to_global = src_pose @ calib
        global_to_dst = np.linalg.inv(dst_pose @ calib)
        transform = global_to_dst @ src_to_global
        return apply_se3_to_points(points, transform)

    def _augment_pointcloud(self, points: np.ndarray, flip_x: bool, angle: float) -> np.ndarray:
        pts = points.copy()
        if flip_x:
            pts[:, 0] = -pts[:, 0]
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x_rot = pts[:, 0] * cos_a - pts[:, 1] * sin_a
        y_rot = pts[:, 0] * sin_a + pts[:, 1] * cos_a
        pts[:, 0] = x_rot
        pts[:, 1] = y_rot
        return pts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        info = self.samples[idx]

        if info.get("stub"):
            return self._get_stub_item(idx)

        frame_idx = info["frame_idx"]
        seq_path = info["seq_path"]
        poses = info["poses"]
        calib = info["calib"]

        current_pose = poses[frame_idx]

        past_indices = list(range(frame_idx - self.past_frames + 1, frame_idx + 1))
        future_indices = list(range(frame_idx + 1, frame_idx + 1 + self.future_frames))

        do_flip = self.augment and self.aug_flip_x and np.random.rand() < 0.5
        rot_angle = 0.0
        if self.augment:
            rot_angle = np.random.uniform(-self.aug_rotate_range, self.aug_rotate_range)

        past_voxels, past_coords, past_num_pts, past_timestamps = [], [], [], []

        for t, fidx in enumerate(past_indices):
            pts = self._load_lidar_frame(seq_path, fidx)
            pts = self._transform_points(pts, poses[fidx], current_pose, calib)

            if self.augment:
                pts = self._augment_pointcloud(pts, do_flip, rot_angle)

            v, c, n = points_to_voxel_grid(
                pts, self.POINT_CLOUD_RANGE, self.VOXEL_SIZE,
                self.GRID_SIZE, self.MAX_POINTS_PER_PILLAR, self.MAX_PILLARS,
            )
            past_voxels.append(torch.from_numpy(v).float())
            past_coords.append(torch.from_numpy(c).long())
            past_num_pts.append(torch.from_numpy(n).long())

            dt = float(fidx - frame_idx) * 0.1
            past_timestamps.append(dt)

        future_occ_grids = []
        future_poses = []

        for fidx in future_indices:
            rel_pose = np.linalg.inv(current_pose) @ poses[fidx]
            future_poses.append(rel_pose)

            pts = self._load_lidar_frame(seq_path, fidx)
            pts = self._transform_points(pts, poses[fidx], current_pose, calib)

            occ = build_occupancy_grid(pts, self.POINT_CLOUD_RANGE, self.VOXEL_SIZE, self.GRID_SIZE)
            future_occ_grids.append(torch.from_numpy(occ).float())

        return {
            "past_voxels": past_voxels,
            "past_coords": past_coords,
            "past_num_points": past_num_pts,
            "past_timestamps": torch.tensor(past_timestamps, dtype=torch.float32),
            "current_pose": torch.from_numpy(current_pose).float(),
            "future_poses": torch.from_numpy(np.stack(future_poses)).float(),
            "future_occ": torch.stack(future_occ_grids),
            "seq": info["seq"],
            "frame_idx": frame_idx,
        }

    def _get_stub_item(self, idx: int) -> Dict[str, Any]:
        T_past = self.past_frames
        T_fut = self.future_frames
        P = 100
        N = self.MAX_POINTS_PER_PILLAR
        G = self.GRID_SIZE

        return {
            "past_voxels": [torch.zeros(P, N, 4) for _ in range(T_past)],
            "past_coords": [torch.zeros(P, 2, dtype=torch.long) for _ in range(T_past)],
            "past_num_points": [torch.ones(P, dtype=torch.long) for _ in range(T_past)],
            "past_timestamps": torch.linspace(-0.4, 0.0, T_past),
            "current_pose": torch.eye(4),
            "future_poses": torch.eye(4).unsqueeze(0).expand(T_fut, -1, -1),
            "future_occ": torch.zeros(T_fut, *G),
            "seq": "stub",
            "frame_idx": idx,
        }
