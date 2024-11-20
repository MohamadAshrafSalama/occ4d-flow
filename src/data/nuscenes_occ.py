import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.voxel_utils import (
    points_to_voxel_grid,
    build_occupancy_grid,
    apply_se3_to_points,
    compose_ego_transforms,
)


class NuScenesOccupancyDataset(Dataset):
    """
    nuScenes occupancy dataset loader.

    Returns temporal sequences of:
      - Past N frames of voxelized LiDAR point clouds (pillar format)
      - Ego-motion poses for past and future frames
      - Future occupancy grid labels

    Voxel grid: 200 x 200 x 16, voxel size 0.5m.
    Coverage: x in [-50, 50], y in [-50, 50], z in [-5, 3] meters.

    Expected data layout:
        data_root/
            v1.0-trainval/
                scene.json, sample.json, sample_data.json, etc.
            samples/
                LIDAR_TOP/
                    <filename>.pcd.bin
            occupancy/
                <token>.npy   (if pre-generated occupancy labels exist)
    """

    POINT_CLOUD_RANGE = (-50.0, -50.0, -5.0, 50.0, 50.0, 3.0)
    GRID_SIZE = (200, 200, 16)
    VOXEL_SIZE = 0.5
    MAX_POINTS_PER_PILLAR = 20
    MAX_PILLARS = 30000

    def __init__(
        self,
        data_root: str,
        version: str = "v1.0-trainval",
        split: str = "train",
        past_frames: int = 5,
        future_frames: int = 5,
        augment: bool = True,
        aug_flip_x: bool = True,
        aug_flip_y: bool = True,
        aug_rotate_range: float = 0.3927,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.version = version
        self.split = split
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.augment = augment and (split == "train")
        self.aug_flip_x = aug_flip_x
        self.aug_flip_y = aug_flip_y
        self.aug_rotate_range = aug_rotate_range

        self.samples = self._load_split_samples()

    def _load_split_samples(self) -> List[Dict[str, Any]]:
        """
        Load sample tokens and metadata for the requested split.

        Returns a list of dicts, each describing a key frame with enough
        context to load past_frames and future_frames around it.
        """
        try:
            from nuscenes.nuscenes import NuScenes

            nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=False)
            self._nusc = nusc

            split_scenes = self._get_split_scenes(nusc)

            samples = []
            for scene in split_scenes:
                sample_token = scene["first_sample_token"]
                scene_samples = []

                while sample_token:
                    sample = nusc.get("sample", sample_token)
                    scene_samples.append(sample)
                    sample_token = sample["next"]

                total = len(scene_samples)
                for i in range(self.past_frames - 1, total - self.future_frames):
                    samples.append({
                        "scene_token": scene["token"],
                        "past_tokens": [s["token"] for s in scene_samples[i - self.past_frames + 1:i + 1]],
                        "future_tokens": [s["token"] for s in scene_samples[i + 1:i + 1 + self.future_frames]],
                        "current_token": scene_samples[i]["token"],
                    })

            return samples

        except ImportError:
            return self._load_stub_samples()

    def _get_split_scenes(self, nusc) -> list:
        """Return scene list for train/val split."""
        val_scene_names = {
            "scene-0003", "scene-0012", "scene-0013", "scene-0014", "scene-0015",
            "scene-0016", "scene-0017", "scene-0018", "scene-0035", "scene-0036",
        }
        if self.split == "val":
            return [s for s in nusc.scene if s["name"] in val_scene_names]
        return [s for s in nusc.scene if s["name"] not in val_scene_names]

    def _load_stub_samples(self) -> List[Dict]:
        """Fallback stub for testing without nuscenes-devkit installed."""
        return [{"stub": True, "idx": i} for i in range(100)]

    def _load_lidar(self, sample_token: str) -> np.ndarray:
        """Load raw lidar sweep as (N, 4) array."""
        sample = self._nusc.get("sample", sample_token)
        sd_token = sample["data"]["LIDAR_TOP"]
        sd = self._nusc.get("sample_data", sd_token)
        lidar_path = os.path.join(self.data_root, sd["filename"])

        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
        return points[:, :4]

    def _get_pose(self, sample_token: str) -> np.ndarray:
        """Get 4x4 global ego pose for this sample."""
        sample = self._nusc.get("sample", sample_token)
        sd_token = sample["data"]["LIDAR_TOP"]
        sd = self._nusc.get("sample_data", sd_token)
        ep = self._nusc.get("ego_pose", sd["ego_pose_token"])

        pose = np.eye(4, dtype=np.float32)
        from pyquaternion import Quaternion
        q = Quaternion(ep["rotation"])
        pose[:3, :3] = q.rotation_matrix.astype(np.float32)
        pose[:3, 3] = np.array(ep["translation"], dtype=np.float32)
        return pose

    def _get_timestamp(self, sample_token: str) -> float:
        """Return timestamp in seconds."""
        sample = self._nusc.get("sample", sample_token)
        return sample["timestamp"] / 1e6

    def _transform_to_current(
        self, points: np.ndarray, src_pose: np.ndarray, dst_pose: np.ndarray
    ) -> np.ndarray:
        """Transform points from src_pose frame to dst_pose frame."""
        src_to_global = src_pose
        global_to_dst = np.linalg.inv(dst_pose)
        transform = global_to_dst @ src_to_global
        return apply_se3_to_points(points, transform)

    def _augment(self, points_list: List[np.ndarray]) -> List[np.ndarray]:
        """Apply random augmentations to a list of point clouds."""
        flip_x = self.aug_flip_x and np.random.rand() < 0.5
        flip_y = self.aug_flip_y and np.random.rand() < 0.5
        angle = np.random.uniform(-self.aug_rotate_range, self.aug_rotate_range)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        augmented = []
        for pts in points_list:
            pts = pts.copy()
            if flip_x:
                pts[:, 0] = -pts[:, 0]
            if flip_y:
                pts[:, 1] = -pts[:, 1]
            x_rot = pts[:, 0] * cos_a - pts[:, 1] * sin_a
            y_rot = pts[:, 0] * sin_a + pts[:, 1] * cos_a
            pts[:, 0] = x_rot
            pts[:, 1] = y_rot
            augmented.append(pts)

        return augmented

    def _voxelize(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return points_to_voxel_grid(
            points,
            self.POINT_CLOUD_RANGE,
            self.VOXEL_SIZE,
            self.GRID_SIZE,
            self.MAX_POINTS_PER_PILLAR,
            self.MAX_PILLARS,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_info = self.samples[idx]

        if sample_info.get("stub"):
            return self._get_stub_item(idx)

        current_token = sample_info["current_token"]
        current_pose = self._get_pose(current_token)
        current_ts = self._get_timestamp(current_token)

        past_points_list = []
        past_timestamps = []

        for token in sample_info["past_tokens"]:
            pts = self._load_lidar(token)
            pose = self._get_pose(token)
            pts_current = self._transform_to_current(pts, pose, current_pose)
            past_points_list.append(pts_current)
            past_timestamps.append(self._get_timestamp(token) - current_ts)

        if self.augment:
            past_points_list = self._augment(past_points_list)

        past_voxels, past_coords, past_num_pts = [], [], []
        for pts in past_points_list:
            v, c, n = self._voxelize(pts)
            past_voxels.append(torch.from_numpy(v).float())
            past_coords.append(torch.from_numpy(c).long())
            past_num_pts.append(torch.from_numpy(n).long())

        future_occ_grids = []
        future_poses = []

        for token in sample_info["future_tokens"]:
            pose = self._get_pose(token)
            rel_pose = np.linalg.inv(current_pose) @ pose
            future_poses.append(rel_pose)

            pts_future = self._load_lidar(token)
            pts_in_current = self._transform_to_current(pts_future, pose, current_pose)
            occ = build_occupancy_grid(pts_in_current, self.POINT_CLOUD_RANGE, self.VOXEL_SIZE, self.GRID_SIZE)
            future_occ_grids.append(torch.from_numpy(occ).float())

        future_poses_tensor = torch.from_numpy(np.stack(future_poses)).float()
        future_occ_tensor = torch.stack(future_occ_grids, dim=0)

        past_ts_tensor = torch.tensor(past_timestamps, dtype=torch.float32)
        current_pose_tensor = torch.from_numpy(current_pose).float()

        return {
            "past_voxels": past_voxels,
            "past_coords": past_coords,
            "past_num_points": past_num_pts,
            "past_timestamps": past_ts_tensor,
            "current_pose": current_pose_tensor,
            "future_poses": future_poses_tensor,
            "future_occ": future_occ_tensor,
            "current_token": current_token,
        }

    def _get_stub_item(self, idx: int) -> Dict[str, Any]:
        """Return a zeroed-out stub sample for testing without data."""
        T_past = self.past_frames
        T_fut = self.future_frames
        P = 100
        N = self.MAX_POINTS_PER_PILLAR
        G = self.GRID_SIZE

        return {
            "past_voxels": [torch.zeros(P, N, 4) for _ in range(T_past)],
            "past_coords": [torch.zeros(P, 2, dtype=torch.long) for _ in range(T_past)],
            "past_num_points": [torch.ones(P, dtype=torch.long) for _ in range(T_past)],
            "past_timestamps": torch.linspace(-2.0, 0.0, T_past),
            "current_pose": torch.eye(4),
            "future_poses": torch.eye(4).unsqueeze(0).expand(T_fut, -1, -1),
            "future_occ": torch.zeros(T_fut, *G),
            "current_token": f"stub_{idx}",
        }
