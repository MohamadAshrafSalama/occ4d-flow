from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def points_to_voxel_grid(
    points: np.ndarray,
    point_cloud_range: Tuple[float, ...],
    voxel_size: float,
    grid_size: Tuple[int, int, int],
    max_points_per_voxel: int = 20,
    max_voxels: int = 30000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Voxelize a point cloud into a sparse pillar/voxel representation.

    Args:
        points: (N, 4) array of x, y, z, intensity
        point_cloud_range: (x_min, y_min, z_min, x_max, y_max, z_max)
        voxel_size: uniform voxel edge length in meters
        grid_size: (X, Y, Z) number of voxels per dimension
        max_points_per_voxel: truncation per voxel
        max_voxels: maximum number of non-empty voxels

    Returns:
        voxels: (V, max_pts, 4) point features per voxel
        coords: (V, 3) voxel indices (xi, yi, zi)
        num_points: (V,) actual point count per voxel
    """
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    gx, gy, gz = grid_size

    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] < z_max)
    )
    points = points[mask]

    xi = np.floor((points[:, 0] - x_min) / voxel_size).astype(np.int32)
    yi = np.floor((points[:, 1] - y_min) / voxel_size).astype(np.int32)
    zi = np.floor((points[:, 2] - z_min) / voxel_size).astype(np.int32)

    xi = np.clip(xi, 0, gx - 1)
    yi = np.clip(yi, 0, gy - 1)
    zi = np.clip(zi, 0, gz - 1)

    voxel_idx = xi * gy * gz + yi * gz + zi

    unique_idx, inverse_map = np.unique(voxel_idx, return_inverse=True)

    num_voxels = min(len(unique_idx), max_voxels)
    voxels = np.zeros((num_voxels, max_points_per_voxel, points.shape[1]), dtype=np.float32)
    num_points_per_voxel = np.zeros(num_voxels, dtype=np.int32)

    flat_idx = unique_idx[:num_voxels]
    coords_xi = (flat_idx // (gy * gz)).astype(np.int32)
    coords_yi = ((flat_idx % (gy * gz)) // gz).astype(np.int32)
    coords_zi = (flat_idx % gz).astype(np.int32)
    coords = np.stack([coords_xi, coords_yi, coords_zi], axis=1)

    for i in range(len(points)):
        v = inverse_map[i]
        if v >= num_voxels:
            continue
        n = num_points_per_voxel[v]
        if n < max_points_per_voxel:
            voxels[v, n] = points[i]
            num_points_per_voxel[v] += 1

    return voxels, coords, num_points_per_voxel


def voxel_coord_to_metric(
    coords: np.ndarray,
    point_cloud_range: Tuple[float, ...],
    voxel_size: float,
) -> np.ndarray:
    """
    Convert integer voxel coordinates to metric space (voxel center).

    Args:
        coords: (..., 3) integer indices (xi, yi, zi)
        point_cloud_range: (x_min, y_min, z_min, x_max, y_max, z_max)
        voxel_size: meters per voxel

    Returns:
        metric: (..., 3) x, y, z in meters
    """
    x_min, y_min, z_min = point_cloud_range[:3]
    origin = np.array([x_min, y_min, z_min], dtype=np.float32)
    return coords * voxel_size + origin + voxel_size / 2.0


def metric_to_voxel_coord(
    points_m: np.ndarray,
    point_cloud_range: Tuple[float, ...],
    voxel_size: float,
    grid_size: Tuple[int, int, int],
) -> np.ndarray:
    """
    Convert metric coordinates to integer voxel indices.

    Args:
        points_m: (..., 3) x, y, z in meters
        point_cloud_range: (x_min, y_min, z_min, ...)
        voxel_size: meters per voxel
        grid_size: (X, Y, Z) for clipping

    Returns:
        coords: (..., 3) integer voxel indices
    """
    x_min, y_min, z_min = point_cloud_range[:3]
    origin = np.array([x_min, y_min, z_min], dtype=np.float32)
    coords = np.floor((points_m - origin) / voxel_size).astype(np.int32)
    coords[..., 0] = np.clip(coords[..., 0], 0, grid_size[0] - 1)
    coords[..., 1] = np.clip(coords[..., 1], 0, grid_size[1] - 1)
    coords[..., 2] = np.clip(coords[..., 2], 0, grid_size[2] - 1)
    return coords


def compose_ego_transforms(poses: np.ndarray) -> np.ndarray:
    """
    Compose a sequence of SE(3) poses into relative transforms.

    Given absolute poses [T_0, T_1, ..., T_{N-1}], compute
    relative transforms T_ref_to_i = T_i_inv @ T_{i-1} for each consecutive pair.

    Args:
        poses: (N, 4, 4) sequence of SE(3) transformation matrices

    Returns:
        rel_poses: (N-1, 4, 4) relative transforms
    """
    N = poses.shape[0]
    rel_poses = np.zeros((N - 1, 4, 4), dtype=np.float32)
    for i in range(1, N):
        rel_poses[i - 1] = np.linalg.inv(poses[i]) @ poses[i - 1]
    return rel_poses


def build_occupancy_grid(
    points: np.ndarray,
    point_cloud_range: Tuple[float, ...],
    voxel_size: float,
    grid_size: Tuple[int, int, int],
) -> np.ndarray:
    """
    Build a binary occupancy grid from a point cloud.

    Args:
        points: (N, 3+) point cloud
        point_cloud_range: bounding box
        voxel_size: resolution
        grid_size: (X, Y, Z)

    Returns:
        grid: (X, Y, Z) binary float32 array
    """
    gx, gy, gz = grid_size
    grid = np.zeros((gx, gy, gz), dtype=np.float32)

    coords = metric_to_voxel_coord(points[:, :3], point_cloud_range, voxel_size, grid_size)
    valid = (
        (coords[:, 0] >= 0) & (coords[:, 0] < gx) &
        (coords[:, 1] >= 0) & (coords[:, 1] < gy) &
        (coords[:, 2] >= 0) & (coords[:, 2] < gz)
    )
    coords = coords[valid]
    grid[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
    return grid


def apply_se3_to_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 SE(3) matrix to a (N, 3) point cloud.

    Args:
        points: (N, 3) or (N, 4+)
        transform: (4, 4) SE(3) matrix

    Returns:
        transformed: same shape as points
    """
    xyz = points[:, :3]
    ones = np.ones((len(xyz), 1), dtype=xyz.dtype)
    homogeneous = np.concatenate([xyz, ones], axis=1)
    transformed = (transform @ homogeneous.T).T[:, :3]
    result = points.copy()
    result[:, :3] = transformed
    return result
