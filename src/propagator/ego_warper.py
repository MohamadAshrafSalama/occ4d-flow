import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def se2_to_affine_matrix(dx: torch.Tensor, dy: torch.Tensor, dyaw: torch.Tensor) -> torch.Tensor:
    """
    Build a 2x3 affine matrix for SE(2) transformation.

    The transformation is: p' = R * p + t
    where R is a 2D rotation by dyaw, and t = (dx, dy).

    For use with F.affine_grid, the coordinates are normalized to [-1, 1].

    Args:
        dx:   (B,) translation along x
        dy:   (B,) translation along y
        dyaw: (B,) rotation angle in radians

    Returns:
        mat: (B, 2, 3) affine matrix
    """
    cos_yaw = torch.cos(dyaw)
    sin_yaw = torch.sin(dyaw)

    mat = torch.stack([
        cos_yaw, -sin_yaw, dx,
        sin_yaw,  cos_yaw, dy,
    ], dim=-1).reshape(-1, 2, 3)

    return mat


class EgoWarper(nn.Module):
    """
    SE(2) ego-motion warping for 3D voxel grids.

    Applies a planar rigid-body transformation to a voxel grid by warping
    each height slice independently. The Z axis is not warped — the
    transformation is purely in the XY (BEV) plane, which is physically
    appropriate for ground-plane ego-motion.

    The transform compensates for ego-vehicle movement between two frames,
    aligning observations from different timestamps into a common frame.

    Uses F.grid_sample with bilinear interpolation for differentiable warping.
    """

    def __init__(
        self,
        interp_mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        self.interp_mode = interp_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(
        self,
        voxel: torch.Tensor,
        dx: torch.Tensor,
        dy: torch.Tensor,
        dyaw: torch.Tensor,
        grid_x_range: float = 1.0,
        grid_y_range: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            voxel: (B, C, X, Y, Z) voxel feature grid
            dx:    (B,) x translation in normalized grid units [-1, 1]
            dy:    (B,) y translation in normalized grid units [-1, 1]
            dyaw:  (B,) rotation in radians (positive = counter-clockwise)

        Returns:
            warped: (B, C, X, Y, Z)
        """
        B, C, X, Y, Z = voxel.shape

        mat = se2_to_affine_matrix(dx, dy, dyaw)

        bev_slice = voxel[:, :, :, :, 0]
        grid = F.affine_grid(mat, (B, C, X, Y), align_corners=self.align_corners)

        warped_slices = []
        for z in range(Z):
            bev_z = voxel[:, :, :, :, z]
            warped_z = F.grid_sample(
                bev_z, grid,
                mode=self.interp_mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )
            warped_slices.append(warped_z)

        warped = torch.stack(warped_slices, dim=-1)
        return warped

    def warp_from_pose(
        self,
        voxel: torch.Tensor,
        pose_current: torch.Tensor,
        pose_reference: torch.Tensor,
        voxel_size: float = 0.5,
        grid_size: int = 200,
    ) -> torch.Tensor:
        """
        Warp voxel from pose_reference frame to pose_current frame.

        Args:
            voxel: (B, C, X, Y, Z) in pose_reference coordinates
            pose_current: (B, 4, 4) current frame SE(3) pose
            pose_reference: (B, 4, 4) reference frame SE(3) pose
            voxel_size: meters per voxel
            grid_size: number of voxels along X and Y

        Returns:
            warped: (B, C, X, Y, Z) in pose_current coordinates
        """
        B = voxel.shape[0]
        device = voxel.device

        rel_pose = torch.linalg.inv(pose_current) @ pose_reference

        dx_m = rel_pose[:, 0, 3]
        dy_m = rel_pose[:, 1, 3]
        dyaw = torch.atan2(rel_pose[:, 1, 0], rel_pose[:, 0, 0])

        grid_half = (grid_size * voxel_size) / 2.0
        dx_norm = dx_m / grid_half
        dy_norm = dy_m / grid_half

        return self.forward(voxel, dx_norm, dy_norm, dyaw)
