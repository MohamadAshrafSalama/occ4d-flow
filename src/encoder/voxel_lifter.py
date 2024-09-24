import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeightBinWeightMLP(nn.Module):
    """
    Learns per-BEV-cell weight distributions over height bins.

    Input: BEV feature vector at each spatial location.
    Output: softmax weight over num_height_bins.

    This allows the network to learn which height bins are likely
    to be occupied given the local 2D context, effectively performing
    learned depth distribution prediction along the vertical axis.
    """

    def __init__(self, bev_channels: int, hidden_dim: int, num_height_bins: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bev_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_height_bins),
        )
        self.num_height_bins = num_height_bins

    def forward(self, bev_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev_feat: (B, C, H, W)
        Returns:
            weights: (B, num_height_bins, H, W) — softmax normalized
        """
        B, C, H, W = bev_feat.shape
        x = bev_feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
        logits = self.net(x)
        weights = F.softmax(logits, dim=-1)
        weights = weights.reshape(B, H, W, self.num_height_bins)
        weights = weights.permute(0, 3, 1, 2)
        return weights


class VoxelLifter(nn.Module):
    """
    Lifts 2D BEV features to a 3D voxel grid.

    For each spatial location (x, y), the BEV feature vector is broadcast
    along Z via learned per-height-bin weights. The weights are produced by a
    small MLP conditioned on the local BEV features, acting as a learned
    occupancy prior over height.

    The output voxel grid has shape (B, voxel_channels, X, Y, Z).

    Channel reduction from bev_channels to voxel_channels is done via a
    1x1 conv before lifting, so the expensive computation happens in the
    lower-dimensional space.
    """

    def __init__(
        self,
        bev_channels: int = 256,
        voxel_channels: int = 128,
        num_height_bins: int = 16,
        height_mlp_hidden: int = 64,
        grid_x: int = 200,
        grid_y: int = 200,
        grid_z: int = 16,
    ) -> None:
        super().__init__()
        self.voxel_channels = voxel_channels
        self.num_height_bins = num_height_bins
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z

        assert num_height_bins == grid_z, (
            f"num_height_bins ({num_height_bins}) must match grid_z ({grid_z})"
        )

        self.feat_proj = nn.Sequential(
            nn.Conv2d(bev_channels, voxel_channels, 1, bias=False),
            nn.GroupNorm(min(32, voxel_channels), voxel_channels),
            nn.ReLU(inplace=True),
        )

        self.height_weight_mlp = HeightBinWeightMLP(
            bev_channels, height_mlp_hidden, num_height_bins
        )

        self.voxel_refine = nn.Sequential(
            nn.Conv3d(voxel_channels, voxel_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, voxel_channels), voxel_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(voxel_channels, voxel_channels, 1, bias=False),
            nn.GroupNorm(min(32, voxel_channels), voxel_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev: (B, bev_channels, H, W) — bird's-eye-view feature map.
                 H and W should match grid_x and grid_y.

        Returns:
            voxels: (B, voxel_channels, X, Y, Z)
        """
        B, C, H, W = bev.shape
        assert H == self.grid_x and W == self.grid_y, (
            f"BEV spatial dims ({H},{W}) do not match grid ({self.grid_x},{self.grid_y})"
        )

        feat = self.feat_proj(bev)
        weights = self.height_weight_mlp(bev)

        feat_z = feat.unsqueeze(2)
        weights_z = weights.unsqueeze(1)

        voxels = feat_z * weights_z

        voxels = self.voxel_refine(voxels)

        return voxels

    def extra_repr(self) -> str:
        return (
            f"grid=({self.grid_x},{self.grid_y},{self.grid_z}), "
            f"channels={self.voxel_channels}"
        )
