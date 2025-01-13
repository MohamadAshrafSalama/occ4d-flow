import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RadialRangeEncoding(nn.Module):
    """Sinusoidal positional encoding over radial distance r = sqrt(x^2 + y^2)."""

    def __init__(self, encoding_dim: int = 32, max_freq: int = 8) -> None:
        super().__init__()
        assert encoding_dim % 2 == 0, "encoding_dim must be even"
        self.encoding_dim = encoding_dim
        freqs = torch.linspace(0.0, max_freq - 1, encoding_dim // 2)
        self.register_buffer("freqs", freqs)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        r = r.unsqueeze(-1)
        phases = r * (2.0 * math.pi * self.freqs)
        return torch.cat([phases.sin(), phases.cos()], dim=-1)


class PillarFeatureNet(nn.Module):
    """
    Per-pillar MLP that maps raw point features to a pillar-level descriptor.

    Input features per point:
        x_c, y_c, z_c  — coordinates relative to pillar centroid
        intensity       — lidar return intensity
        r               — radial distance from sensor
        z_abs           — absolute height

    Total: 6 features per point.
    """

    def __init__(
        self,
        in_channels: int = 6,
        feat_channels: List[int] = (64, 128),
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_channels
        for ch in feat_channels:
            layers.extend([
                nn.Linear(prev, ch, bias=False),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
            ])
            prev = ch
        self.net = nn.Sequential(*layers)
        self.out_channels = prev

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, P, N, C = features.shape
        x = features.reshape(B * P * N, C)
        x = self.net(x)
        x = x.reshape(B, P, N, -1)
        x = x.max(dim=2).values
        return x


class PillarScatter(nn.Module):
    """Scatter pillar descriptors back onto a 2D BEV canvas."""

    def __init__(self, grid_x: int = 200, grid_y: int = 200) -> None:
        super().__init__()
        self.grid_x = grid_x
        self.grid_y = grid_y

    def forward(
        self,
        pillar_features: torch.Tensor,
        pillar_coords: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        C = pillar_features.shape[-1]
        canvas = pillar_features.new_zeros(batch_size, C, self.grid_x, self.grid_y)

        batch_idx = pillar_coords[:, 0].long()
        x_idx = pillar_coords[:, 1].long().clamp(0, self.grid_x - 1)
        y_idx = pillar_coords[:, 2].long().clamp(0, self.grid_y - 1)

        canvas[batch_idx, :, x_idx, y_idx] = pillar_features
        return canvas


class PillarEncoder(nn.Module):
    """
    Pillar-based LiDAR encoder.

    For each non-empty pillar in the voxelized BEV grid:
      - Compute per-point features (relative xyz, intensity, radial distance, abs z)
      - Apply shared PointNet-style MLP
      - Max-pool over points within each pillar
      - Scatter back to BEV canvas
      - Apply radial range encoding to each BEV cell
      - Final projection conv to out_channels
    """

    def __init__(
        self,
        in_channels: int = 6,
        pillar_feat_channels: List[int] = (64, 128),
        bev_channels: int = 128,
        radial_encoding_dim: int = 32,
        radial_max_freq: int = 8,
        out_channels: int = 256,
        grid_x: int = 200,
        grid_y: int = 200,
        voxel_x: float = 0.5,
        voxel_y: float = 0.5,
        x_offset: float = -50.0,
        y_offset: float = -50.0,
    ) -> None:
        super().__init__()
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.voxel_x = voxel_x
        self.voxel_y = voxel_y
        self.x_offset = x_offset
        self.y_offset = y_offset

        self.pillar_net = PillarFeatureNet(in_channels, list(pillar_feat_channels))
        pillar_out = self.pillar_net.out_channels

        self.scatter = PillarScatter(grid_x, grid_y)

        self.radial_encoding = RadialRangeEncoding(radial_encoding_dim, radial_max_freq)

        self.bev_proj = nn.Sequential(
            nn.Conv2d(pillar_out, bev_channels, 1, bias=False),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
        )

        bev_with_radial = bev_channels + radial_encoding_dim
        self.out_proj = nn.Sequential(
            nn.Conv2d(bev_with_radial, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.out_channels = out_channels
        self._init_radial_grid()

    def _init_radial_grid(self) -> None:
        xs = torch.arange(self.grid_x) * self.voxel_x + self.x_offset + self.voxel_x / 2
        ys = torch.arange(self.grid_y) * self.voxel_y + self.y_offset + self.voxel_y / 2
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        r = torch.sqrt(grid_x ** 2 + grid_y ** 2)
        self.register_buffer("radial_grid", r)

    def _compute_pillar_features(
        self, points: torch.Tensor, pillar_coords: torch.Tensor, num_pillars: int
    ) -> torch.Tensor:
        B, P, N, _ = points.shape

        centroid_x = points[..., 0].mean(dim=2, keepdim=True)
        centroid_y = points[..., 1].mean(dim=2, keepdim=True)
        centroid_z = points[..., 2].mean(dim=2, keepdim=True)

        xc = points[..., 0:1] - centroid_x
        yc = points[..., 1:2] - centroid_y
        zc = points[..., 2:3] - centroid_z
        intensity = points[..., 3:4]
        r = torch.sqrt(points[..., 0:1] ** 2 + points[..., 1:2] ** 2).clamp(min=1e-3)
        z_abs = points[..., 2:3]

        features = torch.cat([xc, yc, zc, intensity, r, z_abs], dim=-1)
        return features

    def forward(
        self,
        points: torch.Tensor,
        pillar_coords: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Args:
            points: (B, P, N, 4) — x, y, z, intensity per point in each pillar.
                    P = max pillars, N = max points per pillar.
            pillar_coords: (B*P_valid, 3) — batch_idx, x_idx, y_idx in BEV grid.
            batch_size: int

        Returns:
            bev: (B, out_channels, grid_x, grid_y)
        """
        feats = self._compute_pillar_features(points, pillar_coords, points.shape[1])
        pillar_desc = self.pillar_net(feats)

        B, P, C = pillar_desc.shape
        flat_desc = pillar_desc.reshape(B * P, C)

        flat_coords = []
        for b in range(B):
            b_idx = torch.full((P, 1), b, dtype=pillar_coords.dtype, device=pillar_coords.device)
            coords_b = torch.cat([b_idx, pillar_coords[b * P:(b + 1) * P]], dim=1)
            flat_coords.append(coords_b)
        flat_coords = torch.cat(flat_coords, dim=0)

        bev = self.scatter(flat_desc, flat_coords, batch_size)
        bev = self.bev_proj(bev)

        r_enc = self.radial_encoding(self.radial_grid)
        r_enc = r_enc.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)

        bev = torch.cat([bev, r_enc], dim=1)
        bev = self.out_proj(bev)

        return bev


def build_pillar_encoder(cfg) -> PillarEncoder:
    return PillarEncoder(
        in_channels=cfg.in_channels,
        pillar_feat_channels=cfg.pillar_feat_channels,
        bev_channels=cfg.bev_channels,
        radial_encoding_dim=cfg.radial_encoding_dim,
        out_channels=cfg.out_channels,
        grid_x=cfg.grid_x,
        grid_y=cfg.grid_y,
        voxel_x=cfg.voxel_x,
        voxel_y=cfg.voxel_y,
        x_offset=cfg.x_offset,
        y_offset=cfg.y_offset,
    )

