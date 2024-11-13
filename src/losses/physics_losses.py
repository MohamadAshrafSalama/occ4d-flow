from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RayFreespaceLoss(nn.Module):
    """
    Ray-traced free-space loss.

    For each sensor ray that terminates at a LiDAR return, all voxels
    between the sensor origin and the first return should be free (unoccupied).
    If the model predicts occupancy in those free-space voxels, it is penalized.

    Implementation:
      - Cast rays from the sensor origin along precomputed ray directions
      - For each ray, compute a cumulative mask of voxels that are "before the hit"
      - Penalize the mean occupancy probability within that mask

    For efficiency, rays are sampled from a precomputed ray index tensor
    rather than computing exact Bresenham paths, which would be slow in PyTorch.
    The ray mask is computed using a soft front-to-back cumulative product.
    """

    def __init__(self, weight: float = 1.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = weight
        self.eps = eps

    def _build_freespace_mask(
        self,
        depth_indices: torch.Tensor,
        grid_z: int,
    ) -> torch.Tensor:
        """
        Build a binary mask (1 = free space, 0 = at or beyond hit).

        Args:
            depth_indices: (B, N_rays) integer index of hit voxel along ray
            grid_z: total number of voxels along ray axis

        Returns:
            mask: (B, N_rays, grid_z)
        """
        B, N = depth_indices.shape
        arange = torch.arange(grid_z, device=depth_indices.device)
        arange = arange.unsqueeze(0).unsqueeze(0).expand(B, N, grid_z)
        depth_exp = depth_indices.unsqueeze(-1).expand(B, N, grid_z)
        mask = (arange < depth_exp).float()
        return mask

    def forward(
        self,
        occ_prob: torch.Tensor,
        ray_origins: torch.Tensor,
        ray_hit_coords: torch.Tensor,
        voxel_size: float = 0.5,
        grid_origin: Tuple[float, float, float] = (-50.0, -50.0, -5.0),
    ) -> torch.Tensor:
        """
        Args:
            occ_prob: (B, 1, X, Y, Z) predicted occupancy probability
            ray_origins: (B, 3) sensor origin in metric space
            ray_hit_coords: (B, N_rays, 3) 3D coordinates of first LiDAR returns
            voxel_size: meters per voxel
            grid_origin: (ox, oy, oz) metric coordinates of grid corner

        Returns:
            loss: scalar free-space penalty
        """
        B, _, X, Y, Z = occ_prob.shape
        N = ray_hit_coords.shape[1]
        device = occ_prob.device

        ox, oy, oz = grid_origin
        origin_grid = (ray_origins - ray_origins.new_tensor([ox, oy, oz])) / voxel_size

        hit_grid = (ray_hit_coords - ray_hit_coords.new_tensor([ox, oy, oz]).unsqueeze(0)) / voxel_size

        direction = hit_grid - origin_grid.unsqueeze(1)
        ray_length = direction.norm(dim=-1, keepdim=True).clamp(min=1.0)
        direction_norm = direction / ray_length

        num_steps = int(ray_length.max().item()) + 1
        num_steps = min(num_steps, max(X, Y, Z))

        t_vals = torch.linspace(0.0, 1.0, num_steps, device=device)

        ray_start = origin_grid.unsqueeze(1).unsqueeze(2)
        ray_end = hit_grid.unsqueeze(2)

        sample_coords = ray_start + t_vals.reshape(1, 1, num_steps, 1) * (ray_end - ray_start)

        sample_norm = sample_coords.clone()
        sample_norm[..., 0] = 2.0 * sample_coords[..., 0] / X - 1.0
        sample_norm[..., 1] = 2.0 * sample_coords[..., 1] / Y - 1.0
        sample_norm[..., 2] = 2.0 * sample_coords[..., 2] / Z - 1.0

        grid_5d = sample_norm.reshape(B, N * num_steps, 1, 1, 3)

        occ_grid = occ_prob.expand(B, 1, X, Y, Z)
        sampled = F.grid_sample(
            occ_grid, grid_5d,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )
        sampled = sampled.reshape(B, N, num_steps)

        freespace_mask = torch.zeros(B, N, num_steps, device=device)
        freespace_mask[:, :, :-1] = 1.0

        loss = (sampled * freespace_mask).sum() / (freespace_mask.sum() + self.eps)
        return self.weight * loss

    def forward_simple(
        self,
        occ_prob: torch.Tensor,
        freespace_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simplified version: directly takes a precomputed binary freespace mask.

        Args:
            occ_prob: (B, 1, X, Y, Z)
            freespace_mask: (B, 1, X, Y, Z) binary; 1 = should be free

        Returns:
            loss: scalar
        """
        freespace_occ = occ_prob * freespace_mask
        return self.weight * freespace_occ.mean()


class MassConservationLoss(nn.Module):
    """
    Mass conservation loss across future timesteps.

    Physical scene dynamics conserve the total amount of occupied space
    approximately — objects don't appear or disappear instantly. This
    loss penalizes large absolute changes in the predicted total mass
    between consecutive future steps.

        L_mass = (1/T) * sum_t |sum_{xyz} occ_t - sum_{xyz} occ_{t-1}|
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, occ_sequence: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            occ_sequence: T-list of (B, 1, X, Y, Z) occupancy probabilities

        Returns:
            loss: scalar
        """
        T = len(occ_sequence)
        if T < 2:
            return occ_sequence[0].new_zeros(1).squeeze()

        mass_sequence = [occ.sum(dim=(1, 2, 3, 4)) for occ in occ_sequence]

        total = occ_sequence[0].new_zeros(1)
        for t in range(1, T):
            diff = (mass_sequence[t] - mass_sequence[t - 1]).abs().mean()
            total = total + diff

        return self.weight * total / (T - 1)


class PhysicsLoss(nn.Module):
    """Combined physics-based loss: ray free-space + mass conservation."""

    def __init__(
        self,
        ray_freespace_weight: float = 0.3,
        mass_conservation_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.ray_loss = RayFreespaceLoss(weight=ray_freespace_weight)
        self.mass_loss = MassConservationLoss(weight=mass_conservation_weight)

    def forward(
        self,
        occ_sequence: List[torch.Tensor],
        freespace_masks: Optional[List[torch.Tensor]] = None,
    ) -> dict:
        mass_loss = self.mass_loss(occ_sequence)

        ray_loss = occ_sequence[0].new_zeros(1).squeeze()
        if freespace_masks is not None:
            for occ, mask in zip(occ_sequence, freespace_masks):
                ray_loss = ray_loss + self.ray_loss.forward_simple(occ, mask)
            ray_loss = ray_loss / len(occ_sequence)

        total = mass_loss + ray_loss

        return {
            "ray_freespace": ray_loss,
            "mass_conservation": mass_loss,
            "physics_total": total,
        }
