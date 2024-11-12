from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss between consecutive predicted occupancy maps.

    Warps the t-1 prediction into the coordinate frame of t using the
    known ego-motion transformation, then penalizes the L2 distance
    between the warped prediction and the t prediction.

    This encourages smooth, physically consistent temporal evolution
    of the predicted occupancy rather than independent per-frame prediction.

    Expected to be computed between consecutive future steps.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def _warp_bev(
        self,
        occ_prev: torch.Tensor,
        dx: torch.Tensor,
        dy: torch.Tensor,
        dyaw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Warp occupancy from t-1 to t frame via SE(2) grid sampling.

        Args:
            occ_prev: (B, 1, X, Y, Z)
            dx, dy:   (B,) translation in normalized coords
            dyaw:     (B,) rotation in radians

        Returns:
            warped: (B, 1, X, Y, Z)
        """
        B, C, X, Y, Z = occ_prev.shape
        cos_yaw = torch.cos(dyaw)
        sin_yaw = torch.sin(dyaw)

        mat = torch.stack([
            cos_yaw, -sin_yaw, dx,
            sin_yaw,  cos_yaw, dy,
        ], dim=-1).reshape(B, 2, 3)

        warped_slices = []
        for z in range(Z):
            bev_z = occ_prev[:, :, :, :, z]
            grid = F.affine_grid(mat, (B, C, X, Y), align_corners=True)
            w = F.grid_sample(bev_z, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            warped_slices.append(w)

        return torch.stack(warped_slices, dim=-1)

    def forward(
        self,
        occ_t: torch.Tensor,
        occ_prev: torch.Tensor,
        dx: Optional[torch.Tensor] = None,
        dy: Optional[torch.Tensor] = None,
        dyaw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            occ_t:    (B, 1, X, Y, Z) occupancy at current step
            occ_prev: (B, 1, X, Y, Z) occupancy at previous step
            dx, dy:   (B,) ego-motion translation (normalized)
            dyaw:     (B,) ego-motion yaw delta

        Returns:
            loss: scalar
        """
        if dx is not None and dy is not None and dyaw is not None:
            warped_prev = self._warp_bev(occ_prev, dx, dy, dyaw)
        else:
            warped_prev = occ_prev

        diff = (occ_t - warped_prev) ** 2

        if self.reduction == "mean":
            return diff.mean()
        return diff.sum()

    def forward_sequence(
        self,
        occ_sequence: List[torch.Tensor],
        dx_sequence: Optional[List[torch.Tensor]] = None,
        dy_sequence: Optional[List[torch.Tensor]] = None,
        dyaw_sequence: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute consistency loss across all consecutive pairs in a sequence."""
        T = len(occ_sequence)
        total = occ_sequence[0].new_zeros(1)

        for t in range(1, T):
            dx = dx_sequence[t - 1] if dx_sequence else None
            dy = dy_sequence[t - 1] if dy_sequence else None
            dyaw = dyaw_sequence[t - 1] if dyaw_sequence else None
            total = total + self.forward(occ_sequence[t], occ_sequence[t - 1], dx, dy, dyaw)

        return total / max(T - 1, 1)


class VelocitySmoothnessLoss(nn.Module):
    """
    Spatial smoothness regularization on predicted velocity fields.

    Computes the L2 norm of spatial gradients of the velocity field v:
        L = sum_{axis} ||dv/d_axis||^2

    Applied to the velocity predictions from the flow matching U-Net.
    Encourages spatially smooth velocity fields, preventing sharp
    discontinuities that are physically unrealistic.
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Args:
            velocity: (B, C, X, Y, Z) predicted velocity field

        Returns:
            loss: scalar
        """
        gx = (velocity[:, :, 1:, :, :] - velocity[:, :, :-1, :, :]) ** 2
        gy = (velocity[:, :, :, 1:, :] - velocity[:, :, :, :-1, :]) ** 2
        gz = (velocity[:, :, :, :, 1:] - velocity[:, :, :, :, :-1]) ** 2

        n_elements = gx.numel() + gy.numel() + gz.numel()
        penalty = (gx.sum() + gy.sum() + gz.sum()) / n_elements

        return self.weight * penalty


class TemporalLoss(nn.Module):
    """Combined temporal loss: consistency + velocity smoothness."""

    def __init__(
        self,
        consistency_weight: float = 0.5,
        smoothness_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.consistency = TemporalConsistencyLoss()
        self.smoothness = VelocitySmoothnessLoss()
        self.consistency_weight = consistency_weight
        self.smoothness_weight = smoothness_weight

    def forward(
        self,
        occ_sequence: List[torch.Tensor],
        velocity_sequence: Optional[List[torch.Tensor]] = None,
        dx_sequence: Optional[List[torch.Tensor]] = None,
        dy_sequence: Optional[List[torch.Tensor]] = None,
        dyaw_sequence: Optional[List[torch.Tensor]] = None,
    ) -> dict:
        cons_loss = self.consistency.forward_sequence(
            occ_sequence, dx_sequence, dy_sequence, dyaw_sequence
        )

        smooth_loss = occ_sequence[0].new_zeros(1)
        if velocity_sequence is not None:
            for v in velocity_sequence:
                smooth_loss = smooth_loss + self.smoothness(v)
            smooth_loss = smooth_loss / len(velocity_sequence)

        total = (
            self.consistency_weight * cons_loss
            + self.smoothness_weight * smooth_loss
        )

        return {
            "temporal_consistency": cons_loss,
            "velocity_smoothness": smooth_loss,
            "temporal_total": total,
        }
