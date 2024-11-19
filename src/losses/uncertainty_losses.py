import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyNLLLoss(nn.Module):
    """
    Gaussian negative log-likelihood loss weighted by predicted uncertainty.

    For a Gaussian model p(y | x) = N(mu, sigma^2), the NLL is:
        NLL = 0.5 * log(2*pi*sigma^2) + (y - mu)^2 / (2*sigma^2)
            = 0.5 * logvar + 0.5 * (y - mu)^2 * exp(-logvar) + const

    where logvar = log(sigma^2) is the predicted log-variance.

    The network simultaneously predicts the mean and uncertainty.
    Well-calibrated uncertainty means the model assigns high uncertainty
    where it makes large prediction errors.

    A clipping is applied to logvar to avoid numerical instability.
    """

    def __init__(
        self,
        min_logvar: float = -10.0,
        max_logvar: float = 10.0,
    ) -> None:
        super().__init__()
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        logvar: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, X, Y, Z) predicted occupancy probability or logit
            target: (B, 1, X, Y, Z) ground-truth labels
            logvar: (B, 1, X, Y, Z) predicted log-variance
            mask: optional (B, 1, X, Y, Z) binary mask of valid voxels

        Returns:
            loss: scalar
        """
        logvar = logvar.clamp(self.min_logvar, self.max_logvar)
        target = target.float()

        sq_err = (pred - target) ** 2
        nll = 0.5 * logvar + 0.5 * sq_err * torch.exp(-logvar)

        if mask is not None:
            nll = nll * mask
            return nll.sum() / (mask.sum() + 1e-8)

        return nll.mean()


class RangeWeightedLoss(nn.Module):
    """
    Range-weighted occupancy loss.

    Voxels closer to the sensor receive higher loss weight because
    LiDAR point density is higher at short range and the observations
    are more accurate. Distant voxels are down-weighted due to sparsity
    and range-noise effects.

    Weight function (linear decay with distance):
        w(r) = w_max - (w_max - w_min) * min(r / r_cutoff, 1.0)

    where r is the radial distance of each voxel from sensor origin.
    """

    def __init__(
        self,
        weight_max: float = 3.0,
        weight_min: float = 1.0,
        range_cutoff: float = 40.0,
        voxel_size: float = 0.5,
        grid_origin: Tuple[float, float] = (-50.0, -50.0),
        grid_size: Tuple[int, int] = (200, 200),
    ) -> None:
        super().__init__()
        self.weight_max = weight_max
        self.weight_min = weight_min
        self.range_cutoff = range_cutoff

        xs = torch.arange(grid_size[0]) * voxel_size + grid_origin[0] + voxel_size / 2
        ys = torch.arange(grid_size[1]) * voxel_size + grid_origin[1] + voxel_size / 2
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        r = torch.sqrt(grid_x ** 2 + grid_y ** 2)

        t = (r / range_cutoff).clamp(0.0, 1.0)
        weight_map = weight_max - (weight_max - weight_min) * t

        self.register_buffer("weight_map", weight_map)

    def get_voxel_weights(self, batch_size: int) -> torch.Tensor:
        """
        Returns (B, 1, X, Y, 1) weight tensor for broadcasting.
        """
        w = self.weight_map.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        return w.expand(batch_size, 1, -1, -1, -1)

    def forward(
        self,
        loss_per_voxel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply range weights to a per-voxel loss tensor.

        Args:
            loss_per_voxel: (B, 1, X, Y, Z) unreduced loss

        Returns:
            weighted_loss: scalar
        """
        B = loss_per_voxel.shape[0]
        weights = self.get_voxel_weights(B)

        Z = loss_per_voxel.shape[-1]
        if weights.shape[-1] == 1:
            weights = weights.expand_as(loss_per_voxel)

        return (loss_per_voxel * weights).mean()


class UncertaintyLoss(nn.Module):
    """
    Combined uncertainty loss: Gaussian NLL + range-weighted BCE.

    Encourages the model to:
      1. Predict accurate uncertainty estimates (via NLL)
      2. Give higher priority to nearby well-observed voxels (via range weighting)
    """

    def __init__(
        self,
        nll_weight: float = 0.5,
        range_weight_max: float = 3.0,
        range_weight_min: float = 1.0,
        range_cutoff: float = 40.0,
        voxel_size: float = 0.5,
        grid_origin: Tuple[float, float] = (-50.0, -50.0),
        grid_size: Tuple[int, int] = (200, 200),
    ) -> None:
        super().__init__()
        self.nll = UncertaintyNLLLoss()
        self.range_weighter = RangeWeightedLoss(
            weight_max=range_weight_max,
            weight_min=range_weight_min,
            range_cutoff=range_cutoff,
            voxel_size=voxel_size,
            grid_origin=grid_origin,
            grid_size=grid_size,
        )
        self.nll_weight = nll_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        logvar: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict:
        nll_loss = self.nll(pred, target, logvar, mask)

        target_f = target.float()
        bce = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), target_f, reduction="none")
        range_loss = self.range_weighter(bce)

        total = self.nll_weight * nll_loss + (1.0 - self.nll_weight) * range_loss

        return {
            "uncertainty_nll": nll_loss,
            "range_weighted_bce": range_loss,
            "uncertainty_total": total,
        }
