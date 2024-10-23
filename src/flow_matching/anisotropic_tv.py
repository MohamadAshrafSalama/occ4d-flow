from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnisotropicTotalVariation(nn.Module):
    """
    Anisotropic total variation regularization for 3D voxel grids.

    Penalizes absolute differences between adjacent voxels along each axis
    independently. Each axis has its own weight, allowing the regularizer
    to encode geometric priors:

        TV(x) = w_x * sum|x[i+1,j,k] - x[i,j,k]|
              + w_y * sum|x[i,j+1,k] - x[i,j,k]|
              + w_z * sum|x[i,j,k+1] - x[i,j,k]|

    The Z axis (height / gravity axis) gets a higher weight to preserve
    vertical structure — occupancy tends to have sharp horizontal boundaries
    (floor, ceiling, object tops), which TV should not smooth across.

    Unlike isotropic TV (which penalizes the L2 norm of the gradient vector),
    anisotropic TV penalizes each directional difference independently and
    admits sparser solutions.
    """

    def __init__(
        self,
        weight_x: float = 0.01,
        weight_y: float = 0.01,
        weight_z: float = 0.05,
        reduction: str = "mean",
        eps: float = 1e-8,
        use_smooth_abs: bool = False,
    ) -> None:
        super().__init__()
        self.weight_x = weight_x
        self.weight_y = weight_y
        self.weight_z = weight_z
        self.reduction = reduction
        self.eps = eps
        self.use_smooth_abs = use_smooth_abs

    def _abs(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_smooth_abs:
            return torch.sqrt(x ** 2 + self.eps)
        return x.abs()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, X, Y, Z) or (B, X, Y, Z)

        Returns:
            tv_loss: scalar
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)

        B, C, X, Y, Z = x.shape

        diff_x = self._abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
        diff_y = self._abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
        diff_z = self._abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])

        if self.reduction == "mean":
            tv_x = diff_x.mean()
            tv_y = diff_y.mean()
            tv_z = diff_z.mean()
        elif self.reduction == "sum":
            tv_x = diff_x.sum()
            tv_y = diff_y.sum()
            tv_z = diff_z.sum()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        return self.weight_x * tv_x + self.weight_y * tv_y + self.weight_z * tv_z

    def per_axis(self, x: torch.Tensor):
        """Returns TV losses for each axis separately, as a dict."""
        if x.dim() == 4:
            x = x.unsqueeze(1)

        diff_x = self._abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).mean()
        diff_y = self._abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).mean()
        diff_z = self._abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).mean()

        return {
            "tv_x": self.weight_x * diff_x,
            "tv_y": self.weight_y * diff_y,
            "tv_z": self.weight_z * diff_z,
        }

    def extra_repr(self) -> str:
        return (
            f"weight_x={self.weight_x}, weight_y={self.weight_y}, "
            f"weight_z={self.weight_z}, reduction={self.reduction}"
        )


class SpatialGradientPenalty(nn.Module):
    """
    Isotropic spatial gradient penalty (L2 norm of 3D gradient vector).

    Used for velocity smoothness regularization in the flow matching loss.
    Unlike AnisotropicTotalVariation, this penalizes the magnitude of the
    full gradient vector, resulting in smoother transitions.
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v: (B, C, X, Y, Z) velocity field

        Returns:
            penalty: scalar
        """
        gx = (v[:, :, 1:, :, :] - v[:, :, :-1, :, :]) ** 2
        gy = (v[:, :, :, 1:, :] - v[:, :, :, :-1, :]) ** 2
        gz = (v[:, :, :, :, 1:] - v[:, :, :, :, :-1]) ** 2

        mean_size = gx.numel() + gy.numel() + gz.numel()
        penalty = (gx.sum() + gy.sum() + gz.sum()) / mean_size

        return self.weight * penalty
