from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(min(norm_groups, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(min(norm_groups, out_channels), out_channels),
            nn.ReLU(inplace=True),
        )
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(min(norm_groups, out_channels), out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.block(x) + self.skip(x), inplace=True)


class DiscriminativeDecoder(nn.Module):
    """
    Direct (non-generative) occupancy decoder.

    Produces per-voxel predictions from the propagated feature volume:
      - Occupancy logits: raw log-odds of voxel being occupied
      - Log-variance: predicted aleatoric uncertainty per voxel

    Architecture:
      - Series of Conv3D blocks with residual connections
      - Two parallel output heads:
          (a) occupancy head: (B, 1, X, Y, Z) logits
          (b) uncertainty head: (B, 1, X, Y, Z) log-variance

    The uncertainty head enables the ensemble blender to weight the
    discriminative prediction relative to the generative one.
    Softplus is applied to log-variance to ensure positivity.
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: List[int] = (128, 64, 32),
        num_classes: int = 1,
        uncertainty_head: bool = True,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.uncertainty_head = uncertainty_head

        layers = []
        prev = in_channels
        for ch in hidden_channels:
            layers.append(ConvBlock3D(prev, ch, norm_groups=norm_groups))
            prev = ch
        self.backbone = nn.Sequential(*layers)

        self.occ_head = nn.Sequential(
            nn.Conv3d(prev, prev // 2, 3, padding=1, bias=False),
            nn.GroupNorm(min(norm_groups, prev // 2), prev // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(prev // 2, num_classes, 1),
        )

        if uncertainty_head:
            self.logvar_head = nn.Sequential(
                nn.Conv3d(prev, prev // 2, 3, padding=1, bias=False),
                nn.GroupNorm(min(norm_groups, prev // 2), prev // 2),
                nn.ReLU(inplace=True),
                nn.Conv3d(prev // 2, num_classes, 1),
            )
            nn.init.constant_(self.logvar_head[-1].bias, -2.0)

        nn.init.zeros_(self.occ_head[-1].weight)
        nn.init.zeros_(self.occ_head[-1].bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, X, Y, Z) propagated voxel features

        Returns:
            dict with:
                'logits': (B, num_classes, X, Y, Z)
                'occ_prob': (B, num_classes, X, Y, Z) occupancy probability
                'logvar': (B, num_classes, X, Y, Z) log-variance (if enabled)
                'uncertainty': (B, num_classes, X, Y, Z) variance = softplus(logvar)
        """
        feat = self.backbone(x)

        logits = self.occ_head(feat)
        occ_prob = torch.sigmoid(logits)

        out = {"logits": logits, "occ_prob": occ_prob}

        if self.uncertainty_head:
            logvar = self.logvar_head(feat)
            uncertainty = F.softplus(logvar) + 1e-6
            out["logvar"] = logvar
            out["uncertainty"] = uncertainty

        return out

    def forward_sequence(
        self,
        feature_list: List[torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:
        """Apply decoder to each element of a future feature sequence."""
        return [self.forward(feat) for feat in feature_list]
