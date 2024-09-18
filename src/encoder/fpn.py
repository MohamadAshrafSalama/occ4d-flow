from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class DeformableConvBlock(nn.Module):
    """
    2D deformable convolution block.

    Predicts per-pixel 2D offsets and modulation masks, then applies
    torchvision.ops.deform_conv2d for spatially-adaptive feature sampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        deformable_groups: int = 1,
        norm_groups: int = 32,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.deformable_groups = deformable_groups

        offset_channels = deformable_groups * 2 * kernel_size * kernel_size
        mask_channels = deformable_groups * kernel_size * kernel_size

        self.offset_conv = nn.Conv2d(
            in_channels,
            offset_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.mask_conv = nn.Conv2d(
            in_channels,
            mask_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // deformable_groups, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.norm = nn.GroupNorm(norm_groups, out_channels)
        self.act = nn.ReLU(inplace=True)

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        nn.init.constant_(self.mask_conv.weight, 0)
        nn.init.constant_(self.mask_conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        mask = self.mask_conv(x).sigmoid()

        out = deform_conv2d(
            x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            mask=mask,
        )
        out = self.norm(out)
        out = self.act(out)
        return out


class FPNBlock(nn.Module):
    """Single FPN encoder block with optional stride-2 downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        deformable_groups: int = 1,
        use_deformable: bool = True,
        norm_groups: int = 32,
    ) -> None:
        super().__init__()
        self.downsample: Optional[nn.Module] = None

        if use_deformable:
            self.conv1 = DeformableConvBlock(
                in_channels, out_channels,
                kernel_size=3, stride=stride, padding=1,
                deformable_groups=deformable_groups,
                norm_groups=norm_groups,
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(norm_groups, out_channels),
                nn.ReLU(inplace=True),
            )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_channels),
            nn.ReLU(inplace=True),
        )

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(norm_groups, out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


class LateralConnection(nn.Module):
    """1x1 lateral projection for cross-level feature merging."""

    def __init__(self, in_channels: int, out_channels: int, norm_groups: int = 32) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(norm_groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FeaturePyramidNetwork(nn.Module):
    """
    3-level feature pyramid with deformable convolutions.

    Bottom-up pathway produces feature maps at 1x, 2x, 4x downsampling.
    Top-down pathway merges them via lateral connections and upsampling.
    All levels are projected to out_channels and summed into a single
    BEV feature map at the original resolution.

    Strides relative to input resolution:
      level0 — stride 1  (full resolution)
      level1 — stride 2
      level2 — stride 4
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        num_levels: int = 3,
        deformable_groups: int = 1,
        use_deformable: bool = True,
        norm_groups: int = 32,
    ) -> None:
        super().__init__()
        assert num_levels == 3, "Only 3-level FPN is supported"
        self.out_channels = out_channels

        self.layer0 = FPNBlock(
            in_channels, out_channels, stride=1,
            deformable_groups=deformable_groups,
            use_deformable=use_deformable,
            norm_groups=norm_groups,
        )
        self.layer1 = FPNBlock(
            out_channels, out_channels, stride=2,
            deformable_groups=deformable_groups,
            use_deformable=use_deformable,
            norm_groups=norm_groups,
        )
        self.layer2 = FPNBlock(
            out_channels, out_channels, stride=2,
            deformable_groups=deformable_groups,
            use_deformable=use_deformable,
            norm_groups=norm_groups,
        )

        self.lateral2 = LateralConnection(out_channels, out_channels, norm_groups)
        self.lateral1 = LateralConnection(out_channels, out_channels, norm_groups)
        self.lateral0 = LateralConnection(out_channels, out_channels, norm_groups)

        self.smooth2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_channels),
            nn.ReLU(inplace=True),
        )
        self.smooth1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_channels),
            nn.ReLU(inplace=True),
        )
        self.smooth0 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_channels),
            nn.ReLU(inplace=True),
        )

        self.merge = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) BEV feature map

        Returns:
            out: (B, out_channels, H, W) merged multi-scale features
        """
        c0 = self.layer0(x)
        c1 = self.layer1(c0)
        c2 = self.layer2(c1)

        p2 = self.smooth2(self.lateral2(c2))
        p1 = self.smooth1(
            self.lateral1(c1) + F.interpolate(p2, size=c1.shape[2:], mode="bilinear", align_corners=False)
        )
        p0 = self.smooth0(
            self.lateral0(c0) + F.interpolate(p1, size=c0.shape[2:], mode="bilinear", align_corners=False)
        )

        out = self.merge(p0)
        return out

    def forward_multiscale(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return all three pyramid levels before merging."""
        c0 = self.layer0(x)
        c1 = self.layer1(c0)
        c2 = self.layer2(c1)

        p2 = self.smooth2(self.lateral2(c2))
        p1 = self.smooth1(
            self.lateral1(c1) + F.interpolate(p2, size=c1.shape[2:], mode="bilinear", align_corners=False)
        )
        p0 = self.smooth0(
            self.lateral0(c0) + F.interpolate(p1, size=c0.shape[2:], mode="bilinear", align_corners=False)
        )

        return [p0, p1, p2]
