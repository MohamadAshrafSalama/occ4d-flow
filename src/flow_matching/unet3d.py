import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class TimestepMLP(nn.Module):
    """Sinusoidal timestep embedding followed by MLP projection."""

    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, embed_dim // 2, dtype=torch.float32) / (embed_dim // 2)
        )
        self.register_buffer("freqs", freqs)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        phases = t.unsqueeze(-1) * self.freqs.unsqueeze(0)
        emb = torch.cat([phases.cos(), phases.sin()], dim=-1)
        return self.mlp(emb)


class AdaGN(nn.Module):
    """
    Adaptive Group Normalization.

    Applies GroupNorm then modulates with scale and shift derived from
    a timestep embedding. Conditioning is: out = scale * GN(x) + shift.
    """

    def __init__(self, num_channels: int, num_groups: int, emb_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.proj = nn.Linear(emb_dim, 2 * num_channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        scale_shift = self.proj(emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = scale.reshape(scale.shape[0], -1, 1, 1, 1)
        shift = shift.reshape(shift.shape[0], -1, 1, 1, 1)
        return x * (1.0 + scale) + shift


class ResBlock3D(nn.Module):
    """3D residual block with AdaGN timestep conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        norm_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = AdaGN(in_channels, norm_groups, emb_dim)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU()

        self.norm2 = AdaGN(out_channels, norm_groups, emb_dim)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip_proj = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.skip_proj = nn.Identity()

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x, emb))
        h = self.conv1(h)
        h = self.act(self.norm2(h, emb))
        h = self.drop(h)
        h = self.conv2(h)
        return h + self.skip_proj(x)


class DownBlock3D(nn.Module):
    """Encoder block: res blocks + strided downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        num_res_blocks: int = 2,
        norm_groups: int = 8,
        downsample: bool = True,
    ) -> None:
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResBlock3D(
                in_channels if i == 0 else out_channels,
                out_channels,
                emb_dim, norm_groups,
            )
            for i in range(num_res_blocks)
        ])
        if downsample:
            self.down = nn.Conv3d(out_channels, out_channels, kernel_size=(2, 2, 1), stride=(2, 2, 1))
        else:
            self.down = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for blk in self.res_blocks:
            x = blk(x, emb)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock3D(nn.Module):
    """Decoder block: upsampling + skip connection + res blocks."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        emb_dim: int,
        num_res_blocks: int = 2,
        norm_groups: int = 8,
        upsample: bool = True,
    ) -> None:
        super().__init__()
        if upsample:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=(2, 2, 1), mode="trilinear", align_corners=False),
                nn.Conv3d(in_channels, in_channels, 3, padding=1),
            )
        else:
            self.up = nn.Identity()

        self.res_blocks = nn.ModuleList([
            ResBlock3D(
                (in_channels + skip_channels) if i == 0 else out_channels,
                out_channels,
                emb_dim, norm_groups,
            )
            for i in range(num_res_blocks)
        ])

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        for blk in self.res_blocks:
            x = blk(x, emb)
        return x


class UNet3D(nn.Module):
    """
    3D U-Net velocity field predictor for flow matching.

    Architecture:
      - Timestep embedding (sinusoidal + MLP)
      - Conditioning projection (context features from propagator)
      - 4 encoder stages with strided downsampling
      - Bottleneck
      - 4 decoder stages with trilinear upsampling + skip connections
      - Final projection to in_channels (velocity field)

    All normalization uses AdaGN conditioned on the diffusion timestep.
    Supports gradient checkpointing on all residual blocks.
    """

    def __init__(
        self,
        in_channels: int = 128,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        norm_groups: int = 8,
        timestep_embed_dim: int = 256,
        timestep_mlp_hidden: int = 512,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.use_grad_checkpoint = use_gradient_checkpointing
        ch_list = [base_channels * m for m in channel_mults]

        self.timestep_mlp = TimestepMLP(timestep_embed_dim, timestep_mlp_hidden)
        emb_dim = timestep_mlp_hidden

        self.in_proj = nn.Conv3d(in_channels * 2, ch_list[0], 3, padding=1)

        self.down_blocks = nn.ModuleList()
        in_ch = ch_list[0]
        for i, out_ch in enumerate(ch_list):
            self.down_blocks.append(DownBlock3D(
                in_ch, out_ch, emb_dim, num_res_blocks, norm_groups,
                downsample=(i < len(ch_list) - 1),
            ))
            in_ch = out_ch

        self.bottleneck = nn.ModuleList([
            ResBlock3D(ch_list[-1], ch_list[-1], emb_dim, norm_groups),
            ResBlock3D(ch_list[-1], ch_list[-1], emb_dim, norm_groups),
        ])

        self.up_blocks = nn.ModuleList()
        rev_ch = list(reversed(ch_list))
        for i in range(len(rev_ch)):
            skip_ch = rev_ch[i]
            out_ch = rev_ch[min(i + 1, len(rev_ch) - 1)]
            self.up_blocks.append(UpBlock3D(
                rev_ch[i], skip_ch, out_ch, emb_dim, num_res_blocks, norm_groups,
                upsample=(i < len(rev_ch) - 1),
            ))

        self.out_norm = nn.GroupNorm(norm_groups, rev_ch[-1])
        self.out_act = nn.SiLU()
        self.out_proj = nn.Conv3d(rev_ch[-1], in_channels, 3, padding=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _res_block_fn(self, block: ResBlock3D, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.use_grad_checkpoint and self.training:
            return checkpoint(block, x, emb, use_reentrant=False)
        return block(x, emb)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, X, Y, Z) noisy state at timestep t
            t: (B,) diffusion timestep in [0, 1]
            context: (B, C, X, Y, Z) conditioning from propagator

        Returns:
            velocity: (B, C, X, Y, Z) predicted velocity field
        """
        emb = self.timestep_mlp(t)

        h = self.in_proj(torch.cat([x, context], dim=1))

        skips = []
        for blk in self.down_blocks:
            h, skip = blk(h, emb)
            skips.append(skip)

        for blk in self.bottleneck:
            h = self._res_block_fn(blk, h, emb)

        for i, blk in enumerate(self.up_blocks):
            skip = skips[-(i + 1)]
            h = blk(h, skip, emb)

        h = self.out_act(self.out_norm(h))
        velocity = self.out_proj(h)
        return velocity
