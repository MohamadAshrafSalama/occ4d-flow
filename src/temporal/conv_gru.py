from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGRU3DCell(nn.Module):
    """
    3D convolutional GRU cell.

    Applies all gate and candidate computations with 3D convolutions over
    a volumetric hidden state, enabling spatially-local temporal reasoning
    within the voxel grid without flattening spatial structure.

    Gates:
        z (update): controls interpolation between old and new state
        r (reset): gates how much past state enters the candidate
        h_tilde (candidate): new state proposal
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        self.reset_gate = nn.Sequential(
            nn.Conv3d(
                in_channels + hidden_channels, hidden_channels,
                kernel_size, padding=padding, bias=False,
            ),
            nn.GroupNorm(min(32, hidden_channels), hidden_channels),
        )

        self.update_gate = nn.Sequential(
            nn.Conv3d(
                in_channels + hidden_channels, hidden_channels,
                kernel_size, padding=padding, bias=False,
            ),
            nn.GroupNorm(min(32, hidden_channels), hidden_channels),
        )

        self.candidate = nn.Sequential(
            nn.Conv3d(
                in_channels + hidden_channels, hidden_channels,
                kernel_size, padding=padding, bias=False,
            ),
            nn.GroupNorm(min(32, hidden_channels), hidden_channels),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.orthogonal_(module.weight)

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, X, Y, Z) input features
            h: (B, hidden_channels, X, Y, Z) previous hidden state, or None

        Returns:
            h_new: (B, hidden_channels, X, Y, Z)
        """
        B, C, X, Y, Z = x.shape

        if h is None:
            h = x.new_zeros(B, self.hidden_channels, X, Y, Z)

        xh = torch.cat([x, h], dim=1)

        r = torch.sigmoid(self.reset_gate(xh))
        z = torch.sigmoid(self.update_gate(xh))

        xh_reset = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.candidate(xh_reset))

        h_new = (1.0 - z) * h + z * h_tilde
        return h_new

    def init_hidden(self, batch_size: int, spatial_size: Tuple[int, int, int]) -> torch.Tensor:
        X, Y, Z = spatial_size
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        return torch.zeros(batch_size, self.hidden_channels, X, Y, Z, device=device, dtype=dtype)


class ConvGRU3D(nn.Module):
    """
    Multi-layer 3D ConvGRU.

    Stacks ConvGRU3DCell layers with residual connections between layers.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.cells = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            self.cells.append(ConvGRU3DCell(in_ch, hidden_channels, kernel_size))

        if in_channels != hidden_channels:
            self.input_proj = nn.Sequential(
                nn.Conv3d(in_channels, hidden_channels, 1, bias=False),
                nn.GroupNorm(min(32, hidden_channels), hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.input_proj = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Args:
            x: (B, in_channels, X, Y, Z)
            hidden_states: list of per-layer hidden states or None

        Returns:
            out: (B, hidden_channels, X, Y, Z)
            new_hidden_states: list of updated hidden tensors
        """
        if hidden_states is None:
            hidden_states = [None] * self.num_layers

        current = x
        new_hidden_states = []

        for i, cell in enumerate(self.cells):
            h_new = cell(current, hidden_states[i])
            new_hidden_states.append(h_new)

            if i == 0:
                residual = self.input_proj(x)
            else:
                residual = current

            current = h_new + residual

        return current, new_hidden_states
