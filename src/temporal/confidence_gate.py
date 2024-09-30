import torch
import torch.nn as nn


class ConfidenceGate(nn.Module):
    """
    Delay-aware confidence gating for temporal feature aggregation.

    Computes a per-channel gate value in (0, 1) as a function of the time
    delta between the current frame and an observation. Older observations
    receive lower gate values, suppressing their contribution to the
    aggregated feature state.

    The gate is conditioned on the raw time delta as well as the timestamp
    embedding vector, so the network can learn non-trivial decay profiles
    that depend on both elapsed time and the absolute time context.

    Gate shape: (B, C, 1, 1, 1) for broadcasting against (B, C, X, Y, Z).
    """

    def __init__(
        self,
        voxel_channels: int = 128,
        timestamp_dim: int = 64,
        hidden_dim: int = 64,
        max_time_delta: float = 5.0,
    ) -> None:
        super().__init__()
        self.max_time_delta = max_time_delta

        self.time_proj = nn.Sequential(
            nn.Linear(timestamp_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.gate_head = nn.Linear(hidden_dim, voxel_channels)

        self.channel_norm = nn.LayerNorm(voxel_channels)

        nn.init.constant_(self.gate_head.bias, 1.0)

    def forward(
        self,
        voxel_feat: torch.Tensor,
        time_delta: torch.Tensor,
        timestamp_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            voxel_feat: (B, C, X, Y, Z)
            time_delta: (B,) time difference in seconds (positive = in the past)
            timestamp_emb: (B, timestamp_dim) precomputed timestamp embedding

        Returns:
            gated_feat: (B, C, X, Y, Z)
        """
        B = voxel_feat.shape[0]

        delta_norm = (time_delta / self.max_time_delta).clamp(0.0, 1.0).unsqueeze(-1)
        x = torch.cat([timestamp_emb, delta_norm], dim=-1)

        h = self.time_proj(x)
        gate_logit = self.gate_head(h)
        gate_logit = self.channel_norm(gate_logit)

        gate = torch.sigmoid(gate_logit)
        gate = gate.reshape(B, -1, 1, 1, 1)

        return voxel_feat * gate

    def extra_repr(self) -> str:
        return f"max_time_delta={self.max_time_delta}"


class AdaptiveConfidenceGate(nn.Module):
    """
    Extended confidence gate that also learns a spatial attention map
    in addition to the channel gate. Used as a drop-in replacement for
    ConfidenceGate when richer spatial gating is needed.
    """

    def __init__(
        self,
        voxel_channels: int = 128,
        timestamp_dim: int = 64,
        hidden_dim: int = 64,
        spatial_kernel: int = 3,
        max_time_delta: float = 5.0,
    ) -> None:
        super().__init__()
        self.channel_gate = ConfidenceGate(
            voxel_channels, timestamp_dim, hidden_dim, max_time_delta
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv3d(voxel_channels + 1, hidden_dim, spatial_kernel, padding=spatial_kernel // 2, bias=False),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        voxel_feat: torch.Tensor,
        time_delta: torch.Tensor,
        timestamp_emb: torch.Tensor,
    ) -> torch.Tensor:
        B, C, X, Y, Z = voxel_feat.shape

        ch_gated = self.channel_gate(voxel_feat, time_delta, timestamp_emb)

        delta_map = (time_delta / self.channel_gate.max_time_delta).clamp(0, 1)
        delta_map = delta_map.reshape(B, 1, 1, 1, 1).expand(B, 1, X, Y, Z)

        sp_input = torch.cat([ch_gated, delta_map], dim=1)
        sp_gate = self.spatial_gate(sp_input)

        return ch_gated * sp_gate
