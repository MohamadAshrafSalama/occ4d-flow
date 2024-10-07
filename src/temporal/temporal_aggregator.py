from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from src.temporal.timestamp_embedding import TimestampEmbedding
from src.temporal.confidence_gate import ConfidenceGate
from src.temporal.conv_gru import ConvGRU3D


class TemporalAggregator(nn.Module):
    """
    Continuous-time temporal aggregation over a sequence of voxel feature frames.

    Pipeline per input frame:
      1. Embed the continuous timestamp via sinusoidal + MLP encoding.
      2. Apply delay-aware confidence gating to suppress stale observations.
      3. Concatenate timestamp embedding as a spatial conditioning signal.
      4. Update the 3D ConvGRU hidden state.

    The GRU processes frames in chronological order (oldest to newest).
    The final hidden state is the aggregated temporal context.

    Handles variable sequence lengths and irregular timestamp intervals.
    """

    def __init__(
        self,
        voxel_channels: int = 128,
        timestamp_dim: int = 64,
        timestamp_max_freq: int = 8,
        timestamp_mlp_hidden: int = 128,
        confidence_hidden: int = 64,
        max_time_delta: float = 5.0,
        gru_hidden_channels: int = 128,
        gru_kernel_size: int = 3,
        num_gru_layers: int = 2,
    ) -> None:
        super().__init__()
        self.voxel_channels = voxel_channels
        self.gru_hidden_channels = gru_hidden_channels

        self.timestamp_embed = TimestampEmbedding(
            out_dim=timestamp_dim,
            max_freq=timestamp_max_freq,
            mlp_hidden=timestamp_mlp_hidden,
        )

        self.confidence_gate = ConfidenceGate(
            voxel_channels=voxel_channels,
            timestamp_dim=timestamp_dim,
            hidden_dim=confidence_hidden,
            max_time_delta=max_time_delta,
        )

        ts_spatial_channels = 8
        self.ts_spatial_proj = nn.Sequential(
            nn.Linear(timestamp_dim, ts_spatial_channels),
            nn.ReLU(inplace=True),
        )
        self.ts_spatial_channels = ts_spatial_channels

        gru_in_channels = voxel_channels + ts_spatial_channels
        self.gru = ConvGRU3D(
            in_channels=gru_in_channels,
            hidden_channels=gru_hidden_channels,
            kernel_size=gru_kernel_size,
            num_layers=num_gru_layers,
        )

        self.output_proj = nn.Sequential(
            nn.Conv3d(gru_hidden_channels, voxel_channels, 1, bias=False),
            nn.GroupNorm(min(32, voxel_channels), voxel_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        voxel_sequence: List[torch.Tensor],
        timestamps: torch.Tensor,
        current_timestamp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            voxel_sequence: list of T tensors, each (B, C, X, Y, Z),
                            ordered from oldest to newest
            timestamps: (B, T) timestamps in seconds for each frame
            current_timestamp: (B,) timestamp of the current frame.
                               If None, uses timestamps[:, -1].

        Returns:
            agg: (B, C, X, Y, Z) aggregated voxel feature
        """
        T = len(voxel_sequence)
        B, C, X, Y, Z = voxel_sequence[0].shape
        assert timestamps.shape == (B, T), (
            f"Expected timestamps shape ({B}, {T}), got {timestamps.shape}"
        )

        if current_timestamp is None:
            current_timestamp = timestamps[:, -1]

        hidden_states = None

        for t_idx in range(T):
            frame_feat = voxel_sequence[t_idx]
            ts = timestamps[:, t_idx]
            time_delta = current_timestamp - ts

            ts_emb = self.timestamp_embed(ts)

            gated_feat = self.confidence_gate(frame_feat, time_delta, ts_emb)

            ts_spatial = self.ts_spatial_proj(ts_emb)
            ts_spatial = ts_spatial.reshape(B, self.ts_spatial_channels, 1, 1, 1)
            ts_spatial = ts_spatial.expand(B, self.ts_spatial_channels, X, Y, Z)

            gru_input = torch.cat([gated_feat, ts_spatial], dim=1)

            out, hidden_states = self.gru(gru_input, hidden_states)

        agg = self.output_proj(out)
        return agg

    def forward_with_states(
        self,
        voxel_sequence: List[torch.Tensor],
        timestamps: torch.Tensor,
        current_timestamp: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list]:
        """Same as forward but also returns the raw GRU hidden states for further use."""
        T = len(voxel_sequence)
        B, C, X, Y, Z = voxel_sequence[0].shape

        if current_timestamp is None:
            current_timestamp = timestamps[:, -1]

        hidden_states = None
        last_out = None

        for t_idx in range(T):
            frame_feat = voxel_sequence[t_idx]
            ts = timestamps[:, t_idx]
            time_delta = current_timestamp - ts

            ts_emb = self.timestamp_embed(ts)
            gated_feat = self.confidence_gate(frame_feat, time_delta, ts_emb)

            ts_spatial = self.ts_spatial_proj(ts_emb)
            ts_spatial = ts_spatial.reshape(B, self.ts_spatial_channels, 1, 1, 1)
            ts_spatial = ts_spatial.expand(B, self.ts_spatial_channels, X, Y, Z)

            gru_input = torch.cat([gated_feat, ts_spatial], dim=1)
            last_out, hidden_states = self.gru(gru_input, hidden_states)

        agg = self.output_proj(last_out)
        return agg, hidden_states
