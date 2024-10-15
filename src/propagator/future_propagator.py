from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from src.temporal.conv_gru import ConvGRU3DCell
from src.propagator.ego_warper import EgoWarper


class FuturePropagator(nn.Module):
    """
    ConvGRU-based future state propagation with ego-motion decoupling.

    Given the aggregated current-frame voxel hidden state and a sequence
    of future ego-motion transforms, propagates features forward in time.

    For each future step:
      1. Warp the current state to account for ego-motion (the ego decoupled
         representation keeps the scene in world-relative coordinates).
      2. Feed through a ConvGRU cell to model scene dynamics beyond ego-motion.
      3. Return the sequence of future states.

    This separation is important: ego-motion is a deterministic, known
    transformation. Scene dynamics (other agents, weather, etc.) are modeled
    stochastically by the GRU and the downstream flow matching head.
    """

    def __init__(
        self,
        voxel_channels: int = 128,
        hidden_channels: int = 128,
        gru_kernel_size: int = 3,
        num_future_steps: int = 5,
        voxel_size: float = 0.5,
        grid_size: int = 200,
        ego_interp_mode: str = "bilinear",
        ego_padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.num_future_steps = num_future_steps
        self.voxel_size = voxel_size
        self.grid_size = grid_size

        self.ego_warper = EgoWarper(
            interp_mode=ego_interp_mode,
            padding_mode=ego_padding_mode,
        )

        self.gru_cell = ConvGRU3DCell(
            in_channels=voxel_channels,
            hidden_channels=hidden_channels,
            kernel_size=gru_kernel_size,
        )

        if voxel_channels != hidden_channels:
            self.state_proj = nn.Sequential(
                nn.Conv3d(hidden_channels, voxel_channels, 1, bias=False),
                nn.GroupNorm(min(32, voxel_channels), voxel_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.state_proj = nn.Identity()

        self.step_embed = nn.Embedding(num_future_steps + 1, voxel_channels)

        self.conditioning_proj = nn.Sequential(
            nn.Conv3d(voxel_channels * 2, voxel_channels, 1, bias=False),
            nn.GroupNorm(min(32, voxel_channels), voxel_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        init_state: torch.Tensor,
        future_poses: torch.Tensor,
        current_pose: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Args:
            init_state: (B, C, X, Y, Z) aggregated current voxel state
            future_poses: (B, T, 4, 4) future ego poses in world coordinates
            current_pose: (B, 4, 4) current ego pose; used for relative warping.
                          If None, identity is assumed (poses already relative).

        Returns:
            future_states: list of T tensors, each (B, C, X, Y, Z)
        """
        B, C, X, Y, Z = init_state.shape
        T = future_poses.shape[1]
        device = init_state.device

        if T > self.num_future_steps:
            future_poses = future_poses[:, :self.num_future_steps]
            T = self.num_future_steps

        if current_pose is None:
            current_pose = torch.eye(4, device=device, dtype=init_state.dtype).unsqueeze(0).expand(B, -1, -1)

        h = init_state
        gru_h = None
        future_states = []

        for step in range(T):
            pose_t = future_poses[:, step]
            warped = self.ego_warper.warp_from_pose(
                h, pose_t, current_pose,
                voxel_size=self.voxel_size,
                grid_size=self.grid_size,
            )

            step_idx = torch.full((B,), step, dtype=torch.long, device=device)
            step_emb = self.step_embed(step_idx)
            step_emb = step_emb.reshape(B, C, 1, 1, 1).expand(B, C, X, Y, Z)

            conditioned = self.conditioning_proj(
                torch.cat([warped, step_emb], dim=1)
            )

            gru_h = self.gru_cell(conditioned, gru_h)
            h = self.state_proj(gru_h)

            future_states.append(h)

            current_pose = pose_t

        return future_states

    def rollout(
        self,
        init_state: torch.Tensor,
        future_poses: torch.Tensor,
        current_pose: Optional[torch.Tensor] = None,
        return_all: bool = True,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Runs propagation and returns states plus a concatenated tensor.

        Returns:
            states: list of future states
            stacked: (B, T, C, X, Y, Z) stacked future states
        """
        states = self.forward(init_state, future_poses, current_pose)
        stacked = torch.stack(states, dim=1)
        return states, stacked
