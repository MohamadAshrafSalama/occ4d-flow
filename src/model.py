from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from config.default import ModelConfig
from src.encoder.pillar_encoder import PillarEncoder
from src.encoder.fpn import FeaturePyramidNetwork
from src.encoder.voxel_lifter import VoxelLifter
from src.temporal.temporal_aggregator import TemporalAggregator
from src.propagator.ego_warper import EgoWarper
from src.propagator.future_propagator import FuturePropagator
from src.flow_matching.flow_matching_head import FlowMatchingHead
from src.decoder.discriminative_decoder import DiscriminativeDecoder
from src.decoder.ensemble_blender import EnsembleBlender


class Occ4DFlow(nn.Module):
    """
    4D occupancy forecasting model with continuous-time flow matching.

    Pipeline:
        1. Encode each past frame: pillar encoder -> FPN -> voxel lifter
        2. Aggregate temporal history via continuous-time ConvGRU
        3. Warp aggregated state to current ego frame
        4. Propagate into the future using ConvGRU + known ego poses
        5. For each future step, produce two predictions:
           a. Generative: sample from flow matching head
           b. Discriminative: direct regression + uncertainty
        6. Blend the two predictions via uncertainty-aware ensemble

    At training time, the flow matching head computes the CFM loss
    and all loss terms are returned in a dict.
    At inference time, the model samples future occupancy volumes.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder = PillarEncoder(
            in_channels=cfg.encoder.in_channels,
            pillar_feat_channels=cfg.encoder.pillar_feat_channels,
            bev_channels=cfg.encoder.bev_channels,
            radial_encoding_dim=cfg.encoder.radial_encoding_dim,
            out_channels=cfg.encoder.out_channels,
            grid_x=cfg.encoder.grid_x,
            grid_y=cfg.encoder.grid_y,
            voxel_x=cfg.encoder.voxel_x,
            voxel_y=cfg.encoder.voxel_y,
            x_offset=cfg.encoder.x_offset,
            y_offset=cfg.encoder.y_offset,
        )

        self.fpn = FeaturePyramidNetwork(
            in_channels=cfg.encoder.out_channels,
            out_channels=cfg.fpn.out_channels,
            num_levels=cfg.fpn.num_levels,
            deformable_groups=cfg.fpn.deformable_groups,
            use_deformable=cfg.fpn.use_deformable,
            norm_groups=cfg.fpn.norm_groups,
        )

        self.voxel_lifter = VoxelLifter(
            bev_channels=cfg.fpn.out_channels,
            voxel_channels=cfg.voxel_lifter.voxel_channels,
            num_height_bins=cfg.voxel_lifter.num_height_bins,
            height_mlp_hidden=cfg.voxel_lifter.height_mlp_hidden,
            grid_x=cfg.voxel_lifter.grid_x,
            grid_y=cfg.voxel_lifter.grid_y,
            grid_z=cfg.voxel_lifter.grid_z,
        )

        self.temporal_aggregator = TemporalAggregator(
            voxel_channels=cfg.temporal.voxel_channels,
            timestamp_dim=cfg.temporal.timestamp_dim,
            timestamp_max_freq=cfg.temporal.timestamp_max_freq,
            timestamp_mlp_hidden=cfg.temporal.timestamp_mlp_hidden,
            confidence_hidden=cfg.temporal.confidence_hidden,
            max_time_delta=cfg.temporal.max_time_delta,
            gru_hidden_channels=cfg.temporal.gru_hidden_channels,
            gru_kernel_size=cfg.temporal.gru_kernel_size,
            num_gru_layers=cfg.temporal.num_gru_layers,
        )

        self.ego_warper = EgoWarper()

        self.future_propagator = FuturePropagator(
            voxel_channels=cfg.propagator.voxel_channels,
            hidden_channels=cfg.propagator.hidden_channels,
            gru_kernel_size=cfg.propagator.gru_kernel_size,
            num_future_steps=cfg.propagator.num_future_steps,
        )

        self.flow_head = FlowMatchingHead(
            in_channels=cfg.flow_matching.in_channels,
            base_channels=cfg.flow_matching.base_channels,
            channel_mults=cfg.flow_matching.channel_mults,
            num_res_blocks=cfg.flow_matching.num_res_blocks,
            norm_groups=cfg.flow_matching.norm_groups,
            timestep_embed_dim=cfg.flow_matching.timestep_embed_dim,
            timestep_mlp_hidden=cfg.flow_matching.timestep_mlp_hidden,
            sigma_min=cfg.flow_matching.sigma_min,
            ot_transport=cfg.flow_matching.ot_transport,
            tv_weight_xy=cfg.flow_matching.tv_weight_xy,
            tv_weight_z=cfg.flow_matching.tv_weight_z,
            num_euler_steps=cfg.flow_matching.num_euler_steps,
        )

        self.disc_decoder = DiscriminativeDecoder(
            in_channels=cfg.decoder.in_channels,
            hidden_channels=cfg.decoder.hidden_channels,
            num_classes=cfg.decoder.num_classes,
            uncertainty_head=cfg.decoder.uncertainty_head,
            norm_groups=cfg.decoder.norm_groups,
        )

        self.blender = EnsembleBlender(
            channels=cfg.decoder.num_classes,
            use_learned_correction=True,
        )

    def _encode_frame(
        self,
        voxels: torch.Tensor,
        coords: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Encode a single LiDAR frame from pillar format to voxel features."""
        bev = self.encoder(voxels, coords, batch_size)
        bev = self.fpn(bev)
        voxel_feat = self.voxel_lifter(bev)
        return voxel_feat

    def encode_sequence(
        self,
        past_voxels: List[torch.Tensor],
        past_coords: List[torch.Tensor],
        batch_size: int,
    ) -> List[torch.Tensor]:
        """Encode all past frames into voxel feature tensors."""
        return [
            self._encode_frame(v, c, batch_size)
            for v, c in zip(past_voxels, past_coords)
        ]

    def forward(
        self,
        past_voxels: List[torch.Tensor],
        past_coords: List[torch.Tensor],
        past_timestamps: torch.Tensor,
        future_poses: torch.Tensor,
        future_occ_gt: Optional[torch.Tensor] = None,
        current_pose: Optional[torch.Tensor] = None,
        freespace_masks: Optional[List[torch.Tensor]] = None,
    ) -> Dict:
        """
        Full model forward pass.

        Args:
            past_voxels: T-list of (B, P, N, 4) pillar point tensors
            past_coords: T-list of (B*P, 2) pillar BEV coordinate tensors
            past_timestamps: (B, T) timestamps in seconds relative to current
            future_poses: (B, T_fut, 4, 4) future ego poses
            future_occ_gt: (B, T_fut, X, Y, Z) ground-truth future occupancy (train)
            current_pose: (B, 4, 4) current ego pose
            freespace_masks: T_fut-list of (B, 1, X, Y, Z) ray freespace masks

        Returns:
            dict with predictions and losses (losses only in training mode)
        """
        B = past_timestamps.shape[0]
        T_past = len(past_voxels)
        T_fut = future_poses.shape[1]

        voxel_sequence = self.encode_sequence(past_voxels, past_coords, B)

        aggregated = self.temporal_aggregator(
            voxel_sequence, past_timestamps
        )

        future_states = self.future_propagator(
            aggregated, future_poses, current_pose
        )

        if self.training and future_occ_gt is not None:
            target_list = [
                future_occ_gt[:, t].unsqueeze(1).expand_as(future_states[t])
                for t in range(T_fut)
            ]

            flow_losses = self.flow_head.forward_train_sequence(
                future_states, target_list
            )
        else:
            flow_losses = {}

        if self.training:
            p_gen_list = self.flow_head.sample_sequence(future_states)
        else:
            p_gen_list = self.flow_head.sample_sequence(future_states)

        disc_out_list = self.disc_decoder.forward_sequence(future_states)

        blended_list = self.blender.forward_sequence(p_gen_list, disc_out_list)

        output = {
            "blended": blended_list,
            "p_gen": p_gen_list,
            "disc_out": disc_out_list,
        }

        if flow_losses:
            output.update(flow_losses)

        return output

    def predict(
        self,
        past_voxels: List[torch.Tensor],
        past_coords: List[torch.Tensor],
        past_timestamps: torch.Tensor,
        future_poses: torch.Tensor,
        current_pose: Optional[torch.Tensor] = None,
        num_flow_steps: int = 20,
    ) -> List[torch.Tensor]:
        """
        Inference-only prediction of future occupancy volumes.

        Returns:
            List of T_future (B, 1, X, Y, Z) occupancy probability tensors.
        """
        self.eval()
        with torch.no_grad():
            B = past_timestamps.shape[0]
            voxel_sequence = self.encode_sequence(past_voxels, past_coords, B)
            aggregated = self.temporal_aggregator(voxel_sequence, past_timestamps)
            future_states = self.future_propagator(aggregated, future_poses, current_pose)
            p_gen_list = self.flow_head.sample_sequence(future_states, num_steps=num_flow_steps)
            disc_out_list = self.disc_decoder.forward_sequence(future_states)
            blended = self.blender.forward_sequence(p_gen_list, disc_out_list)
        return [b["blend"] for b in blended]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
