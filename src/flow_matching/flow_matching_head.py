from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.flow_matching.unet3d import UNet3D
from src.flow_matching.scheduler import OTCFMScheduler
from src.flow_matching.anisotropic_tv import AnisotropicTotalVariation


class FlowMatchingHead(nn.Module):
    """
    Full generative head: 3D U-Net + OT-CFM scheduler + TV regularization.

    During training:
      - Sample diffusion time t ~ U(0, 1)
      - Sample noisy state x_t via OT-CFM interpolant
      - Predict velocity with U-Net conditioned on context + t
      - Return CFM loss + TV penalty on predicted velocity

    During inference:
      - Sample Gaussian noise x1
      - Integrate ODE with N Euler steps from t=1 to t=0
      - Return predicted clean occupancy volume

    The context features come from the FuturePropagator output — one
    context tensor per future timestep. The head processes each step
    independently, sharing U-Net weights across timesteps.
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
        sigma_min: float = 1e-4,
        ot_transport: bool = True,
        tv_weight_xy: float = 0.01,
        tv_weight_z: float = 0.05,
        num_euler_steps: int = 10,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_euler_steps = num_euler_steps

        self.unet = UNet3D(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            norm_groups=norm_groups,
            timestep_embed_dim=timestep_embed_dim,
            timestep_mlp_hidden=timestep_mlp_hidden,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        self.scheduler = OTCFMScheduler(
            sigma_min=sigma_min,
            ot_transport=ot_transport,
        )

        self.tv = AnisotropicTotalVariation(
            weight_x=tv_weight_xy,
            weight_y=tv_weight_xy,
            weight_z=tv_weight_z,
        )

        self.output_proj = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, in_channels // 2),
            nn.SiLU(),
            nn.Conv3d(in_channels // 2, 1, 1),
            nn.Sigmoid(),
        )

    def _model_fn(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.unet(x, t, context)

    def forward_train(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass for a single future timestep.

        Args:
            context: (B, C, X, Y, Z) propagated features
            target: (B, C, X, Y, Z) ground-truth occupancy features (or logits)
            t: (B,) optional pre-sampled timestep; sampled if None

        Returns:
            dict with 'cfm_loss', 'tv_loss', 'total_loss'
        """
        B = context.shape[0]
        device = context.device

        if t is None:
            t = self.scheduler.sample_t(B, device)

        x_t, v_t, x1 = self.scheduler.q_sample(target, t)

        v_pred = self.unet(x_t, t, context)

        cfm_loss = self.scheduler.compute_loss(v_pred, v_t)
        tv_loss = self.tv(v_pred)

        return {
            "cfm_loss": cfm_loss,
            "tv_loss": tv_loss,
            "total_loss": cfm_loss + tv_loss,
        }

    def forward_train_sequence(
        self,
        context_list: List[torch.Tensor],
        target_list: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Training over a sequence of future steps, sharing U-Net weights.

        Args:
            context_list: T-list of (B, C, X, Y, Z) context tensors
            target_list:  T-list of (B, C, X, Y, Z) target tensors

        Returns:
            dict with aggregated losses
        """
        T = len(context_list)
        B = context_list[0].shape[0]
        device = context_list[0].device

        t = self.scheduler.sample_t(B, device)

        total_cfm = context_list[0].new_zeros(1)
        total_tv = context_list[0].new_zeros(1)

        for ctx, tgt in zip(context_list, target_list):
            losses = self.forward_train(ctx, tgt, t)
            total_cfm = total_cfm + losses["cfm_loss"]
            total_tv = total_tv + losses["tv_loss"]

        total_cfm = total_cfm / T
        total_tv = total_tv / T

        return {
            "cfm_loss": total_cfm,
            "tv_loss": total_tv,
            "total_loss": total_cfm + total_tv,
        }

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        num_steps: Optional[int] = None,
        use_heun: bool = False,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Generate occupancy volume via ODE integration.

        Args:
            context: (B, C, X, Y, Z) conditioning features
            num_steps: number of Euler steps; defaults to self.num_euler_steps
            use_heun: use Heun integrator instead of Euler
            return_trajectory: if True, return all intermediate states

        Returns:
            occ: (B, 1, X, Y, Z) predicted occupancy probability
        """
        steps = num_steps if num_steps is not None else self.num_euler_steps

        if use_heun:
            raw = self.scheduler.heun_integrate(
                self._model_fn, context, num_steps=steps
            )
        else:
            raw = self.scheduler.euler_integrate(
                self._model_fn, context, num_steps=steps
            )

        occ = self.output_proj(raw)
        return occ

    @torch.no_grad()
    def sample_sequence(
        self,
        context_list: List[torch.Tensor],
        num_steps: Optional[int] = None,
        use_heun: bool = False,
    ) -> List[torch.Tensor]:
        """Sample occupancy for each future timestep."""
        return [
            self.sample(ctx, num_steps, use_heun)
            for ctx in context_list
        ]

    def forward(
        self,
        context: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Unified forward: training mode if target provided, else inference."""
        if self.training and target is not None:
            return self.forward_train(context, target, t)
        return {"occ": self.sample(context)}
