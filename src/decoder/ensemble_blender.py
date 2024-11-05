from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleBlender(nn.Module):
    """
    Uncertainty-aware blending of generative and discriminative predictions.

    Given:
      - generative occupancy probability p_gen (from flow matching head)
      - discriminative occupancy probability p_disc (from discriminative decoder)
      - predicted uncertainty sigma^2 (from discriminative decoder)

    The blend weight for the discriminative prediction is:
        w_disc = 1 / (1 + sigma^2)

    And the generative weight is its complement:
        w_gen = 1 - w_disc = sigma^2 / (1 + sigma^2)

    Both weights sum to 1. When uncertainty is low, the discriminative
    (direct regression) prediction is trusted more. When uncertainty is
    high (e.g., distant voxels, occluded regions), the generative
    (distribution-sampling) prediction contributes more.

    An optional learned correction network post-processes the blend.
    """

    def __init__(
        self,
        channels: int = 1,
        use_learned_correction: bool = True,
        correction_hidden: int = 16,
        norm_groups: int = 4,
    ) -> None:
        super().__init__()
        self.use_learned_correction = use_learned_correction

        if use_learned_correction:
            in_ch = channels * 3 + 1
            self.correction = nn.Sequential(
                nn.Conv3d(in_ch, correction_hidden, 3, padding=1, bias=False),
                nn.GroupNorm(min(norm_groups, correction_hidden), correction_hidden),
                nn.ReLU(inplace=True),
                nn.Conv3d(correction_hidden, channels, 1),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.correction[-2].weight)
            nn.init.zeros_(self.correction[-2].bias)

    def _compute_weights(self, uncertainty: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w_disc = 1.0 / (1.0 + uncertainty)
        w_gen = 1.0 - w_disc
        return w_disc, w_gen

    def forward(
        self,
        p_gen: torch.Tensor,
        p_disc: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            p_gen: (B, C, X, Y, Z) generative occupancy probability
            p_disc: (B, C, X, Y, Z) discriminative occupancy probability
            uncertainty: (B, C, X, Y, Z) variance from discriminative head (>= 0)

        Returns:
            dict with:
                'blend': (B, C, X, Y, Z) blended prediction
                'w_disc': (B, C, X, Y, Z) discriminative weight map
                'w_gen': (B, C, X, Y, Z) generative weight map
        """
        w_disc, w_gen = self._compute_weights(uncertainty)

        blend = w_disc * p_disc + w_gen * p_gen

        if self.use_learned_correction:
            correction_input = torch.cat([p_gen, p_disc, uncertainty, blend], dim=1)
            delta = self.correction(correction_input)
            blend = blend * delta + blend * (1.0 - delta)
            blend = blend.clamp(0.0, 1.0)

        return {
            "blend": blend,
            "w_disc": w_disc,
            "w_gen": w_gen,
        }

    def forward_sequence(
        self,
        p_gen_list: List[torch.Tensor],
        decoder_out_list: List[Dict[str, torch.Tensor]],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Blend predictions for each future timestep.

        Args:
            p_gen_list: T-list of (B, 1, X, Y, Z) generative probabilities
            decoder_out_list: T-list of dicts from DiscriminativeDecoder

        Returns:
            T-list of blend dicts
        """
        results = []
        for p_gen, dec_out in zip(p_gen_list, decoder_out_list):
            p_disc = dec_out["occ_prob"]
            uncertainty = dec_out.get(
                "uncertainty",
                torch.ones_like(p_disc) * 0.5,
            )
            results.append(self.forward(p_gen, p_disc, uncertainty))
        return results

    def extra_repr(self) -> str:
        return f"learned_correction={self.use_learned_correction}"
