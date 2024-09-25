import math

import torch
import torch.nn as nn


class SinusoidalTimestampEncoding(nn.Module):
    """
    NeRF-style sinusoidal encoding of scalar timestamps.

    For a timestamp t in [0, max_t], produces a vector:
        [sin(2^0 * pi * t), cos(2^0 * pi * t),
         sin(2^1 * pi * t), cos(2^1 * pi * t),
         ...,
         sin(2^(L-1) * pi * t), cos(2^(L-1) * pi * t)]

    Output dimension is 2 * num_freqs.
    """

    def __init__(self, num_freqs: int = 8) -> None:
        super().__init__()
        self.num_freqs = num_freqs
        freqs = 2.0 ** torch.arange(num_freqs).float() * math.pi
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self) -> int:
        return 2 * self.num_freqs

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (...,) scalar timestamps (any shape)

        Returns:
            enc: (..., 2 * num_freqs)
        """
        t = t.unsqueeze(-1)
        phases = t * self.freqs
        return torch.cat([phases.sin(), phases.cos()], dim=-1)


class TimestampEmbedding(nn.Module):
    """
    Full timestamp embedding: sinusoidal encoding followed by learnable MLP.

    Accepts continuous timestamps (in seconds) relative to the current frame.
    Negative values indicate past frames, positive values indicate future.

    The sinusoidal component captures periodic patterns, and the MLP adds
    non-linear capacity to transform the encoding into the feature space.
    """

    def __init__(
        self,
        out_dim: int = 64,
        max_freq: int = 8,
        mlp_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.sin_enc = SinusoidalTimestampEncoding(num_freqs=max_freq)
        sin_dim = self.sin_enc.out_dim

        self.mlp = nn.Sequential(
            nn.Linear(sin_dim + 1, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) or (B, T) timestamps in seconds

        Returns:
            emb: same leading dims as t, last dim = out_dim
        """
        sin_feats = self.sin_enc(t)
        t_scalar = t.unsqueeze(-1)
        x = torch.cat([sin_feats, t_scalar], dim=-1)
        return self.mlp(x)
