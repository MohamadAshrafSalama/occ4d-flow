from typing import Optional, Tuple

import torch
import torch.nn as nn


class OTCFMScheduler(nn.Module):
    """
    Optimal Transport Conditional Flow Matching (OT-CFM) scheduler.

    OT-CFM defines straight-line interpolant paths between data and noise:
        x_t = (1 - t) * x_0 + t * x_1
        v_t = x_1 - x_0   (constant velocity along the straight path)

    where:
        x_0 ~ p_data  (clean occupancy)
        x_1 ~ N(0, I)  (Gaussian noise)
        t ~ U(0, 1)

    Training: predict v_t from x_t and t, minimize MSE(predicted, v_t).

    Inference: numerically integrate the ODE dx/dt = v_theta(x_t, t)
    starting from x_1 ~ N(0, I) backward to t=0 using Euler steps.
    The sign convention follows: from t=1 (noise) toward t=0 (data),
    so each Euler step subtracts dt * predicted_velocity.
    """

    def __init__(
        self,
        sigma_min: float = 1e-4,
        ot_transport: bool = True,
    ) -> None:
        super().__init__()
        self.sigma_min = sigma_min
        self.ot_transport = ot_transport

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample diffusion times t ~ U(0, 1)."""
        return torch.rand(batch_size, device=device)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample noisy state x_t and target velocity v_t.

        Args:
            x0: (B, C, X, Y, Z) clean data sample
            t: (B,) diffusion times
            noise: optional pre-sampled noise; if None, samples fresh

        Returns:
            x_t: interpolated state at time t
            v_t: target velocity (x1 - x0)
            x1: the noise sample used
        """
        if noise is None:
            x1 = torch.randn_like(x0)
        else:
            x1 = noise

        if self.ot_transport:
            x1 = self._ot_plan(x0, x1)

        t_shape = t.reshape(-1, *([1] * (x0.dim() - 1)))
        sigma = self.sigma_min

        mu_t = (1.0 - t_shape) * x0 + t_shape * x1
        std_t = sigma * torch.ones_like(mu_t)

        x_t = mu_t + std_t * torch.randn_like(mu_t)
        v_t = x1 - x0

        return x_t, v_t, x1

    def _ot_plan(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        Approximate mini-batch OT plan by sorting along the batch dimension.

        A full Wasserstein plan is intractable for large batches. This
        approximation computes L2 distances within the batch and greedily
        matches noise samples to data samples to reduce transport cost.

        Args:
            x0: (B, ...) data samples
            x1: (B, ...) noise samples

        Returns:
            x1_permuted: (B, ...) permuted noise
        """
        B = x0.shape[0]
        x0_flat = x0.reshape(B, -1).detach()
        x1_flat = x1.reshape(B, -1).detach()

        with torch.no_grad():
            cost = torch.cdist(x0_flat.float(), x1_flat.float(), p=2)
            _, col_ind = cost.min(dim=1)

        used = set()
        perm = []
        for i in range(B):
            j = col_ind[i].item()
            if j not in used:
                used.add(j)
                perm.append(j)
            else:
                for k in range(B):
                    if k not in used:
                        used.add(k)
                        perm.append(k)
                        break

        perm_tensor = torch.tensor(perm, device=x1.device, dtype=torch.long)
        return x1[perm_tensor]

    def compute_loss(
        self,
        predicted_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss between predicted and target velocity."""
        return ((predicted_velocity - target_velocity) ** 2).mean()

    @torch.no_grad()
    def euler_integrate(
        self,
        model_fn,
        context: torch.Tensor,
        x1: Optional[torch.Tensor] = None,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Euler integration from t=1 (noise) to t=0 (data).

        Args:
            model_fn: callable (x_t, t, context) -> predicted_velocity
            context: (B, C, X, Y, Z) conditioning features
            x1: (B, C, X, Y, Z) initial noise; sampled if None
            num_steps: number of Euler steps

        Returns:
            x0: (B, C, X, Y, Z) predicted clean state
        """
        B = context.shape[0]
        device = context.device
        dtype = context.dtype

        if x1 is None:
            x1 = torch.randn_like(context)

        x = x1.clone()

        ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)
        dt = ts[0] - ts[1]

        for i in range(num_steps):
            t_val = ts[i]
            t_batch = t_val.expand(B)

            v = model_fn(x, t_batch, context)
            x = x - dt * v

        return x

    @torch.no_grad()
    def heun_integrate(
        self,
        model_fn,
        context: torch.Tensor,
        x1: Optional[torch.Tensor] = None,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Heun (2nd-order Runge-Kutta) integration for higher accuracy.

        More accurate than Euler at the same number of function evaluations
        doubled (2 model evals per step).
        """
        B = context.shape[0]
        device = context.device
        dtype = context.dtype

        if x1 is None:
            x1 = torch.randn_like(context)

        x = x1.clone()
        ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)

        for i in range(num_steps):
            t_val = ts[i]
            t_next = ts[i + 1]
            dt = t_next - t_val

            t_batch = t_val.expand(B)
            v1 = model_fn(x, t_batch, context)
            x_euler = x + dt * v1

            t_next_batch = t_next.expand(B)
            v2 = model_fn(x_euler, t_next_batch, context)
            x = x + dt * 0.5 * (v1 + v2)

        return x
