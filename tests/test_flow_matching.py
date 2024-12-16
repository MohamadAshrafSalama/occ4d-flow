import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from src.flow_matching.scheduler import OTCFMScheduler
from src.flow_matching.unet3d import UNet3D
from src.flow_matching.anisotropic_tv import AnisotropicTotalVariation, SpatialGradientPenalty
from src.flow_matching.flow_matching_head import FlowMatchingHead


B = 2
C = 32
X, Y, Z = 20, 20, 8


def make_voxel(channels=C):
    return torch.randn(B, channels, X, Y, Z)


def test_scheduler_q_sample_shapes():
    sched = OTCFMScheduler(sigma_min=1e-4, ot_transport=False)
    x0 = make_voxel()
    t = sched.sample_t(B, torch.device("cpu"))
    x_t, v_t, x1 = sched.q_sample(x0, t)

    assert x_t.shape == x0.shape, f"x_t shape mismatch: {x_t.shape}"
    assert v_t.shape == x0.shape, f"v_t shape mismatch: {v_t.shape}"
    assert x1.shape == x0.shape, f"x1 shape mismatch: {x1.shape}"


def test_scheduler_velocity_is_x1_minus_x0():
    sched = OTCFMScheduler(sigma_min=0.0, ot_transport=False)
    x0 = torch.zeros(B, C, X, Y, Z)
    x1 = torch.ones(B, C, X, Y, Z)
    noise = x1.clone()
    t = torch.zeros(B)

    x_t, v_t, _ = sched.q_sample(x0, t, noise=noise)
    expected_v = x1 - x0
    assert torch.allclose(v_t, expected_v, atol=1e-5), "Velocity should be x1 - x0"


def test_scheduler_interpolant_at_t0():
    sched = OTCFMScheduler(sigma_min=0.0, ot_transport=False)
    x0 = torch.randn(B, C, X, Y, Z)
    x1 = torch.randn(B, C, X, Y, Z)
    t = torch.zeros(B)

    x_t, _, _ = sched.q_sample(x0, t, noise=x1)
    assert torch.allclose(x_t, x0, atol=1e-5), "At t=0, x_t should equal x0"


def test_scheduler_loss_non_negative():
    sched = OTCFMScheduler()
    pred = torch.randn(B, C, X, Y, Z)
    target = torch.randn(B, C, X, Y, Z)
    loss = sched.compute_loss(pred, target)
    assert loss.item() >= 0, "CFM loss must be non-negative"


def test_scheduler_euler_integration_shape():
    sched = OTCFMScheduler(sigma_min=1e-4, ot_transport=False)
    context = make_voxel()

    def model_fn(x, t, ctx):
        return torch.zeros_like(x)

    x0 = sched.euler_integrate(model_fn, context, num_steps=3)
    assert x0.shape == context.shape, f"Euler output shape mismatch: {x0.shape}"


def test_unet3d_output_shape():
    net = UNet3D(
        in_channels=C,
        base_channels=16,
        channel_mults=(1, 2),
        num_res_blocks=1,
        norm_groups=4,
        timestep_embed_dim=64,
        timestep_mlp_hidden=128,
    )
    x = make_voxel()
    context = make_voxel()
    t = torch.rand(B)

    with torch.no_grad():
        v = net(x, t, context)

    assert v.shape == x.shape, f"UNet3D velocity shape mismatch: {v.shape}"


def test_unet3d_no_nans():
    net = UNet3D(
        in_channels=C,
        base_channels=8,
        channel_mults=(1, 2),
        num_res_blocks=1,
        norm_groups=4,
        timestep_embed_dim=32,
        timestep_mlp_hidden=64,
    )
    x = make_voxel()
    context = make_voxel()
    t = torch.rand(B)

    with torch.no_grad():
        v = net(x, t, context)

    assert not torch.isnan(v).any(), "NaN in UNet3D output"


def test_anisotropic_tv_positive():
    tv = AnisotropicTotalVariation(weight_x=0.1, weight_y=0.1, weight_z=0.5)
    x = torch.randn(B, 1, X, Y, Z)
    loss = tv(x)
    assert loss.item() >= 0, "TV loss must be non-negative"


def test_anisotropic_tv_zero_for_constant():
    tv = AnisotropicTotalVariation(weight_x=1.0, weight_y=1.0, weight_z=1.0, use_smooth_abs=False)
    x = torch.ones(B, 1, X, Y, Z)
    loss = tv(x)
    assert loss.item() < 1e-6, "TV loss for constant field should be near zero"


def test_flow_matching_head_train_loss():
    head = FlowMatchingHead(
        in_channels=C,
        base_channels=8,
        channel_mults=(1, 2),
        num_res_blocks=1,
        norm_groups=4,
        timestep_embed_dim=32,
        timestep_mlp_hidden=64,
        num_euler_steps=3,
    )
    head.train()

    context = make_voxel()
    target = make_voxel()

    losses = head.forward_train(context, target)
    assert "cfm_loss" in losses
    assert "tv_loss" in losses
    assert losses["cfm_loss"].item() >= 0
    assert losses["tv_loss"].item() >= 0


def test_flow_matching_head_sample_shape():
    head = FlowMatchingHead(
        in_channels=C,
        base_channels=8,
        channel_mults=(1, 2),
        num_res_blocks=1,
        norm_groups=4,
        timestep_embed_dim=32,
        timestep_mlp_hidden=64,
        num_euler_steps=2,
    )
    head.eval()
    context = make_voxel()

    with torch.no_grad():
        occ = head.sample(context, num_steps=2)

    assert occ.shape == (B, 1, X, Y, Z), f"Sample shape mismatch: {occ.shape}"
    assert occ.min() >= 0.0 and occ.max() <= 1.0, "Sample should be probability in [0, 1]"
