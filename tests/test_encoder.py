import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from src.encoder.pillar_encoder import PillarEncoder, RadialRangeEncoding
from src.encoder.fpn import FeaturePyramidNetwork
from src.encoder.voxel_lifter import VoxelLifter


GRID_X = 200
GRID_Y = 200
GRID_Z = 16
B = 2
P = 50
N = 10
BEV_CH = 256
VOXEL_CH = 128


def make_pillar_inputs(batch_size=B, num_pillars=P, max_pts=N, bev_ch=BEV_CH):
    points = torch.randn(batch_size, num_pillars, max_pts, 4)
    coords_per_batch = torch.stack([
        torch.randint(0, GRID_X, (num_pillars,)),
        torch.randint(0, GRID_Y, (num_pillars,)),
    ], dim=1)
    coords_list = []
    for b in range(batch_size):
        b_idx = torch.full((num_pillars, 1), b, dtype=torch.long)
        coords_list.append(torch.cat([b_idx, coords_per_batch], dim=1))
    flat_coords = torch.cat(coords_list, dim=0)
    return points, flat_coords


def test_radial_range_encoding_shape():
    enc = RadialRangeEncoding(encoding_dim=32, max_freq=8)
    r = torch.rand(B, GRID_X, GRID_Y)
    out = enc(r)
    assert out.shape == (B, GRID_X, GRID_Y, 32), f"Expected (B, X, Y, 32), got {out.shape}"


def test_radial_range_encoding_range():
    enc = RadialRangeEncoding(encoding_dim=16, max_freq=4)
    r = torch.rand(100)
    out = enc(r)
    assert out.min() >= -1.01 and out.max() <= 1.01, "Sinusoidal outputs should be in [-1, 1]"


def test_pillar_encoder_output_shape():
    encoder = PillarEncoder(
        in_channels=6,
        pillar_feat_channels=[64, 128],
        bev_channels=128,
        radial_encoding_dim=32,
        out_channels=BEV_CH,
        grid_x=GRID_X,
        grid_y=GRID_Y,
    )
    points, flat_coords = make_pillar_inputs()
    bev = encoder(points, flat_coords, B)
    assert bev.shape == (B, BEV_CH, GRID_X, GRID_Y), f"Unexpected BEV shape: {bev.shape}"


def test_pillar_encoder_no_nans():
    encoder = PillarEncoder(
        in_channels=6,
        pillar_feat_channels=[64],
        bev_channels=64,
        radial_encoding_dim=16,
        out_channels=128,
        grid_x=GRID_X,
        grid_y=GRID_Y,
    )
    points, flat_coords = make_pillar_inputs()
    bev = encoder(points, flat_coords, B)
    assert not torch.isnan(bev).any(), "NaN in pillar encoder output"
    assert not torch.isinf(bev).any(), "Inf in pillar encoder output"


def test_fpn_output_shape():
    fpn = FeaturePyramidNetwork(
        in_channels=BEV_CH,
        out_channels=BEV_CH,
        num_levels=3,
        use_deformable=True,
        norm_groups=32,
    )
    x = torch.randn(B, BEV_CH, GRID_X, GRID_Y)
    out = fpn(x)
    assert out.shape == (B, BEV_CH, GRID_X, GRID_Y), f"FPN output shape mismatch: {out.shape}"


def test_fpn_multiscale():
    fpn = FeaturePyramidNetwork(
        in_channels=BEV_CH,
        out_channels=BEV_CH,
        num_levels=3,
        use_deformable=False,
        norm_groups=32,
    )
    x = torch.randn(B, BEV_CH, GRID_X, GRID_Y)
    levels = fpn.forward_multiscale(x)
    assert len(levels) == 3
    assert levels[0].shape == (B, BEV_CH, GRID_X, GRID_Y)
    assert levels[1].shape == (B, BEV_CH, GRID_X // 2, GRID_Y // 2)
    assert levels[2].shape == (B, BEV_CH, GRID_X // 4, GRID_Y // 4)


def test_voxel_lifter_output_shape():
    lifter = VoxelLifter(
        bev_channels=BEV_CH,
        voxel_channels=VOXEL_CH,
        num_height_bins=GRID_Z,
        height_mlp_hidden=64,
        grid_x=GRID_X,
        grid_y=GRID_Y,
        grid_z=GRID_Z,
    )
    bev = torch.randn(B, BEV_CH, GRID_X, GRID_Y)
    voxels = lifter(bev)
    assert voxels.shape == (B, VOXEL_CH, GRID_X, GRID_Y, GRID_Z), (
        f"Voxel lifter shape mismatch: {voxels.shape}"
    )


def test_voxel_lifter_no_nans():
    lifter = VoxelLifter(
        bev_channels=BEV_CH,
        voxel_channels=VOXEL_CH,
        num_height_bins=GRID_Z,
        height_mlp_hidden=32,
        grid_x=GRID_X,
        grid_y=GRID_Y,
        grid_z=GRID_Z,
    )
    bev = torch.randn(B, BEV_CH, GRID_X, GRID_Y)
    voxels = lifter(bev)
    assert not torch.isnan(voxels).any(), "NaN in voxel lifter output"


def test_full_encoder_pipeline():
    encoder = PillarEncoder(
        in_channels=6, pillar_feat_channels=[64, 128],
        bev_channels=128, radial_encoding_dim=32, out_channels=BEV_CH,
        grid_x=GRID_X, grid_y=GRID_Y,
    )
    fpn = FeaturePyramidNetwork(in_channels=BEV_CH, out_channels=BEV_CH, use_deformable=False)
    lifter = VoxelLifter(BEV_CH, VOXEL_CH, GRID_Z, 32, GRID_X, GRID_Y, GRID_Z)

    points, flat_coords = make_pillar_inputs()
    bev = encoder(points, flat_coords, B)
    bev = fpn(bev)
    voxels = lifter(bev)

    assert voxels.shape == (B, VOXEL_CH, GRID_X, GRID_Y, GRID_Z)
    assert not torch.isnan(voxels).any()
