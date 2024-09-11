from config.default import (
    DataConfig,
    EncoderConfig,
    FPNConfig,
    VoxelLifterConfig,
    TemporalConfig,
    PropagatorConfig,
    FlowMatchingConfig,
    DecoderConfig,
    LossConfig,
    TrainConfig,
    ModelConfig,
    Occ4DFlowConfig,
)


def get_nuscenes_config(data_root: str = "/data/nuscenes") -> Occ4DFlowConfig:
    data = DataConfig(
        dataset="nuscenes",
        data_root=data_root,
        version="v1.0-trainval",
        voxel_size=0.5,
        point_cloud_range=(-50.0, -50.0, -5.0, 50.0, 50.0, 3.0),
        grid_size=(200, 200, 16),
        past_frames=5,
        future_frames=5,
        max_points_per_pillar=20,
        max_pillars=30000,
        num_classes=1,
        lidar_range=70.0,
        aug_flip_x=True,
        aug_flip_y=True,
        aug_rotate_range=0.3927,
    )

    encoder = EncoderConfig(
        in_channels=6,
        pillar_feat_channels=[64, 128],
        bev_channels=128,
        radial_encoding_dim=32,
        out_channels=256,
        grid_x=200,
        grid_y=200,
        voxel_x=0.5,
        voxel_y=0.5,
        x_offset=-50.0,
        y_offset=-50.0,
    )

    train = TrainConfig(
        epochs=24,
        batch_size=2,
        lr=2e-4,
        warmup_epochs=2,
        ema_decay=0.999,
        wandb_project="occ4d-flow",
        wandb_tags=["nuscenes"],
    )

    model = ModelConfig(
        encoder=encoder,
        fpn=FPNConfig(),
        voxel_lifter=VoxelLifterConfig(),
        temporal=TemporalConfig(),
        propagator=PropagatorConfig(),
        flow_matching=FlowMatchingConfig(),
        decoder=DecoderConfig(),
    )

    return Occ4DFlowConfig(
        model=model,
        data=data,
        loss=LossConfig(),
        train=train,
        experiment_name="occ4d_flow_nuscenes",
    )
