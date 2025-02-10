from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DataConfig:
    dataset: str = "nuscenes"
    data_root: str = "/data/nuscenes"
    version: str = "v1.0-trainval"

    voxel_size: float = 0.5
    point_cloud_range: Tuple[float, ...] = (-50.0, -50.0, -5.0, 50.0, 50.0, 3.0)
    grid_size: Tuple[int, int, int] = (200, 200, 16)

    past_frames: int = 5
    future_frames: int = 5
    max_points_per_pillar: int = 20
    max_pillars: int = 30000

    num_classes: int = 1
    lidar_channels: int = 32
    lidar_range: float = 70.0

    augment_train: bool = True
    aug_flip_x: bool = True
    aug_flip_y: bool = True
    aug_rotate_range: float = 0.3927
    aug_scale_range: Tuple[float, float] = (0.95, 1.05)

    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass
class EncoderConfig:
    in_channels: int = 6
    pillar_feat_channels: List[int] = field(default_factory=lambda: [64, 128])
    bev_channels: int = 128
    radial_encoding_dim: int = 32
    radial_max_freq: int = 8
    out_channels: int = 256

    grid_x: int = 200
    grid_y: int = 200
    voxel_x: float = 0.5
    voxel_y: float = 0.5
    x_offset: float = -50.0
    y_offset: float = -50.0


@dataclass
class FPNConfig:
    in_channels: int = 256
    out_channels: int = 256
    num_levels: int = 3
    deformable_groups: int = 1
    use_deformable: bool = True
    norm_groups: int = 32


@dataclass
class VoxelLifterConfig:
    bev_channels: int = 256
    voxel_channels: int = 128
    num_height_bins: int = 16
    height_mlp_hidden: int = 64
    grid_x: int = 200
    grid_y: int = 200
    grid_z: int = 16


@dataclass
class TemporalConfig:
    voxel_channels: int = 128
    timestamp_dim: int = 64
    timestamp_max_freq: int = 8
    timestamp_mlp_hidden: int = 128
    confidence_hidden: int = 64
    max_time_delta: float = 5.0
    gru_hidden_channels: int = 128
    gru_kernel_size: int = 3
    num_gru_layers: int = 2


@dataclass
class PropagatorConfig:
    voxel_channels: int = 128
    hidden_channels: int = 128
    gru_kernel_size: int = 3
    num_future_steps: int = 5
    ego_interp_mode: str = "bilinear"
    ego_padding_mode: str = "zeros"


@dataclass
class FlowMatchingConfig:
    in_channels: int = 128
    base_channels: int = 64
    channel_mults: Tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    norm_groups: int = 8
    timestep_embed_dim: int = 256
    timestep_mlp_hidden: int = 512
    num_euler_steps: int = 10
    sigma_min: float = 1e-4
    ot_transport: bool = True
    tv_weight_xy: float = 0.01
    tv_weight_z: float = 0.05


@dataclass
class DecoderConfig:
    in_channels: int = 128
    hidden_channels: List[int] = field(default_factory=lambda: [128, 64, 32])
    num_classes: int = 1
    uncertainty_head: bool = True
    norm_groups: int = 8


@dataclass
class LossConfig:
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    focal_weight: float = 1.0

    lovasz_weight: float = 1.0

    dice_weight: float = 0.5
    dice_smooth: float = 1.0

    sparsity_weight: float = 0.1
    sparsity_target: float = 0.05

    temporal_consistency_weight: float = 0.5
    velocity_smoothness_weight: float = 0.1

    ray_freespace_weight: float = 0.3
    mass_conservation_weight: float = 0.1

    uncertainty_nll_weight: float = 0.5
    range_weight_max: float = 3.0
    range_weight_min: float = 1.0
    range_weight_cutoff: float = 40.0

    flow_matching_weight: float = 1.0
    tv_weight: float = 0.01


@dataclass
class TrainConfig:
    epochs: int = 24
    batch_size: int = 2
    num_workers: int = 4

    optimizer: str = "adamw"
    lr: float = 2e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    grad_clip: float = 35.0

    scheduler: str = "cosine"
    warmup_epochs: int = 2
    min_lr: float = 1e-6

    amp: bool = True
    grad_checkpoint: bool = True
    sync_bn: bool = True

    ema_decay: float = 0.999
    ema_update_every: int = 10

    log_interval: int = 50
    val_interval: int = 1
    save_interval: int = 2
    keep_last_n: int = 3

    seed: int = 42
    find_unused_parameters: bool = False

    wandb_project: str = "occ4d-flow"
    wandb_entity: str = ""
    wandb_tags: List[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    fpn: FPNConfig = field(default_factory=FPNConfig)
    voxel_lifter: VoxelLifterConfig = field(default_factory=VoxelLifterConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    propagator: PropagatorConfig = field(default_factory=PropagatorConfig)
    flow_matching: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


@dataclass
class Occ4DFlowConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    output_dir: str = "checkpoints/run"
    experiment_name: str = "occ4d_flow_default"
    resume: Optional[str] = None
    eval_only: bool = False

