# occ4d-flow

4D occupancy forecasting using continuous-time flow matching. The system predicts the future volumetric occupancy of the scene from a sequence of past LiDAR sweeps and ego-motion poses.

---

## Overview

Autonomous driving requires accurate predictions of how the occupied regions of 3D space evolve over time. Standard voxel-based approaches discretize this into a fixed-horizon classification problem, losing the continuous temporal structure of scene dynamics. This system addresses that by combining three ideas:

1. **Pillar-based BEV encoding** with radial range encoding — processes raw LiDAR points into compact bird's-eye-view feature maps without losing radial distance context.
2. **Continuous-time temporal aggregation** — embeds observation timestamps as continuous values, applies delay-aware confidence gating, and aggregates history through a 3D ConvGRU. This handles irregular sampling intervals and missing frames naturally.
3. **3D flow matching for future generation** — uses an optimal-transport conditional flow matching objective to train a 3D U-Net that generates plausible future occupancy volumes. Combined with a discriminative head, the final prediction is an uncertainty-weighted ensemble.

---

## Architecture

```
LiDAR Sweeps (T frames)
        │
        ▼
┌───────────────────┐
│   Pillar Encoder  │  scatter → BEV grid, radial range encoding
└────────┬──────────┘
         │  (B, C, H, W) per frame
         ▼
┌───────────────────┐
│       FPN         │  3-level pyramid, deformable conv
└────────┬──────────┘
         │  multi-scale BEV features
         ▼
┌───────────────────┐
│   Voxel Lifter    │  learned per-height-bin projection → (B, C, X, Y, Z)
└────────┬──────────┘
         │
         ▼
┌───────────────────────────────────┐
│      Temporal Aggregator          │
│  timestamp embed → conf gate →    │
│  3D ConvGRU (across T frames)     │
└────────┬──────────────────────────┘
         │  aggregated hidden state
         ▼
┌───────────────────┐
│    Ego Warper     │  SE(2) warp to current ego frame
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Future Propagator │  ConvGRU steps for each future timestep
└────────┬──────────┘
         │  (B, C, X, Y, Z) future feature per step
         │
    ┌────┴─────┐
    │          │
    ▼          ▼
┌────────┐  ┌──────────────────┐
│ Discr. │  │  Flow Matching   │
│ Decoder│  │  Head (3D U-Net) │
└───┬────┘  └────────┬─────────┘
    │  logits+σ²     │  sampled occupancy
    └──────┬─────────┘
           ▼
  ┌─────────────────┐
  │ Ensemble Blender│  uncertainty-weighted fusion
  └────────┬────────┘
           ▼
  Predicted occupancy (B, X, Y, Z, T_future)
```

---

## Installation

```bash
git clone https://github.com/mohamadashrafsalama/occ4d-flow.git
cd occ4d-flow
pip install -e ".[dev]"
```

With Docker:

```bash
docker build -t occ4d-flow .
docker run --gpus all -v /data:/data occ4d-flow
```

---

## Data Preparation

### nuScenes

Download nuScenes from https://www.nuscenes.org. Set the root path in `config/nuscenes.py` or pass `--data-root` at the command line. The dataset loader expects the standard nuScenes directory layout with `v1.0-trainval` annotations.

### SemanticKITTI

Download SemanticKITTI from http://www.semantic-kitti.org. Set `data_root` in the loader. Sequences 00–10 are used for training; 08 for validation.

---

## Training

Single-node multi-GPU training with DDP:

```bash
torchrun --nproc_per_node=8 scripts/train.py \
    --config config/nuscenes.py \
    --data-root /data/nuscenes \
    --output-dir checkpoints/nuscenes_run0 \
    --batch-size 2 \
    --epochs 24 \
    --lr 2e-4
```

Resume from checkpoint:

```bash
torchrun --nproc_per_node=8 scripts/train.py \
    --resume checkpoints/nuscenes_run0/epoch_12.pth \
    ...
```

Key training flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--batch-size` | 2 | Per-GPU batch size |
| `--epochs` | 24 | Total training epochs |
| `--lr` | 2e-4 | Peak learning rate |
| `--warmup-epochs` | 2 | Linear warmup duration |
| `--amp` | True | Automatic mixed precision |
| `--grad-checkpoint` | True | Gradient checkpointing on U-Net |
| `--ema-decay` | 0.999 | EMA model decay |
| `--past-frames` | 5 | Number of past LiDAR sweeps |
| `--future-frames` | 5 | Number of future steps to predict |
| `--flow-steps` | 10 | Euler integration steps at inference |

---

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/nuscenes_run0/best.pth \
    --data-root /data/nuscenes \
    --split val \
    --flow-steps 20
```

Reported metrics:

- **mIoU** — mean intersection-over-union across occupancy classes
- **VPQ** — video panoptic quality over future horizons
- **Ray precision / recall** — sensor-ray-based evaluation of free-space accuracy

---

## Visualization

```bash
python scripts/visualize.py \
    --checkpoint checkpoints/nuscenes_run0/best.pth \
    --data-root /data/nuscenes \
    --scene-token <token> \
    --output-dir vis/
```

Renders predicted and ground-truth voxel grids using Open3D point cloud visualization. Each future timestep is saved as a separate frame.

---

## Configuration

All hyperparameters are defined as Python dataclasses in `config/default.py`. Dataset-specific overrides are in `config/nuscenes.py` and can be extended for other datasets.

Key model dimensions:

| Parameter | Value |
|-----------|-------|
| Voxel grid (nuScenes) | 200 × 200 × 16 |
| Voxel size | 0.5 m |
| BEV feature channels | 256 |
| 3D feature channels | 128 |
| Past frames | 5 |
| Future frames | 5 |
| U-Net base channels | 64 |
| Flow matching steps (train) | 1 (CFM) |
| Flow matching steps (infer) | 10–20 |

---

## Loss Functions

The total loss is a weighted sum of:

- **Focal BCE** — class-imbalance-aware binary cross entropy
- **Lovasz** — differentiable surrogate for IoU
- **Dice** — soft Dice loss for voxel overlap
- **Sparsity** — L1 penalty to match observed occupancy ratio
- **Temporal consistency** — L2 between warped t-1 prediction and t prediction
- **Velocity smoothness** — spatial gradient regularization on predicted velocity
- **Ray free-space** — penalizes occupancy between sensor and first observed return
- **Mass conservation** — penalizes large changes in total predicted mass across steps
- **Uncertainty NLL** — Gaussian negative log-likelihood weighted by predicted uncertainty
- **Flow matching MSE** — velocity prediction loss inside the CFM objective

---

## License

Experimental research code. Not for redistribution without explicit permission.

---

## Author

Mohamed Ashraf Salama
Queen's University, Kingston, Ontario
