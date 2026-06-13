# 🚗 Multi-Modal Sensor Fusion for All-Weather ADAS Object Detection

> **Production-grade PyTorch implementation fusing Camera + LiDAR + RADAR + Thermal IR into a unified BEV representation for robust 3D object detection under any weather condition.**

---

## 📌 What It Does

Single-sensor perception fails in rain, fog, night, and glare — the exact conditions where ADAS systems are needed most. This system fuses **four complementary sensor modalities** into a shared **512×512 Bird's Eye View (BEV) grid**, then applies weather-adaptive attention to dynamically up-weight the sensors that are still reliable in degraded conditions.

The result: a 3D detection model that degrades gracefully under any weather scenario instead of failing catastrophically.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        Per-Sensor Encoders                               │
│                                                                          │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │    CAMERA     │  │    LiDAR      │  │    RADAR    │  │ Thermal IR │ │
│  │               │  │               │  │             │  │            │ │
│  │  Swin-T       │  │ PointPillars  │  │ Doppler +   │  │  Lightweight│ │
│  │  backbone     │  │ voxel encoder │  │ RCS →       │  │  CNN →     │ │
│  │      ↓        │  │      ↓        │  │ sparse BEV  │  │  heat feat │ │
│  │  Lift-Splat-  │  │  BEV pillars  │  │  heatmap    │  │   map      │ │
│  │  Shoot BEV    │  │               │  │             │  │            │ │
│  └──────┬────────┘  └──────┬────────┘  └──────┬──────┘  └─────┬──────┘ │
└─────────┼──────────────────┼──────────────────┼───────────────┼────────┘
          │                  │                  │               │
          └──────────────────┴──────────────────┴───────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │    Weather-Adaptive Gating       │
                    │                                  │
                    │  Soft attention weights computed  │
                    │  per modality based on estimated  │
                    │  sensor reliability in current    │
                    │  weather condition                │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │     BEV Fusion Transformer       │
                    │                                  │
                    │  Cross-attention over unified    │
                    │  512 × 512 BEV grid              │
                    │  (13M parameters)                │
                    └────────────────┬────────────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          │                          │                           │
          ▼                          ▼                           ▼
  ┌──────────────┐          ┌──────────────┐          ┌──────────────────┐
  │  3D Detection│          │ Segmentation │          │ Velocity +       │
  │     Head     │          │    Head      │          │ Trajectory Head  │
  │              │          │              │          │                  │
  │ CenterPoint- │          │ BEV semantic │          │ Per-object vel.  │
  │ style anchor-│          │ class masks  │          │ + 3s trajectory  │
  │ free boxes   │          │              │          │ prediction       │
  └──────────────┘          └──────────────┘          └──────────────────┘
```

---

## 🌦️ Why Four Modalities?

| Condition | Camera | LiDAR | RADAR | Thermal IR |
|---|:---:|:---:|:---:|:---:|
| Clear day | ✅ | ✅ | ✅ | ✅ |
| Night | ❌ | ✅ | ✅ | ✅ |
| Heavy rain | ⚠️ | ⚠️ | ✅ | ✅ |
| Dense fog | ❌ | ⚠️ | ✅ | ✅ |
| Glare / sun | ❌ | ✅ | ✅ | ✅ |
| Snow | ⚠️ | ⚠️ | ✅ | ⚠️ |

The **weather-adaptive gating module** learns to suppress failing modalities at inference time — no manual switching required.

---

## ✨ Key Features

| Feature | Detail |
|---|---|
| 📷 **Camera BEV projection** | Swin-T backbone + Lift-Splat-Shoot — image features lifted to 3D then collapsed to BEV |
| 📡 **LiDAR voxelisation** | PointPillars encoder → fast, memory-efficient BEV pillar features |
| 🔊 **RADAR Doppler fusion** | Doppler velocity + RCS cross-section → sparse BEV heatmap (unique velocity prior) |
| 🌡️ **Thermal IR channel** | Lightweight CNN extracts heat signatures invisible to optical cameras |
| 🌧️ **Adaptive gating** | Soft per-modality attention — degrades gracefully rather than failing hard |
| 🔄 **Cross-attention fusion** | Transformer over unified BEV grid; sensors attend to each other's spatial features |
| 🎯 **Multi-task heads** | Simultaneous 3D detection, segmentation, velocity estimation, and trajectory prediction |
| 📦 **TorchScript export** | Deployable on embedded ADAS hardware (Jetson / Orin) via TorchScript |
| 📊 **Multi-dataset support** | nuScenes, CADC (Canadian Adverse Conditions), RADIATE, FLIR ADAS |

---

## 🚀 Quick Start

### Prerequisites

- CUDA-capable GPU (≥8 GB VRAM recommended)
- Conda / Miniconda

### Setup

```bash
git clone https://github.com/RRN2507/Multi-Modal-Sensor-Fusion.git
cd Multi-Modal-Sensor-Fusion
conda env create -f environment.yml
conda activate adas-fusion
pip install -e .
```

### Download data

```bash
# nuScenes (primary dataset)
python scripts/download_data.py --dataset nuscenes

# Optional adverse-weather datasets
python scripts/download_data.py --dataset cadc      # Canadian winter driving
python scripts/download_data.py --dataset radiate   # Radar + camera adverse conditions
python scripts/download_data.py --dataset flir_adas # Thermal IR
```

### Train

```bash
python training/train.py --config configs/nuscenes.yaml
```

### Evaluate

```bash
python training/evaluate.py --config configs/nuscenes.yaml --checkpoint checkpoints/best.pt
```

### Export for deployment

```bash
python scripts/export.py --checkpoint checkpoints/best.pt --format torchscript
# → exports/fusion_model.ts  (Jetson / Orin deployable)
```

---

## 📁 Project Structure

```
Multi-Modal-Sensor-Fusion/
├── configs/
│   ├── nuscenes.yaml          # Primary training config
│   ├── cadc.yaml              # Canadian adverse conditions
│   └── radiate.yaml           # RADAR-heavy dataset config
├── data/
│   ├── nuscenes/
│   │   ├── maps/  samples/  sweeps/  v1.0-trainval/
│   ├── cadc/
│   ├── radiate/
│   └── flir_adas/
├── models/
│   ├── encoders/
│   │   ├── camera.py          # Swin-T + Lift-Splat-Shoot BEV
│   │   ├── lidar.py           # PointPillars voxel encoder
│   │   ├── radar.py           # Doppler + RCS → BEV heatmap
│   │   └── thermal.py         # Lightweight CNN thermal encoder
│   ├── fusion/
│   │   ├── gating.py          # Weather-adaptive soft attention
│   │   └── transformer.py     # BEV cross-attention fusion
│   ├── heads/
│   │   ├── detection.py       # CenterPoint-style 3D anchor-free head
│   │   ├── segmentation.py    # BEV semantic segmentation
│   │   └── trajectory.py      # Velocity + 3s trajectory prediction
│   └── fusion_model.py        # Full model assembly
├── training/
│   ├── train.py               # Training loop + mixed precision
│   ├── evaluate.py            # mAP / NDS evaluation harness
│   └── losses.py              # Multi-task loss with uncertainty weighting
├── scripts/
│   ├── download_data.py       # Dataset download + preprocessing
│   └── export.py              # TorchScript export for deployment
├── environment.yml
└── README.md
```

---

## 🔧 Model Specs

| Component | Architecture | Parameters |
|---|---|---|
| Camera encoder | Swin-T + LSS BEV projection | ~28M |
| LiDAR encoder | PointPillars | ~4M |
| RADAR encoder | Sparse BEV CNN | ~0.5M |
| Thermal encoder | Lightweight CNN | ~0.5M |
| Gating module | 4-head soft attention | ~0.1M |
| BEV transformer | 6-layer cross-attention, 512×512 grid | ~18M |
| Detection head | CenterPoint-style, anchor-free | ~2M |
| Segmentation head | BEV semantic masks | ~1M |
| Velocity + trajectory | Per-object MLP + GRU | ~0.5M |
| **Total** | | **~13M** |

---

## 📊 Datasets

| Dataset | Modalities | Conditions | Use |
|---|---|---|---|
| [nuScenes](https://nuscenes.org) | Camera, LiDAR, RADAR | Mixed | Primary training + eval |
| [CADC](https://cadcd.uwaterloo.ca) | Camera, LiDAR | Snow / winter | Adverse-weather fine-tune |
| [RADIATE](https://pro.hw.ac.uk/radiate) | Camera, LiDAR, RADAR | Fog, rain, night | RADAR-specific eval |
| [FLIR ADAS](https://www.flir.com/oem/adas/adas-dataset-form/) | Camera, Thermal | Night, heat | Thermal ablation |

---

## 🗓️ Implementation Timeline

| Day | Focus |
|---|---|
| 1 | Repo setup, conda environment, multi-dataset data loaders |
| 2 | Per-sensor encoders — camera (Swin-T + LSS), LiDAR (PointPillars), RADAR, Thermal |
| 3 | Weather-adaptive gating module + ablation harness |
| 4 | BEV fusion transformer — cross-attention over unified grid |
| 5 | Multi-task heads (detection, segmentation, velocity, trajectory) + loss functions |
| 6 | Training loop, mixed-precision, evaluation (NDS / mAP), TorchScript export |

---

## 📄 License

MIT — free to use for learning or production.

---

<p align="center">
  Built by <a href="https://github.com/RRN2507">Rushikesh R. Navale</a> ·
  <a href="https://linkedin.com/in/rrn2507">LinkedIn</a>
</p>
