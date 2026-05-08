# Multi-Modal Sensor Fusion for All-Weather Object Detection (ADAS)

A production-grade PyTorch implementation of camera + LiDAR + RADAR + Thermal IR fusion
for robust 3D object detection under all weather conditions.

## Architecture
- **Camera encoder**: Swin-T backbone → Lift-Splat-Shoot BEV projection
- **LiDAR encoder**: PointPillars → BEV pillars
- **RADAR encoder**: Doppler + RCS → sparse BEV heatmap
- **Thermal IR encoder**: Lightweight CNN → heat feature map
- **Weather-adaptive gating**: Soft attention weights per modality
- **BEV fusion transformer**: Cross-attention over unified 512×512 BEV grid
- **Multi-task heads**: 3D detection, segmentation, velocity, trajectory

## 6-Day Implementation Plan
| Day | Focus |
|-----|-------|
| 1   | Repo setup, environment, data loading pipeline |
| 2   | Per-sensor encoders (camera, LiDAR, RADAR, thermal) |
| 3   | Weather-adaptive gating module |
| 4   | BEV fusion transformer |
| 5   | Multi-task detection heads + losses |
| 6   | Training loop, evaluation, inference & deployment |

## Quick Start
```bash
git clone https://github.com/<your-org>/adas-sensor-fusion.git
cd adas-sensor-fusion
conda env create -f environment.yml
conda activate adas-fusion
pip install -e .
python scripts/download_data.py --dataset nuscenes
python training/train.py --config configs/nuscenes.yaml
```

## Dataset structure expected
```
data/
  nuscenes/
    maps/  samples/  sweeps/  v1.0-trainval/
  cadc/
  radiate/
  flir_adas/
```

## License
MIT
