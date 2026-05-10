import torch

ckpt = torch.load("checkpoints/mock_model.pth")

print("=" * 50)
print("MODEL CHECKPOINT INSPECTION")
print("=" * 50)

total_params = 0

for component, state_dict in ckpt.items():
    n_params = sum(p.numel() for p in state_dict.values())
    total_params += n_params
    print(f"\n[{component.upper()}]")
    print(f"  Parameters : {n_params:,}")
    print(f"  Layers     : {len(state_dict)}")
    print(f"  Layer names (first 3):")
    for i, (k, v) in enumerate(state_dict.items()):
        if i >= 3:
            print(f"    ... and {len(state_dict)-3} more")
            break
        print(f"    {k:50s} {str(v.shape)}")

print("\n" + "=" * 50)
print(f"TOTAL PARAMETERS: {total_params:,}")
print(f"MODEL SIZE:       {total_params * 4 / 1e6:.1f} MB (float32)")
print("=" * 50)

# Load and run a forward pass with saved weights
print("\nRunning inference with saved weights...")

from data.mock_dataset import build_mock_dataloader
from models.encoders.sensor_encoders import LiDAREncoder, RADAREncoder
from models.fusion.weather_gating import WeatherAdaptiveGating
from models.fusion.bev_fusion_transformer import BEVFusion
from models.heads.detection_heads import MultiModalDetector, CenterPointHead

device = torch.device("cpu")

lidar_enc = LiDAREncoder().to(device)
radar_enc = RADAREncoder().to(device)
gate      = WeatherAdaptiveGating().to(device)
fusion    = BEVFusion(bev_ch=128, embed_dim=256, out_ch=256, n_layers=2).to(device)
detector  = MultiModalDetector(in_ch=256).to(device)

lidar_enc.load_state_dict(ckpt["lidar_enc"])
radar_enc.load_state_dict(ckpt["radar_enc"])
gate.load_state_dict(ckpt["gate"])
fusion.load_state_dict(ckpt["fusion"])
detector.load_state_dict(ckpt["detector"])

lidar_enc.eval()
radar_enc.eval()
gate.eval()
fusion.eval()
detector.eval()

loader = build_mock_dataloader(batch_size=1, n_samples=5)
batch  = next(iter(loader))

with torch.no_grad():
    lidar_bev      = lidar_enc(batch["lidar"])
    radar_bev      = radar_enc(batch["radar"])
    bev_maps       = {"lidar": lidar_bev, "radar": radar_bev}
    gated, aux     = gate(bev_maps)
    fused_bev      = fusion(gated)
    preds          = detector(fused_bev)
    decoded        = CenterPointHead.decode_boxes(
        {k: preds[k] for k in ["heatmap","offset","height","size","yaw"]},
        score_thresh=0.1
    )

print("\n" + "=" * 50)
print("INFERENCE RESULTS")
print("=" * 50)
print(f"Fused BEV shape  : {fused_bev.shape}")
print(f"Heatmap shape    : {preds['heatmap'].shape}")
print(f"Detections found : {len(decoded[0]['boxes'])}")
print(f"Weather weights  : { {k: round(v.item(), 3) for k, v in aux['scalar_weights'].items()} }")

if len(decoded[0]["boxes"]) > 0:
    print(f"\nTop detection:")
    print(f"  Box   : {decoded[0]['boxes'][0].numpy().round(2)}")
    print(f"  Score : {decoded[0]['scores'][0].item():.4f}")
    print(f"  Class : {decoded[0]['classes'][0].item()}")

print("\n✅ Model loaded and inference successful!")