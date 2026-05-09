import torch
from data.mock_dataset import build_mock_dataloader
from models.encoders.sensor_encoders import LiDAREncoder, RADAREncoder
from models.fusion.weather_gating import WeatherAdaptiveGating
from models.fusion.bev_fusion_transformer import BEVFusion
from models.heads.detection_heads import MultiModalDetector, MultiTaskLoss, build_targets

device = torch.device("cpu")
print("Device:", device)

lidar_enc = LiDAREncoder().to(device)
radar_enc = RADAREncoder().to(device)
gate      = WeatherAdaptiveGating().to(device)
fusion    = BEVFusion(bev_ch=128, embed_dim=256, out_ch=256, n_layers=2).to(device)
detector  = MultiModalDetector(in_ch=256).to(device)
loss_fn   = MultiTaskLoss().to(device)

params = (list(lidar_enc.parameters()) + list(radar_enc.parameters()) +
          list(gate.parameters()) + list(fusion.parameters()) +
          list(detector.parameters()) + list(loss_fn.parameters()))
optimizer = torch.optim.AdamW(params, lr=2e-4)
loader    = build_mock_dataloader(batch_size=2, n_samples=10)
print(f"Parameters: {sum(p.numel() for p in params):,}")

for epoch in range(3):
    total = 0.0
    for step, batch in enumerate(loader):
        lidar   = batch["lidar"].to(device)
        radar   = batch["radar"].to(device)
        weather = batch["weather"].to(device)

        lidar_bev      = lidar_enc(lidar)
        radar_bev      = radar_enc(radar)
        bev_maps       = {"lidar": lidar_bev, "radar": radar_bev}
        gated, gat_aux = gate(bev_maps, weather)
        fused_bev      = fusion(gated)
        preds          = detector(fused_bev)

        B = lidar.shape[0]
        tlist = [build_targets(batch["gt_boxes"][b], batch["gt_classes"][b],
                 batch["gt_velocities"][b], device=str(device)) for b in range(B)]
        targets    = {k: torch.stack([t[k] for t in tlist]).to(device) for k in tlist[0]}
        seg_target = torch.zeros(B, 128, 128, dtype=torch.long, device=device)

        losses = loss_fn(
            det_preds   = {k: preds[k] for k in ["heatmap","offset","height","size","yaw"]},
            det_targets = {k: targets[k] for k in ["heatmap","offset","height","size","yaw"]},
            vel_pred    = preds["velocity"],
            vel_target  = targets["velocity"],
            center_mask = targets["mask"],
            seg_pred    = preds["seg_logits"],
            seg_target  = seg_target,
            weather_losses = gate.gating_loss(gat_aux, weather),
        )

        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        optimizer.step()
        total += losses["total"].item()
        print(f"Epoch {epoch+1} | Step {step+1}/{len(loader)} | Loss: {losses['total'].item():.4f}")

    print(f"Epoch {epoch+1} avg loss: {total/len(loader):.4f}")
    print("-" * 40)

print("Training complete!")
torch.save({"lidar_enc": lidar_enc.state_dict(),
            "radar_enc": radar_enc.state_dict(),
            "gate":      gate.state_dict(),
            "fusion":    fusion.state_dict(),
            "detector":  detector.state_dict()},
           "checkpoints/mock_model.pth")
print("Model saved to checkpoints/mock_model.pth")