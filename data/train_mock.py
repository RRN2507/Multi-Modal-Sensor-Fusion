"""
Full training run on mock data — no dataset download needed.
Tests the complete pipeline end to end.
"""

import torch
from data.mock_dataset import build_mock_dataloader
from models.encoders.sensor_encoders import LiDAREncoder, RADAREncoder
from models.fusion.weather_gating import WeatherAdaptiveGating
from models.fusion.bev_fusion_transformer import BEVFusion
from models.heads.detection_heads import (
    MultiModalDetector, MultiTaskLoss, build_targets
)

# ── Config ──────────────────────────────────────────
EPOCHS     = 5
BATCH_SIZE = 2
LR         = 2e-4
BEV_H = BEV_W = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Build models ────────────────────────────────────
lidar_enc = LiDAREncoder().to(device)
radar_enc = RADAREncoder().to(device)
gate      = WeatherAdaptiveGating().to(device)
fusion    = BEVFusion(bev_ch=128, embed_dim=256, out_ch=256, n_layers=2).to(device)
detector  = MultiModalDetector(in_ch=256).to(device)
loss_fn   = MultiTaskLoss().to(device)

params = (
    list(lidar_enc.parameters()) +
    list(radar_enc.parameters()) +
    list(gate.parameters()) +
    list(fusion.parameters()) +
    list(detector.parameters()) +
    list(loss_fn.parameters())
)
optimizer = torch.optim.AdamW(params, lr=LR)
loader    = build_mock_dataloader(batch_size=BATCH_SIZE, n_samples=20)

print(f"Total parameters: {sum(p.numel() for p in params):,}")
print(f"Batches per epoch: {len(loader)}")
print("-" * 50)

# ── Training loop ───────────────────────────────────
for epoch in range(EPOCHS):
    total_loss = 0.0

    for step, batch in enumerate(loader):
        lidar  = batch["lidar"].to(device)
        radar  = batch["radar"].to(device)
        weather = batch["weather"].to(device)

        # Forward pass
        lidar_bev = lidar_enc(lidar)
        radar_bev = radar_enc(radar)

        bev_maps       = {"lidar": lidar_bev, "radar": radar_bev}
        gated, gat_aux = gate(bev_maps, weather)
        fused_bev      = fusion(gated)
        preds          = detector(fused_bev)

        # Build targets from mock GT
        B = lidar.shape[0]
        targets_list = []
        for b in range(B):
            t = build_targets(
                batch["gt_boxes"][b],
                batch["gt_classes"][b],
                batch["gt_velocities"][b],
                device=str(device),
            )
            targets_list.append(t)
        targets = {k: torch.stack([t[k] for t in targets_list]).to(device)
                   for k in targets_list[0]}

        seg_target = torch.zeros(B, BEV_H, BEV_W,
                                 dtype=torch.long, device=device)

        # Compute loss
        weather_losses = gate.gating_loss(gat_aux, weather)
        losses = loss_fn(
            det_preds   = {k: preds[k] for k in ["heatmap","offset","height","size","yaw"]},
            det_targets = {k: targets[k] for k in ["heatmap","offset","height","size","yaw"]},
            vel_pred    = preds["velocity"],
            vel_target  = targets["velocity"],
            center_mask = targets["mask"],
            seg_pred    = preds["seg_logits"],
            seg_target  = seg_target,
            weather_losses = weather_losses,
        )

        # Backward pass
        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        optimizer.step()

        total_loss += losses["total"].item()

        print(f"  Epoch {epoch+1} | Step {step+1}/{len(loader)} "
              f"| Loss: {losses['total'].item():.4f} "
              f"| Heat: {losses['heatmap'].item():.4f} "
              f"| Vel: {losses['velocity'].item():.4f}")

    avg = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} complete — avg loss: {avg:.4f}")
    print("-" * 50)

print("Training complete!")
torch.save({
    "lidar_enc": lidar_enc.state_dict(),
    "radar_enc": radar_enc.state_dict(),
    "gate":      gate.state_dict(),
    "fusion":    fusion.state_dict(),
    "detector":  detector.state_dict(),
}, "checkpoints/mock_model.pth")
print("Model saved to checkpoints/mock_model.pth")