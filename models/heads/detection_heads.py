"""
Day 5 — Multi-task detection heads + loss functions.

Heads operating on the fused BEV (B, 256, 128, 128):
  DetectionHead   : 3-D bounding box regression + classification
  VelocityHead    : per-object velocity (from RADAR Doppler)
  SegmentationHead: BEV semantic segmentation (free space / lane)
  PredictionHead  : 3-second trajectory prediction

All losses collected in MultiTaskLoss with learnable uncertainty weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


N_CLASSES  = 9   # nuScenes detection categories
BEV_H = BEV_W = 128
SEG_CLASSES = 3  # background, free-space, lane


# ──────────────────────────────────────────────────────
# 5.1  Shared BEV backbone neck (FPN-style)
# ──────────────────────────────────────────────────────

class BEVNeck(nn.Module):
    """
    Light FPN neck over the fused BEV.
    Produces multi-scale feature maps for the detection head.
    """
    def __init__(self, in_ch: int = 256, out_ch: int = 128):
        super().__init__()
        self.down2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.up4_proj = nn.Conv2d(out_ch, out_ch, 1)
        self.up2_proj = nn.Conv2d(out_ch, out_ch, 1)
        self.out_proj = nn.Conv2d(in_ch + out_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_ch, H, W) → (B, out_ch, H, W)"""
        p1 = x                              # (B, 256, 128, 128)
        p2 = self.down2(p1)                 # (B, 128, 64, 64)
        p4 = self.down4(p2)                 # (B, 128, 32, 32)

        up_p4 = F.interpolate(self.up4_proj(p4), scale_factor=2, mode="bilinear", align_corners=False)
        p2    = p2 + up_p4
        up_p2 = F.interpolate(self.up2_proj(p2), scale_factor=2, mode="bilinear", align_corners=False)

        # Concatenate original scale + upsampled
        fused = torch.cat([p1, up_p2], dim=1)   # (B, 256+128, 128, 128)
        return self.out_proj(fused)              # (B, 128, 128, 128)


# ──────────────────────────────────────────────────────
# 5.2  Heatmap-based 3-D detection head (CenterPoint-style)
# ──────────────────────────────────────────────────────

class CenterPointHead(nn.Module):
    """
    For each BEV cell, predicts:
      heatmap  (N_CLASSES,)    — Gaussian heatmap for center probability
      offset   (2,)            — sub-voxel center offset Δx, Δy
      height   (1,)            — z center
      size     (3,)            — w, l, h
      yaw      (2,)            — (sin θ, cos θ) for cyclic angle
    Total output channels: N_CLASSES + 2 + 1 + 3 + 2 = N_CLASSES + 8
    """

    def __init__(self, in_ch: int = 128, n_classes: int = N_CLASSES):
        super().__init__()
        self.n_cls = n_classes

        def _head(out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=1),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, 1),
            )

        self.heatmap = _head(n_classes)
        self.offset  = _head(2)
        self.height  = _head(1)
        self.size    = _head(3)
        self.yaw     = _head(2)

    def forward(self, bev: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "heatmap": self.heatmap(bev).sigmoid(),   # (B, N_cls, H, W)
            "offset":  self.offset(bev),              # (B, 2, H, W)
            "height":  self.height(bev),              # (B, 1, H, W)
            "size":    self.size(bev).exp(),          # (B, 3, H, W) always positive
            "yaw":     self.yaw(bev),                 # (B, 2, H, W) — atan2 to recover θ
        }

    @staticmethod
    def decode_boxes(
        preds: Dict[str, torch.Tensor],
        score_thresh: float = 0.1,
        voxel_size: float = 0.8,
        pc_range: Tuple = (-51.2, -51.2),
        max_detections: int = 200,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Post-process heatmaps into (B,) lists of detection dicts.
        Each dict: boxes (M,7), scores (M,), classes (M,).
        """
        B, N_cls, H, W = preds["heatmap"].shape
        results = []
        for b in range(B):
            heat  = preds["heatmap"][b]       # (N_cls, H, W)
            max_h = F.max_pool2d(heat.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
            mask  = (heat == max_h) & (heat > score_thresh)

            boxes_b, scores_b, classes_b = [], [], []
            for cls in range(N_cls):
                ys, xs = mask[cls].nonzero(as_tuple=True)
                if len(xs) == 0:
                    continue
                sc = heat[cls, ys, xs]
                dx = preds["offset"][b, 0, ys, xs]
                dy = preds["offset"][b, 1, ys, xs]
                cx = (xs.float() + dx) * voxel_size + pc_range[0]
                cy = (ys.float() + dy) * voxel_size + pc_range[1]
                cz = preds["height"][b, 0, ys, xs]
                w  = preds["size"][b, 0, ys, xs]
                l  = preds["size"][b, 1, ys, xs]
                h  = preds["size"][b, 2, ys, xs]
                sa = preds["yaw"][b, 0, ys, xs]
                co = preds["yaw"][b, 1, ys, xs]
                yaw = torch.atan2(sa, co)

                box = torch.stack([cx, cy, cz, w, l, h, yaw], dim=1)  # (M,7)
                boxes_b.append(box)
                scores_b.append(sc)
                classes_b.append(torch.full((len(sc),), cls, dtype=torch.long))

            if boxes_b:
                boxes_b   = torch.cat(boxes_b)[:max_detections]
                scores_b  = torch.cat(scores_b)[:max_detections]
                classes_b = torch.cat(classes_b)[:max_detections]
            else:
                boxes_b   = torch.zeros(0, 7)
                scores_b  = torch.zeros(0)
                classes_b = torch.zeros(0, dtype=torch.long)

            results.append({"boxes": boxes_b, "scores": scores_b, "classes": classes_b})
        return results


# ──────────────────────────────────────────────────────
# 5.3  Velocity estimation head
# ──────────────────────────────────────────────────────

class VelocityHead(nn.Module):
    """
    Regresses 2-D velocity (vx, vy) per BEV cell.
    Loss is masked to ground-truth object centers only.
    """
    def __init__(self, in_ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 2, 1),   # (vx, vy)
        )

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        return self.net(bev)   # (B, 2, H, W)


# ──────────────────────────────────────────────────────
# 5.4  BEV Segmentation head
# ──────────────────────────────────────────────────────

class SegmentationHead(nn.Module):
    """
    Per-cell BEV semantic segmentation.
    Classes: 0=background, 1=free-space, 2=lane-marking.
    """
    def __init__(self, in_ch: int = 128, n_seg: int = SEG_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1),
            nn.BatchNorm2d(in_ch // 2), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, n_seg, 1),
        )

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        return self.net(bev)   # (B, n_seg, H, W)


# ──────────────────────────────────────────────────────
# 5.5  Trajectory prediction head (3-second horizon)
# ──────────────────────────────────────────────────────

T_FUTURE = 6   # 6 × 0.5 s = 3 seconds

class TrajectoryHead(nn.Module):
    """
    Predicts K=6 diverse future trajectories (multi-modal prediction)
    for the top-K detected agents.
    Each trajectory: T_FUTURE × 2 (Δx, Δy relative to current position).
    """
    def __init__(self, in_ch: int = 128, K: int = 6, T: int = T_FUTURE):
        super().__init__()
        self.K = K
        self.T = T
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, K * T * 2 + K, 1),  # trajectories + mode scores
        )

    def forward(self, bev: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.net(bev)   # (B, K*T*2 + K, H, W)
        traj  = raw[:, :self.K * self.T * 2]   # (B, K*T*2, H, W)
        modes = raw[:, self.K * self.T * 2:]   # (B, K, H, W)
        B, _, H, W = bev.shape
        traj  = traj.view(B, self.K, self.T, 2, H, W)
        modes = modes.softmax(dim=1)
        return {"traj": traj, "mode_scores": modes}


# ──────────────────────────────────────────────────────
# 5.6  Loss functions
# ──────────────────────────────────────────────────────

def gaussian_focal_loss(
    pred: torch.Tensor,    # (B, N_cls, H, W) predictions (already sigmoid)
    target: torch.Tensor,  # (B, N_cls, H, W) Gaussian heatmap targets [0,1]
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """
    Modified focal loss for heatmap regression (CornerNet / CenterPoint style).
    """
    pos_mask = target.eq(1.0)
    neg_mask = ~pos_mask
    neg_weights = (1 - target[neg_mask]).pow(beta)

    pos_loss = -torch.log(pred[pos_mask].clamp(1e-6)).pow(alpha) * torch.log(pred[pos_mask].clamp(1e-6))
    neg_loss = -neg_weights * torch.log(1 - pred[neg_mask].clamp(0, 1 - 1e-6)) * (pred[neg_mask]).pow(alpha)

    n_pos = pos_mask.float().sum().clamp(1)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def reg_l1_loss(
    pred: torch.Tensor,    # (B, C, H, W)
    target: torch.Tensor,  # (B, C, H, W)
    mask: torch.Tensor,    # (B, 1, H, W) — 1 at object centers
) -> torch.Tensor:
    mask = mask.expand_as(pred)
    n = mask.float().sum().clamp(1)
    return (F.l1_loss(pred, target, reduction="none") * mask).sum() / n


class MultiTaskLoss(nn.Module):
    """
    Combines all task losses with learnable uncertainty weighting.
    Reference: Kendall et al. "Multi-Task Learning Using Uncertainty" 2018.
    σ² for each task is a learnable log-variance.
    """

    TASK_NAMES = ["heatmap", "offset", "height", "size", "yaw",
                  "velocity", "segmentation", "trajectory"]

    def __init__(self):
        super().__init__()
        # Log-variance per task (learnable)
        self.log_vars = nn.ParameterDict({
            t: nn.Parameter(torch.tensor(0.0)) for t in self.TASK_NAMES
        })

    def _weighted(self, loss: torch.Tensor, task: str) -> torch.Tensor:
        """Applies uncertainty-weighted scaling: L / (2σ²) + log σ"""
        lv = self.log_vars[task]
        return loss / (2 * lv.exp()) + lv / 2

    def forward(
        self,
        det_preds:  Dict[str, torch.Tensor],
        det_targets: Dict[str, torch.Tensor],
        vel_pred:   torch.Tensor,
        vel_target: torch.Tensor,
        center_mask: torch.Tensor,
        seg_pred:   torch.Tensor,
        seg_target: torch.Tensor,
        traj_pred:  Optional[Dict] = None,
        traj_target: Optional[torch.Tensor] = None,
        weather_losses: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:

        losses = {}

        # 1. Heatmap focal loss
        losses["heatmap"] = self._weighted(
            gaussian_focal_loss(det_preds["heatmap"], det_targets["heatmap"]),
            "heatmap",
        )

        # 2. Offset L1
        losses["offset"] = self._weighted(
            reg_l1_loss(det_preds["offset"], det_targets["offset"], center_mask),
            "offset",
        )

        # 3. Height L1
        losses["height"] = self._weighted(
            reg_l1_loss(det_preds["height"], det_targets["height"], center_mask),
            "height",
        )

        # 4. Size L1 (log-space)
        losses["size"] = self._weighted(
            reg_l1_loss(
                det_preds["size"].log().clamp(-5, 5),
                det_targets["size"].log().clamp(-5, 5),
                center_mask,
            ),
            "size",
        )

        # 5. Yaw L1 (both sin and cos components)
        losses["yaw"] = self._weighted(
            reg_l1_loss(det_preds["yaw"], det_targets["yaw"], center_mask),
            "yaw",
        )

        # 6. Velocity L1
        losses["velocity"] = self._weighted(
            reg_l1_loss(vel_pred, vel_target, center_mask),
            "velocity",
        )

        # 7. Segmentation cross-entropy
        losses["segmentation"] = self._weighted(
            F.cross_entropy(seg_pred, seg_target.long()),
            "segmentation",
        )

        # 8. Trajectory NLL (best-of-K)
        if traj_pred is not None and traj_target is not None:
            # Compute L2 distance to ground truth for each mode
            traj_modes = traj_pred["traj"]   # (B, K, T, 2, H, W)
            B, K, T, _, H, W = traj_modes.shape
            gt = traj_target.unsqueeze(1).unsqueeze(4).unsqueeze(5)  # (B,1,T,2,1,1)
            err = ((traj_modes - gt) ** 2).sum(3).mean(2)  # (B,K,H,W)
            min_err = err.min(dim=1).values                # (B,H,W) best mode
            losses["trajectory"] = self._weighted(
                (min_err * center_mask.squeeze(1)).sum() / center_mask.sum().clamp(1),
                "trajectory",
            )
        else:
            losses["trajectory"] = torch.tensor(0.0, device=det_preds["heatmap"].device)

        # Auxiliary weather losses
        if weather_losses:
            losses["weather_ce"] = 0.3 * weather_losses["weather_ce"]
            losses["prior_kl"]   = 0.1 * weather_losses["prior_kl"]

        losses["total"] = sum(losses.values())
        return losses


# ──────────────────────────────────────────────────────
# 5.7  Full detection model
# ──────────────────────────────────────────────────────

class MultiModalDetector(nn.Module):
    """
    Ties together: BEVNeck → all task heads.
    Receives fused BEV from BEVFusion (Day 4).
    """

    def __init__(self, in_ch: int = 256):
        super().__init__()
        self.neck = BEVNeck(in_ch=in_ch, out_ch=128)
        self.det  = CenterPointHead(in_ch=128)
        self.vel  = VelocityHead(in_ch=128)
        self.seg  = SegmentationHead(in_ch=128)
        self.traj = TrajectoryHead(in_ch=128)

    def forward(self, fused_bev: torch.Tensor) -> Dict[str, torch.Tensor]:
        neck_feat = self.neck(fused_bev)
        return {
            **self.det(neck_feat),
            "velocity":   self.vel(neck_feat),
            "seg_logits": self.seg(neck_feat),
            **{f"traj_{k}": v for k, v in self.traj(neck_feat).items()},
        }


# ── Smoke test ────────────────────────────────────────
if __name__ == "__main__":
    B, H, W = 2, 128, 128
    fused_bev = torch.randn(B, 256, H, W)
    model = MultiModalDetector(in_ch=256)
    out   = model(fused_bev)

    print("Detection outputs:")
    for k, v in out.items():
        print(f"  {k}: {v.shape}")

    loss_fn = MultiTaskLoss()
    print(f"Loss params: {sum(p.numel() for p in loss_fn.parameters())}")
