"""
Day 2 — Per-sensor encoders.
  CameraEncoder   : Swin-T → Lift-Splat-Shoot BEV projection
  LiDAREncoder    : PointPillars → pillar BEV feature map
  RADAREncoder    : MLP on sparse points → dense BEV heatmap
  ThermalEncoder  : Lightweight CNN → heat feature map
All encoders output a (B, C, H_bev, W_bev) BEV tensor
with consistent spatial resolution (default 128×128).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange
from typing import Dict, Tuple


BEV_H = 128   # BEV grid height (Y axis, forward)
BEV_W = 128   # BEV grid width  (X axis, lateral)
BEV_C = 128   # feature channels out of each encoder


# ═══════════════════════════════════════════════════════
# 2.1  CAMERA ENCODER  (Swin-T + Lift-Splat-Shoot)
# ═══════════════════════════════════════════════════════

class DepthHead(nn.Module):
    """Predicts depth distribution D over D_bins for each pixel."""

    def __init__(self, in_ch: int = 192, d_bins: int = 64):
        super().__init__()
        self.d_bins = d_bins
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, d_bins, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax depth distribution (B, D_bins, H, W)."""
        return self.net(x).softmax(dim=1)


class LiftSplatShoot(nn.Module):
    """
    Projects 2-D camera features into the ego BEV plane.
    Reference: Lift, Splat, Shoot (Philion & Fidler 2020).
    """

    def __init__(
        self,
        d_bins: int = 64,
        d_min: float = 1.0,
        d_max: float = 60.0,
        feat_ch: int = 192,
        bev_h: int = BEV_H,
        bev_w: int = BEV_W,
        bev_ch: int = BEV_C,
        x_bound: Tuple[float, float] = (-51.2, 51.2),
        y_bound: Tuple[float, float] = (-51.2, 51.2),
    ):
        super().__init__()
        self.d_bins = d_bins
        self.depths = torch.linspace(d_min, d_max, d_bins)  # (D,)
        self.bev_h, self.bev_w = bev_h, bev_w
        self.x_bound = x_bound
        self.y_bound = y_bound

        self.depth_head = DepthHead(feat_ch, d_bins)
        self.feat_proj   = nn.Conv2d(feat_ch, bev_ch, 1)
        self.bev_pool    = nn.Sequential(
            nn.Conv2d(bev_ch, bev_ch, 3, padding=1),
            nn.BatchNorm2d(bev_ch),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        feats: torch.Tensor,           # (B*N_cams, C, Hf, Wf)
        intrinsics: torch.Tensor,      # (B, N_cams, 3, 3)
        extrinsics: torch.Tensor,      # (B, N_cams, 4, 4)
    ) -> torch.Tensor:                 # (B, bev_ch, BEV_H, BEV_W)
        B_N, C, Hf, Wf = feats.shape
        N_cams = intrinsics.shape[1]
        B = B_N // N_cams

        depth_dist = self.depth_head(feats)              # (B*N, D, Hf, Wf)
        img_feats  = self.feat_proj(feats)               # (B*N, bev_ch, Hf, Wf)

        # Outer product: each pixel gets D weighted copies along depth rays
        # Shape: (B*N, bev_ch, D, Hf, Wf)
        voxel_feats = img_feats.unsqueeze(2) * depth_dist.unsqueeze(1)

        # Build 3-D frustum points in camera coords → ego coords → BEV grid
        device = feats.device
        depths = self.depths.to(device)

        xs = torch.linspace(0, Wf - 1, Wf, device=device)
        ys = torch.linspace(0, Hf - 1, Hf, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (Hf,Wf)

        # Lift pixel grid to 3-D frustum (D, Hf, Wf, 3)
        pts = torch.stack([
            grid_x.unsqueeze(0) * depths[:, None, None],
            grid_y.unsqueeze(0) * depths[:, None, None],
            depths[:, None, None].expand(-1, Hf, Wf),
        ], dim=-1)  # (D, Hf, Wf, 3)
        pts = pts.view(1, 1, -1, 3)   # (1, 1, D*Hf*Wf, 3)

        # Unproject through intrinsics
        K_inv = torch.inverse(intrinsics.view(B * N_cams, 3, 3))  # (B*N,3,3)
        pts_h = torch.cat([pts.expand(B * N_cams, -1, -1, -1).squeeze(1),
                           torch.ones(B * N_cams, pts.shape[2], 1, device=device)], dim=-1)
        # (B*N, D*Hf*Wf, 4) — simplified; full impl would use K_inv
        # For brevity we scatter into BEV directly using the ego-xy coordinates
        ego_x = pts[..., 0].squeeze(0).squeeze(0).view(-1)  # (D*Hf*Wf,)
        ego_y = pts[..., 1].squeeze(0).squeeze(0).view(-1)

        # Normalize to BEV grid [0,1]
        bev_u = (ego_x - self.x_bound[0]) / (self.x_bound[1] - self.x_bound[0])
        bev_v = (ego_y - self.y_bound[0]) / (self.y_bound[1] - self.y_bound[0])

        # Flatten voxel features → (B*N, bev_ch, D*Hf*Wf)
        flat_feats = voxel_feats.view(B * N_cams, -1, D * Hf * Wf
                                      if (D := self.d_bins) else 1)

        # Splat: accumulate into BEV via bilinear splatting (simplified)
        bev = torch.zeros(B, N_cams, -1, self.bev_h, self.bev_w,
                          device=device).squeeze(2)
        # In a full implementation: use voxel_pooling CUDA kernel or grid_sample
        # Here we use average pooling as a structural placeholder
        voxel_mean = voxel_feats.mean(dim=[2, 3, 4])   # (B*N, bev_ch)
        bev_flat = voxel_mean.view(B, N_cams, -1).mean(dim=1)  # (B, bev_ch)
        bev_out  = bev_flat.unsqueeze(-1).unsqueeze(-1).expand(
            B, -1, self.bev_h, self.bev_w)
        return self.bev_pool(bev_out)   # (B, bev_ch, H_bev, W_bev)


class CameraEncoder(nn.Module):
    """
    Swin-T backbone shared across all N_cams cameras,
    followed by Lift-Splat-Shoot BEV projection.
    """

    def __init__(self, n_cams: int = 6, bev_ch: int = BEV_C):
        super().__init__()
        self.n_cams = n_cams
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            features_only=True,
            out_indices=[2],   # stride-16 feature map: C=192
        )
        self.lss = LiftSplatShoot(feat_ch=192, bev_ch=bev_ch)

    def forward(
        self,
        images: Dict[str, torch.Tensor],    # cam → (B,3,H,W)
        intrinsics: Dict[str, torch.Tensor],
        extrinsics: Dict[str, torch.Tensor],
    ) -> torch.Tensor:                       # (B, bev_ch, H_bev, W_bev)
        cam_names = list(images.keys())
        B = next(iter(images.values())).shape[0]

        imgs = torch.stack([images[c] for c in cam_names], dim=1)  # (B,N,3,H,W)
        K    = torch.stack([intrinsics[c] for c in cam_names], dim=1)  # (B,N,3,3)
        E    = torch.stack([extrinsics[c] for c in cam_names], dim=1)  # (B,N,4,4)

        imgs_flat = imgs.view(B * self.n_cams, *imgs.shape[2:])
        feats = self.backbone(imgs_flat)[-1]   # (B*N, 192, Hf, Wf)
        return self.lss(feats, K, E)           # (B, bev_ch, H_bev, W_bev)


# ═══════════════════════════════════════════════════════
# 2.2  LiDAR ENCODER  (PointPillars)
# ═══════════════════════════════════════════════════════

class PillarFeatureNet(nn.Module):
    """
    Encodes a single pillar's point set into a fixed-size feature vector.
    """
    def __init__(self, in_ch: int = 9, out_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Linear(64, out_ch), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (P*Np, in_ch) → (P*Np, out_ch)
        return self.net(x)


class PointPillarsEncoder(nn.Module):
    """
    PointPillars: voxelizes point cloud into pillars → sparse 2-D pseudo-image.
    """

    def __init__(
        self,
        voxel_size: Tuple[float, float] = (0.8, 0.8),  # meters per pillar
        pc_range: Tuple = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        max_pts_per_pillar: int = 32,
        pillar_ch: int = 64,
        bev_ch: int = BEV_C,
    ):
        super().__init__()
        self.vx, self.vy = voxel_size
        self.pc_range    = pc_range
        self.max_pts     = max_pts_per_pillar
        self.nx = int((pc_range[3] - pc_range[0]) / self.vx)  # 128
        self.ny = int((pc_range[4] - pc_range[1]) / self.vy)  # 128

        self.pfn = PillarFeatureNet(in_ch=9, out_ch=pillar_ch)
        self.scatter_conv = nn.Sequential(
            nn.Conv2d(pillar_ch, bev_ch, 3, padding=1),
            nn.BatchNorm2d(bev_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_ch, bev_ch, 3, padding=1),
            nn.BatchNorm2d(bev_ch),
            nn.ReLU(inplace=True),
        )

    def voxelize(self, points: torch.Tensor):
        """
        Converts (B, N, 5) point cloud into pillar indices and features.
        Returns pillar_feats (total_pillars, Np, 9), pillar_indices (total_pillars, 2).
        Simplified CPU-side version; production uses CUDA mmdet3d voxelize.
        """
        B, N, _ = points.shape
        x, y, z, intensity = [points[..., i] for i in range(4)]

        # Quantize to pillar coords
        px = ((x - self.pc_range[0]) / self.vx).long().clamp(0, self.nx - 1)
        py = ((y - self.pc_range[1]) / self.vy).long().clamp(0, self.ny - 1)

        # Group by pillar — simplified: return raw coords for scatter
        return px, py, points  # placeholder for full voxelization

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        points: (B, N, 5) — x,y,z,intensity,ring
        Returns: (B, bev_ch, H_bev, W_bev)
        """
        B = points.shape[0]
        px, py, raw = self.voxelize(points)

        # Build a simple BEV density map as a structural stand-in
        # (full impl uses scatter_max into pillar canvas)
        bev = torch.zeros(B, 64, self.ny, self.nx, device=points.device)
        for b in range(B):
            ix = px[b].clamp(0, self.nx - 1)
            iy = py[b].clamp(0, self.ny - 1)
            vals = raw[b, :, 2]   # use Z as proxy feature
            bev[b, 0].scatter_add_(
                0,
                iy * self.nx + ix,
                vals,
            ) if False else None  # placeholder — real impl uses voxel_pooling

            # Use point count per pillar as feature
            flat_idx = (iy * self.nx + ix).clamp(0, self.ny * self.nx - 1)
            count = torch.bincount(flat_idx, minlength=self.ny * self.nx)
            bev[b, 0] = count.view(self.ny, self.nx).float() / 32.0

        return self.scatter_conv(bev)   # (B, bev_ch, H_bev, W_bev)


class LiDAREncoder(nn.Module):
    def __init__(self, bev_ch: int = BEV_C):
        super().__init__()
        self.pillars = PointPillarsEncoder(bev_ch=bev_ch)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.pillars(points)


# ═══════════════════════════════════════════════════════
# 2.3  RADAR ENCODER
# ═══════════════════════════════════════════════════════

class RADAREncoder(nn.Module):
    """
    Encodes sparse RADAR points (x,y,z,vx,vy,rcs,dist) into a BEV feature map.
    Uses a PointNet-style per-point MLP → scatter to BEV grid.
    """

    def __init__(
        self,
        in_ch: int = 7,
        bev_ch: int = BEV_C,
        pc_range: Tuple = (-51.2, -51.2, 51.2, 51.2),
        bev_h: int = BEV_H,
        bev_w: int = BEV_W,
    ):
        super().__init__()
        self.bev_h, self.bev_w = bev_h, bev_w
        self.pc_range = pc_range

        self.point_mlp = nn.Sequential(
            nn.Linear(in_ch, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 64),   nn.ReLU(inplace=True),
            nn.Linear(64, bev_ch),
        )
        self.bev_conv = nn.Sequential(
            nn.Conv2d(bev_ch, bev_ch, 3, padding=1),
            nn.BatchNorm2d(bev_ch), nn.ReLU(inplace=True),
        )

    def forward(self, radar_pts: torch.Tensor) -> torch.Tensor:
        """
        radar_pts: (B, N, 7)
        Returns:   (B, bev_ch, H_bev, W_bev)
        """
        B, N, _ = radar_pts.shape
        device   = radar_pts.device

        # Per-point features
        pt_feats = self.point_mlp(radar_pts.view(B * N, -1))  # (B*N, bev_ch)
        pt_feats = pt_feats.view(B, N, -1)                    # (B, N, bev_ch)

        # Scatter to BEV
        x = radar_pts[..., 0]   # (B, N)
        y = radar_pts[..., 1]
        bev_u = ((x - self.pc_range[0]) / (self.pc_range[2] - self.pc_range[0])
                 * self.bev_w).long().clamp(0, self.bev_w - 1)
        bev_v = ((y - self.pc_range[1]) / (self.pc_range[3] - self.pc_range[1])
                 * self.bev_h).long().clamp(0, self.bev_h - 1)

        bev = torch.zeros(B, pt_feats.shape[-1], self.bev_h, self.bev_w, device=device)
        flat_idx = bev_v * self.bev_w + bev_u   # (B, N)
        for b in range(B):
            fi = flat_idx[b]   # (N,)
            bev[b].view(-1, self.bev_h * self.bev_w).scatter_add_(
                1,
                fi.unsqueeze(0).expand(pt_feats.shape[-1], -1),
                pt_feats[b].T,
            )
        return self.bev_conv(bev)   # (B, bev_ch, H_bev, W_bev)


# ═══════════════════════════════════════════════════════
# 2.4  THERMAL IR ENCODER
# ═══════════════════════════════════════════════════════

class ThermalEncoder(nn.Module):
    """
    Lightweight CNN for LWIR thermal images → BEV projection.
    1-channel input (grayscale thermal intensity).
    """

    def __init__(self, bev_ch: int = BEV_C):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, bev_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(bev_ch), nn.ReLU(inplace=True),
        )
        # Adaptive pool to (BEV_H, BEV_W)
        self.bev_pool = nn.AdaptiveAvgPool2d((BEV_H, BEV_W))

    def forward(self, thermal_img: torch.Tensor) -> torch.Tensor:
        """
        thermal_img: (B, 1, H, W) — LWIR single channel
        Returns:     (B, bev_ch, BEV_H, BEV_W)
        """
        x = self.backbone(thermal_img)
        return self.bev_pool(x)


# ═══════════════════════════════════════════════════════
# 2.5  Encoder registry
# ═══════════════════════════════════════════════════════

class SensorEncoderBundle(nn.Module):
    """
    Wraps all encoders and returns a dict of BEV feature maps.
    """

    def __init__(self, bev_ch: int = BEV_C, n_cams: int = 6):
        super().__init__()
        self.camera  = CameraEncoder(n_cams=n_cams, bev_ch=bev_ch)
        self.lidar   = LiDAREncoder(bev_ch=bev_ch)
        self.radar   = RADAREncoder(bev_ch=bev_ch)
        # Thermal is optional (FLIR dataset only); disabled by default
        self.thermal = ThermalEncoder(bev_ch=bev_ch)
        self.has_thermal = False

    def forward(self, batch: dict) -> Dict[str, torch.Tensor]:
        bev = {}
        bev["camera"] = self.camera(
            batch["images"], batch["intrinsics"], batch["extrinsics"]
        )
        bev["lidar"]  = self.lidar(batch["lidar"])
        bev["radar"]  = self.radar(batch["radar"])
        if self.has_thermal and "thermal" in batch:
            bev["thermal"] = self.thermal(batch["thermal"])
        return bev   # dict[str → (B, C, H_bev, W_bev)]


# ── Smoke test ─────────────────────────────────────────
if __name__ == "__main__":
    B = 2
    mock_batch = {
        "images":     {"CAM_FRONT": torch.randn(B, 3, 448, 800)},
        "intrinsics": {"CAM_FRONT": torch.eye(3).unsqueeze(0).repeat(B, 1, 1)},
        "extrinsics": {"CAM_FRONT": torch.eye(4).unsqueeze(0).repeat(B, 1, 1)},
        "lidar":      torch.randn(B, 34000, 5),
        "radar":      torch.randn(B, 1024, 7),
    }

    lidar_enc = LiDAREncoder()
    radar_enc = RADAREncoder()
    lidar_bev = lidar_enc(mock_batch["lidar"])
    radar_bev = radar_enc(mock_batch["radar"])

    print(f"LiDAR BEV: {lidar_bev.shape}")   # (B,128,128,128)
    print(f"RADAR BEV: {radar_bev.shape}")   # (B,128,128,128)
