"""
Day 1 — Multi-modal nuScenes data loader.
Loads synchronized camera, LiDAR, RADAR, and weather metadata
for a single scene/sample.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion
import cv2
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────
# 1.1  Weather condition label mapper
# ─────────────────────────────────────────────
WEATHER_MAP = {
    "clear":  0,
    "rain":   1,
    "fog":    2,
    "snow":   3,
    "night":  4,
}

def infer_weather(nusc, sample_token: str) -> int:
    """
    Heuristic: nuScenes has no official weather labels.
    We map scene descriptions to conditions.
    Returns an integer condition index.
    """
    sample = nusc.get("sample", sample_token)
    scene  = nusc.get("scene", sample["scene_token"])
    desc   = scene["description"].lower()
    for key in WEATHER_MAP:
        if key in desc:
            return WEATHER_MAP[key]
    return WEATHER_MAP["clear"]   # default


# ─────────────────────────────────────────────
# 1.2  Camera image loader
# ─────────────────────────────────────────────
CAMERA_NAMES = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK",  "CAM_BACK_LEFT",  "CAM_BACK_RIGHT",
]

def load_cameras(
    nusc: NuScenes,
    sample: dict,
    img_size: Tuple[int, int] = (448, 800),
) -> Dict[str, torch.Tensor]:
    """
    Returns dict cam_name → (3, H, W) float32 tensor in [0,1].
    Also returns intrinsic / extrinsic matrices.
    """
    imgs, intrinsics, extrinsics = {}, {}, {}
    for cam in CAMERA_NAMES:
        sd_token  = sample["data"][cam]
        sd_record = nusc.get("sample_data", sd_token)
        img_path  = os.path.join(nusc.dataroot, sd_record["filename"])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size[1], img_size[0]))
        img = img.astype(np.float32) / 255.0
        imgs[cam] = torch.from_numpy(img).permute(2, 0, 1)  # (3,H,W)

        # Calibration
        cs_record = nusc.get(
            "calibrated_sensor", sd_record["calibrated_sensor_token"]
        )
        K = np.array(cs_record["camera_intrinsic"], dtype=np.float32)
        intrinsics[cam] = torch.from_numpy(K)                # (3,3)

        # Camera-to-ego transform (rotation + translation)
        rot = Quaternion(cs_record["rotation"]).rotation_matrix
        trans = np.array(cs_record["translation"])
        E = np.eye(4, dtype=np.float32)
        E[:3, :3] = rot
        E[:3,  3] = trans
        extrinsics[cam] = torch.from_numpy(E)                # (4,4)

    return {"images": imgs, "intrinsics": intrinsics, "extrinsics": extrinsics}


# ─────────────────────────────────────────────
# 1.3  LiDAR point cloud loader
# ─────────────────────────────────────────────
def load_lidar(
    nusc: NuScenes,
    sample: dict,
    max_points: int = 34000,
) -> torch.Tensor:
    """
    Returns (N, 5) tensor: x, y, z, intensity, ring_index.
    Pads or truncates to max_points.
    """
    sd_token  = sample["data"]["LIDAR_TOP"]
    sd_record = nusc.get("sample_data", sd_token)
    pc_path   = os.path.join(nusc.dataroot, sd_record["filename"])

    scan = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 5)
    scan = scan[:max_points]

    # Pad if fewer points than max
    if len(scan) < max_points:
        pad = np.zeros((max_points - len(scan), 5), dtype=np.float32)
        scan = np.concatenate([scan, pad], axis=0)

    return torch.from_numpy(scan)   # (N, 5)


# ─────────────────────────────────────────────
# 1.4  RADAR point cloud loader (all 5 radars)
# ─────────────────────────────────────────────
RADAR_NAMES = [
    "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT",
]

def load_radar(
    nusc: NuScenes,
    sample: dict,
    max_points: int = 1024,
) -> torch.Tensor:
    """
    Returns (N, 7) tensor: x, y, z, vx, vy, rcs, distance.
    All five radars merged into ego frame.
    """
    all_pts = []
    for radar in RADAR_NAMES:
        sd_token  = sample["data"][radar]
        sd_record = nusc.get("sample_data", sd_token)
        pc_path   = os.path.join(nusc.dataroot, sd_record["filename"])

        # nuScenes RADAR .pcd binary: 18 fields, we use xyz + vx_comp + vy_comp + rcs
        pts = np.fromfile(pc_path, dtype=np.float32)
        if pts.size == 0:
            continue
        pts = pts.reshape(-1, 18)
        xyz = pts[:, :3]
        vx  = pts[:, 6:7]    # compensated velocity x
        vy  = pts[:, 7:8]    # compensated velocity y
        rcs = pts[:, 15:16]  # radar cross section
        dist = np.linalg.norm(xyz[:, :2], axis=1, keepdims=True)
        merged = np.concatenate([xyz, vx, vy, rcs, dist], axis=1)
        all_pts.append(merged)

    if not all_pts:
        return torch.zeros(max_points, 7)

    all_pts = np.concatenate(all_pts, axis=0).astype(np.float32)
    all_pts = all_pts[:max_points]
    if len(all_pts) < max_points:
        pad = np.zeros((max_points - len(all_pts), 7), dtype=np.float32)
        all_pts = np.concatenate([all_pts, pad], axis=0)

    return torch.from_numpy(all_pts)   # (N, 7)


# ─────────────────────────────────────────────
# 1.5  Ground truth 3-D box loader
# ─────────────────────────────────────────────
DETECTION_CLASSES = [
    "car", "truck", "bus", "bicycle", "motorcycle",
    "pedestrian", "traffic_cone", "barrier", "construction_vehicle",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(DETECTION_CLASSES)}

def load_annotations(
    nusc: NuScenes,
    sample: dict,
) -> Dict[str, torch.Tensor]:
    """
    Returns dict with:
      boxes   (M, 7)  x,y,z,w,l,h,yaw  in ego frame
      classes (M,)    int64 class indices
      velocities (M, 2)  vx, vy
    """
    boxes, classes, vels = [], [], []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        cat = ann["category_name"].split(".")[0]
        if cat not in CLASS_TO_IDX:
            continue

        # Convert to ego frame
        center = np.array(ann["translation"], dtype=np.float32)
        size   = np.array(ann["size"], dtype=np.float32)       # w, l, h
        q      = Quaternion(ann["rotation"])
        yaw    = float(q.yaw_pitch_roll[0])

        box = np.array([*center, *size, yaw], dtype=np.float32)
        boxes.append(box)
        classes.append(CLASS_TO_IDX[cat])

        # Velocity from nuScenes helper
        vel = nusc.box_velocity(ann_token)[:2]
        vels.append(vel.astype(np.float32))

    if not boxes:
        return {
            "boxes": torch.zeros(0, 7),
            "classes": torch.zeros(0, dtype=torch.long),
            "velocities": torch.zeros(0, 2),
        }

    return {
        "boxes":      torch.from_numpy(np.stack(boxes)),
        "classes":    torch.tensor(classes, dtype=torch.long),
        "velocities": torch.from_numpy(np.stack(vels)),
    }


# ─────────────────────────────────────────────
# 1.6  Dataset class
# ─────────────────────────────────────────────
class NuScenesMultiModalDataset(Dataset):
    """
    Loads one nuScenes sample → dict of tensors for all modalities.
    """

    def __init__(
        self,
        dataroot: str,
        version: str = "v1.0-trainval",
        split: str = "train",
        img_size: Tuple[int, int] = (448, 800),
        max_lidar_pts: int = 34000,
        max_radar_pts: int = 1024,
        transforms=None,
    ):
        self.nusc   = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.img_sz = img_size
        self.max_lidar = max_lidar_pts
        self.max_radar = max_radar_pts
        self.transforms = transforms

        splits  = create_splits_scenes()
        scene_names = set(splits[split])
        self.samples = [
            s for s in self.nusc.sample
            if self.nusc.get("scene", s["scene_token"])["name"] in scene_names
        ]
        print(f"[Dataset] {split}: {len(self.samples)} samples loaded.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        cam_data  = load_cameras(self.nusc, sample, self.img_sz)
        lidar_pts = load_lidar(self.nusc, sample, self.max_lidar)
        radar_pts = load_radar(self.nusc, sample, self.max_radar)
        anns      = load_annotations(self.nusc, sample)
        weather   = infer_weather(self.nusc, sample["token"])

        item = {
            **cam_data,
            "lidar":        lidar_pts,      # (N, 5)
            "radar":        radar_pts,      # (M, 7)
            "gt_boxes":     anns["boxes"],
            "gt_classes":   anns["classes"],
            "gt_velocities": anns["velocities"],
            "weather":      torch.tensor(weather, dtype=torch.long),
            "sample_token": sample["token"],
        }

        if self.transforms:
            item = self.transforms(item)

        return item


# ─────────────────────────────────────────────
# 1.7  Collate function (handles variable-size anns)
# ─────────────────────────────────────────────
def collate_fn(batch: List[dict]) -> dict:
    """Custom collate that pads variable-length annotation tensors."""
    out = {}
    keys = batch[0].keys()
    for k in keys:
        if k in ("gt_boxes", "gt_classes", "gt_velocities"):
            out[k] = [b[k] for b in batch]   # list of tensors
        elif k in ("images", "intrinsics", "extrinsics"):
            # dict of cam_name → tensor; stack per-cam
            out[k] = {
                cam: torch.stack([b[k][cam] for b in batch])
                for cam in batch[0][k]
            }
        elif k == "sample_token":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


# ─────────────────────────────────────────────
# 1.8  Factory function
# ─────────────────────────────────────────────
def build_dataloader(
    dataroot: str,
    split: str = "train",
    batch_size: int = 4,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    ds = NuScenesMultiModalDataset(dataroot=dataroot, split=split, **kwargs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == "train"),
    )


# ─────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", default="./data/nuscenes")
    args = p.parse_args()

    loader = build_dataloader(args.dataroot, split="train", batch_size=2)
    batch  = next(iter(loader))

    print("Camera tensors:")
    for cam, t in batch["images"].items():
        print(f"  {cam}: {t.shape}")           # (B,3,H,W)
    print(f"LiDAR:   {batch['lidar'].shape}")  # (B,N,5)
    print(f"RADAR:   {batch['radar'].shape}")  # (B,M,7)
    print(f"Weather: {batch['weather']}")
    print(f"GT boxes (sample 0): {batch['gt_boxes'][0].shape}")
