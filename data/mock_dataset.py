"""
Mock dataset — generates fake camera, LiDAR, RADAR tensors.
No real data needed. Use this to test the full pipeline.
"""

import torch
from torch.utils.data import Dataset, DataLoader

class MockMultiModalDataset(Dataset):
    def __init__(self, n_samples=100):
        self.n = n_samples

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "images": {
                "CAM_FRONT":       torch.randn(3, 448, 800),
                "CAM_FRONT_LEFT":  torch.randn(3, 448, 800),
                "CAM_FRONT_RIGHT": torch.randn(3, 448, 800),
                "CAM_BACK":        torch.randn(3, 448, 800),
                "CAM_BACK_LEFT":   torch.randn(3, 448, 800),
                "CAM_BACK_RIGHT":  torch.randn(3, 448, 800),
            },
            "intrinsics": {c: torch.eye(3) for c in [
                "CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT",
                "CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"
            ]},
            "extrinsics": {c: torch.eye(4) for c in [
                "CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT",
                "CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"
            ]},
            "lidar":          torch.randn(34000, 5),
            "radar":          torch.randn(1024, 7),
            "gt_boxes":       torch.randn(5, 7),
            "gt_classes":     torch.randint(0, 9, (5,)),
            "gt_velocities":  torch.randn(5, 2),
            "weather":        torch.randint(0, 5, (1,)).squeeze(),
        }

def build_mock_dataloader(batch_size=2, n_samples=100):
    ds = MockMultiModalDataset(n_samples)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

if __name__ == "__main__":
    loader = build_mock_dataloader(batch_size=2)
    batch  = next(iter(loader))
    print("Mock dataset OK")
    print(f"  LiDAR:   {batch['lidar'].shape}")
    print(f"  RADAR:   {batch['radar'].shape}")
    print(f"  Weather: {batch['weather']}")
    print(f"  GT boxes:{batch['gt_boxes'][0].shape}")