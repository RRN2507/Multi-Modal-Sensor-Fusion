import torch
from data.mock_dataset import build_mock_dataloader
from models.encoders.sensor_encoders import LiDAREncoder, RADAREncoder
from models.fusion.weather_gating import WeatherAdaptiveGating
from models.heads.detection_heads import MultiModalDetector

loader = build_mock_dataloader(batch_size=2)
batch  = next(iter(loader))

lidar_enc = LiDAREncoder()
radar_enc = RADAREncoder()
gate      = WeatherAdaptiveGating()
detector  = MultiModalDetector(in_ch=256)

lidar_bev = lidar_enc(batch['lidar'])
radar_bev = radar_enc(batch['radar'])
bev_maps  = {'lidar': lidar_bev, 'radar': radar_bev}
gated, _  = gate(bev_maps)

print('Pipeline OK')
print('LiDAR BEV:', lidar_bev.shape)
print('RADAR BEV:', radar_bev.shape)
print('Gated mods:', list(gated.keys()))
