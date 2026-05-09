"""
Day 3 — Weather-Adaptive Attention Gating.

Given BEV feature maps from N modalities and a weather condition signal,
this module learns to re-weight each modality's contribution to the
fused representation.

Key idea:
  - A small WeatherClassifier infers condition (clear/rain/fog/snow/night)
    from a fused global feature or from raw image stats.
  - An AttentionGate produces a per-modality soft weight in (0,1).
  - Weights are applied element-wise to each modality's BEV map before fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ──────────────────────────────────────────────────────
# 3.1  Weather classifier
#      Takes a stacked global feature → predicts weather
# ──────────────────────────────────────────────────────

N_WEATHER_CLASSES = 5   # clear, rain, fog, snow, night


class WeatherClassifier(nn.Module):
    """
    Classifies driving weather from a concatenated global descriptor
    computed from all modality BEV maps (global average pooled).
    Also used as auxiliary loss during training.
    """

    def __init__(
        self,
        in_ch_per_mod: int = 128,
        n_modalities: int = 3,         # camera, lidar, radar
        hidden: int = 256,
        n_classes: int = N_WEATHER_CLASSES,
    ):
        super().__init__()
        self.n_mod = n_modalities
        self.gap   = nn.AdaptiveAvgPool2d(1)
        in_dim = in_ch_per_mod * n_modalities

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )

    def forward(
        self,
        bev_maps: Dict[str, torch.Tensor],   # mod → (B, C, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          logits  (B, n_classes)
          probs   (B, n_classes)  — softmax confidence per weather class
        """
        descriptors = []
        for bev in bev_maps.values():
            g = self.gap(bev).squeeze(-1).squeeze(-1)   # (B, C)
            descriptors.append(g)

        # Pad if fewer modalities than expected (e.g. thermal absent)
        while len(descriptors) < self.n_mod:
            descriptors.append(torch.zeros_like(descriptors[0]))

        feat   = torch.cat(descriptors[:self.n_mod], dim=1)  # (B, C*n_mod)
        logits = self.classifier(feat)                        # (B, n_classes)
        return logits, logits.softmax(dim=-1)


# ──────────────────────────────────────────────────────
# 3.2  Modality attention gate
#      Produces per-modality spatial + channel weights
# ──────────────────────────────────────────────────────

class ModalityAttentionGate(nn.Module):
    """
    For each modality, predicts a scalar weight α ∈ (0,1) conditioned on:
      1. The modality's own BEV feature (self-assessment of quality)
      2. The predicted weather probabilities

    Additionally predicts a spatial confidence mask (B,1,H,W) to
    suppress regions where the modality is unreliable (e.g. LiDAR points
    in heavy rain that contain ghost reflections).
    """

    def __init__(
        self,
        bev_ch: int = 128,
        n_weather: int = N_WEATHER_CLASSES,
        n_modalities: int = 3,
    ):
        super().__init__()
        self.n_mod = n_modalities

        # Scalar weight network (global)
        self.scalar_net = nn.Sequential(
            nn.Linear(bev_ch + n_weather, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Spatial mask network (per-pixel reliability)
        self.spatial_net = nn.Sequential(
            nn.Conv2d(bev_ch + n_weather, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        bev: torch.Tensor,          # (B, C, H, W) — single modality
        weather_probs: torch.Tensor,   # (B, n_weather)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          scalar_w  (B, 1, 1, 1)   — global modality weight
          spatial_m (B, 1, H, W)   — per-pixel mask
        """
        B, C, H, W = bev.shape

        # Global feature for scalar gate
        g = bev.mean(dim=[2, 3])   # (B, C)
        g_in = torch.cat([g, weather_probs], dim=1)   # (B, C+n_weather)
        scalar_w = self.scalar_net(g_in).view(B, 1, 1, 1)

        # Spatial mask: broadcast weather to every pixel
        w_expand = weather_probs[:, :, None, None].expand(-1, -1, H, W)  # (B,n,H,W)
        s_in = torch.cat([bev, w_expand], dim=1)   # (B, C+n, H, W)
        spatial_m = self.spatial_net(s_in)           # (B, 1, H, W)

        return scalar_w, spatial_m


# ──────────────────────────────────────────────────────
# 3.3  Full gating module
# ──────────────────────────────────────────────────────

# Known-best modality per weather condition (prior for loss shaping)
# Shape: (n_weather, n_modalities)  — columns: camera, lidar, radar
WEATHER_PRIOR = torch.tensor([
    [0.50, 0.35, 0.15],   # clear  — camera dominates
    [0.25, 0.35, 0.40],   # rain   — radar most reliable
    [0.15, 0.20, 0.65],   # fog    — radar heavily favoured
    [0.20, 0.35, 0.45],   # snow   — radar + lidar
    [0.30, 0.50, 0.20],   # night  — lidar + camera (thermal)
], dtype=torch.float32)


class WeatherAdaptiveGating(nn.Module):
    """
    Complete gating pipeline:
      1. Classify weather from multi-modal BEV global features.
      2. Gate each modality with a learned scalar + spatial mask.
      3. Apply gates and return weighted BEV maps ready for fusion.
    """

    def __init__(
        self,
        bev_ch: int = 128,
        modality_names: Tuple[str, ...] = ("camera", "lidar", "radar"),
        n_weather: int = N_WEATHER_CLASSES,
    ):
        super().__init__()
        self.modality_names = modality_names
        n_mod = len(modality_names)
        self.mod_idx = {m: i for i, m in enumerate(modality_names)}

        self.weather_clf = WeatherClassifier(
            in_ch_per_mod=bev_ch,
            n_modalities=n_mod,
            n_classes=n_weather,
        )
        self.gates = nn.ModuleDict({
            m: ModalityAttentionGate(bev_ch=bev_ch, n_weather=n_weather, n_modalities=n_mod)
            for m in modality_names
        })

        self.register_buffer("weather_prior", WEATHER_PRIOR)

    def forward(
        self,
        bev_maps: Dict[str, torch.Tensor],   # mod → (B, C, H, W)
        gt_weather: Optional[torch.Tensor] = None,  # (B,) int labels for training
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Returns:
          gated_bevs  dict: mod → (B, C, H, W)  — after gating
          aux         dict: weather_logits, scalar_weights, spatial_masks
        """
        # Step 1: classify weather
        logits, weather_probs = self.weather_clf(bev_maps)

        # Step 2: compute per-modality gates
        scalar_weights, spatial_masks, gated_bevs = {}, {}, {}
        for mod in self.modality_names:
            if mod not in bev_maps:
                continue
            sw, sm = self.gates[mod](bev_maps[mod], weather_probs)
            gated = bev_maps[mod] * sw * sm        # element-wise gating
            scalar_weights[mod] = sw
            spatial_masks[mod]  = sm
            gated_bevs[mod]     = gated

        aux = {
            "weather_logits":  logits,
            "weather_probs":   weather_probs,
            "scalar_weights":  scalar_weights,
            "spatial_masks":   spatial_masks,
        }
        return gated_bevs, aux

    def gating_loss(
        self,
        aux: dict,
        gt_weather: torch.Tensor,    # (B,) int class indices
    ) -> Dict[str, torch.Tensor]:
        """
        Two auxiliary losses:
          weather_ce  — cross-entropy for weather classification
          prior_kl    — KL divergence between learned scalar weights
                        and the WEATHER_PRIOR (encourages correct dominance)
        """
        B = gt_weather.shape[0]

        # Weather classification loss
        weather_ce = F.cross_entropy(aux["weather_logits"], gt_weather)

        # Prior KL loss (soft target)
        prior_target = self.weather_prior[gt_weather]   # (B, n_mod)
        scalar_stack = torch.cat([
            aux["scalar_weights"][m].view(B, 1)
            for m in self.modality_names
            if m in aux["scalar_weights"]
        ], dim=1)   # (B, n_mod)
        scalar_probs = scalar_stack / (scalar_stack.sum(dim=1, keepdim=True) + 1e-6)
        prior_kl = F.kl_div(
            scalar_probs.log().clamp(-10, 0),
            prior_target.to(scalar_probs.device),
            reduction="batchmean",
        )

        return {"weather_ce": weather_ce, "prior_kl": prior_kl}


# ── Smoke test ────────────────────────────────────────
if __name__ == "__main__":
    B, C, H, W = 2, 128, 128, 128

    bev_maps = {
        "camera": torch.randn(B, C, H, W),
        "lidar":  torch.randn(B, C, H, W),
        "radar":  torch.randn(B, C, H, W),
    }
    gt_weather = torch.randint(0, 5, (B,))

    gate = WeatherAdaptiveGating(bev_ch=C)
    gated, aux = gate(bev_maps, gt_weather)

    print("Gated BEV shapes:")
    for k, v in gated.items():
        print(f"  {k}: {v.shape}")

    losses = gate.gating_loss(aux, gt_weather)
    print(f"Weather CE: {losses['weather_ce'].item():.4f}")
    print(f"Prior KL:   {losses['prior_kl'].item():.4f}")
    print("Scalar weights:", {k: v.squeeze().item() for k, v in aux["scalar_weights"].items()})
