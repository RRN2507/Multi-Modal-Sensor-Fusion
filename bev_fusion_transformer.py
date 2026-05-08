"""
Day 4 — BEV Fusion Transformer.

Takes gated BEV feature maps from N modalities and fuses them into a
single, unified BEV representation via a cross-attention transformer.

Architecture:
  1. Modality-specific linear projection → common embedding dim
  2. Positional encoding over BEV grid
  3. Multi-head self-attention across modalities (tokens = BEV cells × modalities)
  4. Feed-forward refinement
  5. Weighted sum + convolutional head → final fused BEV
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────
# 4.1  Learned 2-D positional encoding for BEV grid
# ──────────────────────────────────────────────────────

class BEVPositionalEncoding(nn.Module):
    """
    Sinusoidal + learned positional encoding over a 2-D BEV grid.
    Encodes row (Y / forward) and column (X / lateral) independently.
    """

    def __init__(self, embed_dim: int, bev_h: int = 128, bev_w: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        half = embed_dim // 2

        # Sinusoidal encoding
        pe = torch.zeros(1, embed_dim, bev_h, bev_w)

        y_pos = torch.arange(bev_h, dtype=torch.float32).unsqueeze(1)
        x_pos = torch.arange(bev_w, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, half, 2, dtype=torch.float32)
            * -(math.log(10000.0) / half)
        )

        # Y axis → first half of channels
        pe[0, 0:half:2, :, :] = (torch.sin(y_pos * div_term)
                                  .unsqueeze(1).expand(-1, bev_w, -1)
                                  .permute(2, 1, 0).unsqueeze(0))
        pe[0, 1:half:2, :, :] = (torch.cos(y_pos * div_term)
                                  .unsqueeze(1).expand(-1, bev_w, -1)
                                  .permute(2, 1, 0).unsqueeze(0))

        # X axis → second half
        pe[0, half::2,   :, :] = (torch.sin(x_pos * div_term)
                                   .unsqueeze(0).expand(bev_h, -1, -1)
                                   .permute(2, 0, 1).unsqueeze(0))
        pe[0, half+1::2, :, :] = (torch.cos(x_pos * div_term)
                                   .unsqueeze(0).expand(bev_h, -1, -1)
                                   .permute(2, 0, 1).unsqueeze(0))
        self.register_buffer("pe", pe)   # (1, D, H, W)

        # Tiny learned residual
        self.learned_pe = nn.Parameter(torch.zeros(1, embed_dim, bev_h, bev_w) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D, H, W) → (B, D, H, W) + positional bias"""
        return x + self.pe + self.learned_pe


# ──────────────────────────────────────────────────────
# 4.2  Modality token embedder
# ──────────────────────────────────────────────────────

class ModalityEmbedder(nn.Module):
    """
    Projects each modality's BEV from bev_ch → embed_dim
    and appends a learnable modality-type embedding.
    """

    def __init__(
        self,
        bev_ch: int = 128,
        embed_dim: int = 256,
        modality_names: Tuple[str, ...] = ("camera", "lidar", "radar"),
        bev_h: int = 128,
        bev_w: int = 128,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.bev_h = bev_h
        self.bev_w = bev_w

        # Per-modality linear projection
        self.proj = nn.ModuleDict({
            m: nn.Sequential(
                nn.Conv2d(bev_ch, embed_dim, 1),
                nn.GroupNorm(8, embed_dim),
                nn.GELU(),
            )
            for m in modality_names
        })

        # Learnable modality-type token (like BERT segment embeddings)
        self.mod_embed = nn.ParameterDict({
            m: nn.Parameter(torch.randn(1, embed_dim, 1, 1) * 0.02)
            for m in modality_names
        })

        self.pos_enc = BEVPositionalEncoding(embed_dim, bev_h, bev_w)

    def forward(
        self,
        bev_maps: Dict[str, torch.Tensor],   # mod → (B, bev_ch, H, W)
    ) -> torch.Tensor:
        """
        Returns: (B, N_tokens, embed_dim)
        where N_tokens = n_modalities × H × W
        """
        tokens = []
        for mod, proj in self.proj.items():
            if mod not in bev_maps:
                continue
            x = proj(bev_maps[mod])                # (B, embed_dim, H, W)
            x = self.pos_enc(x)                    # + positional bias
            x = x + self.mod_embed[mod]            # + modality type
            B, D, H, W = x.shape
            tokens.append(x.view(B, D, H * W))     # (B, D, H*W)

        # Stack modalities → (B, D, N_mod * H*W) → (B, N_tokens, D)
        stacked = torch.cat(tokens, dim=2)         # (B, D, N_tokens)
        return stacked.transpose(1, 2)             # (B, N_tokens, D)


# ──────────────────────────────────────────────────────
# 4.3  Multi-modal BEV Transformer
# ──────────────────────────────────────────────────────

class BEVFusionTransformer(nn.Module):
    """
    A standard pre-norm transformer applied over the flattened
    multi-modal token sequence.  After attention, tokens are
    re-shaped back to BEV and pooled across modalities.

    Memory note: H=W=128, n_mod=3 → 49152 tokens per sample.
    Use window attention (BEVFormer-style) in production.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        bev_h: int = 128,
        bev_w: int = 128,
        n_modalities: int = 3,
        out_ch: int = 256,
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.n_mod = n_modalities
        self.embed_dim = embed_dim

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-norm for stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Learned per-modality readout weight
        self.mod_weight = nn.Parameter(torch.ones(n_modalities) / n_modalities)

        # Output projection from (embed_dim) → (out_ch, H_bev, W_bev)
        self.out_proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, N_mod * H * W, embed_dim)
        Returns fused BEV: (B, out_ch, H, W)
        """
        B, N_total, D = tokens.shape
        n_per_mod = self.bev_h * self.bev_w

        # Transformer attention across all tokens
        fused = self.transformer(tokens)    # (B, N_total, D)

        # Split back by modality
        mod_tokens = fused.split(n_per_mod, dim=1)    # list of (B, H*W, D)
        weights    = self.mod_weight.softmax(dim=0)    # (n_mod,) normalized

        # Weighted sum across modalities
        bev_flat = sum(w * t for w, t in zip(weights, mod_tokens))  # (B, H*W, D)

        # Reshape to spatial
        bev_2d = bev_flat.transpose(1, 2).view(B, D, self.bev_h, self.bev_w)

        return self.out_proj(bev_2d)   # (B, out_ch, H, W)


# ──────────────────────────────────────────────────────
# 4.4  Window-partitioned efficient BEV attention
#      (production alternative to global attention)
# ──────────────────────────────────────────────────────

class WindowBEVAttention(nn.Module):
    """
    Applies multi-head attention within non-overlapping windows
    (Swin-style) to make attention O(N) in the BEV resolution.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_heads: int = 8,
        window_size: int = 16,      # 16×16 window → 256 tokens per window
        bev_h: int = 128,
        bev_w: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ws = window_size
        self.n_win_h = bev_h // window_size
        self.n_win_w = bev_w // window_size
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn  = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D, H, W) → (B, D, H, W)"""
        B, D, H, W = x.shape
        ws = self.ws

        # Partition into windows
        x_win = x.view(B, D, H // ws, ws, W // ws, ws)
        x_win = x_win.permute(0, 2, 4, 3, 5, 1)   # (B, nwH, nwW, ws, ws, D)
        n_win = (H // ws) * (W // ws)
        x_win = x_win.reshape(B * n_win, ws * ws, D)

        # Self-attention within each window
        x_normed = self.norm(x_win)
        attn_out, _ = self.attn(x_normed, x_normed, x_normed)
        x_win = x_win + attn_out

        # FFN
        x_win = x_win + self.ffn(self.norm2(x_win))

        # Merge windows back
        x_win = x_win.view(B, H // ws, W // ws, ws, ws, D)
        x_win = x_win.permute(0, 5, 1, 3, 2, 4)   # (B, D, nwH, ws, nwW, ws)
        return x_win.reshape(B, D, H, W)


# ──────────────────────────────────────────────────────
# 4.5  Full BEV Fusion module
# ──────────────────────────────────────────────────────

class BEVFusion(nn.Module):
    """
    Top-level fusion module. Wraps:
      ModalityEmbedder → BEVFusionTransformer → output BEV
    """

    def __init__(
        self,
        bev_ch: int = 128,
        embed_dim: int = 256,
        out_ch: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        bev_h: int = 128,
        bev_w: int = 128,
        modality_names: Tuple[str, ...] = ("camera", "lidar", "radar"),
        use_window_attn: bool = True,
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.use_window_attn = use_window_attn

        self.embedder = ModalityEmbedder(
            bev_ch=bev_ch,
            embed_dim=embed_dim,
            modality_names=modality_names,
            bev_h=bev_h,
            bev_w=bev_w,
        )

        if use_window_attn:
            # Stack multiple window-attention blocks
            self.window_attn_blocks = nn.ModuleList([
                WindowBEVAttention(
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                    window_size=16,
                    bev_h=bev_h,
                    bev_w=bev_w,
                )
                for _ in range(n_layers)
            ])
            self.out_proj = nn.Sequential(
                nn.Conv2d(embed_dim, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), nn.GELU(),
            )
        else:
            self.transformer = BEVFusionTransformer(
                embed_dim=embed_dim,
                n_heads=n_heads,
                n_layers=n_layers,
                out_ch=out_ch,
                bev_h=bev_h,
                bev_w=bev_w,
                n_modalities=len(modality_names),
            )

    def forward(
        self,
        gated_bevs: Dict[str, torch.Tensor],   # mod → (B, bev_ch, H, W)
    ) -> torch.Tensor:                          # (B, out_ch, H, W)
        if self.use_window_attn:
            # Embed and stack modalities spatially
            tokens = self.embedder(gated_bevs)           # (B, N_tokens, D)
            B, N, D = tokens.shape
            n_mod = len(gated_bevs)
            n_per = self.bev_h * self.bev_w
            mod_tokens = tokens.split(n_per, dim=1)

            # Average modalities → single (B, D, H, W)
            x = sum(t for t in mod_tokens) / n_mod
            x = x.transpose(1, 2).view(B, D, self.bev_h, self.bev_w)

            for block in self.window_attn_blocks:
                x = block(x)
            return self.out_proj(x)
        else:
            tokens = self.embedder(gated_bevs)
            return self.transformer(tokens)


# ── Smoke test ────────────────────────────────────────
if __name__ == "__main__":
    B, C, H, W = 2, 128, 128, 128
    gated = {
        "camera": torch.randn(B, C, H, W),
        "lidar":  torch.randn(B, C, H, W),
        "radar":  torch.randn(B, C, H, W),
    }
    fusion = BEVFusion(bev_ch=C, embed_dim=256, out_ch=256, n_layers=2, use_window_attn=True)
    out = fusion(gated)
    print(f"Fused BEV: {out.shape}")   # (B, 256, 128, 128)
    print(f"Params: {sum(p.numel() for p in fusion.parameters()):,}")
