# Ultralytics - Attention blocks
# Save as: ultralytics/nn/modules/attention.py

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SE", "CBAM"]


# ---------------- Squeeze-and-Excitation ----------------
class SE(nn.Module):
    """
    Squeeze-and-Excitation block (Hu et al., 2018).
    Args:
        c (int): Number of input channels.
        r (int): Reduction ratio. Default = 16.
    """
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        assert c > 0, "SE: channels must be > 0"
        mid = max(c // r, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, c, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.avg(x))
        return x * w


# ---------------- CBAM (optional for compatibility) ----------------
class ChannelGate(nn.Module):
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        mid = max(c // r, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(c, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, c, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        w = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * w

class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * s

class CBAM(nn.Module):
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        self.channel = ChannelGate(c, r=r)
        self.spatial = SpatialGate()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))
