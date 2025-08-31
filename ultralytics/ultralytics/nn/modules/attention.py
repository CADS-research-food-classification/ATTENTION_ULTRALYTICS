# Ultralytics - Attention blocks
# Save as: ultralytics/nn/modules/attention.py

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = "SE"


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



