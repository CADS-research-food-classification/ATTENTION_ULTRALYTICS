import torch
import torch.nn as nn
import torch.nn.functional as F

class SE(nn.Module):
    def __init__(self, c, r=16):  # c = input channels, r = reduction ratio
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, max(c // r, 4), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(c // r, 4), c, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w
