"""Residual 3D U-Net (primary model)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ResidualBlock


class ResUNet3D(nn.Module):
    """3D residual U-Net with two pooling stages (dim_in must be multiple of 4)."""

    def __init__(
        self,
        in_channels: int = 1,
        n_class: int = 8,
        filters=(48, 64, 128),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        chs = [in_channels] + list(filters[:-1])
        for i, f in enumerate(filters[:-1]):
            self.downs.append(ResidualBlock(chs[i], f, dropout))
            self.pools.append(nn.MaxPool3d(2))

        bot_f = filters[-1]
        self.bottleneck = nn.Sequential(
            ResidualBlock(chs[-1], bot_f, dropout),
            ResidualBlock(bot_f, bot_f, dropout),
            ResidualBlock(bot_f, bot_f, dropout),
            ResidualBlock(bot_f, bot_f, dropout),
        )

        self.ups = nn.ModuleList()
        rev = list(reversed(filters[:-1]))
        up_chs = [bot_f] + list(rev[:-1])
        for i, f in enumerate(rev):
            self.ups.append(nn.ConvTranspose3d(up_chs[i], f, 2, stride=2))
            self.ups.append(ResidualBlock(f * 2, f, dropout))
            self.ups.append(ResidualBlock(f, f, dropout))
        self.head = nn.Conv3d(f, n_class, 1)

    def forward(self, x):
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        skips = list(reversed(skips))
        idx = 0
        for skip in skips:
            x = self.ups[idx](x)  # upsample
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(
                    x, size=skip.shape[2:], mode="trilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
            x = self.ups[idx + 1](x)
            x = self.ups[idx + 2](x)
            idx += 3
        return self.head(x)