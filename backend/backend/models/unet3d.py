"""Plain 3D U-Net."""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBlock


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        n_class: int = 8,
        filters=(32, 48, 64),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pools = nn.ModuleList()
        chs = [in_channels] + list(filters[:-1])
        for i, f in enumerate(filters[:-1]):
            self.downs.append(ConvBlock(chs[i], f, dropout))
            self.pools.append(nn.MaxPool3d(2))
        # bottleneck
        bot_f = filters[-1]
        self.bottleneck = nn.Sequential(
            ConvBlock(chs[-1], bot_f, dropout),
            ConvBlock(bot_f, bot_f, dropout),
        )
        # decoder
        rev = list(reversed(filters[:-1]))
        up_chs = [bot_f] + list(rev[:-1])
        for i, f in enumerate(rev):
            self.ups.append(nn.ConvTranspose3d(up_chs[i], f, 2, stride=2))
            self.ups.append(ConvBlock(f * 2, f, dropout))
        self.head = nn.Conv3d(f, n_class, 1)

    def forward(self, x):
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        skips = list(reversed(skips))
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(
                    x, size=skip.shape[2:], mode="trilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
            x = self.ups[i + 1](x)
        return self.head(x)