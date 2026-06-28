"""3D building blocks: conv blocks, residual blocks, attention blocks.

Normalization uses GroupNorm (not BatchNorm) so training is stable at the small
per-GPU batches typical of 3D segmentation (e.g. 4/GPU on 2x T4 with DataParallel).
GroupNorm is independent of batch size and avoids running-stat drift between
train and eval.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _norm(channels: int, groups: int = 8) -> nn.Module:
    """GroupNorm with at most `groups` groups (clamped to channel count)."""
    g = min(groups, channels)
    # ensure groups divides channels; if not, fall back to a divisor
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels)


class ConvBlock(nn.Module):
    """Two 3x3x3 convs with GroupNorm + LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            _norm(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            _norm(out_ch),
            nn.LeakyReLU(inplace=True),
        ]
        if dropout and dropout > 0:
            layers.append(nn.Dropout3d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual conv block with 1x1x1 shortcut when channels differ.

    Uses GroupNorm instead of BatchNorm for stability at small per-GPU batches.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = _norm(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = _norm(out_ch)
        self.act = nn.LeakyReLU(inplace=True)
        self.shortcut = (
            nn.Conv3d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )
        self.drop = nn.Dropout3d(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.drop(out)
        return self.act(out + identity)


class AttentionBlock(nn.Module):
    """Gating attention block for attention U-Net."""

    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
        super().__init__()
        self.theta = nn.Conv3d(skip_ch, inter_ch, 1, bias=False)
        self.phi = nn.Conv3d(gate_ch, inter_ch, 1, bias=False)
        self.psi = nn.Conv3d(inter_ch, 1, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)

    def forward(self, x, g):
        theta = self.theta(x)
        phi = self.phi(g)
        # upsample gate to match skip spatial size
        phi = self.upsample(phi)
        if phi.shape[2:] != theta.shape[2:]:
            phi = nn.functional.interpolate(
                phi, size=theta.shape[2:], mode="trilinear", align_corners=False
            )
        h = self.act(theta + phi)
        psi = self.gate(self.psi(h))
        # broadcast psi to skip channels
        psi = psi.expand(-1, x.shape[1], -1, -1, -1)
        return x * psi