"""Model factory."""

from __future__ import annotations

import torch.nn as nn

from .unet3d import UNet3D
from .res_unet3d import ResUNet3D
from .attention_unet3d import AttentionUNet3D


_REGISTRY = {
    "unet": UNet3D,
    "res_unet": ResUNet3D,
    "attention_unet": AttentionUNet3D,
}


def build_model(
    name: str,
    in_channels: int = 1,
    n_class: int = 8,
    filters=(48, 64, 128),
    dropout: float = 0.0,
) -> nn.Module:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choices: {list(_REGISTRY)}")
    cls = _REGISTRY[name]
    return cls(in_channels=in_channels, n_class=n_class, filters=tuple(filters), dropout=dropout)