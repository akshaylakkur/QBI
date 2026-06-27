"""3D data augmentations operating on torch tensors (image + integer label)."""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


class Augment3D:
    """Apply a random subset of 3D augmentations to (image, label) patches.

    image: torch.FloatTensor of shape (1, D, D, D)
    label: torch.LongTensor  of shape (D, D, D) with integer class indices
    """

    def __init__(
        self,
        p_flip: float = 0.5,
        p_rot180: float = 0.5,
        p_noise: float = 0.3,
        p_blur: float = 0.3,
        p_brightness: float = 0.4,
        p_contrast: float = 0.4,
        p_intensity: float = 0.4,
        noise_std: float = 0.05,
        blur_sigma: Tuple[float, float] = (0.5, 1.25),
    ):
        self.p_flip = p_flip
        self.p_rot180 = p_rot180
        self.p_noise = p_noise
        self.p_blur = p_blur
        self.p_brightness = p_brightness
        self.p_contrast = p_contrast
        self.p_intensity = p_intensity
        self.noise_std = noise_std
        self.blur_sigma = blur_sigma

    def __call__(
        self, image: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Random flips along any of the 3 spatial axes (cheap, label-safe)
        # image: (1,1,D,D,D) -> spatial dims 2,3,4 ; label: (D,D,D) -> 0,1,2
        for ax in (0, 1, 2):
            if random.random() < self.p_flip:
                image = torch.flip(image, dims=[ax + 2])
                label = torch.flip(label, dims=[ax])

        # 180-degree rotation around a random pair of spatial axes (label-safe)
        if random.random() < self.p_rot180:
            pair = random.choice([(0, 1), (0, 2), (1, 2)])
            label = torch.rot90(label, k=2, dims=list(pair))
            image = torch.rot90(image, k=2, dims=[d + 2 for d in pair])

        # Intensity augmentations (image only)
        if random.random() < self.p_brightness:
            image = image + float(np.random.uniform(-0.3, 0.3))
        if random.random() < self.p_intensity:
            image = image * float(np.random.uniform(0.8, 1.2))
        if random.random() < self.p_contrast:
            mean = image.mean()
            image = mean + float(np.random.uniform(0.7, 1.3)) * (image - mean)
        if random.random() < self.p_noise:
            std = float(np.random.uniform(0.0, self.noise_std))
            image = image + torch.randn_like(image) * std
        if random.random() < self.p_blur:
            sigma = float(np.random.uniform(*self.blur_sigma))
            image = _gaussian_blur_3d(image, sigma)

        return image, label


def _gaussian_blur_3d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable 3D gaussian blur on (1,1,D,D,D)."""
    if sigma <= 0:
        return x
    radius = max(1, int(3 * sigma))
    ax = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    k1d = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    k1d = k1d / k1d.sum()
    # reshape to conv kernels (1,1,K,1,1),(1,1,1,K,1),(1,1,1,1,K)
    pad = radius
    k = k1d.view(1, 1, -1, 1, 1)
    x = F.conv3d(x, k, padding=(pad, 0, 0))
    k = k1d.view(1, 1, 1, -1, 1)
    x = F.conv3d(x, k, padding=(0, pad, 0))
    k = k1d.view(1, 1, 1, 1, -1)
    x = F.conv3d(x, k, padding=(0, 0, pad))
    return x