"""Uniformly distributed unit directions on the sphere (Fibonacci / golden spiral)."""

from __future__ import annotations

import numpy as np

_GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))


def fibonacci_directions(n: int) -> np.ndarray:
    """Return ``(n, 3)`` unit vectors uniformly distributed on S^2.

    Uses a golden-angle spiral (Fibonacci sphere), which avoids polar clustering
    that naive latitude/longitude sampling introduces.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    i = np.arange(n, dtype=np.float64)
    y = 1.0 - (2.0 * i + 1.0) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = _GOLDEN_ANGLE * i
    x = np.cos(theta) * r
    z = np.sin(theta) * r
    dirs = np.stack([x, y, z], axis=1)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs / norms
