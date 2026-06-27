"""Shared datatypes for downstream geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .constants import radius_for_type


@dataclass
class ExposureConfig:
    """Parameters for Monte Carlo steric exposure raycasting."""

    n_rays: int = 1000
    probe_radius: float = 5.0
    surface_epsilon: float = 2.0
    neighbor_margin: float = 0.0


@dataclass
class ParticleCloud:
    """One tomogram's particle detections in Angstrom coordinates."""

    tomo_id: str
    types: list[str]
    coords: np.ndarray
    radii: np.ndarray
    confidence: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=np.float64)
        self.radii = np.asarray(self.radii, dtype=np.float64)
        if self.coords.ndim != 2 or self.coords.shape[1] != 3:
            raise ValueError(f"coords must be (N, 3), got {self.coords.shape}")
        n = len(self.types)
        if self.coords.shape[0] != n or self.radii.shape[0] != n:
            raise ValueError("types, coords, and radii must have the same length")

    @property
    def n_particles(self) -> int:
        return len(self.types)

    @classmethod
    def from_arrays(
        cls,
        tomo_id: str,
        types: list[str],
        coords: np.ndarray,
        confidence: Optional[np.ndarray] = None,
    ) -> "ParticleCloud":
        radii = np.array([radius_for_type(t) for t in types], dtype=np.float64)
        return cls(
            tomo_id=tomo_id,
            types=types,
            coords=coords,
            radii=radii,
            confidence=confidence,
        )


@dataclass
class ExposureResult:
    """Steric exposure output for one particle."""

    index: int
    steric_exposure: float
    n_rays: int
    n_neighbors_considered: int
    n_rays_blocked: int = 0

    @property
    def blocked_fraction(self) -> float:
        return 1.0 - self.steric_exposure
