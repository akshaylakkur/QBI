"""CZII particle radii (Angstrom) — matches backend copick config."""

from __future__ import annotations

# name -> radius in Angstrom
RADIUS_ANGSTROM: dict[str, float] = {
    "apo-ferritin": 60.0,
    "beta-amylase": 65.0,
    "beta-galactosidase": 90.0,
    "ribosome": 150.0,
    "thyroglobulin": 130.0,
    "virus-like-particle": 135.0,
}

PARTICLE_TYPES: tuple[str, ...] = tuple(RADIUS_ANGSTROM.keys())


def radius_for_type(particle_type: str) -> float:
    """Return hard-sphere radius in Angstrom for a copick particle name."""
    key = particle_type.strip()
    if key not in RADIUS_ANGSTROM:
        known = ", ".join(sorted(RADIUS_ANGSTROM))
        raise ValueError(f"Unknown particle type '{particle_type}'. Known: {known}")
    return RADIUS_ANGSTROM[key]
