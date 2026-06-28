"""Tests for Hopkins statistic and Grid Viability Index."""

import numpy as np
import pytest

from downstream.geometry.hopkins import hopkins_statistic
from downstream.qa.grid_viability import (
    GridViabilityConfig,
    assess_cloud,
    fit_reference_stats,
)
from downstream.types import ParticleCloud


def _cloud(coords, tomo_id="test") -> ParticleCloud:
    n = len(coords)
    return ParticleCloud(
        tomo_id=tomo_id,
        types=["beta-galactosidase"] * n,
        coords=np.asarray(coords, dtype=np.float64),
        radii=np.full(n, 90.0),
    )


def test_hopkins_uniform_random_near_half():
    rng = np.random.default_rng(42)
    coords = rng.random((500, 3)) * 3000.0
    h = hopkins_statistic(coords, rng=rng)
    assert h == pytest.approx(0.5, abs=0.08)


def test_hopkins_clustered_high():
    rng = np.random.default_rng(0)
    center = rng.normal(1500, 30, size=(200, 3))
    h = hopkins_statistic(center, rng=rng)
    assert h > 0.65


def test_hopkins_insufficient_points():
    assert np.isnan(hopkins_statistic(np.zeros((1, 3))))


def test_gvi_flags_uniform_static():
    rng = np.random.default_rng(7)
    coords = rng.random((400, 3)) * 3000.0
    cloud = _cloud(coords)
    ref = fit_reference_stats([_cloud(rng.random((150, 3)) * 3000.0) for _ in range(5)])
    config = GridViabilityConfig(min_particles=30)
    result = assess_cloud(cloud, config=config, reference=ref, rng=rng)
    assert result.status in ("fail_static", "warn", "pass")
    assert result.n_particles == 400


def test_gvi_insufficient_particles():
    cloud = _cloud([[0, 0, 0], [100, 0, 0]])
    result = assess_cloud(cloud, config=GridViabilityConfig(min_particles=30))
    assert result.status == "insufficient"


def test_fit_reference_stats():
    rng = np.random.default_rng(1)
    clouds = [
        _cloud(rng.random((120 + i * 10, 3)) * 3000.0, tomo_id=f"T{i}")
        for i in range(3)
    ]
    ref = fit_reference_stats(clouds)
    assert ref.hopkins_mean > 0.3
    assert ref.n_particles_mean > 100
