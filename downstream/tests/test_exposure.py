"""Tests for steric exposure raycasting."""

import numpy as np
import pytest

from downstream.geometry.exposure import steric_exposure_batch, steric_exposure_one
from downstream.types import ExposureConfig, ParticleCloud


def _cloud(coords, radii, types=None) -> ParticleCloud:
    n = len(coords)
    if types is None:
        types = ["beta-galactosidase"] * n
    return ParticleCloud(
        tomo_id="test",
        types=types,
        coords=np.asarray(coords, dtype=np.float64),
        radii=np.asarray(radii, dtype=np.float64),
    )


def test_isolated_particle_full_exposure():
    cloud = _cloud([[0.0, 0.0, 0.0]], [90.0])
    config = ExposureConfig(n_rays=2000)
    res = steric_exposure_one(cloud, 0, config=config)
    assert res.steric_exposure == pytest.approx(1.0, abs=0.02)
    assert res.n_neighbors_considered == 0


def test_touching_pair_reduces_exposure():
    r = 90.0
    cloud = _cloud([[0.0, 0.0, 0.0], [2 * r, 0.0, 0.0]], [r, r])
    config = ExposureConfig(n_rays=2000, probe_radius=0.0)
    a = steric_exposure_one(cloud, 0, config=config)
    b = steric_exposure_one(cloud, 1, config=config)
    assert a.steric_exposure < 1.0
    assert b.steric_exposure < 1.0
    assert a.steric_exposure == pytest.approx(b.steric_exposure, abs=0.05)


def test_occluded_hemisphere_low_exposure():
    small_r, large_r = 45.0, 150.0
    # Overlapping placement: large sphere engulfs much of the small target's outward hemisphere.
    separation = 80.0
    cloud = _cloud(
        [[0.0, 0.0, 0.0], [separation, 0.0, 0.0]],
        [small_r, large_r],
    )
    config = ExposureConfig(n_rays=2000, probe_radius=0.0, surface_epsilon=2.0)
    res = steric_exposure_one(cloud, 0, config=config)
    assert res.steric_exposure < 0.85
    assert res.n_rays_blocked > 300


def test_large_neighbor_blocks_small_target():
    cloud = _cloud(
        [[0.0, 0.0, 0.0], [0.0, 200.0, 0.0]],
        [45.0, 150.0],
        types=["beta-galactosidase", "ribosome"],
    )
    config = ExposureConfig(n_rays=2000, probe_radius=5.0)
    res = steric_exposure_one(cloud, 0, config=config)
    assert res.steric_exposure < steric_exposure_one(cloud, 1, config=config).steric_exposure


def test_batch_matches_single():
    cloud = _cloud(
        [[0.0, 0.0, 0.0], [300.0, 0.0, 0.0], [0.0, 400.0, 0.0]],
        [90.0, 90.0, 130.0],
    )
    config = ExposureConfig(n_rays=500)
    batch = steric_exposure_batch(cloud, config=config)
    for i, res in enumerate(batch):
        single = steric_exposure_one(cloud, i, config=config)
        assert res.steric_exposure == pytest.approx(single.steric_exposure, abs=1e-12)
