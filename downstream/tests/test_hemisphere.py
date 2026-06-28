"""Tests for directional exposure hemisphere metrics and connectivity calibration."""

import numpy as np
import pytest

from downstream.geometry.directions import fibonacci_directions, fibonacci_ray_adjacency
from downstream.geometry.exposure import steric_exposure_batch, steric_exposure_one
from downstream.geometry.hemisphere import (
    analyze_hemisphere,
    connected_components_unblocked,
)
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


@pytest.mark.parametrize("n_rays", [200, 500, 2000])
def test_synthetic_hemisphere_is_single_component(n_rays: int):
    directions = fibonacci_directions(n_rays)
    edge_pairs, _ = fibonacci_ray_adjacency(n_rays)
    axis = np.array([1.0, 0.0, 0.0])
    blocked = directions @ axis <= 0.0

    components, n_comp = connected_components_unblocked(blocked, edge_pairs)
    n_open = int((~blocked).sum())
    largest = max((len(c) for c in components), default=0)

    assert n_comp == 1, f"expected 1 open component at n={n_rays}, got {n_comp}"
    assert largest == n_open


@pytest.mark.parametrize("n_rays", [200, 500, 2000])
def test_connect_angle_tracks_nn_spacing(n_rays: int):
    _, angle_deg = fibonacci_ray_adjacency(n_rays)
    assert angle_deg > 0
    if n_rays == 200:
        assert 12.0 < angle_deg < 22.0
    elif n_rays == 2000:
        assert 4.0 < angle_deg < 9.0


def test_two_separated_open_caps_not_merged():
    n_rays = 2000
    directions = fibonacci_directions(n_rays)
    edge_pairs, _ = fibonacci_ray_adjacency(n_rays)
    axis = np.array([0.0, 0.0, 1.0])
    dots = directions @ axis
    blocked = np.abs(dots) < 0.35

    components, n_comp = connected_components_unblocked(blocked, edge_pairs)
    assert n_comp == 2
    assert all(len(c) > 50 for c in components)


def test_isolated_particle_full_metrics():
    cloud = _cloud([[0.0, 0.0, 0.0]], [90.0])
    config = ExposureConfig(n_rays=2000)
    res = steric_exposure_one(cloud, 0, config=config)
    assert res.steric_exposure == pytest.approx(1.0, abs=0.02)
    assert res.clean_extraction_score == pytest.approx(1.0, abs=0.02)
    assert res.anisotropy_index == pytest.approx(0.0, abs=0.05)
    assert res.n_open_components == 1


def test_hemispheric_block_high_clean_extraction():
    """Large neighbor on +X blocks one hemisphere; open side stays one component."""
    small_r, large_r = 45.0, 150.0
    separation = 80.0
    cloud = _cloud(
        [[0.0, 0.0, 0.0], [separation, 0.0, 0.0]],
        [small_r, large_r],
    )
    config = ExposureConfig(n_rays=2000, probe_radius=0.0, surface_epsilon=2.0)
    res = steric_exposure_one(cloud, 0, config=config)

    assert res.steric_exposure < 0.85
    assert res.clean_extraction_score > 0.35
    assert res.anisotropy_index < 0.15
    assert res.n_open_components == 1


def test_uniform_ring_low_clean_extraction():
    """Six neighbors on a ring give ~half scalar exposure but fragmented open patches."""
    r = 90.0
    ring_radius = 220.0
    angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    ring_coords = [
        [ring_radius * np.cos(a), ring_radius * np.sin(a), 0.0] for a in angles
    ]
    cloud = _cloud([[0.0, 0.0, 0.0], *ring_coords], [r] * (1 + len(ring_coords)))
    config = ExposureConfig(n_rays=2000, probe_radius=5.0)
    res = steric_exposure_one(cloud, 0, config=config)

    assert 0.25 < res.steric_exposure < 0.75
    assert res.clean_extraction_score < res.steric_exposure * 0.85
    assert res.anisotropy_index > 0.05


def test_hemispheric_vs_uniform_discrimination():
    """Same ballpark scalar exposure, different clean extraction / anisotropy."""
    small_r, large_r = 45.0, 150.0
    hemispheric = _cloud(
        [[0.0, 0.0, 0.0], [80.0, 0.0, 0.0]],
        [small_r, large_r],
    )
    r = 90.0
    ring_radius = 220.0
    angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    ring_coords = [
        [ring_radius * np.cos(a), ring_radius * np.sin(a), 0.0] for a in angles
    ]
    uniform = _cloud([[0.0, 0.0, 0.0], *ring_coords], [r] * (1 + len(ring_coords)))

    config = ExposureConfig(n_rays=2000, probe_radius=5.0)
    h = steric_exposure_one(hemispheric, 0, config=config)
    u = steric_exposure_one(uniform, 0, config=config)

    assert h.clean_extraction_score > u.clean_extraction_score + 0.1
    assert u.anisotropy_index > h.anisotropy_index


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
        assert res.clean_extraction_score == pytest.approx(
            single.clean_extraction_score, abs=1e-12
        )
