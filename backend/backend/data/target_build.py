"""Build sphere-based segmentation targets from copick picks (port of DeepFindET)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from tqdm import tqdm

from . import copick_io


def make_sphere(diameter: int, radius: float) -> np.ndarray:
    """Return a cubic uint8 mask of a filled sphere."""
    d = int(diameter)
    if d <= 0:
        return np.zeros((0, 0, 0), dtype=np.uint8)
    z = np.arange(d) - d // 2
    y = np.arange(d) - d // 2
    x = np.arange(d) - d // 2
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    dist = np.sqrt(zz ** 2 + yy ** 2 + xx ** 2)
    return (dist <= radius).astype(np.uint8)


def stamp_sphere(
    target: np.ndarray, cx: int, cy: int, cz: int, sphere: np.ndarray, label: int
) -> None:
    """Stamp a sphere mask (in-place) into target at integer center (x,y,z)."""
    if sphere.size == 0:
        return
    d = sphere.shape[0]
    half = d // 2

    x0, x1 = cx - half, cx - half + d
    y0, y1 = cy - half, cy - half + d
    z0, z1 = cz - half, cz - half + d

    # bounds inside target (Z,Y,X)
    tz0, tz1 = max(z0, 0), min(z1, target.shape[0])
    ty0, ty1 = max(y0, 0), min(y1, target.shape[1])
    tx0, tx1 = max(x0, 0), min(x1, target.shape[2])

    if tz0 >= tz1 or ty0 >= ty1 or tx0 >= tx1:
        return

    # corresponding region in sphere
    sz0, sz1 = tz0 - z0, tz1 - z0
    sy0, sy1 = ty0 - y0, ty1 - y0
    sx0, sx1 = tx0 - x0, tx1 - x0

    region = target[tz0:tz1, ty0:ty1, tx0:tx1]
    mask = sphere[sz0:sz1, sy0:sy1, sx0:sx1].astype(bool)
    np.maximum(region, label * mask.astype(region.dtype), out=region, where=mask)


def build_targets_for_run(
    config_path: str,
    tomo_id: str,
    voxel_size: float = 10.0,
    particle_targets: dict | None = None,
    seg_targets: dict | None = None,
) -> np.ndarray:
    """Build a uint8 segmentation target volume for a single run.

    Args:
        config_path: copick config path.
        tomo_id: run name.
        voxel_size: voxel size in Angstrom.
        particle_targets: dict {object_name: {"label":..,"user_id":..,"session_id":..,"radius":..}}.
            If None, all particle pickable objects are used.
        seg_targets: dict {seg_name: {"label":..,"user_id":..,"session_id":..}} for membrane-like
            pre-existing segmentations to overlay.
    """
    import zarr

    root = copick_io.get_copick_root(config_path)

    # Determine target objects
    if particle_targets is None:
        particle_targets = {}
        for obj in root.pickable_objects:
            if obj.is_particle:
                r = getattr(obj, "radius", None)
                particle_targets[obj.name] = {
                    "label": obj.label,
                    "user_id": None,
                    "session_id": None,
                    "radius": (r / voxel_size) if r else 0.0,
                }

    target_vol = copick_io.get_empty_target(config_path, tomo_id, voxel_size)

    # Overlay existing segmentations (e.g. membrane) if provided
    if seg_targets:
        for name, info in seg_targets.items():
            segs = root.get_run(tomo_id).get_segmentations(
                name=name,
                user_id=info.get("user_id"),
                session_id=info.get("session_id"),
                voxel_size=voxel_size,
                is_multilabel=False,
            )
            for seg in segs:
                vol = zarr.open(seg.zarr(), mode="r")["0"][:]
                np.maximum(target_vol, vol.astype(np.uint8) * info["label"], out=target_vol)

    # Precompute sphere masks per class (radius in voxels)
    spheres = {}
    for name, info in particle_targets.items():
        r = info["radius"]
        if r <= 0:
            continue
        diameter = int(np.ceil(2 * r)) + 2
        spheres[info["label"]] = make_sphere(diameter, r)

    # Stamp particles
    for name, info in particle_targets.items():
        label = info["label"]
        sphere = spheres.get(label)
        if sphere is None or sphere.size == 0:
            continue
        coords = copick_io.get_picks(
            config_path, tomo_id, name,
            user_id=info.get("user_id"),
            session_id=info.get("session_id"),
        )
        if coords.shape[0] == 0:
            continue
        # Angstrom -> voxel
        vcoords = coords / voxel_size
        for (xa, ya, za) in vcoords:
            stamp_sphere(target_vol, int(round(xa)), int(round(ya)), int(round(za)),
                         sphere, label)
    return target_vol


def build_targets(
    config_path: str,
    tomo_ids: list[str] | None = None,
    voxel_size: float = 10.0,
    out_name: str = "pytargets",
    out_user_id: str = "pytorch",
    out_session_id: str = "0",
    particle_targets: dict | None = None,
    seg_targets: dict | None = None,
) -> None:
    """Build and write segmentation targets for all (or given) runs."""
    root = copick_io.get_copick_root(config_path)
    if tomo_ids is None:
        tomo_ids = [run.name for run in root.runs]

    if particle_targets is None:
        particle_targets = {}
        for obj in root.pickable_objects:
            if obj.is_particle:
                r = getattr(obj, "radius", None)
                particle_targets[obj.name] = {
                    "label": obj.label,
                    "user_id": None,
                    "session_id": None,
                    "radius": (r / voxel_size) if r else 0.0,
                }

    for tomo_id in tqdm(tomo_ids, desc="Building targets"):
        target = build_targets_for_run(
            config_path, tomo_id, voxel_size, particle_targets, seg_targets
        )
        copick_io.write_ome_zarr_segmentation(
            config_path, tomo_id, target, voxel_size,
            name=out_name, user_id=out_user_id, session_id=out_session_id,
            multilabel=True,
        )