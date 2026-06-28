"""Copick/Zarr I/O helpers for CryoET tomograms and picks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import zarr


# ---------------------------------------------------------------------------
# Config blob (czii 2024 challenge layout)
# ---------------------------------------------------------------------------
CZII_CONFIG_BLOB = """{{
    "name": "czii_cryoet_mlchallenge_2024",
    "description": "2024 CZII CryoET ML Challenge training data.",
    "version": "1.0.0",
    "pickable_objects": [
        {{"name": "apo-ferritin",        "is_particle": true, "pdb_id": "4V1W", "label": 1, "color": [0,117,220,128],   "radius": 60,  "map_threshold": 0.0418}},
        {{"name": "beta-amylase",        "is_particle": true, "pdb_id": "1FA2", "label": 2, "color": [153,63,0,128],    "radius": 65,  "map_threshold": 0.035}},
        {{"name": "beta-galactosidase",  "is_particle": true, "pdb_id": "6X1Q", "label": 3, "color": [76,0,92,128],     "radius": 90,  "map_threshold": 0.0578}},
        {{"name": "ribosome",            "is_particle": true, "pdb_id": "6EK0", "label": 4, "color": [0,92,49,128],     "radius": 150, "map_threshold": 0.0374}},
        {{"name": "thyroglobulin",       "is_particle": true, "pdb_id": "6SCJ", "label": 5, "color": [43,206,72,128],   "radius": 130, "map_threshold": 0.0278}},
        {{"name": "virus-like-particle", "is_particle": true,                  "label": 6, "color": [255,204,153,128], "radius": 135, "map_threshold": 0.201}},
        {{"name": "membrane",            "is_particle": false,                 "label": 8, "color": [100,100,100,128]}},
        {{"name": "background",          "is_particle": false,                 "label": 9, "color": [10,150,200,128]}}
    ],
    "overlay_root": "{overlay_root}",
    "overlay_fs_args": {{"auto_mkdir": true}},
    "static_root": "{static_root}"
}}
"""


def write_copick_config(
    config_path: str,
    static_root: str,
    overlay_root: str,
) -> str:
    """Write a copick config JSON for the czii challenge layout."""
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    blob = CZII_CONFIG_BLOB.format(overlay_root=overlay_root, static_root=static_root)
    with open(config_path, "w") as f:
        f.write(blob)
    return config_path


def get_copick_root(config_path: str):
    """Load a copick root from a config file (lazy import of copick)."""
    import copick

    return copick.from_file(config_path)


def list_runs(config_path: str) -> List[str]:
    root = get_copick_root(config_path)
    return [run.name for run in root.runs]


# ---------------------------------------------------------------------------
# Tomograms
# ---------------------------------------------------------------------------
def _open_zarr_volume(zarr_path: str):
    """Open a multiscale zarr group and return the level-0 array."""
    group = zarr.open(zarr_path, mode="r")
    # OME-NGFF multiscale: arrays are 0,1,2,... (0 is highest resolution)
    return group["0"]


def _fetch_tomogram(vs, tomo_algorithm: str):
    """Fetch a single tomogram from a voxel-spacing, tolerating both the
    deprecated `get_tomogram` (singular) and the new `get_tomograms` (plural,
    returns a list) copick APIs. Silences the deprecation warning either way."""
    import warnings

    tomo = None
    # New API: get_tomograms returns a list (possibly filtered by name)
    if hasattr(vs, "get_tomograms"):
        try:
            tomos = vs.get_tomograms(tomo_algorithm)
            if tomos:  # non-empty list
                tomo = tomos[0]
        except Exception:
            tomos = None
    # Fallback to deprecated singular API (suppress its DeprecationWarning)
    if tomo is None and hasattr(vs, "get_tomogram"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            tomo = vs.get_tomogram(tomo_algorithm)
    if tomo is None:
        avail = [t.tomo_type for t in vs.tomograms]
        raise ValueError(
            f"Tomogram '{tomo_algorithm}' not found. Available: {avail}"
        )
    return tomo


def get_tomogram(
    config_path: str,
    tomo_id: str,
    voxel_size: float = 10.0,
    tomo_algorithm: str = "denoised",
):
    """Return the level-0 zarr array for a tomogram (lazy, not materialized)."""
    root = get_copick_root(config_path)
    run = root.get_run(tomo_id)
    vs = run.get_voxel_spacing(voxel_size)
    if vs is None:
        raise ValueError(f"Voxel spacing {voxel_size} not found for run {tomo_id}")
    tomo = _fetch_tomogram(vs, tomo_algorithm)
    return _open_zarr_volume(tomo.zarr())


def get_tomogram_shape(config_path: str, tomo_id: str, voxel_size: float = 10.0,
                       tomo_algorithm: str = "denoised") -> Tuple[int, int, int]:
    return get_tomogram(config_path, tomo_id, voxel_size, tomo_algorithm).shape


def get_empty_target(config_path: str, tomo_id: str, voxel_size: float = 10.0,
                     tomo_algorithm: str = "denoised") -> np.ndarray:
    shape = get_tomogram_shape(config_path, tomo_id, voxel_size, tomo_algorithm)
    return np.zeros(shape, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Picks (ground-truth coordinates)
# ---------------------------------------------------------------------------
def get_picks(
    config_path: str,
    tomo_id: str,
    object_name: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> np.ndarray:
    """Return (N,3) pick coordinates in Angstroms (x,y,z)."""
    root = get_copick_root(config_path)
    run = root.get_run(tomo_id)
    picks_list = run.get_picks(object_name, user_id=user_id, session_id=session_id)
    if not picks_list:
        return np.zeros((0, 3), dtype=np.float32)
    picks = picks_list[0]
    coords = np.array(
        [(p.location.x, p.location.y, p.location.z) for p in picks.points],
        dtype=np.float32,
    )
    return coords


def get_pickable_objects(config_path: str) -> List[dict]:
    """Return list of pickable object descriptors with name/label/radius/is_particle."""
    root = get_copick_root(config_path)
    out = []
    for obj in root.pickable_objects:
        out.append(
            {
                "name": obj.name,
                "label": obj.label,
                "radius": getattr(obj, "radius", None),
                "is_particle": obj.is_particle,
                "pdb_id": getattr(obj, "pdb_id", None),
            }
        )
    return out


def get_object_label(config_path: str, object_name: str) -> int:
    root = get_copick_root(config_path)
    return root.get_object(object_name).label


def get_object_radius_voxels(config_path: str, object_name: str,
                             voxel_size: float = 10.0) -> float:
    root = get_copick_root(config_path)
    obj = root.get_object(object_name)
    if getattr(obj, "radius", None) is None:
        return 0.0
    return float(obj.radius) / float(voxel_size)


# ---------------------------------------------------------------------------
# Segmentation read/write
# ---------------------------------------------------------------------------
def write_ome_zarr_segmentation(
    config_path: str,
    tomo_id: str,
    volume: np.ndarray,
    voxel_size: float = 10.0,
    name: str = "segmentation",
    user_id: str = "pytorch",
    session_id: str = "0",
    multilabel: bool = True,
) -> None:
    """Write a segmentation volume into the copick overlay."""
    import ome_zarr.writer

    root = get_copick_root(config_path)
    run = root.get_run(tomo_id)

    segs = run.get_segmentations(name=name, user_id=user_id, session_id=session_id)
    if len(segs) == 0 or segs[0].voxel_size != voxel_size:
        seg = run.new_segmentation(
            voxel_size=voxel_size,
            name=name,
            session_id=session_id,
            is_multilabel=multilabel,
            user_id=user_id,
        )
    else:
        seg = segs[0]

    loc = seg.zarr()
    root_group = zarr.group(loc, overwrite=True)

    axes = [
        {"name": "z", "type": "space", "unit": "angstrom"},
        {"name": "y", "type": "space", "unit": "angstrom"},
        {"name": "x", "type": "space", "unit": "angstrom"},
    ]
    transforms = [{"scale": [voxel_size, voxel_size, voxel_size], "type": "scale"}]
    ome_zarr.writer.write_multiscale(
        [volume],
        group=root_group,
        axes=axes,
        coordinate_transformations=[transforms],
        storage_options=dict(chunks=(256, 256, 256), overwrite=True),
        compute=True,
    )


def write_ome_zarr_scoremap(
    config_path: str,
    tomo_id: str,
    scoremap: np.ndarray,  # shape (C, Z, Y, X) probabilities
    voxel_size: float = 10.0,
    name: str = "pyscoremap",
    user_id: str = "pytorch",
    session_id: str = "0",
    tomo_type: str = "denoised",
) -> None:
    """Write a per-class scoremap as OME-Zarr features (channel-first)."""
    import ome_zarr.writer

    root = get_copick_root(config_path)
    run = root.get_run(tomo_id)
    tomo = _fetch_tomogram(run.get_voxel_spacing(voxel_size), tomo_type)
    feat = tomo.get_features(name)
    if feat is None:
        feat = tomo.new_features(feature_type=name)

    loc = feat.zarr()
    root_group = zarr.group(loc, overwrite=True)

    vol = np.transpose(scoremap, (3, 0, 1, 2))  # -> (X,Y,Z,C)? keep (C,Z,Y,X) per copick convention
    vol = np.ascontiguousarray(vol)

    axes = [
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "angstrom"},
        {"name": "y", "type": "space", "unit": "angstrom"},
        {"name": "x", "type": "space", "unit": "angstrom"},
    ]
    transforms = [
        {"scale": [voxel_size, voxel_size, voxel_size, voxel_size], "type": "scale"}
    ]
    ome_zarr.writer.write_multiscale(
        [vol],
        group=root_group,
        axes=axes,
        coordinate_transformations=[transforms],
        storage_options=dict(chunks=(1, 256, 256, 256), overwrite=True),
        compute=True,
    )


def get_segmentation(
    config_path: str,
    tomo_id: str,
    name: str = "segmentation",
    user_id: str = "pytorch",
    session_id: str = "0",
):
    """Return the level-0 zarr array for a stored segmentation."""
    root = get_copick_root(config_path)
    run = root.get_run(tomo_id)
    segs = run.get_segmentations(name=name, user_id=user_id, session_id=session_id)
    if not segs:
        raise ValueError(f"No segmentation '{name}' for {tomo_id} (user {user_id}, session {session_id})")
    return _open_zarr_volume(segs[0].zarr())