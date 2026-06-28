"""Optional: convert segmentations to particle coordinates (connected components).

The stored segmentation carries *contiguous model-class indices* (see
``labels.py``); this module converts them back to copick object names via
``CLASS_TO_LABEL`` / ``CLASS_TO_NAME`` for writing picks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from .settings import Config
from .utils import get_logger
from .data import copick_io
from .labels import PARTICLE_CLASSES, CLASS_TO_NAME, CLASS_TO_LABEL


def _remove_duplicates(coords: np.ndarray, threshold: float) -> np.ndarray:
    if coords.shape[0] <= 1:
        return coords
    from scipy.spatial import cKDTree

    tree = cKDTree(coords[:, :3])
    pairs = tree.query_pairs(threshold)
    drop = set()
    for a, b in pairs:
        # keep the first; mark the second for removal (lower score later)
        drop.add(b)
    keep = sorted(set(range(coords.shape[0])) - drop)
    return coords[keep]


def _euler_to_matrix(rot: float, tilt: float, psi: float) -> list:
    from scipy.spatial.transform import Rotation as R

    r = R.from_euler("zyz", [rot, tilt, psi], degrees=True).as_matrix()
    m = np.zeros((4, 4))
    m[:3, :3] = r
    m[3, 3] = 1.0
    return np.round(m, 3).tolist()


def _write_copick_picks(name, tomo_id, coords, path_output, user_id, session_id):
    """Write a copick JSON pick file (coords in Angstroms, x/y/z)."""
    pts = []
    for row in coords:
        x, y, z = float(row[0]), float(row[1]), float(row[2])
        score = float(row[3]) if row.shape[0] > 3 else 1.0
        pts.append({
            "location": {"x": x, "y": y, "z": z},
            "transformation_": _euler_to_matrix(0, 0, 0),
            "instance_id": 0,
            "score": score,
        })
    data = {
        "pickable_object_name": name,
        "user_id": user_id,
        "session_id": session_id,
        "run_name": tomo_id,
        "voxel_spacing": None,
        "unit": "angstrom",
        "trust_orientation": "false",
        "points": pts,
    }
    out_dir = Path(path_output) / tomo_id / "Picks"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{user_id}_{session_id}_{name}.json"
    with open(fname, "w") as f:
        json.dump(data, f, indent=2)


def _object_radius_voxels(config_path: str, object_name: str, voxel_size: float) -> float:
    objs = {o["name"]: o for o in copick_io.get_pickable_objects(config_path)}
    o = objs.get(object_name)
    if not o or not o.get("radius"):
        return 0.0
    return float(o["radius"]) / float(voxel_size)


def localize_segmentations(
    cfg: Config,
    tomo_ids: Optional[List[str]] = None,
    segmentation_name: Optional[str] = None,
) -> None:
    """Convert stored segmentations into particle coordinate picks.

    Iterates over *particle* model-class indices only (apo-ferritin=1 through
    virus-like-particle=6), so apo-ferritin is no longer dropped. Background
    (0) and membrane (7) are skipped automatically.
    """
    log = get_logger("localize")
    root = copick_io.get_copick_root(cfg.data.copick_config)
    if tomo_ids is None:
        tomo_ids = [r.name for r in root.runs]

    seg_name = segmentation_name or cfg.inference.segmentation_name
    out_overlay = cfg.inference.out_overlay
    for tid in tomo_ids:
        log.info(f"Localizing {tid}")
        try:
            labelmap = copick_io.get_segmentation(
                cfg.data.copick_config, tid, name=seg_name,
                user_id=cfg.inference.user_id, session_id=cfg.inference.session_id,
            )[:]
        except Exception as e:
            log.warning(f"  no segmentation '{seg_name}' for {tid}: {e}")
            continue

        # Iterate over the six particle classes (1..6).
        for cls in PARTICLE_CLASSES:
            name = CLASS_TO_NAME.get(cls)
            if name is None:
                continue
            r_vox = _object_radius_voxels(cfg.data.copick_config, name, cfg.data.voxel_size)
            if r_vox <= 0:
                continue

            lbl_objs, _ = ndimage.label(labelmap == cls)
            sizes = np.bincount(lbl_objs.ravel())
            min_size = (4 / 3) * np.pi * (r_vox ** 3) * cfg.localize.min_protein_size
            valid = np.where(sizes > min_size)[0]
            valid = valid[valid != 0]  # drop background label 0

            coords = []
            for oid in valid:
                com = ndimage.center_of_mass(lbl_objs == oid)  # (z,y,x)
                x, y, z = com[2], com[1], com[0]
                coords.append((x, y, z, 1.0))

            if not coords:
                continue
            coords = np.array(coords, dtype=np.float32)
            threshold = np.ceil(r_vox * 3)
            coords = _remove_duplicates(coords, threshold)
            # voxel -> angstrom
            coords[:, :3] *= cfg.data.voxel_size

            if cfg.localize.write_copick_picks:
                _write_copick_picks(
                    name, tid, coords, out_overlay,
                    cfg.localize.picks_user_id, cfg.localize.picks_session_id,
                )
            log.info(f"  {name}: {coords.shape[0]} picks")