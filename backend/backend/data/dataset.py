"""PyTorch Datasets for CryoET 3D patch sampling (class-balanced bootstrap)."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

from ..labels import LABEL_TO_CLASS, PARTICLE_CLASSES
from . import copick_io
from .augment import Augment3D


# ---------------------------------------------------------------------------
# Pick indexing across runs
# ---------------------------------------------------------------------------
def index_picks(
    config_path: str,
    tomo_ids: List[str],
    targets: Optional[Dict[str, dict]] = None,
    voxel_size: float = 10.0,
) -> List[Tuple[str, float, float, float, int]]:
    """Flatten all picks across runs into a list of (tomo_id, x, y, z, class).

    Coordinates are in VOXEL units (Angstrom / voxel_size). The returned class
    index is the *contiguous model-class index* (copick label remapped via
    ``LABEL_TO_CLASS``), so it is always in ``0..n_class-1``.
    """
    root = copick_io.get_copick_root(config_path)
    if targets is None:
        targets = {}
        for obj in root.pickable_objects:
            if obj.is_particle:
                targets[obj.name] = {"user_id": None, "session_id": None}

    flat = []
    for tomo_id in tomo_ids:
        for name, info in targets.items():
            obj = root.get_object(name)
            cls = LABEL_TO_CLASS.get(obj.label, 0)  # model-class index
            coords = copick_io.get_picks(
                config_path, tomo_id, name,
                user_id=info.get("user_id"),
                session_id=info.get("session_id"),
            )
            if coords.shape[0] == 0:
                continue
            vcoords = coords / voxel_size  # (N,3) x,y,z in voxels
            for (x, y, z) in vcoords:
                flat.append((tomo_id, float(x), float(y), float(z), int(cls)))
    return flat


def get_class_counts(picks: List[Tuple[str, float, float, float, int]]) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    for *_p, label in picks:
        counts[label] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Patch position sampler
# ---------------------------------------------------------------------------
def patch_position(
    tomo_shape: Tuple[int, int, int],
    p_in: int,
    l_rnd: int,
    cx: float,
    cy: float,
    cz: float,
) -> Tuple[int, int, int]:
    """Compute integer (x, y, z) patch center with random shift, clamped to bounds."""
    x = int(cx) + np.random.randint(-l_rnd, l_rnd + 1)
    y = int(cy) + np.random.randint(-l_rnd, l_rnd + 1)
    z = int(cz) + np.random.randint(-l_rnd, l_rnd + 1)
    # tomo_shape is (Z, Y, X)
    x = min(max(x, p_in), tomo_shape[2] - p_in)
    y = min(max(y, p_in), tomo_shape[1] - p_in)
    z = min(max(z, p_in), tomo_shape[0] - p_in)
    return x, y, z


# ---------------------------------------------------------------------------
# Tomogram pool (in-memory subset, swappable)
# ---------------------------------------------------------------------------
class TomogramPool:
    """Holds a subset of tomograms + targets resident in RAM."""

    def __init__(
        self,
        config_path: str,
        target_name: str,
        target_user_id: str,
        target_session_id: str,
        voxel_size: float = 10.0,
        tomo_algorithm: str = "denoised",
    ):
        self.config_path = config_path
        self.target_name = target_name
        self.target_user_id = target_user_id
        self.target_session_id = target_session_id
        self.voxel_size = voxel_size
        self.tomo_algorithm = tomo_algorithm
        self.data: Dict[str, np.ndarray] = {}   # tomo_id -> denoised vol (np.float32)
        self.targets: Dict[str, np.ndarray] = {}  # tomo_id -> uint8 labels

    def load(self, tomo_ids: List[str]) -> None:
        self.data.clear()
        self.targets.clear()
        for tid in tomo_ids:
            tomo = copick_io.get_tomogram(
                self.config_path, tid, self.voxel_size, self.tomo_algorithm
            )[:]
            self.data[tid] = tomo.astype(np.float32)
            seg = copick_io.get_segmentation(
                self.config_path, tid,
                name=self.target_name,
                user_id=self.target_user_id,
                session_id=self.target_session_id,
            )[:]
            self.targets[tid] = seg.astype(np.uint8)

    def has(self, tomo_id: str) -> bool:
        return tomo_id in self.data

    def shape(self, tomo_id: str) -> Tuple[int, int, int]:
        return self.data[tomo_id].shape

    def get_patch(
        self, tomo_id: str, x: int, y: int, z: int, p_in: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        p = p_in
        data = self.data[tomo_id][z - p:z + p, y - p:y + p, x - p:x + p]
        tgt = self.targets[tomo_id][z - p:z + p, y - p:y + p, x - p:x + p]
        return data.copy(), tgt.copy()


# ---------------------------------------------------------------------------
# Iterable dataset
# ---------------------------------------------------------------------------
class CryoETPatchDataset(IterableDataset):
    """Yields a fixed number of (image, label) patch batches per epoch.

    - Class-balanced bootstrap: each step picks `batch_size` picks, balanced
      across foreground classes, with a fraction of pure-background patches.
    - Tomogram pool swaps every `n_sub_epoch` (handled externally by calling
      `set_pool`); this class just samples from the current pool.
    """

    def __init__(
        self,
        config_path: str,
        tomo_ids: List[str],
        dim_in: int,
        batch_size: int,
        steps: int,
        l_rnd: int = 15,
        background_ratio: float = 0.30,
        voxel_size: float = 10.0,
        tomo_algorithm: str = "denoised",
        target_name: str = "pytargets",
        target_user_id: str = "pytorch",
        target_session_id: str = "0",
        targets: Optional[Dict[str, dict]] = None,
        augment: Optional[Augment3D] = None,
        pool: Optional[TomogramPool] = None,
        n_sub_epoch: int = 10,
        sample_size: int = 5,
        seed: int = 42,
    ):
        super().__init__()
        self.config_path = config_path
        self.tomo_ids = list(tomo_ids)
        self.dim_in = dim_in
        self.p_in = dim_in // 2
        self.batch_size = batch_size
        self.steps = steps
        self.l_rnd = l_rnd
        self.background_ratio = background_ratio
        self.voxel_size = voxel_size
        self.tomo_algorithm = tomo_algorithm
        self.target_name = target_name
        self.target_user_id = target_user_id
        self.target_session_id = target_session_id
        self.targets = targets
        self.augment = augment
        self.n_sub_epoch = n_sub_epoch
        self.sample_size = sample_size
        self.seed = seed
        self._rng = random.Random(seed)

        # Index picks for all provided tomo_ids (class indices already remapped)
        self.picks = index_picks(config_path, tomo_ids, targets, voxel_size)
        # group by class index
        self.by_class: Dict[int, List[int]] = defaultdict(list)
        for i, (*_p, label) in enumerate(self.picks):
            self.by_class[label].append(i)
        self.class_labels = sorted(self.by_class.keys())
        if not self.class_labels:
            raise ValueError("No picks found for the given tomo_ids/targets.")

        # Per-tomogram pick coordinate arrays (voxel units) for background
        # rejection. Keyed by tomo_id -> (N,3) float array of (x,y,z).
        self._picks_by_tomo: Dict[str, np.ndarray] = defaultdict(list)
        for (t, x, y, z, _cls) in self.picks:
            self._picks_by_tomo[t].append((x, y, z))
        self._picks_by_tomo = {
            t: np.asarray(v, dtype=np.float32) for t, v in self._picks_by_tomo.items()
        }
        # Max particle radius in voxels (used as exclusion distance for bg
        # sampling). Default 15 voxels (150 A / 10 A) if unknown.
        self._bg_exclude = max(15.0, self._max_radius_voxels(config_path, voxel_size))

        # Tomogram pool
        self.pool = pool or TomogramPool(
            config_path, target_name, target_user_id, target_session_id,
            voxel_size, tomo_algorithm,
        )
        self._pool_loaded_ids: List[str] = []
        self._swap_counter = 0

    # -- pool management -------------------------------------------------
    def _ensure_pool(self) -> None:
        """Load a fresh random subset of tomograms if pool empty/stale."""
        needed = set(t for (t, *_p) in self.picks)
        if self._pool_loaded_ids and self._swap_counter < self.n_sub_epoch:
            # still in current subset window; ensure needed tomos loaded
            missing = [t for t in self._pool_loaded_ids if not self.pool.has(t)]
            if not missing:
                return
        # pick a new random subset that has picks
        candidates = [t for t in self.tomo_ids if t in needed]
        if not candidates:
            candidates = list(self.tomo_ids)
        k = min(self.sample_size, len(candidates))
        subset = self._rng.sample(candidates, k)
        self.pool.load(subset)
        self._pool_loaded_ids = subset
        self._swap_counter = 0

    def set_pool(self, pool: TomogramPool) -> None:
        self.pool = pool

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _max_radius_voxels(config_path: str, voxel_size: float) -> float:
        """Largest particle radius in voxels across pickable objects."""
        try:
            objs = copick_io.get_pickable_objects(config_path)
        except Exception:
            return 15.0
        r = 0.0
        for o in objs:
            if o.get("is_particle") and o.get("radius"):
                r = max(r, float(o["radius"]) / float(voxel_size))
        return r or 15.0

    # -- sampling --------------------------------------------------------
    def _sample_indices(self) -> List[int]:
        n_bg = int(round(self.batch_size * self.background_ratio))
        n_fg = self.batch_size - n_bg
        chosen: List[int] = []
        if n_fg > 0:
            # cycle through foreground particle classes for balance
            fg_labels = [l for l in self.class_labels if l in PARTICLE_CLASSES]
            if not fg_labels:
                fg_labels = list(self.class_labels)
            for i in range(n_fg):
                lbl = fg_labels[i % len(fg_labels)]
                pool = self.by_class[lbl]
                chosen.append(self._rng.choice(pool))
        return chosen

    def _sample_background_index(self) -> Tuple[str, float, float, float]:
        """Random background location inside a loaded tomo, avoiding picks.

        Rejects candidates within ``_bg_exclude`` voxels (plus ``l_rnd`` jitter)
        of any particle pick so that "background" patches actually contain
        background rather than accidentally overlapping a particle.
        """
        tid = self._rng.choice(self._pool_loaded_ids)
        shape = self.pool.shape(tid)
        p = self.p_in
        picks = self._picks_by_tomo.get(tid)
        excl = self._bg_exclude + self.l_rnd
        for _ in range(16):  # try up to 16 times to find a clean location
            x = self._rng.randint(p, shape[2] - p)
            y = self._rng.randint(p, shape[1] - p)
            z = self._rng.randint(p, shape[0] - p)
            if picks is None or picks.shape[0] == 0:
                return tid, float(x), float(y), float(z)
            d = np.sqrt(((picks - np.array([x, y, z], dtype=np.float32)) ** 2).sum(1))
            if d.min() > excl:
                return tid, float(x), float(y), float(z)
        # fall back to the last candidate if we couldn't find a clean one
        return tid, float(x), float(y), float(z)

    # -- iteration -------------------------------------------------------
    def __iter__(self):
        self._ensure_pool()
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        rng = random.Random(self.seed + worker_id)
        steps_for_worker = self.steps // num_workers + (1 if worker_id < self.steps % num_workers else 0)

        for _ in range(steps_for_worker):
            self._swap_counter += 1
            if self._swap_counter >= self.n_sub_epoch:
                self._ensure_pool()

            imgs = []
            lbls = []
            fg_idxs = self._sample_indices()
            n_bg = self.batch_size - len(fg_idxs)

            for idx in fg_idxs:
                tid, cx, cy, cz, label = self.picks[idx]
                if not self.pool.has(tid):
                    # fallback: load this tomo if pool missing it
                    self.pool.load([tid])
                    self._pool_loaded_ids = list(set(self._pool_loaded_ids + [tid]))
                shape = self.pool.shape(tid)
                x, y, z = patch_position(shape, self.p_in, self.l_rnd, cx, cy, cz)
                data, tgt = self.pool.get_patch(tid, x, y, z, self.p_in)
                # normalize per-patch
                data = (data - data.mean()) / (data.std() + 1e-8)
                imgs.append(data)
                lbls.append(tgt)

            # background patches
            for _ in range(n_bg):
                tid, cx, cy, cz = self._sample_background_index()
                shape = self.pool.shape(tid)
                x, y, z = patch_position(shape, self.p_in, self.l_rnd, cx, cy, cz)
                data, tgt = self.pool.get_patch(tid, x, y, z, self.p_in)
                data = (data - data.mean()) / (data.std() + 1e-8)
                imgs.append(data)
                lbls.append(tgt)

            # shuffle batch so background isn't always last
            order = list(range(len(imgs)))
            rng.shuffle(order)
            imgs = [imgs[i] for i in order]
            lbls = [lbls[i] for i in order]

            img_t = torch.from_numpy(np.stack(imgs)).float().unsqueeze(1)  # (B,1,D,D,D)
            lbl_t = torch.from_numpy(np.stack(lbls)).long()                 # (B,D,D,D)

            if self.augment is not None:
                img_t, lbl_t = self.augment(img_t, lbl_t)

            yield img_t, lbl_t