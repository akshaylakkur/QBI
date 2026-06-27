"""Sliding-window 3D inference producing per-class scoremaps + labelmap."""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import torch

from .settings import Config
from .utils import get_logger
from .models import build_model
from .utils.checkpoint import load_checkpoint
from .data import copick_io


def _cosine_window_3d(d: int, device, dtype) -> torch.Tensor:
    """3D cosine (Hann) window for smooth overlap blending. Returns (d,d,d)."""
    k1 = torch.sin(torch.linspace(0, np.pi, d, device=device, dtype=dtype)) ** 2
    # separable 3D window = outer product along each axis
    w = torch.einsum("i,j,k->ijk", k1, k1, k1)
    return (w / w.max())


@torch.no_grad()
def segment_volume(
    model: torch.nn.Module,
    volume: np.ndarray,  # (Z,Y,X)
    patch_size: int,
    overlap: int,
    pcrop: int,
    n_class: int,
    batch_patches: int = 4,
    amp: bool = True,
    device: torch.device | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment a full tomogram with sliding-window inference.

    Returns:
        labelmap: (Z,Y,X) uint8 argmax classes
        scoremap: (C,Z,Y,X) float32 softmax probabilities
    """
    log = get_logger("inference")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    P = patch_size
    p = P // 2
    lcrop = p - pcrop
    step = P - overlap

    # Normalize and pad
    vol = (volume.astype(np.float32) - volume.mean()) / (volume.std() + 1e-8)
    vol = np.pad(vol, pcrop, mode="constant", constant_values=0)
    dim = vol.shape  # (Z,Y,X) padded

    # patch centers along each axis
    def centers(d):
        c = list(range(p, d - p, step))
        if not c:
            c = [p]
        if c[-1] < d - p:
            c.append(d - p)
        return c

    cz = centers(dim[0])
    cy = centers(dim[1])
    cx = centers(dim[2])
    Npatch = len(cz) * len(cy) * len(cx)
    log.info(f"Segmenting {Npatch} patches (P={P}, overlap={overlap}, pcrop={pcrop})...")

    pred = np.zeros((n_class,) + dim, dtype=np.float32)
    norm = np.zeros(dim, dtype=np.float32)
    window = _cosine_window_3d(P - 2 * pcrop, device, torch.float32).cpu().numpy()

    patches = []
    coords = []
    count = 0
    t0 = time.time()

    for z in cz:
        for y in cy:
            for x in cx:
                patch = vol[z - p:z + p, y - p:y + p, x - p:x + p]
                patches.append(patch)
                coords.append((z, y, x))
                if len(patches) == batch_patches:
                    _flush(model, patches, coords, pred, norm, window, P, p, lcrop,
                           pcrop, n_class, amp, device)
                    patches.clear()
                    coords.clear()
                count += 1
                if count % 20 == 0:
                    log.info(f"  patch {count}/{Npatch} ({(time.time()-t0):.0f}s)")

    if patches:
        _flush(model, patches, coords, pred, norm, window, P, p, lcrop,
               pcrop, n_class, amp, device)

    # normalize overlaps
    norm[norm == 0] = 1.0
    pred = pred / norm[None]
    # unpad
    pred = pred[:, pcrop:-pcrop, pcrop:-pcrop, pcrop:-pcrop]
    labelmap = np.argmax(pred, axis=0).astype(np.uint8)
    log.info(f"Segmentation done in {time.time()-t0:.0f}s")
    return labelmap, pred


def _flush(model, patches, coords, pred, norm, window, P, p, lcrop, pcrop,
           n_class, amp, device):
    batch = np.stack(patches)[:, None]  # (B,1,P,P,P)
    t = torch.from_numpy(batch).float().to(device)
    with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
        out = model(t)
    probs = torch.softmax(out.float(), dim=1).cpu().numpy()
    win = window[None]  # (1,winD,winD,winD) for broadcasting over classes
    d = P - 2 * pcrop  # valid region side length
    for i, (z, y, x) in enumerate(coords):
        # central valid region of the prediction (crop pcrop from each side)
        p = probs[i, :, pcrop:pcrop + d, pcrop:pcrop + d, pcrop:pcrop + d]
        # valid region starts at lcrop within the padded volume
        z0, y0, x0 = z - lcrop, y - lcrop, x - lcrop
        pred[:, z0:z0 + d, y0:y0 + d, x0:x0 + d] += p * win
        norm[z0:z0 + d, y0:y0 + d, x0:x0 + d] += win[0]


def run_inference(cfg: Config, weights_path: str, tomo_ids=None) -> None:
    """Segment all (or given) tomograms and write labelmaps (+ optional scoremaps)."""
    log = get_logger("inference")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        cfg.model.name, in_channels=cfg.model.in_channels,
        n_class=cfg.model.n_class, filters=cfg.model.filters,
        dropout=cfg.model.dropout,
    )
    load_checkpoint(weights_path, model, map_location=device)
    model = model.to(device).eval()

    if tomo_ids is None:
        tomo_ids = copick_io.list_runs(cfg.data.copick_config)
    log.info(f"Segmenting {len(tomo_ids)} tomograms: {tomo_ids}")

    for tid in tomo_ids:
        log.info(f"=== {tid} ===")
        tomo = copick_io.get_tomogram(
            cfg.data.copick_config, tid, cfg.data.voxel_size, cfg.data.tomo_algorithm
        )[:]
        labelmap, scoremap = segment_volume(
            model, tomo,
            patch_size=cfg.inference.patch_size,
            overlap=cfg.inference.overlap,
            pcrop=cfg.inference.pcrop,
            n_class=cfg.model.n_class,
            batch_patches=cfg.inference.batch_patches,
            amp=cfg.inference.amp,
            device=device,
        )
        copick_io.write_ome_zarr_segmentation(
            cfg.data.copick_config, tid, labelmap, cfg.data.voxel_size,
            name=cfg.inference.segmentation_name,
            user_id=cfg.inference.user_id,
            session_id=cfg.inference.session_id,
        )
        if cfg.inference.write_scoremap:
            copick_io.write_ome_zarr_scoremap(
                cfg.data.copick_config, tid, scoremap, cfg.data.voxel_size,
                name=cfg.inference.scoremap_name,
                user_id=cfg.inference.user_id,
                session_id=cfg.inference.session_id,
                tomo_type=cfg.data.tomo_algorithm,
            )