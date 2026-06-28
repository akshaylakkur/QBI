#!/usr/bin/env python3
"""
Run inference on a Zarr tomogram directly (no CZII dataset structure required).

Usage:
    python backend/inference/run_inference_direct.py \
        --checkpoint czii-weights/weights_best.ckpt \
        --zarr_path /path/to/tomogram.zarr \
        --device mps \
        --dtype float32 \
        --window_size 64 64 64 \
        --tiles_per_dim 3 10 10 \
        --output predictions.json

Progress is reported via stderr lines: __QBI_PROGRESS__:<percent>
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.inference.model import (
    SegResNetForObjectDetectionV2,
    decode_detections_with_nms,
)
from backend.inference.data import (
    normalize_volume_to_unit_range,
    compute_tiles,
    CLASS_LABEL_TO_NAME,
    TARGET_SIGMAS,
    ANGSTROMS_IN_PIXEL,
)


def load_volume_from_zarr(zarr_path: str) -> np.ndarray:
    """Load a tomogram volume directly from a Zarr directory.

    Supports OME-Zarr multiscales (finds the highest-res '0' group)
    and flat Zarr arrays.
    """
    import zarr

    store = zarr.DirectoryStore(zarr_path)
    root = zarr.open(store, mode="r")

    # Try OME-Zarr multiscales structure first
    if "0" in root:
        volume = root["0"][:]
    elif hasattr(root, "shape") and len(root.shape) == 3:
        volume = root[:]
    else:
        # Try to find the first 3D array
        for key in root:
            try:
                arr = root[key]
                if hasattr(arr, "shape") and len(arr.shape) == 3:
                    volume = arr[:]
                    break
            except Exception:
                continue
        else:
            raise ValueError(f"Could not find a 3D array in {zarr_path}")

    return np.asarray(volume, dtype=np.float32)


class TileDataset(Dataset):
    """Dataset that yields tiles from a volume."""

    def __init__(
        self,
        volume: np.ndarray,
        window_size: Tuple[int, int, int],
        tiles_per_dim: Tuple[int, int, int],
        dtype: torch.dtype = torch.float32,
    ):
        self.volume = volume
        self.tiles = list(compute_tiles(volume.shape, window_size, tiles_per_dim))
        self.window_size = window_size
        self.dtype = dtype

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        tile = self.tiles[index]
        tile_volume = self.volume[tile[0], tile[1], tile[2]]

        # Pad if tile is smaller than window_size (edge tiles)
        pad_z = self.window_size[0] - tile_volume.shape[0]
        pad_y = self.window_size[1] - tile_volume.shape[1]
        pad_x = self.window_size[2] - tile_volume.shape[2]

        if pad_z > 0 or pad_y > 0 or pad_x > 0:
            tile_volume = np.pad(
                tile_volume,
                ((0, pad_z), (0, pad_y), (0, pad_x)),
                mode="constant",
                constant_values=0,
            )

        tile_offsets = (tile[0].start, tile[1].start, tile[2].start)
        return (
            torch.from_numpy(tile_volume).unsqueeze(0).to(self.dtype),
            torch.tensor(tile_offsets, dtype=torch.long),
        )


class AccumulatedPredictionContainer:
    """
    Accumulates predictions from overlapping tiles into a full-volume score/offset map.
    Uses simple averaging (or weighted averaging) of overlapping predictions.
    """

    def __init__(
        self,
        shape: Tuple[int, int, int],
        num_classes: int,
        strides: List[int],
        window_size: Tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        use_weighted_average: bool = False,
    ):
        self.strides = strides
        self.window_size = window_size
        self.use_weighted_average = use_weighted_average
        self.device = device

        d, h, w = shape
        self.scores = [
            torch.zeros((num_classes, d // s, h // s, w // s), device=device, dtype=dtype)
            for s in strides
        ]
        self.offsets = [
            torch.zeros((3, d // s, h // s, w // s), device=device, dtype=dtype)
            for s in strides
        ]
        self.counters = [
            torch.zeros((d // s, h // s, w // s), device=device, dtype=dtype)
            for s in strides
        ]

        if use_weighted_average:
            self.weight_tensors = [
                self._compute_weight_matrix(
                    torch.zeros((1, d // s, h // s, w // s), device=device)
                )
                for s in strides
            ]
        else:
            self.weight_tensors = [1.0] * len(strides)

    @staticmethod
    def _compute_weight_matrix(scores_volume: torch.Tensor, sigma: float = 15.0) -> torch.Tensor:
        """Gaussian weight matrix centered in the tile."""
        _, D, H, W = scores_volume.shape
        center = torch.tensor([D / 2, H / 2, W / 2], device=scores_volume.device)
        i = torch.arange(D, device=scores_volume.device)
        j = torch.arange(H, device=scores_volume.device)
        k = torch.arange(W, device=scores_volume.device)
        I, J, K = torch.meshgrid(i, j, k, indexing="ij")
        distances = torch.sqrt((I - center[0]) ** 2 + (J - center[1]) ** 2 + (K - center[2]) ** 2)
        weight = torch.exp(-distances / (sigma**2))
        return weight ** 3

    def accumulate(self, scores_list: List[torch.Tensor], offsets_list: List[torch.Tensor], tile_coords_zyx: Tuple[int, int, int]):
        """Accumulate predictions from one tile."""
        for i in range(len(self.scores)):
            stride = self.strides[i]
            scores = scores_list[i]
            offsets = offsets_list[i]

            strided_offset = (
                tile_coords_zyx[0] // stride,
                tile_coords_zyx[1] // stride,
                tile_coords_zyx[2] // stride,
            )

            # Compute ROI in the accumulated volume
            sz, sy, sx = strided_offset
            ez = min(sz + scores.shape[1], self.scores[i].shape[1])
            ey = min(sy + scores.shape[2], self.scores[i].shape[2])
            ex = min(sx + scores.shape[3], self.scores[i].shape[3])

            # Crop scores/offsets to fit
            scores_crop = scores[:, :ez - sz, :ey - sy, :ex - sx]
            offsets_crop = offsets[:, :ez - sz, :ey - sy, :ex - sx]

            if self.use_weighted_average:
                w = self.weight_tensors[i]
                w_crop = w[:ez - sz, :ey - sy, :ex - sx]
            else:
                w_crop = 1.0

            self.counters[i][sz:ez, sy:ey, sx:ex] += w_crop
            self.scores[i][:, sz:ez, sy:ey, sx:ex] += scores_crop * w_crop
            self.offsets[i][:, sz:ez, sy:ey, sx:ex] += offsets_crop * w_crop

    def merge(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Normalize accumulated predictions and return final scores/offsets."""
        for i in range(len(self.scores)):
            c = self.counters[i].unsqueeze(0)
            zero_mask = c.eq(0)
            self.scores[i] /= c
            self.scores[i].masked_fill_(zero_mask, 0.0)
            self.offsets[i] /= c
            self.offsets[i].masked_fill_(zero_mask, 0.0)
        return self.scores, self.offsets


def flip_volume(volume: torch.Tensor, dim: int) -> torch.Tensor:
    """Flip volume along specified dimension."""
    return volume.flip(dim)


def flip_offsets(offsets: torch.Tensor, dim: int, offset_dim: int) -> torch.Tensor:
    """Flip offsets, negating the flipped dimension's offset component."""
    offsets_flip = torch.flip(offsets, [dim]).clone()
    offsets_flip[:, offset_dim] *= -1
    return offsets_flip


@torch.no_grad()
def predict_volume(
    volume: np.ndarray,
    model: torch.nn.Module,
    output_strides: List[int],
    window_size: Tuple[int, int, int],
    tiles_per_dim: Tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 1,
    use_weighted_average: bool = False,
    use_z_flip_tta: bool = False,
    use_y_flip_tta: bool = False,
    use_x_flip_tta: bool = False,
    progress_callback=None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Run model on a full volume using tiled inference with optional TTA.

    Returns:
        scores: List of [C, D/s, H/s, W/s] tensors (one per stride)
        offsets: List of [3, D/s, H/s, W/s] tensors (one per stride)
    """
    volume = normalize_volume_to_unit_range(volume)
    ds = TileDataset(volume, window_size, tiles_per_dim, dtype=dtype)

    # Determine number of classes from a dummy forward pass
    dummy_tile = torch.zeros((1, 1, *window_size), device=device, dtype=dtype)
    logits, _ = model(dummy_tile)
    num_classes = logits[0].shape[1]

    container = AccumulatedPredictionContainer(
        shape=volume.shape,
        num_classes=num_classes,
        strides=output_strides,
        window_size=window_size,
        device=device,
        dtype=dtype,
        use_weighted_average=use_weighted_average,
    )

    loader = DataLoader(
        ds, batch_size=batch_size, num_workers=0, drop_last=False, pin_memory=True
    )

    total_batches = len(loader)
    tta_factor = 1
    if use_z_flip_tta:
        tta_factor += 1
    if use_y_flip_tta:
        tta_factor += 1
    if use_x_flip_tta:
        tta_factor += 1

    for batch_idx, (tile_volume, tile_offsets) in enumerate(loader):
        tile_volume = tile_volume.to(device=device, non_blocking=True)

        # Forward pass
        probas, offsets = model(tile_volume)

        # Accumulate
        for b in range(tile_volume.size(0)):
            scores_list = [p[b] for p in probas]
            offsets_list = [o[b] for o in offsets]
            container.accumulate(scores_list, offsets_list, tuple(tile_offsets[b].tolist()))

        # TTA: Z flip
        if use_z_flip_tta:
            vol_flip = flip_volume(tile_volume, 2)
            probas_f, offsets_f = model(vol_flip)
            probas_f = [flip_volume(p, 2) for p in probas_f]
            offsets_f = [flip_offsets(o, 2, 2) for o in offsets_f]
            for b in range(tile_volume.size(0)):
                container.accumulate(
                    [p[b] for p in probas_f], [o[b] for o in offsets_f],
                    tuple(tile_offsets[b].tolist())
                )

        # TTA: Y flip
        if use_y_flip_tta:
            vol_flip = flip_volume(tile_volume, 3)
            probas_f, offsets_f = model(vol_flip)
            probas_f = [flip_volume(p, 3) for p in probas_f]
            offsets_f = [flip_offsets(o, 3, 1) for o in offsets_f]
            for b in range(tile_volume.size(0)):
                container.accumulate(
                    [p[b] for p in probas_f], [o[b] for o in offsets_f],
                    tuple(tile_offsets[b].tolist())
                )

        # TTA: X flip
        if use_x_flip_tta:
            vol_flip = flip_volume(tile_volume, 4)
            probas_f, offsets_f = model(vol_flip)
            probas_f = [flip_volume(p, 4) for p in probas_f]
            offsets_f = [flip_offsets(o, 4, 0) for o in offsets_f]
            for b in range(tile_volume.size(0)):
                container.accumulate(
                    [p[b] for p in probas_f], [o[b] for o in offsets_f],
                    tuple(tile_offsets[b].tolist())
                )

        # Report progress with tqdm-style bar
        if progress_callback:
            batch_progress = (batch_idx + 1) / total_batches
            progress_callback(batch_progress, batch_idx + 1, total_batches)

    scores, offsets = container.merge()
    return scores, offsets


def run_inference(
    checkpoint_path: str,
    zarr_path: str,
    device: str = "cpu",
    dtype: str = "float32",
    window_size: Tuple[int, int, int] = (64, 64, 64),
    tiles_per_dim: Tuple[int, int, int] = (3, 10, 10),
    score_thresholds: Optional[List[float]] = None,
    iou_threshold: float = 0.85,
    output_path: Optional[str] = None,
    use_centernet_nms: bool = False,
    use_single_label_per_anchor: bool = True,
    pre_nms_top_k: Optional[int] = None,
    use_weighted_average: bool = False,
    use_z_flip_tta: bool = False,
    use_y_flip_tta: bool = False,
    use_x_flip_tta: bool = False,
    progress_callback=None,
) -> dict:
    """
    Run full inference pipeline on a Zarr tomogram.

    Returns:
        Result dict with detections.
    """
    # Resolve device
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU", file=sys.stderr)
        device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU", file=sys.stderr)
        device = "cpu"

    device = torch.device(device)
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32

    if progress_callback:
        progress_callback(0.01)

    print(f"Loading model from {checkpoint_path}...", file=sys.stderr)
    model = SegResNetForObjectDetectionV2.from_checkpoint(checkpoint_path, device=device)
    model = model.to(dtype=torch_dtype)
    model.eval()

    if progress_callback:
        progress_callback(0.05)

    # Determine output strides from model heads
    output_strides = []
    if hasattr(model, "head4") and model.use_stride4:
        output_strides.append(model.head4.stride)
    if hasattr(model, "head2") and model.use_stride2:
        output_strides.append(model.head2.stride)
    print(f"Output strides: {output_strides}", file=sys.stderr)

    print(f"Loading volume from {zarr_path}...", file=sys.stderr)
    volume = load_volume_from_zarr(zarr_path)
    print(f"Volume shape: {volume.shape}", file=sys.stderr)

    if progress_callback:
        progress_callback(0.1)

    # Default score thresholds
    if score_thresholds is None:
        score_thresholds = [0.265, 0.290, 0.195, 0.150, 0.550, 0.3]

    print(f"Running inference with window={window_size}, tiles={tiles_per_dim}...", file=sys.stderr)
    t0 = time.time()

    # Wrap progress_callback to map 0.1-0.85 range
    def inference_progress(p, batch=None, total=None):
        if progress_callback:
            progress_callback(0.1 + p * 0.75, batch, total)

    scores, offsets = predict_volume(
        volume=volume,
        model=model,
        output_strides=output_strides,
        window_size=window_size,
        tiles_per_dim=tiles_per_dim,
        device=device,
        dtype=torch_dtype,
        batch_size=1,
        use_weighted_average=use_weighted_average,
        use_z_flip_tta=use_z_flip_tta,
        use_y_flip_tta=use_y_flip_tta,
        use_x_flip_tta=use_x_flip_tta,
        progress_callback=inference_progress,
    )
    print(f"Inference took {time.time() - t0:.1f}s", file=sys.stderr)

    if progress_callback:
        progress_callback(0.88)

    print("Decoding detections with NMS...", file=sys.stderr)
    centers, labels, confs = decode_detections_with_nms(
        scores=scores,
        offsets=offsets,
        strides=output_strides,
        class_sigmas=TARGET_SIGMAS,
        min_score=score_thresholds,
        iou_threshold=iou_threshold,
        use_centernet_nms=use_centernet_nms,
        use_single_label_per_anchor=use_single_label_per_anchor,
        pre_nms_top_k=pre_nms_top_k,
    )

    if progress_callback:
        progress_callback(0.95)

    # Convert to finetuned-compatible format
    centers_px = centers.float().cpu().numpy()  # voxel coordinates
    centers_angstrom = centers_px * ANGSTROMS_IN_PIXEL
    labels_np = labels.cpu().numpy()
    confs_np = confs.float().cpu().numpy()

    detections = []
    for i in range(len(labels_np)):
        detections.append({
            "particle_type": CLASS_LABEL_TO_NAME[int(labels_np[i])],
            "class_label": int(labels_np[i]),
            "score": float(confs_np[i]),
            "x_angstrom": float(centers_angstrom[i, 0]),
            "y_angstrom": float(centers_angstrom[i, 1]),
            "z_angstrom": float(centers_angstrom[i, 2]),
            "x_pixel": float(centers_px[i, 0]),
            "y_pixel": float(centers_px[i, 1]),
            "z_pixel": float(centers_px[i, 2]),
        })

    result = {
        "zarr_path": zarr_path,
        "voxel_size_angstroms": ANGSTROMS_IN_PIXEL,
        "num_detections": len(detections),
        "detections": detections,
    }

    print(f"Found {result['num_detections']} particles", file=sys.stderr)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved predictions to {output_path}", file=sys.stderr)

    if progress_callback:
        progress_callback(1.0)

    return result


def main():
    parser = argparse.ArgumentParser(description="CryoET Object Detection Inference on Zarr")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt checkpoint file")
    parser.add_argument("--zarr_path", required=True, help="Path to the .zarr tomogram directory")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Device")
    parser.add_argument("--dtype", default="float32", choices=["float16", "float32"], help="Precision")
    parser.add_argument("--window_size", nargs=3, type=int, default=[64, 64, 64], help="Tile window size D H W")
    parser.add_argument("--tiles_per_dim", nargs=3, type=int, default=[3, 10, 10], help="Number of tiles per dimension")
    parser.add_argument("--score_threshold", type=float, default=0.2, help="Score threshold for detections (single value)")
    parser.add_argument("--score_thresholds", nargs="*", type=float, default=None, help="Per-class score thresholds (6 values)")
    parser.add_argument("--iou_threshold", type=float, default=0.85, help="NMS IoU threshold")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--use_centernet_nms", action="store_true", help="Use CenterNet-style NMS")
    parser.add_argument("--pre_nms_top_k", type=int, default=None, help="Top-K before NMS")
    parser.add_argument("--use_weighted_average", action="store_true", help="Use weighted average for tile blending")
    parser.add_argument("--use_z_flip_tta", action="store_true", help="Z-flip TTA")
    parser.add_argument("--use_y_flip_tta", action="store_true", help="Y-flip TTA")
    parser.add_argument("--use_x_flip_tta", action="store_true", help="X-flip TTA")

    args = parser.parse_args()

    # Expand user paths
    checkpoint_path = os.path.expanduser(args.checkpoint)
    zarr_path = os.path.expanduser(args.zarr_path)

    # Resolve score thresholds
    if args.score_thresholds is not None and len(args.score_thresholds) > 0:
        score_thresholds = args.score_thresholds
    else:
        score_thresholds = [args.score_threshold]

    # Progress callback that writes to stderr for the server to capture
    def progress_callback(progress, batch=None, total=None):
        if batch is not None and total is not None:
            bar_len = 30
            filled = int(bar_len * batch / total)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"__QBI_PROGRESS__:{progress:.3f}:{batch}/{total}:{bar}", file=sys.stderr, flush=True)
        else:
            print(f"__QBI_PROGRESS__:{progress:.3f}", file=sys.stderr, flush=True)

    run_inference(
        checkpoint_path=checkpoint_path,
        zarr_path=zarr_path,
        device=args.device,
        dtype=args.dtype,
        window_size=tuple(args.window_size),
        tiles_per_dim=tuple(args.tiles_per_dim),
        score_thresholds=score_thresholds,
        iou_threshold=args.iou_threshold,
        output_path=args.output,
        use_centernet_nms=args.use_centernet_nms,
        use_single_label_per_anchor=True,
        pre_nms_top_k=args.pre_nms_top_k,
        use_weighted_average=args.use_weighted_average,
        use_z_flip_tta=args.use_z_flip_tta,
        use_y_flip_tta=args.use_y_flip_tta,
        use_x_flip_tta=args.use_x_flip_tta,
        progress_callback=progress_callback,
    )


if __name__ == "__main__":
    main()
