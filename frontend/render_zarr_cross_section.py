#!/usr/bin/env python3
"""Render one CryoET Zarr cross-section with CZII pick overlays.

Example:
  python3 render_zarr_cross_section.py \
    --zarr "sample_data/cryo_object detection/train/static/ExperimentRuns/TS_5_4/VoxelSpacing10.000/denoised.zarr" \
    --picks "sample_data/cryo_object detection/train/overlay/ExperimentRuns/TS_5_4/Picks" \
    --level 1 \
    --axis z \
    --slice 45 \
    --output ts_5_4_level1_z45.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont


OBJECT_STYLES = {
    "apo-ferritin": {"color": (255, 196, 0), "radius_angstrom": 60},
    "beta-amylase": {"color": (0, 198, 255), "radius_angstrom": 65},
    "beta-galactosidase": {"color": (155, 99, 255), "radius_angstrom": 90},
    "ribosome": {"color": (255, 74, 74), "radius_angstrom": 150},
    "thyroglobulin": {"color": (0, 220, 135), "radius_angstrom": 130},
    "virus-like-particle": {"color": (255, 136, 0), "radius_angstrom": 135},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", required=True, help="Path to a .zarr root or to a .zarr level folder.")
    parser.add_argument("--picks", required=True, help="Path to TS_5_4/Picks containing object JSON files.")
    parser.add_argument("--level", default="1", help="Zarr level to render when --zarr points at the .zarr root.")
    parser.add_argument("--axis", choices=("x", "y", "z"), default="z", help="Cross-section axis.")
    parser.add_argument("--slice", type=int, default=None, help="Slice index at the selected level. Defaults to middle.")
    parser.add_argument("--output", default="cross_section_overlay.png", help="Output PNG path.")
    parser.add_argument("--window-low", type=float, default=1.0, help="Low percentile for contrast window.")
    parser.add_argument("--window-high", type=float, default=99.0, help="High percentile for contrast window.")
    parser.add_argument(
        "--scan-color",
        choices=("warm", "cyan", "gray"),
        default="warm",
        help="Color treatment for the underlying tomogram slice.",
    )
    parser.add_argument("--labels", action="store_true", help="Draw object names next to visible overlays.")
    parser.add_argument("--marker-outline", action="store_true", help="Draw ring markers on top of the colorized scan.")
    parser.add_argument("--flip-x", action="store_true", help="Flip the rendered overlay horizontally for orientation checks.")
    parser.add_argument("--flip-y", action="store_true", help="Flip the rendered overlay vertically for orientation checks.")
    parser.add_argument("--no-overlay", action="store_true", help="Render only the grayscale slice.")
    return parser.parse_args()


def zarr_level_path(zarr_path: Path, level: str) -> Path:
    if (zarr_path / ".zarray").exists() or (zarr_path / "zarr.json").exists():
        return zarr_path

    level_path = zarr_path / str(level)
    if (level_path / ".zarray").exists() or (level_path / "zarr.json").exists():
        return level_path

    raise FileNotFoundError(
        f"Could not find Zarr array metadata at {zarr_path} or {level_path}. "
        "Point --zarr at a complete .zarr folder, not a single chunk file."
    )


def load_level_scale(zarr_root: Path, level_path: Path, level: str) -> tuple[float, float, float]:
    attrs_path = zarr_root / ".zattrs"
    if not attrs_path.exists():
        return (1.0, 1.0, 1.0)

    attrs = json.loads(attrs_path.read_text())
    for dataset in attrs.get("multiscales", [{}])[0].get("datasets", []):
        if str(dataset.get("path")) != str(level):
            continue
        for transform in dataset.get("coordinateTransformations", []):
            if transform.get("type") == "scale":
                scale = transform.get("scale", [1.0, 1.0, 1.0])
                return (float(scale[0]), float(scale[1]), float(scale[2]))

    level_metadata = level_path / ".zarray"
    root_metadata = zarr_root / "0" / ".zarray"
    if level_metadata.exists() and root_metadata.exists():
        level_shape = json.loads(level_metadata.read_text())["shape"]
        root_shape = json.loads(root_metadata.read_text())["shape"]
        return tuple(float(root_shape[i]) / float(level_shape[i]) for i in range(3))

    return (1.0, 1.0, 1.0)


def read_slice(array: zarr.Array, axis: str, index: int) -> np.ndarray:
    if axis == "z":
        return np.asarray(array[index, :, :])
    if axis == "y":
        return np.asarray(array[:, index, :])
    return np.asarray(array[:, :, index])


def normalize_to_u8(values: np.ndarray, low_percentile: float, high_percentile: float) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)

    low, high = np.percentile(finite, [low_percentile, high_percentile])
    if high <= low:
        high = low + 1.0

    clipped = np.clip(values, low, high)
    normalized = (clipped - low) / (high - low)
    return np.round(normalized * 255).astype(np.uint8)


def colorize_scan(values: np.ndarray, mode: str) -> Image.Image:
    if mode == "gray":
        return Image.fromarray(values, mode="L").convert("RGBA")

    normalized = values.astype(np.float32) / 255.0
    if mode == "cyan":
        red = np.round(18 + normalized * 188).astype(np.uint8)
        green = np.round(36 + normalized * 206).astype(np.uint8)
        blue = np.round(54 + normalized * 196).astype(np.uint8)
    else:
        red = np.round(36 + normalized * 219).astype(np.uint8)
        green = np.round(32 + normalized * 197).astype(np.uint8)
        blue = np.round(42 + normalized * 142).astype(np.uint8)

    alpha = np.full(values.shape, 255, dtype=np.uint8)
    rgba = np.dstack([red, green, blue, alpha])
    return Image.fromarray(rgba, mode="RGBA")


def load_picks(picks_dir: Path) -> list[dict]:
    picks = []
    for path in sorted(picks_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        object_name = payload.get("pickable_object_name") or path.stem
        style = OBJECT_STYLES.get(object_name, {"color": (255, 255, 255), "radius_angstrom": 80})
        for point in payload.get("points", []):
            location = point.get("location", {})
            picks.append({
                "object_name": object_name,
                "x": float(location["x"]),
                "y": float(location["y"]),
                "z": float(location["z"]),
                "color": style["color"],
                "radius_angstrom": style["radius_angstrom"],
            })
    return picks


def project_pick(
    pick: dict,
    axis: str,
    scale_zyx: tuple[float, float, float],
    image_size: tuple[int, int],
    flip_x: bool,
    flip_y: bool,
) -> tuple[float, float, float]:
    z = pick["z"] / scale_zyx[0]
    y = pick["y"] / scale_zyx[1]
    x = pick["x"] / scale_zyx[2]
    if axis == "z":
        px, py, paxis = x, y, z
    elif axis == "y":
        px, py, paxis = x, z, y
    else:
        px, py, paxis = y, z, x

    width, height = image_size
    if flip_x:
        px = (width - 1) - px
    if flip_y:
        py = (height - 1) - py
    return px, py, paxis


def draw_overlay(
    image: Image.Image,
    picks: list[dict],
    axis: str,
    slice_index: int,
    scale_zyx: tuple[float, float, float],
    labels: bool,
    marker_outline: bool,
    flip_x: bool,
    flip_y: bool,
) -> int:
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    visible_count = 0

    for pick in picks:
        px, py, paxis = project_pick(pick, axis, scale_zyx, image.size, flip_x, flip_y)
        axis_scale = {"z": scale_zyx[0], "y": scale_zyx[1], "x": scale_zyx[2]}[axis]
        radius = max(2.0, pick["radius_angstrom"] / axis_scale)
        distance = abs(paxis - slice_index)
        if distance > radius:
            continue

        visible_count += 1
        slice_radius = max(2.0, math.sqrt(max(0.0, radius * radius - distance * distance)))
        opacity = 1.0 - (distance / radius)
        alpha = int(56 + 126 * opacity)
        ring = pick["color"] + (230,)
        fill = pick["color"] + (alpha,)
        halo = pick["color"] + (58,)
        bbox = [px - slice_radius, py - slice_radius, px + slice_radius, py + slice_radius]
        halo_radius = slice_radius + 5
        halo_bbox = [px - halo_radius, py - halo_radius, px + halo_radius, py + halo_radius]
        draw.ellipse(halo_bbox, fill=halo)
        draw.ellipse(bbox, fill=fill)
        if marker_outline:
            draw.ellipse(bbox, outline=ring, width=2)
            draw.ellipse([px - 1.5, py - 1.5, px + 1.5, py + 1.5], fill=(255, 255, 255, 235))
        if labels:
            draw.text((px + slice_radius + 4, py - 6), pick["object_name"], fill=ring, font=font)

    return visible_count


def add_legend(image: Image.Image) -> Image.Image:
    legend_items = list(OBJECT_STYLES.items())
    row_height = 16
    padding = 8
    legend_width = 190
    legend_height = padding * 2 + row_height * len(legend_items)
    output = Image.new("RGB", (image.width + legend_width, image.height), (12, 14, 16))
    output.paste(image.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    x0 = image.width + padding
    y = padding
    for name, style in legend_items:
        color = style["color"]
        draw.ellipse([x0, y + 2, x0 + 10, y + 12], fill=color)
        draw.text((x0 + 16, y + 2), name, fill=(230, 236, 240), font=font)
        y += row_height
    return output


def add_caption(image: Image.Image, text: str) -> Image.Image:
    caption_height = 28
    output = Image.new("RGB", (image.width, image.height + caption_height), (12, 14, 16))
    output.paste(image.convert("RGB"), (0, caption_height))
    draw = ImageDraw.Draw(output)
    draw.text((8, 8), text, fill=(230, 236, 240), font=ImageFont.load_default())
    return output


def main() -> None:
    args = parse_args()
    input_path = Path(args.zarr)
    zarr_root = input_path if input_path.suffix == ".zarr" else input_path.parent
    level_path = zarr_level_path(input_path, str(args.level))
    array = zarr.open(str(level_path), mode="r")

    axis_index = {"z": 0, "y": 1, "x": 2}[args.axis]
    slice_index = args.slice if args.slice is not None else array.shape[axis_index] // 2
    slice_index = max(0, min(array.shape[axis_index] - 1, slice_index))

    image_u8 = normalize_to_u8(read_slice(array, args.axis, slice_index), args.window_low, args.window_high)
    image = colorize_scan(image_u8, args.scan_color)

    scale_zyx = load_level_scale(zarr_root, level_path, str(args.level))
    visible_count = 0
    if not args.no_overlay:
        visible_count = draw_overlay(
            image,
            load_picks(Path(args.picks)),
            args.axis,
            slice_index,
            scale_zyx,
            args.labels,
            args.marker_outline,
            args.flip_x,
            args.flip_y,
        )
        image = add_legend(image)

    caption = (
        f"{level_path} | shape z,y,x={array.shape} | axis={args.axis} | slice={slice_index} | "
        f"scale z,y,x={scale_zyx} | visible picks={visible_count}"
    )
    output = add_caption(image, caption)
    output.save(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
