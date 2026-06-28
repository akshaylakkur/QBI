#!/usr/bin/env python3
"""Split a predictions JSON into pick-style molecule JSON files.

Example:
  python3 format_predictions_to_picks.py \
    --predictions sample_data/predictions/predictions_TS_5_4.json \
    --output-root sample_data/formatted_picks

This writes:
  sample_data/formatted_picks/TS_5_4/apo-ferritin.json
  sample_data/formatted_picks/TS_5_4/beta-amylase.json
  ...
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path


POINT_TRANSFORM = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a predictions JSON into pick-style molecule JSON files.")
    parser.add_argument(
        "--predictions",
        default="sample_data/predictions/predictions_TS_5_4.json",
        help="Path to a predictions JSON file.",
    )
    parser.add_argument(
        "--output-root",
        default="sample_data/formatted_picks",
        help="Directory where the per-sample pick folders will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing output files to be replaced.",
    )
    return parser.parse_args()


def load_predictions(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def group_detections(detections: list[dict]) -> OrderedDict[str, list[dict]]:
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for detection in detections:
        particle_type = detection.get("particle_type")
        if not particle_type:
            continue
        grouped.setdefault(particle_type, []).append(detection)
    return grouped


def build_pick_payload(sample_name: str, molecule_name: str, detections: list[dict]) -> dict:
    points = []
    for detection in detections:
        points.append(
            {
                "location": {
                    "x": detection["x_angstrom"],
                    "y": detection["y_angstrom"],
                    "z": detection["z_angstrom"],
                },
                "transformation_": POINT_TRANSFORM,
                "instance_id": 0,
            }
        )

    return {
        "pickable_object_name": molecule_name,
        "user_id": "curation",
        "session_id": "0",
        "run_name": sample_name,
        "voxel_spacing": None,
        "unit": "angstrom",
        "points": points,
        "trust_orientation": True,
    }


def main() -> int:
    args = parse_args()
    predictions_path = Path(args.predictions).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    payload = load_predictions(predictions_path)
    sample_name = payload.get("study_name") or predictions_path.stem.replace("predictions_", "")
    detections = payload.get("detections", [])
    if not isinstance(detections, list):
        raise ValueError("Expected 'detections' to be a list")

    grouped = group_detections(detections)
    if not grouped:
        raise ValueError(f"No detections with particle_type found in {predictions_path}")

    sample_output_dir = output_root / sample_name
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    written_files = []
    for molecule_name, molecule_detections in grouped.items():
        output_path = sample_output_dir / f"{molecule_name}.json"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{output_path} already exists. Re-run with --overwrite to replace it."
            )

        pick_payload = build_pick_payload(sample_name, molecule_name, molecule_detections)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(pick_payload, handle, indent=2)
            handle.write("\n")
        written_files.append(output_path)

    print(f"Wrote {len(written_files)} molecule files to {sample_output_dir}")
    for path in written_files:
        print(f"  {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
