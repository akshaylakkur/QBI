"""JSON API for frontend graph / GVI / crowding analysis.

Reads a JSON payload on stdin, writes JSON to stdout.
Progress lines on stderr: __QBI_PROGRESS__:<0-1>:<message>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from downstream.constants import RADIUS_ANGSTROM
from downstream.geometry.exposure import steric_exposure_batch, steric_exposure_one
from downstream.gnn.features import EDGE_DIM, FeatureStats, build_edges
from downstream.gnn.graph import build_graph_from_arrays
from downstream.gnn.model import ExposureGNN
from downstream.qa.grid_viability import GridViabilityConfig, assess_cloud
from downstream.types import ExposureConfig, ParticleCloud

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[3] / "runs" / "exposure_gnn.pt"
DEFAULT_N_RAYS = 2000


def _emit_progress(fraction: float, message: str) -> None:
    print(f"__QBI_PROGRESS__:{fraction:.4f}:{message}", file=sys.stderr, flush=True)


def _radius_for_detection(molecule: str, radius_angstrom: Optional[float]) -> float:
    if radius_angstrom is not None and radius_angstrom > 0:
        return float(radius_angstrom)
    key = (molecule or "").strip()
    if key in RADIUS_ANGSTROM:
        return RADIUS_ANGSTROM[key]
    return 90.0


def cloud_from_detections(tomo_id: str, detections: List[dict]) -> ParticleCloud:
    types: list[str] = []
    coords: list[tuple[float, float, float]] = []
    radii: list[float] = []
    ids: list[str] = []

    for det in detections:
        phys = det.get("physical") or {}
        x, y, z = float(phys.get("x", 0)), float(phys.get("y", 0)), float(phys.get("z", 0))
        if not all(np.isfinite(v) for v in (x, y, z)):
            continue
        molecule = det.get("molecule") or det.get("type") or "unknown"
        types.append(str(molecule))
        coords.append((x, y, z))
        radii.append(_radius_for_detection(molecule, det.get("radiusAngstrom")))
        ids.append(str(det.get("id", len(ids))))

    if not coords:
        raise ValueError("No valid pick coordinates in payload")

    cloud = ParticleCloud(
        tomo_id=tomo_id,
        types=types,
        coords=np.asarray(coords, dtype=np.float64),
        radii=np.asarray(radii, dtype=np.float64),
    )
    cloud.pick_ids = ids  # type: ignore[attr-defined]
    return cloud


def _pick_index(cloud: ParticleCloud, pick_id: str) -> int:
    pick_ids: List[str] = list(getattr(cloud, "pick_ids", []))
    if pick_id not in pick_ids:
        raise ValueError(f"Pick id {pick_id!r} not found in cloud")
    return pick_ids.index(pick_id)


def _open_direction_dict(vec: np.ndarray) -> dict:
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    return {"x": round(float(v[0]), 6), "y": round(float(v[1]), 6), "z": round(float(v[2]), 6)}


def _exposure_fields(res) -> dict:
    return {
        "stericExposure": round(float(res.steric_exposure), 4),
        "anisotropyIndex": round(float(res.anisotropy_index), 4),
        "cleanExtractionScore": round(float(res.clean_extraction_score), 4),
        "openDirection": _open_direction_dict(res.open_direction),
        "cleanConeHalfAngleDeg": round(float(res.clean_cone_half_angle_deg), 2),
        "nOpenComponents": int(res.n_open_components),
    }


def run_gvi(cloud: ParticleCloud, mean_exposure: Optional[float] = None) -> dict:
    config = GridViabilityConfig()
    result = assess_cloud(cloud, config=config, reference=None, mean_exposure=mean_exposure)
    return result.to_dict()


def run_hemisphere(
    cloud: ParticleCloud,
    pick_id: str,
    n_rays: int = DEFAULT_N_RAYS,
) -> dict:
    idx = _pick_index(cloud, pick_id)
    config = ExposureConfig(n_rays=n_rays)
    res = steric_exposure_one(cloud, idx, config=config)
    directions = res.directions
    blocked = res.blocked
    if directions is None or blocked is None:
        raise ValueError("Hemisphere data unavailable for pick")

    return {
        "pickId": pick_id,
        "directions": directions.astype(float).tolist(),
        "blocked": blocked.astype(bool).tolist(),
        **_exposure_fields(res),
    }


def _knn_neighbor_ids(
    cloud: ParticleCloud,
    edge_cutoff: float = 500.0,
    knn_k: int = 12,
) -> Dict[str, List[dict]]:
    coords = cloud.coords
    radii = cloud.radii
    n = len(coords)
    pick_ids: List[str] = list(getattr(cloud, "pick_ids", [str(i) for i in range(n)]))

    edge_index, _edge_attr = build_edges(coords, radii, edge_cutoff=edge_cutoff, knn_k=knn_k)

    neighbors: Dict[int, List[tuple[int, float]]] = {i: [] for i in range(n)}
    if edge_index.shape[1] > 0:
        for e in range(edge_index.shape[1]):
            i, j = int(edge_index[0, e]), int(edge_index[1, e])
            dist = float(np.linalg.norm(coords[j] - coords[i]))
            neighbors[i].append((j, dist))

    out: Dict[str, List[dict]] = {}
    for i in range(n):
        seen: set[int] = set()
        nbrs: List[dict] = []
        for j, dist in sorted(neighbors[i], key=lambda t: t[1]):
            if j in seen or j == i:
                continue
            seen.add(j)
            nbrs.append(
                {
                    "id": pick_ids[j],
                    "molecule": cloud.types[j],
                    "distanceAngstrom": round(dist, 2),
                    "radiusAngstrom": round(float(radii[j]), 2),
                    "physical": {
                        "x": float(coords[j, 0]),
                        "y": float(coords[j, 1]),
                        "z": float(coords[j, 2]),
                    },
                }
            )
            if len(nbrs) >= knn_k:
                break
        out[pick_ids[i]] = nbrs
    return out


def run_crowding(
    cloud: ParticleCloud,
    checkpoint_path: Path,
    n_rays: int = DEFAULT_N_RAYS,
) -> dict:
    _emit_progress(0.05, f"Computing steric exposure ({n_rays} rays)…")
    config = ExposureConfig(n_rays=n_rays)
    indices = np.arange(cloud.n_particles, dtype=np.int64)
    exposure_results = steric_exposure_batch(cloud, config=config, target_indices=indices)
    exposure = np.array([r.steric_exposure for r in exposure_results], dtype=np.float32)
    clean_scores = np.array([r.clean_extraction_score for r in exposure_results], dtype=np.float32)
    neighbor_counts = np.array(
        [r.n_neighbors_considered for r in exposure_results], dtype=np.int32,
    )

    pick_ids: List[str] = list(getattr(cloud, "pick_ids", []))
    mean_exposure = float(exposure.mean()) if exposure.size else None
    mean_clean = float(clean_scores.mean()) if clean_scores.size else None

    _emit_progress(0.45, "Running GNN exposure prediction…")
    gnn_exposure = exposure.copy()
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.exists():
        import torch

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        stats = FeatureStats.from_dict(ckpt["feature_stats"])
        model = ExposureGNN(
            in_channels=ckpt["in_channels"],
            hidden_dim=cfg.get("hidden_dim", 64),
            out_channels=cfg.get("out_channels", 32),
            dropout=cfg.get("dropout", 0.25),
            heads=cfg.get("heads", 2),
            edge_dim=ckpt.get("edge_dim", EDGE_DIM),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        edge_cutoff = cfg.get("edge_cutoff", 500.0)
        knn_k = cfg.get("knn_k", 12)
        graph = build_graph_from_arrays(
            cloud.tomo_id,
            cloud.types,
            cloud.coords,
            cloud.radii,
            exposure,
            edge_cutoff=edge_cutoff,
            knn_k=knn_k,
            feature_stats=stats,
        )
        with torch.no_grad():
            gnn_exposure = model(graph.x, graph.edge_index, graph.edge_attr).numpy().ravel()
    else:
        _emit_progress(0.45, f"GNN checkpoint not found at {ckpt_path}; using geometry only")

    _emit_progress(0.75, "Building neighbor graph…")
    neighbor_map = _knn_neighbor_ids(cloud, edge_cutoff=500.0, knn_k=12)

    _emit_progress(0.9, "Summarizing…")
    gvi = run_gvi(cloud, mean_exposure=mean_exposure)

    by_type: Dict[str, dict] = {}
    for i, ptype in enumerate(cloud.types):
        entry = by_type.setdefault(
            ptype,
            {"exposures": [], "cleanScores": [], "anisotropies": []},
        )
        entry["exposures"].append(float(exposure[i]))
        entry["cleanScores"].append(float(clean_scores[i]))
        entry["anisotropies"].append(float(exposure_results[i].anisotropy_index))

    p10_exp = float(np.percentile(exposure, 10)) if exposure.size else 0.0
    p90_exp = float(np.percentile(exposure, 90)) if exposure.size else 0.0
    p10_clean = float(np.percentile(clean_scores, 10)) if clean_scores.size else 0.0
    p90_clean = float(np.percentile(clean_scores, 90)) if clean_scores.size else 0.0

    enriched = []
    for i, pid in enumerate(pick_ids):
        res = exposure_results[i]
        enriched.append(
            {
                "id": pid,
                "gnnExposure": round(float(gnn_exposure[i]), 4),
                "neighborCount": int(neighbor_counts[i]),
                "neighbors": neighbor_map.get(pid, []),
                **_exposure_fields(res),
            }
        )

    summary = {
        "particleCount": cloud.n_particles,
        "nRays": n_rays,
        "meanExposure": round(mean_exposure or 0.0, 4),
        "p10Exposure": round(p10_exp, 4),
        "p90Exposure": round(p90_exp, 4),
        "meanCleanExtraction": round(mean_clean or 0.0, 4),
        "p10CleanExtraction": round(p10_clean, 4),
        "p90CleanExtraction": round(p90_clean, 4),
        "byType": {
            t: {
                "count": len(vals["exposures"]),
                "meanExposure": round(float(np.mean(vals["exposures"])), 4),
                "meanCleanExtraction": round(float(np.mean(vals["cleanScores"])), 4),
                "meanAnisotropy": round(float(np.mean(vals["anisotropies"])), 4),
            }
            for t, vals in sorted(by_type.items())
        },
    }

    _emit_progress(1.0, "Done")
    return {
        "gvi": gvi,
        "summary": summary,
        "picks": enriched,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    payload = json.load(sys.stdin)
    mode = payload.get("mode", "gvi")
    tomo_id = payload.get("tomo_id") or "scan"
    detections = payload.get("detections") or []
    n_rays = int(payload.get("n_rays", DEFAULT_N_RAYS))

    cloud = cloud_from_detections(tomo_id, detections)

    if mode == "gvi":
        result = {"gvi": run_gvi(cloud)}
    elif mode == "crowding":
        checkpoint = Path(payload.get("checkpoint") or DEFAULT_CHECKPOINT)
        result = run_crowding(cloud, checkpoint, n_rays=n_rays)
    elif mode == "hemisphere":
        pick_id = payload.get("pick_id") or payload.get("pickId")
        if not pick_id:
            raise ValueError("pick_id required for hemisphere mode")
        result = run_hemisphere(cloud, str(pick_id), n_rays=n_rays)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    json.dump(_json_safe(result), sys.stdout)


if __name__ == "__main__":
    main()
