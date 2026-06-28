# QBI CryoET Molecule Detection

QBI is a Cryo-Electron Tomography object detection and visualization project for
CZII-style tomogram Zarr files. It loads 3D cryo scan volumes, runs a trained
PyTorch detection model, converts model output into molecule picks, and displays
the result in a browser-based 3D/slice viewer.

The model targets the molecule classes from the CZII CryoET Object Identification
dataset:

| Molecule | Expected difficulty | Scored in CZII challenge | Notes |
| --- | --- | --- | --- |
| `apo-ferritin` | Easy | Yes | Small, high-contrast particle. |
| `beta-amylase` | Impossible | No | Included as a class in this repo, but not scored by the competition. |
| `beta-galactosidase` | Hard | Yes | Lower visibility and harder localization. |
| `ribosome` | Easy | Yes | Large particle with strong signal. |
| `thyroglobulin` | Hard | Yes | Difficult particle with more ambiguous appearance. |
| `virus-like-particle` | Easy | Yes | Large, visually distinct particle. |

Training data: [CZII CryoET Object Identification on Kaggle](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/data).

## What This Repository Contains

```text
.
|-- backend/                 # PyTorch training, segmentation, and inference code
|   |-- backend/             # Training pipeline package
|   |-- configs/default.yaml # Default training/inference config
|   |-- inference/           # Direct Zarr object-detection inference path
|   `-- scripts/             # Setup, target-building, training, segmentation CLIs
|-- downstream/              # Geometry, exposure, graph, and QA utilities
|-- frontend/                # Node.js web server and browser viewer
|   |-- app.js
|   |-- index.html
|   |-- server.js
|   `-- predictions/         # Generated prediction JSON and pick files
|-- runs/                    # Runtime metrics/checkpoints for downstream models
|-- kaggle_cryoet_train.ipynb
`-- README.md
```

There are two main backend paths:

1. `backend/backend/*` and `backend/scripts/*`: a PyTorch-native 3D segmentation
   training pipeline with copick support.
2. `backend/inference/*`: direct object-detection inference for a single `.zarr`
   tomogram using a SegResNet-style checkpoint.

The frontend currently uses the direct inference path.

## Requirements

- Python 3.9+
- Node.js 18+
- PyTorch 2.0+
- A model checkpoint compatible with `backend/inference/model.py`
- Optional GPU acceleration:
  - CUDA on Linux/Windows
  - Apple Silicon MPS on macOS

The web viewer imports Three.js from `unpkg.com`, so the browser needs network
access unless those imports are vendored locally.

## Installation

From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e backend
python -m pip install -e downstream[all]

cd frontend
npm install
cd ..
```

If you only need the viewer and direct inference, `backend` plus the frontend
dependencies are enough:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e backend

cd frontend
npm install
```

## Model Checkpoint

The frontend server expects the object-detection checkpoint at:

```text
~/Downloads/czii-weights/weight_best.ckpt
```

Create the directory and place the checkpoint there:

```sh
mkdir -p ~/Downloads/czii-weights
# put weight_best.ckpt in ~/Downloads/czii-weights/
```

If the checkpoint is missing, the app can still load and visualize scans, but
`Run Model Inference` will report a checkpoint-not-found error.

For direct CLI inference, you can pass any checkpoint path with `--checkpoint`.

## Run The Web App

```sh
cd frontend
npm start
```

Open:

```text
http://localhost:3000
```

Use a different port if needed:

```sh
cd frontend
PORT=3015 npm start
```

### Web Workflow

1. Start the frontend server.
2. Open `http://localhost:3000`.
3. Load a scan using one of:
   - `Upload a .zarr folder`
   - `Local .zarr path`
   - a bundled sample scan, if present under `frontend/sample_data`
4. Prefer `Local .zarr path` for original OME-Zarr folders. Browser folder
   uploads can omit hidden files such as `.zarray` and `.zattrs`, which are
   required for Zarr metadata.
5. Select the desired resolution level and inspect the X/Y/Z slices.
6. Click `Run Model Inference`.
7. When inference finishes, detections appear in the `Detected molecules` panel
   and are overlaid in the viewer.
8. Optionally upload copick-style pick JSON labels for comparison or analysis.

Generated files are written under:

```text
frontend/predictions/
```

The server also stores uploaded scans under:

```text
frontend/uploaded_scans/
```

## Direct Inference From A Zarr File

Run object detection without the browser:

```sh
source .venv/bin/activate
python backend/inference/run_inference_direct.py \
  --checkpoint ~/Downloads/czii-weights/weight_best.ckpt \
  --zarr_path /path/to/denoised.zarr \
  --device cpu \
  --dtype float32 \
  --window_size 64 64 64 \
  --tiles_per_dim 3 10 10 \
  --output predictions.json
```

Device options:

```text
cpu
cuda
mps
```

For Apple Silicon:

```sh
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python backend/inference/run_inference_direct.py \
  --checkpoint ~/Downloads/czii-weights/weight_best.ckpt \
  --zarr_path /path/to/denoised.zarr \
  --device mps \
  --dtype float32 \
  --output predictions.json
```

For CUDA:

```sh
python backend/inference/run_inference_direct.py \
  --checkpoint ~/Downloads/czii-weights/weight_best.ckpt \
  --zarr_path /path/to/denoised.zarr \
  --device cuda \
  --dtype float16 \
  --output predictions.json
```

Useful inference options:

```text
--score_threshold 0.2              Single threshold for all classes
--score_thresholds ...             Per-class thresholds, one value per class
--iou_threshold 0.85               NMS threshold
--pre_nms_top_k 10000              Limit candidates before NMS
--use_weighted_average             Weighted tile blending
--use_z_flip_tta                   Z-axis test-time augmentation
--use_y_flip_tta                   Y-axis test-time augmentation
--use_x_flip_tta                   X-axis test-time augmentation
```

The default class order for direct inference is:

```text
0 apo-ferritin
1 beta-galactosidase
2 ribosome
3 thyroglobulin
4 virus-like-particle
5 beta-amylase
```

## Prediction Output

Direct inference writes JSON:

```json
{
  "zarr_path": "/path/to/denoised.zarr",
  "voxel_size_angstroms": 10.012,
  "num_detections": 123,
  "detections": [
    {
      "particle_type": "ribosome",
      "class_label": 2,
      "score": 0.91,
      "x_angstrom": 1200.5,
      "y_angstrom": 842.1,
      "z_angstrom": 310.4,
      "x_pixel": 119.9,
      "y_pixel": 84.1,
      "z_pixel": 31.0
    }
  ]
}
```

When launched from the web UI, the server also converts predictions into
pick-style JSON files grouped by molecule type under:

```text
frontend/predictions/picks_<scanId>/
```

## Supported Zarr Layouts

`run_inference_direct.py` supports:

- OME-Zarr multiscales where the highest-resolution array is under group `0`
- Flat 3D Zarr arrays
- Zarr groups containing at least one readable 3D array

The viewer expects Zarr array metadata files such as `.zarray` and `.zattrs`.
If folder upload fails because hidden metadata was skipped, use the local path
field with the original directory.

## Training Pipeline

The training code under `backend/` is designed for the Kaggle CZII dataset and
copick overlays. See [backend/README.md](backend/README.md) for deeper training
details.

One-shot training pipeline:

```sh
cd backend
python scripts/run_all.py --config configs/default.yaml \
  --input-root /kaggle/input/competitions/czii-cryo-et-object-identification \
  --output-root /kaggle/working
```

Step-by-step:

```sh
cd backend

python scripts/setup_copick.py \
  --input-root /kaggle/input/competitions/czii-cryo-et-object-identification \
  --output-root /kaggle/working

python scripts/build_targets.py --config configs/default.yaml

python scripts/train_model.py --config configs/default.yaml

python scripts/segment.py --config configs/default.yaml \
  --weights /kaggle/working/train_results/net_weights_BEST.pt \
  --write-scoremap
```

Override config values without editing YAML:

```sh
python scripts/train_model.py --config configs/default.yaml \
  --set "train.epochs 50" \
  --set "train.batch_size 6"
```

The default backend segmentation class layout is:

| Label | Name | Radius |
| ---: | --- | ---: |
| 1 | `apo-ferritin` | 60 A |
| 2 | `beta-amylase` | 65 A |
| 3 | `beta-galactosidase` | 90 A |
| 4 | `ribosome` | 150 A |
| 5 | `thyroglobulin` | 130 A |
| 6 | `virus-like-particle` | 135 A |
| 8 | `membrane` | N/A |
| 9 | `background` | N/A |

## Downstream Analysis

The `downstream/` package contains geometry and graph utilities used by the
frontend for molecule neighborhood and exposure analysis:

- hemisphere/open-direction metrics
- local crowding graph utilities
- grid viability checks
- exposure GNN helpers

Run downstream tests:

```sh
source .venv/bin/activate
python -m pytest downstream/tests
```

## Optional AI Analysis

`frontend/server.js` can call Anthropic's Messages API for molecule-level report
generation when an API key is available. This is optional; the app has fallback
structured summaries when the key is absent.

Create `frontend/.env`:

```text
ANTHROPIC_API_KEY=your_api_key
ANTHROPIC_MODEL=claude-sonnet-4-6
```

## Troubleshooting

### Checkpoint not found

The frontend uses:

```text
~/Downloads/czii-weights/weight_best.ckpt
```

Place the checkpoint there or run direct inference with an explicit
`--checkpoint` path.

### Uploaded Zarr has no metadata

Browser folder upload may skip hidden files. Use `Local .zarr path` and point to
the original folder containing `.zarray`, `.zattrs`, or `zarr.json`.

### CUDA or MPS falls back to CPU

The inference script checks device availability. If CUDA or MPS is unavailable,
it prints a warning and runs on CPU.

### Frontend loads but 3D view is blank

The browser imports Three.js from `https://unpkg.com`. Check network access or
vendor the Three.js files locally and update the import map in
`frontend/index.html`.

### Inference is slow

Large tomograms are processed in tiles. Reduce `--tiles_per_dim`, use a GPU,
or start with a smaller Zarr level for visualization before running full
resolution inference.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
