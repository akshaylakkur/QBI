# CryoSight Frontend

This directory contains the local CryoSight web viewer and its Node.js server.
For the global project overview, dataset layout, backend pipeline, and shared
setup notes, see `../README.md`.

## What Lives Here

```text
index.html                    application markup
styles.css                    application styles
app.js                        browser viewer and UI logic
server.js                     local HTTP server and API routes
render_zarr_cross_section.py  helper for rendering PNG slice previews
format_predictions_to_picks.py
                              helper for converting predictions to CZII picks
assets/                       logo and bundled fonts
sample_data/                  optional Kaggle-style local sample data
uploaded_scans/               runtime browser-uploaded scans
uploaded_labels/              runtime uploaded pick labels
predictions/                  runtime model outputs
```

## Requirements

- Node.js 18 or newer.
- Python 3 with `zarr`, `numpy`, and `Pillow` for slice rendering helpers.
- Optional: `frontend/.env` with `ANTHROPIC_API_KEY` for Claude-backed analysis.
- Optional: a backend checkpoint for model inference from the UI.

## Install

```bash
npm install
python3 -m pip install zarr numpy pillow
```

## Run

```bash
npm start
```

Open:

```text
http://localhost:3000
```

Use another port with:

```bash
PORT=3015 npm start
```

## Data Loading

The server discovers `.zarr` folders in `sample_data/` and exposes them through
the Dataset selector. You can also upload a Zarr folder in the browser or
register an existing local `.zarr` path through the UI.

Browser folder upload may omit hidden Zarr metadata files such as `.zarray`,
`.zattrs`, and `.zgroup`. When that happens, use the local path option.

## Inference Hook

The UI inference button launches:

```text
../backend/inference/run_inference_direct.py
```

The current server expects:

```text
~/Downloads/czii-weights/weight_best.ckpt
```

Generated prediction JSON and converted pick files are written under
`predictions/`.

## Local Analysis

`server.js` reads `frontend/.env` if present:

```bash
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-sonnet-4-6
```

Without an API key, the server returns deterministic local analysis.
