# backend — PyTorch-native CryoET 3D segmentation

A clean PyTorch reimplementation of the DeepFindET pipeline for the CZII CryoET
object-identification challenge. It produces **raw 3D segmentations** (per-class
labelmaps + optional probability scoremaps) that you can postprocess yourself.

## Highlights

- **PyTorch native** — no TensorFlow dependency. AMP (fp16) for fast T4 training.
- **3D residual U-Net** (plus plain U-Net and attention U-Net variants).
- **Class-balanced bootstrap patch sampling** with in-memory tomogram pooling and
  periodic subset swaps (handles tomograms too large to all fit in RAM).
- **Torch-native 3D augmentation** (flips, 180° rotations, intensity, gaussian blur,
  noise).
- **Combined CE + Tversky loss** with inverse-frequency class weights.
- **Weighted sliding-window inference** (cosine/Hann overlap blending) to reduce seams.
- **Single YAML config** driving every stage; CLI overrides via `--set key value`.
- **Multi-GPU** via `torch.nn.DataParallel` (auto-detected).

## Layout

```
backend/
  configs/default.yaml         # all hyperparameters
  backend/
    data/                      # copick IO, target builder, dataset, augmentation, splits
    models/                    # blocks + unet3d / res_unet3d / attention_unet3d + factory
    losses/                    # Tversky, Focal-Tversky, Dice, CE+Tversky
    train.py                   # training loop (AMP, schedulers, TB, checkpointing)
    inference.py               # sliding-window segmentation
    localize.py                # optional: segmentation -> particle coordinates
    pipeline.py                # orchestrator
    settings.py                # typed YAML config
  scripts/
    setup_copick.py            # Stage 0: write copick config + copy overlay picks
    build_targets.py           # Stage 1: sphere segmentation targets
    train_model.py             # Stage 2: train
    segment.py                 # Stage 3: inference (writes raw segmentations)
    localize.py                # Stage 4 (optional): coords from segmentations
    run_all.py                 # one-shot pipeline
```

## Quick start (Kaggle 2× T4)

In a Kaggle notebook cell:

```bash
%%bash
cd /kaggle/working
pip -q install copick copick-utils ome-zarr tensorboard
git clone <your-repo> backend   # or upload backend/ as a dataset
cd backend
```

Then run the full pipeline:

```python
!python scripts/run_all.py --config configs/default.yaml \
    --input-root /kaggle/input/competitions/czii-cryo-et-object-identification \
    --output-root /kaggle/working
```

Or step-by-step:

```bash
# 1. set up copick config + copy train overlay picks into /kaggle/working/overlay
python scripts/setup_copick.py \
    --input-root /kaggle/input/competitions/czii-cryo-et-object-identification \
    --output-root /kaggle/working

# 2. build sphere targets
python scripts/build_targets.py --config configs/default.yaml

# 3. train
python scripts/train_model.py --config configs/default.yaml

# 4. segment (writes raw labelmaps to the overlay)
python scripts/segment.py --config configs/default.yaml \
    --weights /kaggle/working/train_results/net_weights_BEST.pt \
    --write-scoremap
```

Override any knob without editing the yaml:

```bash
python scripts/train_model.py --config configs/default.yaml \
    --set "train.epochs 50" --set "train.batch_size 6"
```

## Output (what you postprocess)

- **Raw labelmap** (OME-Zarr, `int8`, shape `(Z,Y,X)`): argmax over the `n_class`
  per-class softmax. Written to the copick overlay as a segmentation named
  `pysegmentation` (configurable). This is the primary deliverable.
- **Optional scoremap** (OME-Zarr features, `float32`, shape `(C,Z,Y,X)`): the full
  per-class probabilities. Enable with `inference.write_scoremap: true` or
  `--write-scoremap`. Useful for threshold tuning / NMS in your postprocessor.
- **Optional picks** (`localize.py`): connected-component centers as copick JSON,
  if you want a quick coordinate baseline.

## Config notes

`configs/default.yaml` is tuned for Kaggle 2× T4 (15 GB each) at `patch=72³`,
`batch=8/GPU`, `filters=[48,64,128]`, AMP enabled. Key knobs:

- `data.dim_in` — patch size (must be multiple of 4 for the two pooling stages).
- `data.background_ratio` — fraction of each batch drawn from non-particle regions.
- `data.sample_size` / `data.n_sub_epoch` — tomogram pool size and swap cadence.
- `loss.class_weights: inverse` — uses inverse pick frequency (recommended).
- `inference.overlap` / `inference.pcrop` — sliding-window overlap and border crop.
- `inference.write_scoremap` — set `true` to also dump per-class probabilities.

## Class layout (from the copick config)

| label | name                  | radius (Å) |
|------:|-----------------------|-----------|
| 1     | apo-ferritin          | 60        |
| 2     | beta-amylase          | 65 (unscored) |
| 3     | beta-galactosidase    | 90        |
| 4     | ribosome              | 150       |
| 5     | thyroglobulin         | 130       |
| 6     | virus-like-particle   | 135       |
| 8     | membrane              | —         |
| 9     | background            | —         |

The model output has `n_class=8` channels indexed `0..7`, where channel `0` is
background and channels `1..6` are the particle classes (membrane=8 maps to
channel 7 in the 8-class softmax). Adjust `model.n_class` if you want a different
grouping.