"""Training loop for 3D CryoET segmentation (PyTorch, AMP, multi-GPU).

Validation / model selection is driven by a *full-volume* sliding-window pass
over a held-out tomogram (via ``segment_volume``), not patch-level F1 on random
crops, so the chosen checkpoint reflects end-task segmentation quality. Only
*active* classes (background + scored particles) participate in the macro-F1
used for selection; dead heads (e.g. membrane when not trained) are excluded.
"""

from __future__ import annotations

import os
import math
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # tensorboard optional; its import chain (TF/pyOpenSSL) can be broken on Kaggle
    SummaryWriter = None

from .settings import Config
from .utils import seed_everything, get_logger
from .utils.checkpoint import save_checkpoint, load_checkpoint
from .models import build_model
from .losses import build_loss
from .metrics import segmentation_metrics, MetricAccumulator
from .data import copick_io, splits as split_utils
from .data.dataset import CryoETPatchDataset, TomogramPool, index_picks
from .data.augment import Augment3D
from .inference import segment_volume
from .labels import ACTIVE_CLASSES, PARTICLE_CLASSES


def _active_classes_for(cfg: Config) -> List[int]:
    """Classes that carry target signal (background + particles)."""
    if cfg.loss.active_classes is not None:
        return list(cfg.loss.active_classes)
    return list(ACTIVE_CLASSES)


def _inverse_freq_weights(picks, n_class: int, active_classes: List[int]) -> list:
    """Inverse-frequency class weights over ACTIVE classes only.

    Inactive classes get zero weight so dead heads (e.g. membrane) don't pull
    the model. Background (class 0) is down-weighted.
    """
    counts = np.zeros(n_class, dtype=np.float64)
    for *_p, cls in picks:
        counts[cls] += 1
    counts[counts == 0] = 1.0
    w = 1.0 / counts
    # normalize over active classes so the mean active weight is 1.0
    active = np.asarray(active_classes, dtype=np.int64)
    w_active_mean = w[active].mean() if active.size else 1.0
    w = w / max(w_active_mean, 1e-8)
    # zero out inactive classes
    mask = np.zeros(n_class, dtype=np.float64)
    mask[active] = 1.0
    w = w * mask
    # down-weight background relative to the mean particle weight
    if 0 in active:
        particle_w = w[[c for c in active if c != 0]]
        w[0] = 0.1 * (particle_w.mean() if particle_w.size else 1.0)
    return w.tolist()


def _build_optimizer(cfg: Config, model: torch.nn.Module) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.train.optimizer.lower() == "adamw":
        return torch.optim.AdamW(
            params, lr=cfg.train.lr, betas=tuple(cfg.train.betas),
            eps=cfg.train.eps, weight_decay=cfg.train.weight_decay,
        )
    if cfg.train.optimizer.lower() == "adam":
        return torch.optim.Adam(
            params, lr=cfg.train.lr, betas=tuple(cfg.train.betas),
            eps=cfg.train.eps, weight_decay=cfg.train.weight_decay,
        )
    if cfg.train.optimizer.lower() == "sgd":
        return torch.optim.SGD(
            params, lr=cfg.train.lr, momentum=0.9, weight_decay=cfg.train.weight_decay
        )
    raise ValueError(f"Unknown optimizer '{cfg.train.optimizer}'")


def _build_scheduler(cfg: Config, optimizer, steps_per_epoch: int):
    if cfg.train.scheduler == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR

        return CosineAnnealingLR(optimizer, T_max=cfg.train.epochs * steps_per_epoch,
                                 eta_min=cfg.train.min_lr)
    if cfg.train.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.75, patience=6, min_lr=cfg.train.min_lr
        )
    return None


def _warmup_lr(optimizer, step, warmup_steps, base_lr):
    if warmup_steps <= 0:
        return
    lr = base_lr * min(1.0, step / max(1, warmup_steps))
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def _unwrap(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


@torch.no_grad()
def full_volume_metrics(
    model, cfg, tomo_id, active_classes, device, log
) -> dict:
    """Run sliding-window inference on one full tomogram and score it against
    the stored target. Returns per-class + macro F1/IoU over active classes.

    This is a more honest checkpoint-selection signal than patch-level F1 on
    random crops, because it measures end-to-end segmentation quality on a
    realistic volume.
    """
    tomo = copick_io.get_tomogram(
        cfg.data.copick_config, tomo_id, cfg.data.voxel_size, cfg.data.tomo_algorithm
    )[:]
    try:
        tgt = copick_io.get_segmentation(
            cfg.data.copick_config, tomo_id,
            name=cfg.data.target_name,
            user_id=cfg.data.target_user_id,
            session_id=cfg.data.target_session_id,
        )[:]
    except Exception as e:
        log.warning(f"  no target for {tomo_id}: {e}; skipping full-volume eval")
        return {"macro_f1": 0.0, "macro_iou": 0.0}

    labelmap, _ = segment_volume(
        _unwrap(model), tomo,
        patch_size=cfg.inference.patch_size,
        overlap=cfg.inference.overlap,
        pcrop=cfg.inference.pcrop,
        n_class=cfg.model.n_class,
        batch_patches=cfg.inference.batch_patches,
        amp=cfg.inference.amp,
        device=device,
    )
    pred = torch.from_numpy(labelmap.astype(np.int64))[None].to(device)       # (1,Z,Y,X)
    target = torch.from_numpy(tgt.astype(np.int64))[None].to(device)           # (1,Z,Y,X)
    # segmentation_metrics expects logits (B,C,D,H,W) and target (B,D,H,W).
    onehot = torch.nn.functional.one_hot(pred, cfg.model.n_class).permute(0, 4, 1, 2, 3)  # (1,C,Z,Y,X)
    onehot = onehot.float() * 20.0  # large logits so argmax == pred
    m = segmentation_metrics(onehot, target, cfg.model.n_class, cfg.loss.ignore_index, active_classes)
    return m


def train(cfg: Config) -> str:
    """Run training. Returns path to the best checkpoint."""
    log = get_logger("train")
    seed_everything(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    log.info(f"Device: {device} | GPUs: {n_gpu}")
    if n_gpu > 1:
        log.info(f"Using DataParallel across {n_gpu} GPUs")

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_yaml(out_dir / "config_used.yaml")

    # ---- Data: split tomo IDs -----------------------------------------
    all_runs = copick_io.list_runs(cfg.data.copick_config)
    if cfg.data.train_tomo_ids and cfg.data.valid_tomo_ids:
        train_ids = cfg.data.train_tomo_ids
        valid_ids = cfg.data.valid_tomo_ids
    else:
        train_ids, valid_ids, _test = split_utils.split_runs(
            all_runs, cfg.data.train_ratio, cfg.data.val_ratio, cfg.data.test_ratio,
            seed=cfg.train.seed,
        )
        if len(_test) > len(valid_ids):
            valid_ids, _test = _test, valid_ids
    log.info(f"Train tomos ({len(train_ids)}): {train_ids}")
    log.info(f"Valid tomos ({len(valid_ids)}): {valid_ids}")

    # ---- Active classes + class weights ------------------------------
    active_classes = _active_classes_for(cfg)
    log.info(f"Active classes (loss/metric): {active_classes}")
    cfg.loss.active_classes = active_classes

    train_picks = index_picks(cfg.data.copick_config, train_ids, None, cfg.data.voxel_size)
    if cfg.loss.class_weights == "inverse":
        weights = _inverse_freq_weights(train_picks, cfg.model.n_class, active_classes)
        log.info(f"Class weights (inverse-freq, active-only): {[round(w,4) for w in weights]}")
        cfg.loss.class_weights = weights

    # ---- Model / loss / optim ----------------------------------------
    model = build_model(
        cfg.model.name, in_channels=cfg.model.in_channels,
        n_class=cfg.model.n_class, filters=cfg.model.filters,
        dropout=cfg.model.dropout,
    )
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    criterion = build_loss(cfg.loss, n_class=cfg.model.n_class).to(device)
    optimizer = _build_optimizer(cfg, model)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and device.type == "cuda")
    scheduler = _build_scheduler(cfg, optimizer, cfg.train.steps_per_epoch)

    start_epoch = 0
    best_f1 = -1.0

    # ---- Resume logic ----
    resume_path = cfg.train.resume
    if resume_path is None:
        best_path = out_dir / "net_weights_BEST.pt"
        last_path = out_dir / "net_weights_LAST.pt"
        if best_path.exists():
            resume_path = str(best_path)
            log.info(f"Found existing BEST weights -> resuming from {resume_path}")
        elif last_path.exists():
            resume_path = str(last_path)
            log.info(f"Found existing LAST weights -> resuming from {resume_path}")

    if resume_path is not None and Path(resume_path).exists():
        ckpt = load_checkpoint(resume_path, model, optimizer, scheduler, scaler, map_location=device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_f1 = float(ckpt.get("best_metric") or -1.0)
        log.info(f"Resumed from {resume_path} -> start_epoch={start_epoch} best_f1={best_f1:.4f}")
    elif resume_path is not None:
        log.warning(f"Resume path {resume_path} does not exist; training from scratch.")
    else:
        log.info("No existing checkpoint found; training from scratch.")

    # ---- Datasets / loaders ------------------------------------------
    augment = Augment3D()
    train_ds = CryoETPatchDataset(
        cfg.data.copick_config, train_ids, cfg.data.dim_in, cfg.train.batch_size,
        steps=cfg.train.steps_per_epoch, l_rnd=cfg.data.l_rnd,
        background_ratio=cfg.data.background_ratio,
        voxel_size=cfg.data.voxel_size, tomo_algorithm=cfg.data.tomo_algorithm,
        target_name=cfg.data.target_name, target_user_id=cfg.data.target_user_id,
        target_session_id=cfg.data.target_session_id, augment=augment,
        n_sub_epoch=cfg.data.n_sub_epoch, sample_size=cfg.data.sample_size,
        seed=cfg.train.seed,
    )
    # Patch-level validation loader is kept for a quick per-epoch sanity loss,
    # but checkpoint selection uses the full-volume pass below.
    valid_ds = CryoETPatchDataset(
        cfg.data.copick_config, valid_ids, cfg.data.dim_in, cfg.train.batch_size,
        steps=cfg.train.steps_per_valid, l_rnd=cfg.data.l_rnd,
        background_ratio=cfg.data.background_ratio,
        voxel_size=cfg.data.voxel_size, tomo_algorithm=cfg.data.tomo_algorithm,
        target_name=cfg.data.target_name, target_user_id=cfg.data.target_user_id,
        target_session_id=cfg.data.target_session_id, augment=None,
        n_sub_epoch=cfg.data.n_sub_epoch, sample_size=cfg.data.sample_size,
        seed=cfg.train.seed + 1,
    )

    def _loader(ds, shuffle=False):
        return DataLoader(
            ds, batch_size=None, num_workers=cfg.train.n_workers,
            pin_memory=cfg.train.pin_memory and device.type == "cuda",
            worker_init_fn=lambda wid: None,
        )

    # Held-out tomogram for full-volume model selection.
    vol_eval_id = valid_ids[0] if valid_ids else (train_ids[0] if train_ids else None)

    # ---- TensorBoard / history ----------------------------------------
    writer = SummaryWriter(log_dir=str(out_dir / "tensorboard_logs")) if SummaryWriter else None
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_macro_iou": [],
               "vol_macro_f1": [], "lr": []}

    warmup_steps = cfg.train.warmup_epochs * cfg.train.steps_per_epoch
    global_step = start_epoch * cfg.train.steps_per_epoch

    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        t0 = time.time()
        train_loader = _loader(train_ds)
        running_loss = 0.0
        for step, (img, lbl) in enumerate(train_loader):
            img = img.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)

            if global_step < warmup_steps:
                _warmup_lr(optimizer, global_step, warmup_steps, cfg.train.lr)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=cfg.train.amp and device.type == "cuda"):
                out = model(img)
                loss = criterion(out, lbl)
            scaler.scale(loss).backward()
            if cfg.train.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            global_step += 1

            if step % 20 == 0:
                log.info(f"E{epoch} S{step}/{cfg.train.steps_per_epoch} loss={float(loss.item()):.4f} lr={optimizer.param_groups[0]['lr']:.2e}")

        train_loss = running_loss / max(1, cfg.train.steps_per_epoch)

        # ---- Quick patch-level validation loss (sanity) ----
        model.eval()
        acc = MetricAccumulator(cfg.model.n_class, active_classes)
        with torch.no_grad():
            for img, lbl in _loader(valid_ds):
                img = img.to(device, non_blocking=True)
                lbl = lbl.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=cfg.train.amp and device.type == "cuda"):
                    out = model(img)
                    loss = criterion(out, lbl)
                m = segmentation_metrics(out.float(), lbl, cfg.model.n_class, cfg.loss.ignore_index, active_classes)
                acc.update(m, float(loss.item()))
        vm = acc.compute()

        # ---- Full-volume validation (drives checkpoint selection) ----
        # Full-volume inference slides over an entire tomogram and is much
        # costlier than patch-level validation. Run it every `vol_eval_every`
        # epochs (and on the final epoch); on other epochs fall back to patch
        # F1 so checkpoint selection still works.
        is_last_epoch = (epoch == cfg.train.epochs - 1)
        do_vol_eval = (vol_eval_id is not None) and (
            (epoch + 1) % max(1, cfg.train.vol_eval_every) == 0 or is_last_epoch
        )
        vol_f1 = vm["macro_f1"]
        vol_iou = vm["macro_iou"]
        if do_vol_eval:
            try:
                vm_vol = full_volume_metrics(model, cfg, vol_eval_id, active_classes, device, log)
                vol_f1 = vm_vol["macro_f1"]
                vol_iou = vm_vol["macro_iou"]
                log.info(
                    f"[Epoch {epoch}] train_loss={train_loss:.4f} patch_loss={vm['loss']:.4f} "
                    f"patchF1={vm['macro_f1']:.4f} | VOLUME macroF1={vol_f1:.4f} "
                    f"mIoU={vol_iou:.4f} ({time.time()-t0:.0f}s)"
                )
            except Exception as e:
                log.warning(f"  full-volume eval failed: {e}; falling back to patch F1")
                log.info(
                    f"[Epoch {epoch}] train_loss={train_loss:.4f} val_loss={vm['loss']:.4f} "
                    f"macroF1={vm['macro_f1']:.4f} mIoU={vm['macro_iou']:.4f} ({time.time()-t0:.0f}s)"
                )
        else:
            log.info(
                f"[Epoch {epoch}] train_loss={train_loss:.4f} val_loss={vm['loss']:.4f} "
                f"macroF1={vm['macro_f1']:.4f} mIoU={vm['macro_iou']:.4f} "
                f"(patch-only; full-vol eval every {cfg.train.vol_eval_every} eps) "
                f"({time.time()-t0:.0f}s)"
            )

        # ---- LR schedule ----
        if scheduler is not None:
            if cfg.train.scheduler == "plateau":
                scheduler.step(vol_f1)
            else:
                scheduler.step()

        # ---- Logging / checkpointing ----
        history["train_loss"].append(train_loss)
        history["val_loss"].append(vm["loss"])
        history["val_macro_f1"].append(vm["macro_f1"])
        history["val_macro_iou"].append(vm["macro_iou"])
        history["vol_macro_f1"].append(vol_f1)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        import json as _json
        _json.dump(history, open(out_dir / "history.json", "w"))

        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("valid/patch_loss", vm["loss"], epoch)
            writer.add_scalar("valid/patch_macro_f1", vm["macro_f1"], epoch)
            writer.add_scalar("valid/volume_macro_f1", vol_f1, epoch)
            writer.add_scalar("valid/volume_macro_iou", vol_iou, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
            for c in active_classes:
                writer.add_scalar(f"valid/f1_class{c}", vm["f1"][c], epoch)

        ckpt_path = out_dir / "net_weights_LAST.pt"
        save_checkpoint(str(ckpt_path), _unwrap(model),
                        optimizer, scheduler, scaler, epoch, best_f1)

        # Select BEST on the full-volume macro-F1 (the end-task signal).
        if vol_f1 > best_f1:
            best_f1 = vol_f1
            best_path = out_dir / "net_weights_BEST.pt"
            save_checkpoint(str(best_path), _unwrap(model),
                            optimizer, scheduler, scaler, epoch, best_f1)
            log.info(f"  -> new best volume macroF1={best_f1:.4f} saved to {best_path}")

        if (epoch + 1) % max(1, cfg.train.save_every) == 0:
            ep_path = out_dir / f"net_weights_epoch{epoch+1}.pt"
            save_checkpoint(str(ep_path), _unwrap(model),
                            optimizer, scheduler, scaler, epoch, best_f1)

    if writer is not None:
        writer.close()
    log.info(f"Training complete. best volume macroF1={best_f1:.4f}")
    return str(out_dir / "net_weights_BEST.pt")