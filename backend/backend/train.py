"""Training loop for 3D CryoET segmentation (PyTorch, AMP, multi-GPU)."""

from __future__ import annotations

import os
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # tensorboard not installed
    SummaryWriter = None

from .settings import Config
from .utils import seed_everything, get_logger
from .utils.checkpoint import save_checkpoint, load_checkpoint
from .models import build_model
from .losses import build_loss
from .metrics import segmentation_metrics, MetricAccumulator
from .data import copick_io, splits as split_utils
from .data.dataset import CryoETPatchDataset, TomogramPool
from .data.augment import Augment3D


def _class_weight_vector(cfg: Config) -> Optional[list]:
    """Build a class-weight vector from cfg.loss.class_weights."""
    n = cfg.model.n_class
    mode = cfg.loss.class_weights
    if mode == "ones":
        return [1.0] * n
    if isinstance(mode, list):
        return [float(x) for x in mode]
    # "inverse" — derive from pick counts
    # We approximate using object radii as proxies for class frequency rarity.
    # The notebook used raw particle counts; here we use inverse-frequency of
    # particle label occurrences in the train picks (computed lazily below).
    return None  # handled in train() where picks are available


def _inverse_freq_weights(picks, n_class) -> list:
    counts = np.zeros(n_class, dtype=np.float64)
    for *_p, label in picks:
        counts[label] += 1
    counts[counts == 0] = 1.0
    w = 1.0 / counts
    w = w / w.sum() * n_class  # normalize so mean weight = 1
    # background (label 0 in our softmax indexing) gets a small weight
    # NOTE: target labels in the dataset are 1..N (particle labels) plus 0 for
    # background. We use class index 0 = background in the model output, so we
    # must remap. For simplicity we keep label 0..n_class-1 aligned with target
    # integer values (background=0, apo-ferritin=1, ...).
    w[0] = 0.1 * (w[1:].mean() if n_class > 1 else 1.0)
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
        # ensure validation set is at least as large as test (per DeepFindET)
        if len(_test) > len(valid_ids):
            valid_ids, _test = _test, valid_ids
    log.info(f"Train tomos ({len(train_ids)}): {train_ids}")
    log.info(f"Valid tomos ({len(valid_ids)}): {valid_ids}")

    # ---- Class weights ------------------------------------------------
    # index_picks is defined earlier in the notebook global namespace.
    train_picks = index_picks(cfg.data.copick_config, train_ids, None, cfg.data.voxel_size)
    if cfg.loss.class_weights == "inverse":
        weights = _inverse_freq_weights(train_picks, cfg.model.n_class)
        log.info(f"Class weights (inverse-freq): {[round(w,4) for w in weights]}")
        # inject into loss cfg
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
    criterion = build_loss(cfg.loss).to(device)
    optimizer = _build_optimizer(cfg, model)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and device.type == "cuda")
    scheduler = _build_scheduler(cfg, optimizer, cfg.train.steps_per_epoch)

    start_epoch = 0
    best_f1 = -1.0
    if cfg.train.resume:
        ckpt = load_checkpoint(cfg.train.resume, model, optimizer, scheduler, scaler, map_location=device)
        start_epoch = ckpt.get("epoch", 0)
        best_f1 = ckpt.get("best_metric") or -1.0
        log.info(f"Resumed from {cfg.train.resume} at epoch {start_epoch} (best_f1={best_f1:.4f})")

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

    # ---- TensorBoard --------------------------------------------------
    writer = SummaryWriter(log_dir=str(out_dir / "tensorboard_logs")) if SummaryWriter else None
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_macro_iou": [], "lr": []}

    # ---- Training loop ------------------------------------------------
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

        # ---- Validation ----
        model.eval()
        acc = MetricAccumulator(cfg.model.n_class)
        with torch.no_grad():
            for img, lbl in _loader(valid_ds):
                img = img.to(device, non_blocking=True)
                lbl = lbl.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=cfg.train.amp and device.type == "cuda"):
                    out = model(img)
                    loss = criterion(out, lbl)
                m = segmentation_metrics(out.float(), lbl, cfg.model.n_class, cfg.loss.ignore_index)
                acc.update(m, float(loss.item()))
        vm = acc.compute()
        log.info(
            f"[Epoch {epoch}] train_loss={train_loss:.4f} valid_loss={vm['loss']:.4f} "
            f"macroF1={vm['macro_f1']:.4f} mIoU={vm['macro_iou']:.4f} "
            f"perF1={[round(float(x),3) for x in vm['f1']]} ({time.time()-t0:.0f}s)"
        )

        # ---- Per-epoch side-by-side visualization (tomogram | GT | prediction) ----
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # grab one validation patch (already yielded by the iterable dataset)
            viz_img, viz_lbl = None, None
            for vi, vl in _loader(valid_ds):
                viz_img, viz_lbl = vi[0:1], vl[0:1]
                break
            if viz_img is not None:
                viz_img = viz_img.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=cfg.train.amp and device.type == "cuda"):
                    viz_out = model(viz_img)
                viz_img = viz_img[0, 0].detach().cpu().numpy()        # (D,D,D)
                viz_lbl_np = viz_lbl[0].detach().cpu().numpy()       # (D,D,D)
                viz_pred = viz_out[0].argmax(0).detach().cpu().numpy()  # (D,D,D) argmax over class axis
                # mid-slice
                dz = viz_img.shape[0] // 2
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(viz_img[dz], cmap="gray")
                axes[0].set_title(f"tomogram z={dz}"); axes[0].axis("off")
                # use inlined overlay helpers if present, else plain imshow
                if "overlay_slice" in globals() and "legend_for_labels" in globals():
                    overlay_slice(viz_img[dz], viz_lbl_np[dz], alpha=0.55, ax=axes[1],
                                  title=f"GROUND TRUTH z={dz} (epoch {epoch})")
                    legend_for_labels(np.unique(viz_lbl_np[dz]), ax=axes[1])
                    overlay_slice(viz_img[dz], viz_pred[dz], alpha=0.55, ax=axes[2],
                                  title=f"PREDICTION z={dz} (epoch {epoch})")
                    legend_for_labels(np.unique(viz_pred[dz]), ax=axes[2])
                else:
                    from matplotlib.colors import ListedColormap, BoundaryNorm
                    from matplotlib.patches import Patch
                    cmap = ListedColormap([[0,0,0,0],[0,0.46,0.86,0.8],[0.6,0.25,0,0.8],
                                           [0.3,0,0.36,0.8],[0,0.36,0.19,0.8],[0.17,0.81,0.28,0.8],
                                           [1,0.8,0.6,0.8],[1,1,1,0.8]])
                    for a, vol, name in [(axes[1], viz_lbl_np, "GROUND TRUTH"),
                                         (axes[2], viz_pred, "PREDICTION")]:
                        a.imshow(viz_img[dz], cmap="gray")
                        lp = np.unique(vol[dz]); lp = lp[lp > 0]
                        if len(lp):
                            bounds = np.array(list(lp)+[lp[-1]+1], dtype=float) - 0.5
                            a.imshow(vol[dz], cmap=cmap, norm=BoundaryNorm(bounds, cmap.N), alpha=0.55)
                        a.set_title(f"{name} z={dz} (epoch {epoch})"); a.axis("off")
                fig.suptitle(f"Epoch {epoch} — val macroF1={vm['macro_f1']:.3f}", fontsize=12)
                plt.tight_layout()
                viz_dir = out_dir / "viz"
                viz_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(viz_dir / f"epoch_{epoch:03d}.png", dpi=90, bbox_inches="tight")
                plt.show()
                plt.close(fig)
        except Exception as _viz_err:
            log.info(f"(per-epoch viz skipped: {_viz_err})")

        # ---- LR schedule ----
        if scheduler is not None:
            if cfg.train.scheduler == "plateau":
                scheduler.step(vm["macro_f1"])
            else:
                scheduler.step()

        # ---- Logging / checkpointing ----
        history["train_loss"].append(train_loss)
        history["val_loss"].append(vm["loss"])
        history["val_macro_f1"].append(vm["macro_f1"])
        history["val_macro_iou"].append(vm["macro_iou"])
        history["lr"].append(optimizer.param_groups[0]["lr"])
        import json as _json
        _json.dump(history, open(out_dir / "history.json", "w"))

        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("valid/loss", vm["loss"], epoch)
            writer.add_scalar("valid/macro_f1", vm["macro_f1"], epoch)
            writer.add_scalar("valid/macro_iou", vm["macro_iou"], epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
            for c in range(cfg.model.n_class):
                writer.add_scalar(f"valid/f1_class{c}", vm["f1"][c], epoch)

        ckpt_path = out_dir / "net_weights_LAST.pt"
        save_checkpoint(str(ckpt_path), model.module if isinstance(model, torch.nn.DataParallel) else model,
                        optimizer, scheduler, scaler, epoch, best_f1)

        if vm["macro_f1"] > best_f1:
            best_f1 = vm["macro_f1"]
            best_path = out_dir / "net_weights_BEST.pt"
            save_checkpoint(str(best_path), model.module if isinstance(model, torch.nn.DataParallel) else model,
                            optimizer, scheduler, scaler, epoch, best_f1)
            log.info(f"  -> new best macroF1={best_f1:.4f} saved to {best_path}")

        if (epoch + 1) % max(1, cfg.train.save_every) == 0:
            ep_path = out_dir / f"net_weights_epoch{epoch+1}.pt"
            save_checkpoint(str(ep_path), model.module if isinstance(model, torch.nn.DataParallel) else model,
                            optimizer, scheduler, scaler, epoch, best_f1)

    if writer is not None:
        writer.close()
    log.info(f"Training complete. best macroF1={best_f1:.4f}")
    return str(out_dir / "net_weights_BEST.pt")