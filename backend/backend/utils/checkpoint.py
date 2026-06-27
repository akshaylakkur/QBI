"""Checkpoint helpers."""

from pathlib import Path
from typing import Optional, Dict, Any
import torch


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying model, unwrapping DataParallel/DistributedDataParallel."""
    while isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model = model.module
    return model


def _reconcile_state_dict(state_dict: Dict[str, Any], model: torch.nn.Module) -> Dict[str, Any]:
    """Match a checkpoint state_dict's key prefix to the model's wrapping.

    Saves always store the *unwrapped* (no `module.` prefix) state_dict, so we only
    need to add `module.` here when the target model is DataParallel-wrapped.
    """
    is_wrapped = isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
    sd_has_module = any(k.startswith("module.") for k in state_dict)

    if is_wrapped and not sd_has_module:
        # checkpoint is raw, model is wrapped -> add module. prefix
        return {"module." + k: v for k, v in state_dict.items()}
    if not is_wrapped and sd_has_module:
        # checkpoint is wrapped, model is raw -> strip module. prefix
        return {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
    return state_dict


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    epoch: int = 0,
    best_metric: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a checkpoint. The model state_dict is always stored *unwrapped*
    (no `module.` prefix), so it can be loaded into either a raw or DataParallel-wrapped model."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    raw_model = _unwrap(model)
    state = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "extra": extra or {},
    }
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint, reconciling the `module.` prefix so loading works whether
    the target model is raw or DataParallel/DistributedDataParallel-wrapped."""
    ckpt = torch.load(path, map_location=map_location)
    state_dict = _reconcile_state_dict(ckpt["model"], model)
    model.load_state_dict(state_dict)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt