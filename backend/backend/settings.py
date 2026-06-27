"""Typed configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, List, Optional, Union

import yaml


def _as(dataclass_cls):
    """Decorator not needed; kept simple with explicit from_dict methods."""
    return dataclass_cls


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------
@dataclass
class DataCfg:
    copick_config: str = "/kaggle/working/copick.config"
    input_root: str = "/kaggle/input/competitions/czii-cryo-et-object-identification"
    output_root: str = "/kaggle/working"
    voxel_size: float = 10.0
    tomo_algorithm: str = "denoised"
    tomo_type: str = "denoised"
    target_name: str = "pytargets"
    target_user_id: str = "pytorch"
    target_session_id: str = "0"
    dim_in: int = 72
    l_rnd: int = 15
    background_ratio: float = 0.30
    sample_size: int = 5
    n_sub_epoch: int = 10
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    train_tomo_ids: Optional[List[str]] = None
    valid_tomo_ids: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, d: dict) -> "DataCfg":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ModelCfg:
    name: str = "res_unet"
    filters: List[int] = field(default_factory=lambda: [48, 64, 128])
    dropout: float = 0.0
    in_channels: int = 1
    n_class: int = 8

    @classmethod
    def from_dict(cls, d: dict) -> "ModelCfg":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class LossCfg:
    type: str = "ce_tversky"
    alpha: float = 0.3
    beta: float = 0.7
    gamma: float = 2.0
    ce_weight: float = 1.0
    tversky_weight: float = 1.0
    smooth: float = 1.0e-3
    class_weights: Union[str, List[float]] = "inverse"
    ignore_index: int = -1

    @classmethod
    def from_dict(cls, d: dict) -> "LossCfg":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainCfg:
    epochs: int = 70
    steps_per_epoch: int = 150
    batch_size: int = 8
    steps_per_valid: int = 20
    n_workers: int = 4
    pin_memory: bool = True
    optimizer: str = "adamw"
    lr: float = 1.0e-4
    betas: tuple = (0.9, 0.999)
    eps: float = 1.0e-8
    weight_decay: float = 0.0
    scheduler: str = "cosine"
    warmup_epochs: int = 3
    min_lr: float = 1.0e-6
    amp: bool = True
    grad_clip: float = 0.0
    out_dir: str = "/kaggle/working/train_results"
    save_every: int = 10
    resume: Optional[str] = None
    seed: int = 42

    @classmethod
    def from_dict(cls, d: dict) -> "TrainCfg":
        kw = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "betas" in kw and isinstance(kw["betas"], list):
            kw["betas"] = tuple(kw["betas"])
        return cls(**kw)


@dataclass
class InferenceCfg:
    patch_size: int = 72
    overlap: int = 55
    pcrop: int = 25
    batch_patches: int = 4
    amp: bool = True
    write_scoremap: bool = False
    scoremap_name: str = "pyscoremap"
    segmentation_name: str = "pysegmentation"
    user_id: str = "pytorch"
    session_id: str = "0"
    out_overlay: str = "/kaggle/working/predictions"

    @classmethod
    def from_dict(cls, d: dict) -> "InferenceCfg":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class LocalizeCfg:
    min_protein_size: float = 0.8
    write_copick_picks: bool = True
    write_csv: bool = False
    csv_path: str = "/kaggle/working/submission.csv"
    picks_user_id: str = "pytorch"
    picks_session_id: str = "0"

    @classmethod
    def from_dict(cls, d: dict) -> "LocalizeCfg":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Config:
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    inference: InferenceCfg = field(default_factory=InferenceCfg)
    localize: LocalizeCfg = field(default_factory=LocalizeCfg)

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        path = Path(path)
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(
            data=DataCfg.from_dict(d.get("data", {}) or {}),
            model=ModelCfg.from_dict(d.get("model", {}) or {}),
            loss=LossCfg.from_dict(d.get("loss", {}) or {}),
            train=TrainCfg.from_dict(d.get("train", {}) or {}),
            inference=InferenceCfg.from_dict(d.get("inference", {}) or {}),
            localize=LocalizeCfg.from_dict(d.get("localize", {}) or {}),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def save_yaml(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    # convenience: merge CLI overrides (key path like "train.lr")
    def set(self, dotted: str, value: Any) -> "Config":
        parts = dotted.split(".")
        obj = self
        for p in parts[:-1]:
            obj = getattr(obj, p)
        # cast to the field type
        ftype = type(getattr(obj, parts[-1]))
        try:
            if ftype is tuple:
                value = tuple(value)
            else:
                value = ftype(value)
        except (TypeError, ValueError):
            pass
        setattr(obj, parts[-1], value)
        return self