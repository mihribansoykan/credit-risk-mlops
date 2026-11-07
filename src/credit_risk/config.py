"""Typed configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    url: str
    raw_path: Path
    test_size: float = Field(0.2, gt=0, lt=1)
    valid_size: float = Field(0.2, gt=0, lt=1)


class ModelConfig(BaseModel):
    n_trials: int = Field(40, ge=1)
    cv_folds: int = Field(5, ge=2)
    calibration: Literal["auto", "none", "isotonic", "sigmoid"] = "auto"


class CostConfig(BaseModel):
    false_negative: float = Field(5.0, gt=0)
    false_positive: float = Field(1.0, gt=0)


class TrackingConfig(BaseModel):
    enabled: bool = True
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "credit-risk"


class ArtifactConfig(BaseModel):
    model_path: Path
    metrics_path: Path
    figures_dir: Path
    reference_profile_path: Path


class TrainConfig(BaseModel):
    data: DataConfig
    protected_attributes: list[str]
    model: ModelConfig
    costs: CostConfig
    tracking: TrackingConfig
    artifacts: ArtifactConfig
    seed: int = 42


def load_config(path: str | Path = "configs/train.yaml") -> TrainConfig:
    with open(path, encoding="utf-8") as f:
        return TrainConfig.model_validate(yaml.safe_load(f))
