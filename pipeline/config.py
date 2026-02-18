"""
Configuration module for the experimentation pipeline.

Provides Pydantic models for YAML configuration parsing and validation.
"""

import os
from pathlib import Path
from typing import Any, Literal
from enum import Enum

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class DatasetType(str, Enum):
    """Supported dataset types."""

    MNIST = "mnist"
    CIFAR10 = "cifar10"
    CIFAR10_COLOR = "cifar10_color"
    CIFAR100 = "cifar100"
    FASHION_MNIST = "fashionmnist"
    SVHN = "svhn"
    USPS = "usps"


class LayerType(str, Enum):
    """Layer types for network building."""

    CONV = "conv"
    DENSE = "dense"


class LayerConfig(BaseModel):
    """Configuration for a single network layer."""

    type: LayerType
    # Conv layer params
    kernel_size: int = Field(default=3, ge=1)
    stride: int = Field(default=1, ge=1)
    filters: int = Field(default=1, ge=1)
    # Dense layer params
    size: int = Field(default=128, ge=1)
    synapses_per: int | None = None
    # Common params
    connectivity: float = Field(default=0.8, ge=0.0, le=1.0)


class NetworkBuildConfig(BaseModel):
    """Configuration for building a new network."""

    dataset: DatasetType = DatasetType.MNIST
    layers: list[LayerConfig] = Field(default_factory=list)
    inhibitory_signals: bool = False
    rgb_separate_neurons: bool = False


class NetworkConfig(BaseModel):
    """Network configuration - either load existing or build new."""

    source: Literal["file", "build"] = "file"
    path: str | None = None
    build_config: NetworkBuildConfig | None = None

    @model_validator(mode="after")
    def validate_source(self):
        if self.source == "file" and not self.path:
            raise ValueError("path is required when source is 'file'")
        if self.source == "build" and not self.build_config:
            raise ValueError("build_config is required when source is 'build'")
        return self


class ActivityRecordingConfig(BaseModel):
    """Configuration for activity recording step."""

    dataset: DatasetType = DatasetType.MNIST
    ticks_per_image: int = Field(default=20, ge=1)
    images_per_label: int = Field(default=500, ge=1)
    tick_time_ms: int = Field(default=0, ge=0)
    fresh_run_per_label: bool = True
    fresh_run_per_image: bool = True
    use_multiprocessing: bool = True
    binary_format: bool = True
    export_network_states: bool = False
    cifar10_color_normalization: float = Field(default=0.5, ge=0.0, le=1.0)


class DataPreparationConfig(BaseModel):
    """Configuration for data preparation step."""

    feature_types: list[str] = Field(default_factory=lambda: ["avg_S", "firings"])
    train_split: float = Field(default=0.8, ge=0.1, le=0.99)
    scaling_method: Literal["minmax", "zscore", "maxabs", "none"] = "minmax"
    max_ticks: int | None = None

    @field_validator("feature_types")
    @classmethod
    def validate_feature_types(cls, v):
        allowed = {"firings", "avg_S", "avg_t_ref"}
        for ft in v:
            if ft not in allowed:
                raise ValueError(f"Invalid feature type: {ft}. Allowed: {allowed}")
        return v


class TrainingConfig(BaseModel):
    """Configuration for classifier training step."""

    epochs: int = Field(default=20, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=0.0005, gt=0.0)
    test_every: int = Field(default=1, ge=1)
    device: str = "cpu"
    hidden_size: int = Field(default=512, ge=32)


class EvaluationConfig(BaseModel):
    """Configuration for evaluation step."""

    samples: int = Field(default=1000, ge=1)
    device: str = "cpu"
    think_longer: bool = True
    max_thinking_multiplier: int = Field(default=4, ge=1)
    window_size: int = Field(default=80, ge=1)
    dataset_name: str = Field(
        default="mnist",
        description="Dataset for evaluation (mnist, fashionmnist, cifar10, cifar10_color, cifar100)",
    )


class VisualizationTypeConfig(BaseModel):
    """Configuration for a specific visualization type."""

    name: str
    plots: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class VisualizationsConfig(BaseModel):
    """Configuration for visualization steps."""

    enabled: bool = False
    types: list[VisualizationTypeConfig] = Field(default_factory=list)


class WebhookEventType(str, Enum):
    """Events that can trigger webhooks."""

    JOB_STARTED = "job_started"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"


class WebhookConfig(BaseModel):
    """Configuration for a single webhook."""

    url: str
    events: list[WebhookEventType] = Field(
        default_factory=lambda: [
            WebhookEventType.JOB_STARTED,
            WebhookEventType.JOB_COMPLETED,
            WebhookEventType.JOB_FAILED,
        ]
    )
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=30, ge=1, le=300)


class PipelineConfig(BaseModel):
    """Root configuration for the entire pipeline."""

    job_name: str
    network: NetworkConfig
    activity_recording: ActivityRecordingConfig = Field(
        default_factory=ActivityRecordingConfig
    )
    data_preparation: DataPreparationConfig = Field(
        default_factory=DataPreparationConfig
    )
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    visualizations: VisualizationsConfig = Field(default_factory=VisualizationsConfig)
    webhooks: list[WebhookConfig] = Field(default_factory=list)

    # Execution options
    parallel_workers: int = Field(default=1, ge=1)
    skip_steps: list[str] = Field(default_factory=list)

    @field_validator("skip_steps")
    @classmethod
    def validate_skip_steps(cls, v):
        allowed = {
            "network",
            "activity_recording",
            "data_preparation",
            "training",
            "evaluation",
            "visualizations",
        }
        for step in v:
            if step not in allowed:
                raise ValueError(f"Invalid step to skip: {step}. Allowed: {allowed}")
        return v


def load_config(config_path: str | Path) -> PipelineConfig:
    """Load and validate pipeline configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated PipelineConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        pydantic.ValidationError: If config validation fails
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        raise ValueError("Empty configuration file")

    return PipelineConfig(**raw_config)


def load_config_from_string(config_string: str) -> PipelineConfig:
    """Load and validate pipeline configuration from YAML string.

    Args:
        config_string: YAML configuration string

    Returns:
        Validated PipelineConfig instance
    """
    raw_config = yaml.safe_load(config_string)

    if raw_config is None:
        raise ValueError("Empty configuration string")

    return PipelineConfig(**raw_config)
