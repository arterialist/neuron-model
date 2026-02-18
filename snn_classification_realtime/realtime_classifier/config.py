"""Configuration for real-time classification."""

from dataclasses import dataclass
from typing import Any


@dataclass
class RealtimeConfig:
    """Configuration for real-time classification CLI."""

    snn_model_path: str
    neuron_model_path: str
    dataset_name: str
    ticks_per_image: int
    window_size: int
    evaluation_mode: bool
    eval_samples: int
    device: str | None
    think_longer: bool
    max_thinking_multiplier: float
    bistability_rescue: bool
    cifar10_color_upper_bound: float
    enable_web_server: bool
    ablation: str | None
