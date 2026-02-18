"""Configuration dataclasses for activity dataset building."""

from dataclasses import dataclass
from typing import Any


@dataclass
class DatasetConfig:
    """Configuration for the selected vision dataset."""

    dataset: Any  # torchvision dataset
    image_vector_size: int
    num_classes: int
    dataset_name: str
    is_colored_cifar10: bool = False
    cifar10_color_normalization_factor: float = 0.5
