"""Configuration for SNN training."""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Configuration for SNN classifier training."""

    dataset_dir: str
    model_save_path: str
    load_model_path: str | None
    output_dir: str
    epochs: int
    learning_rate: float
    batch_size: int
    test_every: int
    device: str | None
