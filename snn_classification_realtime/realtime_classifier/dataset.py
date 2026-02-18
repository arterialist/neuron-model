"""Dataset loading for real-time classification."""

from snn_classification_realtime.core.config import DatasetConfig
from snn_classification_realtime.core.vision_datasets import load_dataset_by_name


def select_and_load_dataset(
    dataset_name: str,
    cifar10_color_upper_bound: float = 1.0,
) -> DatasetConfig:
    """Load a specified dataset and return DatasetConfig."""
    cfg = load_dataset_by_name(
        dataset_name,
        train=False,
        root="./data",
        cifar10_color_upper_bound=cifar10_color_upper_bound,
    )
    if cfg.is_colored_cifar10:
        img0, _ = cfg.dataset[0]
        print(
            f"Successfully loaded {dataset_name} dataset (colored, "
            f"{img0.shape[1]}x{img0.shape[2]} pixels × 3 channels = {cfg.image_vector_size} synapses)."
        )
        print(
            f"Each color channel normalized to [0, {cifar10_color_upper_bound:.3f}] range"
        )
    else:
        print(f"Successfully loaded {dataset_name} dataset.")
    return cfg
