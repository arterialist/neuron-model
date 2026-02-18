"""Activity dataset builder package.

Builds neural activity datasets from network simulations and vision datasets.
Split into reusable modules: datasets, prompts, vision, network utils,
input mapping, signals, and workers.
"""

from snn_classification_realtime.core.config import DatasetConfig
from snn_classification_realtime.activity_dataset_builder.datasets import HDF5TensorRecorder, LazyActivityDataset

__all__ = [
    "DatasetConfig",
    "HDF5TensorRecorder",
    "LazyActivityDataset",
]
