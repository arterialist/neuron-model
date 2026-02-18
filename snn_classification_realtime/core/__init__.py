"""Shared core components for SNN classification pipeline.

Used by activity_dataset_builder, realtime_classifier, and related modules.
"""

from snn_classification_realtime.core.config import DatasetConfig
from snn_classification_realtime.core.input_mapping import image_to_signals
from snn_classification_realtime.core.network_utils import (
    infer_layers_from_metadata,
    determine_input_mapping,
)
from snn_classification_realtime.core.vision_datasets import load_dataset_by_name

__all__ = [
    "DatasetConfig",
    "image_to_signals",
    "infer_layers_from_metadata",
    "determine_input_mapping",
    "load_dataset_by_name",
]
