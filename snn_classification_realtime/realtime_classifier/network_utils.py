"""Network topology utilities - re-export from core."""

from snn_classification_realtime.core.network_utils import (
    infer_layers_from_metadata,
    determine_input_mapping,
)

__all__ = ["infer_layers_from_metadata", "determine_input_mapping"]
