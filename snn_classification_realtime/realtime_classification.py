"""Real-time classification of neuron network activity using a trained SNN.

Entry point for the real-time classification CLI.
All logic has been refactored into the realtime_classifier package.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snn_classification_realtime.realtime_classifier.run import main

# Backward compatibility: re-export commonly used symbols for any external imports
from snn_classification_realtime.realtime_classifier.scaling import (
    apply_scaling_to_snapshot,
)
from snn_classification_realtime.realtime_classifier.dataset import (
    select_and_load_dataset,
)
from snn_classification_realtime.realtime_classifier.features import (
    collect_features_consistently,
    collect_activity_snapshot,
    collect_multi_feature_snapshot,
)
from snn_classification_realtime.realtime_classifier.model_config import (
    load_model_config,
)
from snn_classification_realtime.realtime_classifier.network_utils import (
    infer_layers_from_metadata,
    determine_input_mapping,
)
from snn_classification_realtime.realtime_classifier.input_mapping import (
    image_to_signals,
)

__all__ = [
    "main",
    "apply_scaling_to_snapshot",
    "select_and_load_dataset",
    "collect_features_consistently",
    "collect_activity_snapshot",
    "collect_multi_feature_snapshot",
    "load_model_config",
    "infer_layers_from_metadata",
    "determine_input_mapping",
    "image_to_signals",
]


if __name__ == "__main__":
    main()
