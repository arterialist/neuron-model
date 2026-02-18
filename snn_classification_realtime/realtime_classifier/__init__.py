"""Real-time classification package for SNN inference on neuron network activity."""

import sys
from pathlib import Path

_parent = Path(__file__).resolve().parent.parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from snn_classification_realtime.realtime_classifier.scaling import apply_scaling_to_snapshot
from snn_classification_realtime.realtime_classifier.dataset import select_and_load_dataset
from snn_classification_realtime.realtime_classifier.features import (
    collect_features_consistently,
    collect_activity_snapshot,
    collect_multi_feature_snapshot,
)
from snn_classification_realtime.realtime_classifier.model_config import load_model_config

__all__ = [
    "apply_scaling_to_snapshot",
    "select_and_load_dataset",
    "collect_features_consistently",
    "collect_activity_snapshot",
    "collect_multi_feature_snapshot",
    "load_model_config",
]
