"""Activity data preparation package for SNN training.

Prepares network activity data from binary (HDF5) or legacy JSON formats.
"""

# Ensure parent path for build_activity_dataset import when used as subpackage
import sys
from pathlib import Path

_parent = Path(__file__).resolve().parent.parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from snn_classification_realtime.activity_preparer.extraction import (
    extract_S_from_layer,
    extract_fired_from_layer,
    extract_t_ref_from_layer,
)
from snn_classification_realtime.activity_preparer.loaders import (
    load_dataset,
    group_by_image,
)
from snn_classification_realtime.activity_preparer.features import (
    extract_firings_time_series,
    extract_avg_S_time_series,
    extract_avg_t_ref_time_series,
    extract_multi_feature_time_series,
)
from snn_classification_realtime.activity_preparer.scaler import FeatureScaler

__all__ = [
    "extract_S_from_layer",
    "extract_fired_from_layer",
    "extract_t_ref_from_layer",
    "load_dataset",
    "group_by_image",
    "extract_firings_time_series",
    "extract_avg_S_time_series",
    "extract_avg_t_ref_time_series",
    "extract_multi_feature_time_series",
    "FeatureScaler",
]
