"""Backward-compatibility shim: run activity dataset builder from new location."""

import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

# Delegate to the moved module
from snn_classification_realtime.build_activity_dataset import (
    main,
    run_build,
    HDF5TensorRecorder,
    LazyActivityDataset,
)

__all__ = ["main", "run_build", "HDF5TensorRecorder", "LazyActivityDataset"]

if __name__ == "__main__":
    main()
