"""Build activity dataset from network simulation.

This script is the entry point for the activity dataset builder.
All logic has been refactored into the activity_dataset_builder package.
"""

import os
import sys

# Ensure workspace root is in path for package imports
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from snn_classification_realtime.activity_dataset_builder import (
    HDF5TensorRecorder,
    LazyActivityDataset,
)
from snn_classification_realtime.activity_dataset_builder.build import run_build

__all__ = ["HDF5TensorRecorder", "LazyActivityDataset", "main", "run_build"]


def main() -> None:
    """Entry point for the activity dataset builder CLI."""
    run_build()


if __name__ == "__main__":
    main()
