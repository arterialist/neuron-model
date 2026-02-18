"""Build activity dataset from network simulation.

This script is the entry point for the activity dataset builder.
All logic has been refactored into the activity_dataset_builder package.
"""

import os
import sys

# Ensure local imports resolve when run from different working directories
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Re-export for backward compatibility with existing imports (noqa: used in __all__)
from activity_dataset_builder import HDF5TensorRecorder, LazyActivityDataset  # noqa: F401
from activity_dataset_builder.build import run_build

__all__ = ["HDF5TensorRecorder", "LazyActivityDataset", "main", "run_build"]


def main() -> None:
    """Entry point for the activity dataset builder CLI."""
    run_build()


if __name__ == "__main__":
    main()
