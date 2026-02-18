"""Dataset metadata loading."""

import json
import os
from typing import Any


def load_dataset_metadata(dataset_dir: str) -> dict[str, Any]:
    """Load dataset metadata to understand feature configuration."""
    metadata_path = os.path.join(dataset_dir, "dataset_metadata.json")
    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "feature_types": ["firings"],
            "num_features": 1,
        }
