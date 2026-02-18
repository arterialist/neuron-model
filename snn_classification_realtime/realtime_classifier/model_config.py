"""Model configuration loading."""

import json


def load_model_config(model_path: str) -> dict:
    """Load model configuration from the config file."""
    config_path = model_path.replace(".pth", "_config.json")
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "feature_types": ["firings"],
            "num_features": 1,
        }
