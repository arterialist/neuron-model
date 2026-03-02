"""Model configuration loading."""

import json
import os


def resolve_best_model_path(path: str) -> str:
    """Resolve path to the model with lowest test loss.

    Accepts a directory or a .pth file. If a file is passed, uses its parent
    directory for the search. Scans for all .pth files with matching _config.json,
    picks the one with lowest test_losses[-1] or final_test_loss.
    Returns the original path if no configs with test loss are found.
    """
    path = os.path.normpath(path)
    if os.path.isfile(path) and path.endswith(".pth"):
        search_dir = os.path.dirname(path)
    elif os.path.isdir(path):
        search_dir = path
    else:
        return path

    candidates: list[tuple[float, int, str]] = []

    for name in os.listdir(search_dir):
        if not name.endswith(".pth"):
            continue
        model_path = os.path.join(search_dir, name)
        config_path = model_path.replace(".pth", "_config.json")
        if not os.path.isfile(config_path):
            continue
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        loss = config.get("final_test_loss")
        if loss is None:
            losses = config.get("test_losses") or []
            loss = losses[-1] if losses else None
        if loss is None:
            continue
        epoch = config.get("checkpoint_epoch") or config.get("completed_epochs")
        if epoch is None:
            epoch = 99999
        candidates.append((loss, epoch, model_path))

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[0][2]
    if os.path.isfile(path) and path.endswith(".pth"):
        return path
    pth_files = [f for f in os.listdir(search_dir) if f.endswith(".pth")]
    if pth_files:
        return os.path.join(search_dir, sorted(pth_files)[0])
    return path


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
