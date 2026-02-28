"""
YAML config loading and validation for the standalone network builder.

Supports standalone format (top-level dataset, layers, output_dir, name)
and pipeline-embedded format (network.build_config).
"""

from pathlib import Path
from typing import Any

import yaml


def load_network_config_yaml(path: str | Path) -> dict[str, Any]:
    """Load and validate network builder config from a YAML file.

    Accepts standalone format with top-level dataset, layers, optional
    name, output_dir, inhibitory_signals, rgb_separate_neurons, input_size.
    Also accepts pipeline-embedded format (nested under network.build_config);
    in that case the returned dict has the build_config content plus optional
    top-level name/output_dir if present.

    Args:
        path: Path to YAML file.

    Returns:
        Config dict suitable for build_network() and for output path resolution
        (name, output_dir). Keys: dataset, layers, optional name, output_dir,
        inhibitory_signals, rgb_separate_neurons, input_size.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError("Empty YAML file")

    if "network" in raw and "build_config" in raw["network"]:
        embedded = raw["network"]["build_config"]
        if hasattr(embedded, "model_dump"):
            cfg = embedded.model_dump()
        elif hasattr(embedded, "dict"):
            cfg = embedded.dict()
        else:
            cfg = dict(embedded)
        if "dataset" in cfg and hasattr(cfg["dataset"], "value"):
            cfg["dataset"] = cfg["dataset"].value
        layers = cfg.get("layers", [])
        cfg["layers"] = [
            layer.model_dump() if hasattr(layer, "model_dump") else (layer.dict() if hasattr(layer, "dict") else dict(layer))
            for layer in layers
        ]
        if isinstance(raw.get("network"), dict):
            if "name" in raw["network"]:
                cfg["name"] = raw["network"]["name"]
            if "output_dir" in raw["network"]:
                cfg["output_dir"] = raw["network"]["output_dir"]
        return cfg

    cfg = {}
    if "dataset" in raw:
        cfg["dataset"] = raw["dataset"]
    else:
        raise ValueError("YAML must contain 'dataset' (or use pipeline format with network.build_config)")
    cfg["layers"] = raw.get("layers", [])
    if not cfg["layers"]:
        raise ValueError("YAML must contain non-empty 'layers'")
    cfg["inhibitory_signals"] = raw.get("inhibitory_signals", False)
    cfg["rgb_separate_neurons"] = raw.get("rgb_separate_neurons", False)
    cfg["input_size"] = raw.get("input_size", 100)
    if "name" in raw:
        cfg["name"] = raw["name"]
    if "output_dir" in raw:
        cfg["output_dir"] = raw["output_dir"]
    return cfg


def get_output_name_and_dir(
    config_path: str | Path,
    config: dict[str, Any],
    *,
    output_dir_override: str | Path | None = None,
    name_override: str | None = None,
) -> tuple[str, Path]:
    """Resolve output filename (without extension) and output directory.

    Name: name_override > config["name"] > config filename stem.
    Output dir: output_dir_override > config["output_dir"]; one must be provided.

    Returns:
        (name_without_extension, output_dir_path).
    """
    name = name_override or config.get("name")
    if not name:
        name = Path(config_path).stem
    output_dir = output_dir_override
    if output_dir is None and config.get("output_dir") is not None:
        output_dir = Path(config["output_dir"])
    elif output_dir is not None:
        output_dir = Path(output_dir)
    if output_dir is None:
        raise ValueError(
            "Output directory must be set via --output-dir or 'output_dir' in YAML"
        )
    return (str(name), Path(output_dir))
