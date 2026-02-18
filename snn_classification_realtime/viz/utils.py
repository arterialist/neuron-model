"""Shared utilities for visualization and analysis scripts."""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class VizConfig:
    """Configuration for visualization exports and caching."""

    plot_image_scale: float = 2.0
    matplotlib_dpi: int = 400
    cache_version: str = "v1"


def compute_dataset_hash(file_path: str) -> str:
    """Return a short MD5 hash for the dataset file.

    Handles both single files (JSON) and directories (binary/HDF5).
    Prefers shell md5sum/md5 when available; falls back to Python hashlib.
    Returns the first 16 hex characters (lowercase).
    """
    if os.path.isdir(file_path):
        h5_path = os.path.join(file_path, "activity_dataset.h5")
        if not os.path.exists(h5_path):
            raise RuntimeError(
                f"Binary dataset directory {file_path} does not contain activity_dataset.h5"
            )
        actual_path = h5_path
    else:
        actual_path = file_path

    try:
        result = subprocess.run(
            ["md5sum", actual_path],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout.strip().split()[0].lower()[:16]
    except FileNotFoundError:
        pass

    try:
        result = subprocess.run(
            ["md5", "-q", actual_path],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout.strip().split()[0].lower()[:16]
    except FileNotFoundError:
        pass

    hasher = hashlib.md5()
    with open(actual_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def get_cache_dir(base_output_dir: str, cache_version: str | None = "v1") -> str:
    """Return (and create) the cache directory for intermediates."""
    subdir = f"cache_{cache_version}" if cache_version else "cache"
    cache_dir = os.path.join(base_output_dir, subdir)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def format_float_token(value: float, decimals: int = 6) -> str:
    """Return a stable, filename-safe token for a float parameter."""
    return f"{float(value):.{decimals}f}"


def log_plot_start(plot_name: str, scope: Optional[str] = None) -> None:
    """Log the start of a plot generation."""
    if scope:
        print(f"\n[Plot] Starting {plot_name} ({scope})...")
    else:
        print(f"\n[Plot] Starting {plot_name}...")


def log_plot_end(plot_name: str, scope: Optional[str] = None) -> None:
    """Log the completion of a plot generation."""
    if scope:
        print(f"[Plot] Completed {plot_name} ({scope})")
    else:
        print(f"[Plot] Completed {plot_name}")
