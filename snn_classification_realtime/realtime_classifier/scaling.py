"""Apply FeatureScaler state to activity snapshots."""

from typing import Any

import torch


def apply_scaling_to_snapshot(
    snapshot: list[float],
    scaler_state: dict[str, Any],
) -> list[float]:
    """Apply saved FeatureScaler state (from prepare_activity_data) to a 1D snapshot.

    The scaler_state is a dict with keys: method, eps, and optionally mean/std, min/max, max_abs.
    """
    if not scaler_state:
        return snapshot
    method = scaler_state.get("method", "none")
    if method == "none":
        return snapshot

    x = torch.tensor(snapshot, dtype=torch.float32)
    eps = float(scaler_state.get("eps", 1e-8))

    if method == "standard":
        mean = scaler_state.get("mean")
        std = scaler_state.get("std")
        if mean is None or std is None:
            return snapshot
        mean_t = mean.to(dtype=torch.float32)
        std_t = std.to(dtype=torch.float32)
        denom = std_t + eps
        y = (x - mean_t) / denom
        return y.tolist()

    if method == "minmax":
        vmin = scaler_state.get("min")
        vmax = scaler_state.get("max")
        if vmin is None or vmax is None:
            return snapshot
        vmin_t = vmin.to(dtype=torch.float32)
        vmax_t = vmax.to(dtype=torch.float32)
        scale = vmax_t - vmin_t
        scale = torch.where(scale == 0, torch.full_like(scale, 1.0), scale)
        y = (x - vmin_t) / (scale + eps)
        return y.tolist()

    if method == "maxabs":
        max_abs = scaler_state.get("max_abs")
        if max_abs is None:
            return snapshot
        max_abs_t = max_abs.to(dtype=torch.float32)
        denom = max_abs_t + eps
        y = x / denom
        return y.tolist()

    return snapshot
