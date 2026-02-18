"""Per-dimension feature scaling for variable-length time series."""

from typing import Any, Optional

import torch


class FeatureScaler:
    """Fits per-dimension scaling statistics over variable-length time series.

    Scaling is computed across ALL timesteps of the training samples for each
    feature dimension independently to avoid temporal information leakage.
    """

    def __init__(self, method: str, eps: float = 1e-8) -> None:
        self.method = method
        self.eps = float(eps)
        self._fitted = False
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None
        self._min: Optional[torch.Tensor] = None
        self._max: Optional[torch.Tensor] = None
        self._max_abs: Optional[torch.Tensor] = None

    def fit(self, samples: list[torch.Tensor]) -> None:
        if self.method == "none":
            self._fitted = True
            return

        if not samples:
            raise ValueError("Cannot fit scaler on empty sample list")

        first = samples[0]
        if first.ndim != 2:
            raise ValueError("Each sample must be a 2D tensor [T, D]")
        feature_dim = int(first.shape[1])

        if self.method == "standard":
            running_sum = torch.zeros(feature_dim, dtype=torch.float32)
            running_sumsq = torch.zeros(feature_dim, dtype=torch.float32)
            total_count = 0
            for ts in samples:
                if ts.numel() == 0:
                    continue
                running_sum += ts.sum(dim=0)
                running_sumsq += (ts * ts).sum(dim=0)
                total_count += int(ts.shape[0])
            if total_count == 0:
                raise ValueError("All training samples are empty; cannot fit scaler")
            mean = running_sum / float(total_count)
            var = torch.clamp(
                running_sumsq / float(total_count) - mean * mean, min=0.0
            )
            self._mean = mean
            self._std = torch.sqrt(var)
            self._fitted = True
            return

        if self.method == "minmax":
            cur_min = torch.full((feature_dim,), float("inf"), dtype=torch.float32)
            cur_max = torch.full((feature_dim,), float("-inf"), dtype=torch.float32)
            for ts in samples:
                if ts.numel() == 0:
                    continue
                tmin = ts.min(dim=0).values
                tmax = ts.max(dim=0).values
                cur_min = torch.minimum(cur_min, tmin)
                cur_max = torch.maximum(cur_max, tmax)
            self._min = cur_min
            self._max = cur_max
            self._fitted = True
            return

        if self.method == "maxabs":
            cur_max_abs = torch.zeros(feature_dim, dtype=torch.float32)
            for ts in samples:
                if ts.numel() == 0:
                    continue
                tmax_abs = ts.abs().max(dim=0).values
                cur_max_abs = torch.maximum(cur_max_abs, tmax_abs)
            self._max_abs = cur_max_abs
            self._fitted = True
            return

        raise ValueError(f"Unknown scaler method: {self.method}")

    def transform(self, sample: torch.Tensor) -> torch.Tensor:
        if not self._fitted or self.method == "none" or sample.numel() == 0:
            return sample
        if sample.ndim != 2:
            raise ValueError("Each sample must be a 2D tensor [T, D]")

        if self.method == "standard":
            assert self._mean is not None and self._std is not None
            denom = self._std + self.eps
            return (sample - self._mean) / denom

        if self.method == "minmax":
            assert self._min is not None and self._max is not None
            scale = self._max - self._min
            scale = torch.where(scale == 0, torch.full_like(scale, 1.0), scale)
            return (sample - self._min) / (scale + self.eps)

        if self.method == "maxabs":
            assert self._max_abs is not None
            denom = self._max_abs + self.eps
            return sample / denom

        return sample

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {"method": self.method, "eps": self.eps}
        if self._mean is not None:
            state["mean"] = self._mean.detach().cpu()
        if self._std is not None:
            state["std"] = self._std.detach().cpu()
        if self._min is not None:
            state["min"] = self._min.detach().cpu()
        if self._max is not None:
            state["max"] = self._max.detach().cpu()
        if self._max_abs is not None:
            state["max_abs"] = self._max_abs.detach().cpu()
        return state
