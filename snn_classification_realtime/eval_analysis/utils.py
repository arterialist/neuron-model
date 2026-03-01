"""Shared helpers for eval analysis metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def safe_float(x: Any) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float("nan")
    return float(x)


def nanpercentile(arr: list[float], p: float) -> float:
    a = np.array([x for x in arr if not math.isnan(x)])
    if len(a) == 0:
        return float("nan")
    return float(np.nanpercentile(a, p))
