"""Timing metrics (ticks to correct, extended thinking)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from snn_classification_realtime.eval_analysis.utils import safe_float


def compute_timing(results: list[dict[str, Any]]) -> dict[str, Any]:
    first_ticks = [
        safe_float(r.get("first_correct_tick"))
        for r in results
        if r.get("first_correct_tick") is not None
    ]
    second_ticks = [
        safe_float(r.get("first_second_correct_tick"))
        for r in results
        if r.get("first_second_correct_tick") is not None
    ]
    third_ticks = [
        safe_float(r.get("first_third_correct_tick"))
        for r in results
        if r.get("first_third_correct_tick") is not None
    ]
    first_ticks = [t for t in first_ticks if not math.isnan(t)]
    second_ticks = [t for t in second_ticks if not math.isnan(t)]
    third_ticks = [t for t in third_ticks if not math.isnan(t)]

    thinking_results = [r for r in results if r.get("used_extended_thinking", False)]
    n = len(results)
    fraction_thinking = len(thinking_results) / n if n else 0

    def stats(arr: list[float]) -> dict[str, float | None]:
        if not arr:
            return {
                "mean": None,
                "std": None,
                "p25": None,
                "p50": None,
                "p75": None,
                "p90": None,
            }
        a = np.array(arr)
        return {
            "mean": round(float(np.mean(a)), 2),
            "std": round(float(np.std(a)), 2) if len(a) > 1 else 0.0,
            "p25": round(float(np.percentile(a, 25)), 2),
            "p50": round(float(np.median(a)), 2),
            "p75": round(float(np.percentile(a, 75)), 2),
            "p90": round(float(np.percentile(a, 90)), 2),
        }

    return {
        "first_correct_tick": stats(first_ticks),
        "first_second_correct_tick": stats(second_ticks),
        "first_third_correct_tick": stats(third_ticks),
        "fraction_using_extended_thinking": round(fraction_thinking, 4),
        "avg_ticks_added": round(
            float(np.mean([r["total_ticks_added"] for r in thinking_results])), 2
        )
        if thinking_results
        else 0,
    }
