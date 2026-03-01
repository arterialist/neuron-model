"""Stability metrics (bistability rescue, appearance vs final)."""

from __future__ import annotations

from typing import Any


def compute_stability(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    errors_first = [r for r in results if not r.get("correct", True)]
    bistab_rescued = sum(
        1 for r in errors_first if r.get("bistability_rescue_correct", False)
    )
    total_errors = len(errors_first)
    bistability_rate = bistab_rescued / total_errors if total_errors > 0 else None

    appeared_but_wrong = [
        r for r in results if r.get("had_correct_appearance_but_wrong_final", False)
    ]
    appeared_correct_final = [
        r
        for r in results
        if r.get("first_correct_appearance_tick") is not None
        and r.get("correct", False)
    ]
    correct_appearances = [
        r for r in results if r.get("first_correct_appearance_tick") is not None
    ]
    stability_rate = (
        100.0 * len(appeared_correct_final) / len(correct_appearances)
        if correct_appearances
        else None
    )

    base_ticks = next(
        (
            r.get("base_ticks_per_image")
            for r in results
            if r.get("base_ticks_per_image") is not None
        ),
        None,
    )
    early_convergence = 0
    if base_ticks is not None:
        early_convergence = sum(
            1
            for r in results
            if r.get("first_correct_tick") is not None
            and r["first_correct_tick"] < base_ticks
        )
    early_convergence_fraction = early_convergence / n if n else 0

    return {
        "bistability_rescue_rate": round(bistability_rate, 4)
        if bistability_rate is not None
        else None,
        "appeared_but_wrong_final_count": len(appeared_but_wrong),
        "appeared_and_correct_final_count": len(appeared_correct_final),
        "total_correct_appearances": len(correct_appearances),
        "correct_prediction_stability_rate": round(stability_rate, 4)
        if stability_rate is not None
        else None,
        "early_convergence_count": early_convergence,
        "early_convergence_fraction": round(early_convergence_fraction, 4),
    }
