"""Accuracy metrics from evaluation results."""

from __future__ import annotations

from typing import Any


def compute_accuracy(results: list[dict[str, Any]], num_classes: int) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {}
    correct = sum(1 for r in results if r.get("correct", False))
    second_correct = sum(
        1 for r in results if r.get("correct") or r.get("second_correct", False)
    )
    third_correct = sum(
        1
        for r in results
        if r.get("correct") or r.get("second_correct") or r.get("third_correct", False)
    )
    second_strict = sum(
        1 for r in results if r.get("correct") or r.get("second_correct_strict", False)
    )
    third_strict = sum(
        1
        for r in results
        if r.get("correct")
        or r.get("second_correct_strict")
        or r.get("third_correct_strict", False)
    )
    bistab_correct = sum(
        1
        for r in results
        if r.get("bistability_rescue_correct", r.get("correct", False))
    )
    first_acc = 100.0 * correct / n
    second_acc = 100.0 * second_correct / n
    third_acc = 100.0 * third_correct / n
    second_strict_acc = 100.0 * second_strict / n
    third_strict_acc = 100.0 * third_strict / n
    bistab_acc = 100.0 * bistab_correct / n
    bistab_improvement = bistab_acc - first_acc

    processed = [r for r in results if r.get("predicted_label") is not None]
    base_correct = sum(1 for r in processed if r.get("base_time_correct", False))
    final_correct = sum(1 for r in processed if r.get("correct", False))
    pn = len(processed) or 1
    base_time_accuracy = 100.0 * base_correct / pn
    final_accuracy = 100.0 * final_correct / pn

    return {
        "first_choice_accuracy": round(first_acc, 4),
        "second_choice_accuracy": round(second_acc, 4),
        "third_choice_accuracy": round(third_acc, 4),
        "strict_second_choice_accuracy": round(second_strict_acc, 4),
        "strict_third_choice_accuracy": round(third_strict_acc, 4),
        "bistability_rescue_accuracy": round(bistab_acc, 4),
        "bistability_rescue_improvement": round(bistab_improvement, 4),
        "total_errors_first_choice": n - correct,
        "total_errors_second_choice": n - second_correct,
        "total_errors_third_choice": n - third_correct,
        "strict_total_errors_second_choice": n - second_strict,
        "strict_total_errors_third_choice": n - third_strict,
        "zero_confidence_contribution_second": second_correct - second_strict,
        "zero_confidence_contribution_third": third_correct - third_strict,
        "base_time_accuracy": round(base_time_accuracy, 4),
        "final_accuracy": round(final_accuracy, 4),
        "accuracy_improvement": round(final_accuracy - base_time_accuracy, 4),
    }
