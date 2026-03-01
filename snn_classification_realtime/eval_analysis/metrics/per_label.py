"""Per-label statistics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from snn_classification_realtime.eval_analysis.utils import nanpercentile, safe_float


def compute_per_label(
    results: list[dict[str, Any]],
    num_classes: int,
    class_labels: list[str],
) -> dict[str, dict[str, Any]]:
    by_label: dict[int, list[dict[str, Any]]] = {i: [] for i in range(num_classes)}
    for r in results:
        al = r.get("actual_label")
        if al is not None and al in by_label:
            by_label[al].append(r)

    out: dict[str, dict[str, Any]] = {}
    for label in range(num_classes):
        rows = by_label[label]
        name = class_labels[label] if label < len(class_labels) else str(label)
        if not rows:
            out[name] = {
                "accuracy": {},
                "confidence": {},
                "timing": {},
                "thinking": {},
                "stability": {},
            }
            continue

        correct = [r for r in rows if r.get("correct", False)]
        incorrect = [r for r in rows if not r.get("correct", True)]
        acc = 100.0 * len(correct) / len(rows)

        confs = [safe_float(r.get("confidence")) for r in rows]
        confs_valid = [c for c in confs if not math.isnan(c)]
        conf_correct = [safe_float(r.get("confidence")) for r in correct]
        conf_incorrect = [safe_float(r.get("confidence")) for r in incorrect]
        conf_correct = [c for c in conf_correct if not math.isnan(c)]
        conf_incorrect = [c for c in conf_incorrect if not math.isnan(c)]

        first_ticks = [
            safe_float(r.get("first_correct_tick"))
            for r in rows
            if r.get("first_correct_tick") is not None
        ]
        first_ticks = [t for t in first_ticks if not math.isnan(t)]

        thinking_used = [r for r in rows if r.get("used_extended_thinking", False)]
        ticks_added = [r.get("total_ticks_added", 0) for r in thinking_used]
        base_correct_count = sum(1 for r in rows if r.get("base_time_correct", False))
        final_correct_count = len(correct)
        base_acc_label = 100.0 * base_correct_count / len(rows)
        final_acc_label = 100.0 * final_correct_count / len(rows)

        appeared_wrong = [
            r for r in rows if r.get("had_correct_appearance_but_wrong_final", False)
        ]
        rescue_second = sum(
            1
            for r in rows
            if r.get("second_correct", False) and not r.get("correct", False)
        )
        rescue_third = sum(
            1
            for r in rows
            if r.get("third_correct", False)
            and not r.get("second_correct", False)
            and not r.get("correct", False)
        )

        out[name] = {
            "accuracy": {
                "mean": round(acc, 4),
                "std": 0.0,
                "total_samples": len(rows),
                "correct_count": len(correct),
            },
            "confidence": {
                "mean": round(float(np.mean(confs_valid)), 4) if confs_valid else None,
                "std": round(float(np.std(confs_valid)), 4)
                if len(confs_valid) > 1
                else None,
                "min": round(min(confs_valid), 4) if confs_valid else None,
                "max": round(max(confs_valid), 4) if confs_valid else None,
                "p25": round(nanpercentile(confs_valid, 25), 4) if confs_valid else None,
                "p75": round(nanpercentile(confs_valid, 75), 4) if confs_valid else None,
                "mean_correct": round(float(np.mean(conf_correct)), 4)
                if conf_correct
                else None,
                "mean_incorrect": round(float(np.mean(conf_incorrect)), 4)
                if conf_incorrect
                else None,
            },
            "timing": {
                "mean_first_correct_tick": round(float(np.mean(first_ticks)), 2)
                if first_ticks
                else None,
                "std_first_correct_tick": round(float(np.std(first_ticks)), 2)
                if len(first_ticks) > 1
                else None,
                "median_first_correct_tick": round(float(np.median(first_ticks)), 2)
                if first_ticks
                else None,
                "p25_first_correct_tick": round(nanpercentile(first_ticks, 25), 2)
                if first_ticks
                else None,
                "p75_first_correct_tick": round(nanpercentile(first_ticks, 75), 2)
                if first_ticks
                else None,
            },
            "thinking": {
                "mean_ticks_added": round(float(np.mean(ticks_added)), 2)
                if ticks_added
                else 0.0,
                "fraction_using_thinking": round(len(thinking_used) / len(rows), 4),
                "base_accuracy": round(base_acc_label, 4),
                "final_accuracy": round(final_acc_label, 4),
                "accuracy_delta": round(final_acc_label - base_acc_label, 4),
            },
            "stability": {
                "had_correct_appearance_but_wrong_final_count": len(appeared_wrong),
                "had_correct_appearance_but_wrong_final_fraction": round(
                    len(appeared_wrong) / len(rows), 4
                ),
                "top2_rescue_count": rescue_second,
                "top3_rescue_count": rescue_third,
            },
        }
    return out
