"""Eval metrics: accuracy, per-label, confusion, timing, concept hierarchy, etc."""

from __future__ import annotations

from typing import Any

from snn_classification_realtime.eval_analysis.metrics import accuracy as _accuracy
from snn_classification_realtime.eval_analysis.metrics import concept_hierarchy as _ch
from snn_classification_realtime.eval_analysis.metrics import confidence_calibration as _cal
from snn_classification_realtime.eval_analysis.metrics import confusion_matrix as _cm
from snn_classification_realtime.eval_analysis.metrics import per_label as _per_label
from snn_classification_realtime.eval_analysis.metrics import specialization as _spec
from snn_classification_realtime.eval_analysis.metrics import stability as _stability
from snn_classification_realtime.eval_analysis.metrics import timing as _timing

compute_accuracy = _accuracy.compute_accuracy
compute_per_label = _per_label.compute_per_label
compute_confusion_matrix = _cm.compute_confusion_matrix
compute_timing = _timing.compute_timing
compute_concept_hierarchy = _ch.compute_concept_hierarchy
compute_stability = _stability.compute_stability
compute_specialization = _spec.compute_specialization
compute_confidence_calibration = _cal.compute_confidence_calibration


def compute_all(
    results: list[dict[str, Any]],
    num_classes: int,
    class_labels: list[str],
) -> dict[str, Any]:
    """Compute all metrics from evaluation results."""
    return {
        "accuracy": compute_accuracy(results, num_classes),
        "per_label": compute_per_label(results, num_classes, class_labels),
        "confusion_matrix": compute_confusion_matrix(
            results, num_classes, class_labels
        ),
        "timing": compute_timing(results),
        "concept_hierarchy": compute_concept_hierarchy(
            results, num_classes, class_labels
        ),
        "stability": compute_stability(results),
        "specialization": compute_specialization(results, num_classes, class_labels),
        "confidence_calibration": compute_confidence_calibration(
            results, num_classes, class_labels
        ),
    }
