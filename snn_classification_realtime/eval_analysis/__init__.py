"""Eval analysis: load JSONL/summary, compute metrics, write JSON/Markdown."""

from __future__ import annotations

from typing import Any

from snn_classification_realtime.eval_analysis.loaders import (
    load_jsonl,
    load_summary,
    resolve_summary_path,
)
from snn_classification_realtime.eval_analysis.metrics import compute_all
from snn_classification_realtime.eval_analysis.output import write_json, write_markdown

__all__ = [
    "run_analysis",
    "load_jsonl",
    "load_summary",
    "resolve_summary_path",
    "compute_all",
    "write_json",
    "write_markdown",
]


def run_analysis(
    jsonl_path: str,
    output_dir: str,
    *,
    summary_path: str | None = None,
    num_classes: int = 10,
    class_labels: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Load JSONL, compute all metrics, return (metrics dict, class_labels).

    Caller may write files with write_json(metrics, path) and write_markdown(metrics, path, class_labels).
    If metadata is provided (e.g. from summary JSON), it is merged into the result.
    class_labels defaults to 0..N-1 or CIFAR-10 names if metadata.dataset_name has 'cifar'.
    """
    results = load_jsonl(jsonl_path)
    if not results:
        raise ValueError("No records in JSONL.")
    summary_path = summary_path or resolve_summary_path(jsonl_path)
    meta: dict[str, Any] = metadata or {}
    if summary_path:
        try:
            summary_data = load_summary(summary_path)
            meta = summary_data.get("evaluation_metadata", {}) or {}
        except OSError:
            pass
    if metadata is not None:
        meta = {**meta, **metadata}

    n = num_classes
    if meta.get("num_classes") is not None:
        n = int(meta["num_classes"])

    labels = class_labels
    if labels is None:
        if meta.get("dataset_name", "").lower().find("cifar") >= 0:
            labels = [
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck",
            ]
        else:
            labels = [str(i) for i in range(n)]
    while len(labels) < n:
        labels.append(str(len(labels)))

    metrics = compute_all(results, n, labels)
    metrics["metadata"] = meta
    return metrics, labels
