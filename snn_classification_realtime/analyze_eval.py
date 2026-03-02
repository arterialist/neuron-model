"""Analyze evaluation JSONL and produce metrics JSON + Markdown.

CLI-only. Delegates to the eval_analysis package for loading, metrics, and output.
"""

from __future__ import annotations

import argparse
import os
import sys

from snn_classification_realtime.eval_analysis import (
    run_analysis,
    write_json,
    write_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze evaluation JSONL and produce metrics JSON + Markdown."
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        required=True,
        help="Path to evaluation JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for metrics.json and metrics.md",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Path to evaluation summary JSON (auto-detected from JSONL if omitted)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of classes (default: 10)",
    )
    parser.add_argument(
        "--class-labels",
        type=str,
        default=None,
        help="Comma-separated class labels, e.g. airplane,automobile,... (default: 0..N-1)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["both", "json", "markdown"],
        default="both",
        help="Output format: both, json, or markdown (default: both)",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Suppress progress logs (for agent context)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.jsonl):
        print(f"JSONL file not found: {args.jsonl}")
        sys.exit(1)

    class_labels_arg: list[str] | None = None
    if args.class_labels:
        class_labels_arg = [s.strip() for s in args.class_labels.split(",")]

    try:
        metrics, class_labels = run_analysis(
            args.jsonl,
            args.output_dir,
            summary_path=args.summary,
            num_classes=args.num_classes,
            class_labels=class_labels_arg,
        )
    except ValueError as e:
        print(e)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, "metrics.json")
    out_md = os.path.join(args.output_dir, "metrics.md")

    if args.format in ("both", "json"):
        write_json(metrics, out_json)
        if not args.silent:
            print(f"Wrote {out_json}")
    if args.format in ("both", "markdown"):
        write_markdown(metrics, out_md, class_labels)
        if not args.silent:
            print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
