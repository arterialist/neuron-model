"""Entry point for concept hierarchy visualization."""

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from snn_classification_realtime.viz.concept_hierarchy.run import run_plot


def main() -> None:
    """Entry point for concept hierarchy CLI."""
    parser = argparse.ArgumentParser(
        description="Create concept hierarchy visualizations from evaluation results"
    )
    parser.add_argument("--json-file", type=str, required=True, help="Summary JSON (or legacy single-file with results)")
    parser.add_argument("--results-file", type=str, default=None, help="JSONL results file (when using split summary+JSONL format)")
    parser.add_argument("--output-dir", type=str, default="concept_hierarchy_output")
    args = parser.parse_args()

    print("=" * 80)
    print("PAULA CONCEPT HIERARCHY VISUALIZATION")
    print("=" * 80)
    print(f"Input file: {args.json_file}")
    if args.results_file:
        print(f"Results file: {args.results_file}")
    print(f"Output directory: {args.output_dir}")
    print()

    output_path = run_plot(args.json_file, args.output_dir, results_file_path=args.results_file)

    print()
    print("✓ Concept hierarchy analysis complete!")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
