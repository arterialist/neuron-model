"""Prepare network activity data for SNN training.

Entry point for the activity data preparation CLI.
All logic has been refactored into the activity_preparer package.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snn_classification_realtime.activity_preparer.config import PrepareConfig
from snn_classification_realtime.activity_preparer import (
    load_dataset,
    group_by_image,
    FeatureScaler,
)

__all__ = ["main", "load_dataset", "group_by_image", "FeatureScaler"]
from snn_classification_realtime.activity_preparer.run import run_prepare


def main() -> None:
    """Entry point for the prepare_activity_data CLI."""
    parser = argparse.ArgumentParser(
        description="Prepare network activity data for SNN training."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the input dataset directory (binary) or JSON file (with --legacy-json).",
    )
    parser.add_argument(
        "--legacy-json",
        action="store_true",
        help="Force loading as legacy JSON format instead of binary",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="prepared_data",
        help="Directory to save the processed datasets.",
    )
    parser.add_argument(
        "--feature-types",
        type=str,
        nargs="+",
        default=["firings"],
        choices=["firings", "avg_S", "avg_t_ref"],
        help=(
            "The temporal characteristics to extract. Can specify multiple features like: "
            "--feature-types firings avg_S avg_t_ref"
        ),
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="The fraction of data to use for the training set.",
    )
    parser.add_argument(
        "--use-streaming",
        action="store_true",
        help="Enable streaming mode for loading large JSON files. Uses ijson for memory-efficient processing.",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="none",
        choices=["none", "standard", "minmax", "maxabs"],
        help=(
            "Feature scaling method applied after the train/test split. "
            "'standard' uses z-score; 'minmax' scales to [0,1]; 'maxabs' scales by max absolute value."
        ),
    )
    parser.add_argument(
        "--scale-eps",
        type=float,
        default=1e-8,
        help="Numerical epsilon to avoid division by zero during scaling.",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="Maximum number of ticks to include per image presentation.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of image samples to process.",
    )
    args = parser.parse_args()

    config = PrepareConfig(
        input_file=args.input_file,
        output_dir=args.output_dir,
        feature_types=args.feature_types,
        train_split=args.train_split,
        use_streaming=args.use_streaming,
        scaler=args.scaler,
        scale_eps=args.scale_eps,
        max_ticks=args.max_ticks,
        max_samples=args.max_samples,
        legacy_json=args.legacy_json,
    )

    run_prepare(config)


if __name__ == "__main__":
    main()
