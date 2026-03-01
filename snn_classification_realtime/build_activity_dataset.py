"""Build activity dataset from network simulation.

CLI-only. All parameters via arguments.
"""

import argparse
import os
import sys

# Ensure workspace root is in path for package imports
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from snn_classification_realtime.activity_dataset_builder import (
    HDF5TensorRecorder,
    LazyActivityDataset,
)
from snn_classification_realtime.activity_dataset_builder.build import run_build
from snn_classification_realtime.activity_dataset_builder.vision_datasets import (
    DATASET_NAMES,
)

__all__ = ["HDF5TensorRecorder", "LazyActivityDataset", "main", "run_build"]


def main() -> None:
    """Entry point for the activity dataset builder CLI."""
    parser = argparse.ArgumentParser(
        description="Build activity dataset from network simulation."
    )
    parser.add_argument(
        "--network-path",
        type=str,
        required=True,
        help="Path to network JSON file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print recommended ticks per image and exit (no recording)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="mnist",
        choices=list(DATASET_NAMES),
        help="Vision dataset name (default: mnist)",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        choices=[
            "none",
            "tref_frozen",
            "retrograde_disabled",
            "weight_update_disabled",
            "thresholds_frozen",
            "directional_error_disabled",
        ],
        help="Neuron ablation variant (default: none)",
    )
    parser.add_argument(
        "--ticks-per-image",
        type=int,
        default=None,
        help="Simulation ticks per image (default: auto from network)",
    )
    parser.add_argument(
        "--images-per-label",
        type=int,
        default=100,
        help="Number of images to present per label (default: 5)",
    )
    parser.add_argument(
        "--tick-ms",
        type=int,
        default=0,
        help="Tick delay in milliseconds, 0 = no delay (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="activity_datasets",
        help="Output directory for dataset (default: activity_datasets)",
    )
    parser.add_argument(
        "--dataset-base",
        type=str,
        default=None,
        help="Dataset name base (default: network filename stem)",
    )
    parser.add_argument(
        "--fresh-run-per-label",
        action="store_true",
        default=True,
        help="Reload network for each label (default: True)",
    )
    parser.add_argument(
        "--fresh-run-per-image",
        action="store_true",
        default=True,
        help="Reload network for each image (default: False)",
    )
    parser.add_argument(
        "--use-multiprocessing",
        action="store_true",
        default=True,
        help="Use multiprocessing for parallel processing",
    )
    parser.add_argument(
        "--export-network-states",
        action="store_true",
        default=False,
        help="Export network state after each sample (for synaptic analysis)",
    )
    parser.add_argument(
        "--start-web-server",
        action="store_true",
        default=False,
        help="Start web visualization server during collection",
    )
    parser.add_argument(
        "--cifar10-color-normalization-factor",
        type=float,
        default=0.5,
        help="CIFAR10 color normalization factor (default: 0.165)",
    )
    args = parser.parse_args()

    from snn_classification_realtime.activity_dataset_builder.network_utils import (
        compute_default_ticks_per_image,
        infer_layers_from_metadata,
    )
    from neuron.network_config import NetworkConfig

    if not os.path.isfile(args.network_path):
        print(f"Network file not found: {args.network_path}")
        sys.exit(1)

    network_sim = NetworkConfig.load_network_config(args.network_path)
    layers = infer_layers_from_metadata(network_sim)
    default_ticks = compute_default_ticks_per_image(network_sim, layers)
    ticks_per_image = (
        args.ticks_per_image if args.ticks_per_image is not None else default_ticks
    )

    if args.dry_run:
        print("--- Dry Run: Recommended Ticks per Image ---")
        print(f"Network: {args.network_path}")
        print(
            f"Detected {len(layers)} layers. Sizes: {[len(layer) for layer in layers]}"
        )
        print(f"Recommended ticks per image: {default_ticks}")
        print("Use --ticks-per-image to override.")
        return

    config = {
        "network_path": args.network_path,
        "dataset_name": args.dataset_name,
        "ablation": args.ablation,
        "ticks_per_image": ticks_per_image,
        "images_per_label": args.images_per_label,
        "tick_ms": args.tick_ms,
        "output_dir": args.output_dir,
        "dataset_base": args.dataset_base,
        "fresh_run_per_label": args.fresh_run_per_label,
        "fresh_run_per_image": args.fresh_run_per_image,
        "use_multiprocessing": args.use_multiprocessing,
        "export_network_states": args.export_network_states,
        "start_web_server": args.start_web_server,
        "cifar10_color_normalization_factor": args.cifar10_color_normalization_factor,
    }
    run_build(config=config)


if __name__ == "__main__":
    main()
