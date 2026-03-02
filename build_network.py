#!/usr/bin/env python3
"""
Standalone entrypoint to build a network from a YAML config file.

Usage:
  python build_network.py --config path/to/config.yaml [--output-dir DIR] [--name NAME]

Output path: {output_dir}/{name}.json
Name: --name > YAML "name" > config filename stem.
Output dir: --output-dir > YAML "output_dir"; one must be provided.
"""

import argparse
import logging
import sys
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neuron.network_config import NetworkConfig

from snn_classification_realtime.network_builder_direct import build_network_config_direct
from snn_classification_realtime.network_builder_config import (
    get_output_name_and_dir,
    load_network_config_yaml,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a neuron network from a YAML config file."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML network config (dataset, layers, optional name, output_dir)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the network JSON (overrides YAML output_dir)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Output filename without extension (overrides YAML name)",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Suppress logs and progress output (for agent context)",
    )
    args = parser.parse_args()

    if args.silent:
        logging.getLogger().setLevel(logging.WARNING)
        logger.setLevel(logging.WARNING)

    config = load_network_config_yaml(args.config)
    name, output_dir = get_output_name_and_dir(
        args.config,
        config,
        output_dir_override=args.output_dir,
        name_override=args.name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}.json"

    config_dict = build_network_config_direct(config, logger=logger)
    NetworkConfig.save_config_dict(config_dict, str(out_path))
    if not args.silent:
        logger.info("Saved network to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
