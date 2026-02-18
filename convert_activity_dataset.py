#!/usr/bin/env python3
"""
Convert old JSON activity datasets to new HDF5 format.

This script takes an activity dataset in the legacy JSON format and converts it
to the new HDF5-based format with proper directory structure.

The old JSON format contains records with:
- image_index, label, tick
- layers[].neurons[] with S, t_ref, F_avg, fired values

The new HDF5 format stores data in compressed HDF5 files with:
- activity_dataset.h5 containing u, t_ref, fr, spikes, labels datasets
- Proper neuron ID mapping and layer structure metadata

Usage:
    python convert_activity_dataset.py <json_file> [--output-dir <dir>] [--force]

Arguments:
    json_file: Path to the input JSON activity dataset file
    --output-dir: Output directory (default: activity_datasets/)
    --force: Overwrite existing output directory if it exists

Example:
    python convert_activity_dataset.py old_dataset.json
    # Creates: activity_datasets/old_dataset/ with activity_dataset.h5
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List


# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from snn_classification_realtime.build_activity_dataset import HDF5TensorRecorder
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running from the neuron-model directory")
    sys.exit(1)


class MockNeuron:
    """Mock neuron class for HDF5TensorRecorder compatibility."""

    def __init__(self, neuron_id, layer=0):
        self.neuron_id = neuron_id
        self.metadata = {"layer": layer}
        self.S = 0.0
        self.t_ref = 0.0
        self.F_avg = 0.0
        self.O = 0  # Output/firing state


class MockNetwork:
    """Mock network class for HDF5TensorRecorder compatibility."""

    def __init__(self, neurons):
        self.neurons = {nid: MockNeuron(nid) for nid in neurons}


def load_json_dataset(json_path: str) -> List[Dict[str, Any]]:
    """Load activity dataset from JSON file."""
    print(f"Loading JSON dataset from: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    # Handle different JSON formats
    if isinstance(data, dict) and "records" in data:
        records = data["records"]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("Unsupported JSON format. Expected list or {'records': [...]}")

    print(f"Loaded {len(records)} records")
    return records


def extract_neuron_ids(records: List[Dict[str, Any]]) -> List[int]:
    """Extract all unique neuron IDs from the dataset."""
    neuron_ids = set()

    for record in records:
        layers = record.get("layers", [])
        for layer in layers:
            # Handle both formats: 'neurons' list of dicts, or 'neuron_ids' array
            if "neurons" in layer:
                # Format: [{'neuron_id': id, ...}, ...]
                neurons = layer.get("neurons", [])
                for neuron in neurons:
                    neuron_ids.add(neuron["neuron_id"])
            elif "neuron_ids" in layer:
                # Format: {'neuron_ids': [id1, id2, ...], 'S': [...], ...}
                neuron_ids.update(layer["neuron_ids"])

    return sorted(list(neuron_ids))


def group_records_by_image(
    records: List[Dict[str, Any]],
) -> Dict[tuple, List[Dict[str, Any]]]:
    """Group records by (label, image_index) pairs."""
    buckets = {}

    for record in records:
        label = record.get("label", 0)
        image_index = record.get("image_index", 0)
        key = (label, image_index)

        if key not in buckets:
            buckets[key] = []
        buckets[key].append(record)

    # Sort records within each bucket by tick
    for key in buckets:
        buckets[key].sort(key=lambda r: r.get("tick", 0))

    return buckets


def convert_json_to_hdf5(json_path: str, output_dir: str, force: bool = False) -> str:
    """Convert JSON dataset to HDF5 format."""
    # Load JSON data
    records = load_json_dataset(json_path)

    if not records:
        raise ValueError("No records found in JSON file")

    # Extract neuron IDs
    neuron_ids = extract_neuron_ids(records)
    print(f"Found {len(neuron_ids)} unique neurons")

    # Create mock network
    mock_network = MockNetwork(neuron_ids)

    # Determine output directory
    json_basename = Path(json_path).stem
    output_dataset_dir = os.path.join(output_dir, json_basename)

    # Check if output directory exists
    if os.path.exists(output_dataset_dir):
        if not force:
            print(f"Output directory already exists: {output_dataset_dir}")
            print("Use --force to overwrite")
            return output_dataset_dir
        else:
            import shutil

            shutil.rmtree(output_dataset_dir)
            print(f"Removed existing directory: {output_dataset_dir}")

    # Group records by image
    image_buckets = group_records_by_image(records)
    print(f"Found {len(image_buckets)} unique images")

    # Create HDF5 recorder
    recorder = HDF5TensorRecorder(output_dataset_dir, mock_network)

    # Process each image
    sample_idx = 0
    for (label, image_idx), image_records in sorted(image_buckets.items()):
        # Determine ticks per image
        ticks = len(image_records)
        recorder.init_buffer(ticks)

        # Process each tick
        for tick in range(ticks):
            record = image_records[tick]

            # Extract neuron data for this tick
            for neuron_id in neuron_ids:
                neuron = mock_network.neurons[neuron_id]

                # Find this neuron in the current record
                found = False
                for layer in record.get("layers", []):
                    if "neurons" in layer:
                        # Handle 'neurons' format: [{'neuron_id': id, ...}, ...]
                        for neuron_data in layer.get("neurons", []):
                            if neuron_data["neuron_id"] == neuron_id:
                                # Set neuron values
                                neuron.S = neuron_data.get("S", 0.0)
                                neuron.t_ref = neuron_data.get("t_ref", 0.0)
                                neuron.F_avg = neuron_data.get("F_avg", 0.0)
                                neuron.O = neuron_data.get(
                                    "fired", 0
                                )  # Set firing state
                                found = True
                                break
                    elif "neuron_ids" in layer:
                        # Handle 'neuron_ids' format: {'neuron_ids': [...], 'S': [...], ...}
                        neuron_ids_in_layer = layer.get("neuron_ids", [])
                        try:
                            neuron_idx = neuron_ids_in_layer.index(neuron_id)
                            # Set neuron values from corresponding arrays
                            neuron.S = (
                                layer.get("S", [0.0])[neuron_idx]
                                if neuron_idx < len(layer.get("S", []))
                                else 0.0
                            )
                            neuron.t_ref = (
                                layer.get("t_ref", [0.0])[neuron_idx]
                                if neuron_idx < len(layer.get("t_ref", []))
                                else 0.0
                            )
                            neuron.F_avg = (
                                layer.get("F_avg", [0.0])[neuron_idx]
                                if neuron_idx < len(layer.get("F_avg", []))
                                else 0.0
                            )
                            neuron.O = (
                                layer.get("fired", [0])[neuron_idx]
                                if neuron_idx < len(layer.get("fired", []))
                                else 0
                            )
                            found = True
                            break
                        except ValueError:
                            # neuron_id not in this layer
                            continue
                    if found:
                        break

                if not found:
                    # Neuron not present in this tick, set defaults
                    neuron.S = 0.0
                    neuron.t_ref = 0.0
                    neuron.F_avg = 0.0
                    neuron.O = 0

            # Capture the tick (this will handle spikes via neuron.O)
            recorder.capture_tick(tick, mock_network.neurons)

        # Save the sample with sequential index
        recorder.save_sample(sample_idx, int(label))
        sample_idx += 1

    # Close the recorder
    recorder.close()

    print(f"Successfully converted dataset to: {output_dataset_dir}")
    print(f"Created {len(image_buckets)} samples with {len(neuron_ids)} neurons each")

    return output_dataset_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert old JSON activity datasets to new HDF5 format"
    )
    parser.add_argument(
        "json_file", type=str, help="Path to the input JSON activity dataset file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="activity_datasets",
        help="Output directory for converted datasets (default: activity_datasets/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory if it exists",
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.json_file):
        print(f"Error: Input file does not exist: {args.json_file}")
        sys.exit(1)

    if not args.json_file.lower().endswith(".json"):
        print(f"Warning: Input file does not have .json extension: {args.json_file}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        output_path = convert_json_to_hdf5(args.json_file, args.output_dir, args.force)
        print("\n✅ Conversion completed successfully!")
        print(f"Output: {output_path}")
        print(f"You can now use this dataset with: --input-file {output_path}")

    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
