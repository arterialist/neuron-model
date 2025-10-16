import os
import json
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from tqdm import tqdm


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Loads the activity dataset from a JSON file."""
    with open(path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "records" in payload:
        return payload["records"]
    if isinstance(payload, list):
        return payload
    raise ValueError(
        "Unsupported dataset JSON format: Expected a list of records or a dict with a 'records' key."
    )


def group_by_image(
    records: List[Dict[str, Any]],
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """Groups records by (label, image_index) to collect per-tick data for each image presentation."""
    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for rec in tqdm(records, desc="Grouping records"):
        label = rec.get("label", -1)
        img_idx = rec.get("image_index", -1)
        if label == -1 or img_idx == -1:
            continue
        key = (int(label), int(img_idx))
        buckets.setdefault(key, []).append(rec)

    # Sort records within each bucket by tick to ensure correct temporal order
    for key in buckets:
        buckets[key].sort(key=lambda r: r.get("tick", 0))
    return buckets


def extract_firings_time_series(image_records: List[Dict[str, Any]]) -> torch.Tensor:
    """Extracts a time series of firings from records of a single image presentation."""
    time_series = []
    if not image_records:
        return torch.empty(0)

    for record in image_records:
        tick_firings = []
        for layer in record.get("layers", []):
            tick_firings.extend(layer.get("fired", []))
        time_series.append(tick_firings)

    return torch.tensor(time_series, dtype=torch.float32)


def extract_avg_S_time_series(image_records: List[Dict[str, Any]]) -> torch.Tensor:
    """Extracts a time series of average membrane potentials (S) from records."""
    time_series = []
    if not image_records:
        return torch.empty(0)

    for record in image_records:
        tick_s_values = []
        for layer in record.get("layers", []):
            tick_s_values.extend(layer.get("S", []))
        time_series.append(tick_s_values)

    return torch.tensor(time_series, dtype=torch.float32)


def extract_multi_feature_time_series(
    image_records: List[Dict[str, Any]], feature_types: List[str]
) -> torch.Tensor:
    """Extracts a time series combining multiple features from records of a single image presentation."""
    if not image_records:
        return torch.empty(0)

    # Extract each feature type separately
    feature_series = {}
    for feature_type in feature_types:
        if feature_type == "firings":
            feature_series[feature_type] = extract_firings_time_series(image_records)
        elif feature_type == "avg_S":
            feature_series[feature_type] = extract_avg_S_time_series(image_records)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

    # Concatenate features along the feature dimension (dim=1)
    # Each feature contributes its own dimension to the feature vector
    combined_series = torch.cat(list(feature_series.values()), dim=1)
    return combined_series


def main():
    parser = argparse.ArgumentParser(
        description="Prepare network activity data for SNN training."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the input JSON activity dataset.",
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
        choices=["firings", "avg_S"],
        help="The temporal characteristics to extract. Can specify multiple features like: --feature-types firings avg_S",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="The fraction of data to use for the training set.",
    )
    args = parser.parse_args()

    # Create a structured output directory based on the input file name, feature types
    input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
    feature_suffix = "_".join(args.feature_types)
    structured_output_dir = os.path.join(
        args.output_dir, f"{input_basename}_{feature_suffix}"
    )
    os.makedirs(structured_output_dir, exist_ok=True)

    print(f"Starting data preparation")
    print(f"Feature types: {args.feature_types}")
    print(f"Output will be saved in: {structured_output_dir}")

    # 1. Load and group data
    records = load_dataset(args.input_file)
    image_buckets = group_by_image(records)

    all_data = []
    all_labels = []

    # 2. Extract features for each image presentation
    for (label, img_idx), image_records in tqdm(
        image_buckets.items(), desc="Extracting features"
    ):
        if len(args.feature_types) == 1:
            # Single feature extraction (backward compatibility)
            if args.feature_types[0] == "firings":
                time_series = extract_firings_time_series(image_records)
            elif args.feature_types[0] == "avg_S":
                time_series = extract_avg_S_time_series(image_records)
            else:
                raise ValueError(f"Unknown feature type: {args.feature_types[0]}")
        else:
            # Multi-feature extraction
            time_series = extract_multi_feature_time_series(
                image_records, args.feature_types
            )

        if time_series.numel() > 0:
            all_data.append(time_series)
            all_labels.append(label)

    if not all_data:
        print("No data was extracted. Check the input file and feature type.")
        return

    # 3. Shuffle and split data
    print(f"Extracted {len(all_data)} samples. Shuffling and splitting data...")

    # Convert to tensors for shuffling
    # Note: Samples can have different time lengths, so we keep them in a list for now.
    # They will be handled by a custom collate function in the training script.
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    # Create a permutation
    indices = torch.randperm(len(all_data))

    # Apply permutation
    shuffled_data = [all_data[i] for i in indices]
    shuffled_labels = labels_tensor[indices]

    # Split
    split_idx = int(len(shuffled_data) * args.train_split)
    train_data = shuffled_data[:split_idx]
    train_labels = shuffled_labels[:split_idx]
    test_data = shuffled_data[split_idx:]
    test_labels = shuffled_labels[split_idx:]

    print(f"Training set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")

    # 5. Save datasets
    torch.save(train_data, os.path.join(structured_output_dir, "train_data.pt"))
    torch.save(train_labels, os.path.join(structured_output_dir, "train_labels.pt"))
    torch.save(test_data, os.path.join(structured_output_dir, "test_data.pt"))
    torch.save(test_labels, os.path.join(structured_output_dir, "test_labels.pt"))

    # 6. Save metadata about the feature configuration and target architecture
    metadata = {
        "feature_types": args.feature_types,
        "num_features": len(args.feature_types),
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "input_file": args.input_file,
        "train_split": args.train_split,
    }

    # Save metadata as JSON
    import json

    metadata_path = os.path.join(structured_output_dir, "dataset_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Successfully saved datasets to {structured_output_dir}")
    print(f"Feature configuration: {args.feature_types}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
