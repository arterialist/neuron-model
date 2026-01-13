import os
import json
import argparse
import sys
import yaml
from typing import Dict, Any, List, Tuple, Iterable, Iterator, Optional

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import binary dataset support
from pipeline.build_activity_dataset import LazyActivityDataset
from pipeline.config import PreparationConfig

class FeatureScaler:
    """Fits per-dimension scaling statistics over variable-length time series."""
    def __init__(self, method: str, eps: float = 1e-8) -> None:
        self.method = method
        self.eps = float(eps)
        self._fitted = False
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None
        self._min: Optional[torch.Tensor] = None
        self._max: Optional[torch.Tensor] = None
        self._max_abs: Optional[torch.Tensor] = None

    def fit(self, samples: List[torch.Tensor]) -> None:
        if self.method == "none":
            self._fitted = True
            return

        if not samples:
            raise ValueError("Cannot fit scaler on empty sample list")

        # Determine feature dimension D
        first = samples[0]
        if first.ndim != 2:
            raise ValueError("Each sample must be a 2D tensor [T, D]")
        feature_dim = int(first.shape[1])

        if self.method == "standard":
            running_sum = torch.zeros(feature_dim, dtype=torch.float32)
            running_sumsq = torch.zeros(feature_dim, dtype=torch.float32)
            total_count = 0
            for ts in samples:
                if ts.numel() == 0:
                    continue
                # Sum over time dimension
                running_sum += ts.sum(dim=0)
                running_sumsq += (ts * ts).sum(dim=0)
                total_count += int(ts.shape[0])
            if total_count == 0:
                raise ValueError("All training samples are empty; cannot fit scaler")
            mean = running_sum / float(total_count)
            var = torch.clamp(running_sumsq / float(total_count) - mean * mean, min=0.0)
            std = torch.sqrt(var)
            self._mean = mean
            self._std = std
            self._fitted = True
            return

        if self.method == "minmax":
            cur_min = torch.full((feature_dim,), float("inf"), dtype=torch.float32)
            cur_max = torch.full((feature_dim,), float("-inf"), dtype=torch.float32)
            for ts in samples:
                if ts.numel() == 0:
                    continue
                # Per-dimension min/max over time
                tmin = ts.min(dim=0).values
                tmax = ts.max(dim=0).values
                cur_min = torch.minimum(cur_min, tmin)
                cur_max = torch.maximum(cur_max, tmax)
            self._min = cur_min
            self._max = cur_max
            self._fitted = True
            return

        if self.method == "maxabs":
            cur_max_abs = torch.zeros(feature_dim, dtype=torch.float32)
            for ts in samples:
                if ts.numel() == 0:
                    continue
                tmax_abs = ts.abs().max(dim=0).values
                cur_max_abs = torch.maximum(cur_max_abs, tmax_abs)
            self._max_abs = cur_max_abs
            self._fitted = True
            return

        raise ValueError(f"Unknown scaler method: {self.method}")

    def transform(self, sample: torch.Tensor) -> torch.Tensor:
        if not self._fitted or self.method == "none" or sample.numel() == 0:
            return sample
        if sample.ndim != 2:
            raise ValueError("Each sample must be a 2D tensor [T, D]")

        if self.method == "standard":
            assert self._mean is not None and self._std is not None
            denom = self._std + self.eps
            return (sample - self._mean) / denom

        if self.method == "minmax":
            assert self._min is not None and self._max is not None
            scale = self._max - self._min
            scale = torch.where(scale == 0, torch.full_like(scale, 1.0), scale)
            return (sample - self._min) / (scale + self.eps)

        if self.method == "maxabs":
            assert self._max_abs is not None
            denom = self._max_abs + self.eps
            return sample / denom

        return sample

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {"method": self.method, "eps": self.eps}
        if self._mean is not None:
            state["mean"] = self._mean.detach().cpu()
        if self._std is not None:
            state["std"] = self._std.detach().cpu()
        if self._min is not None:
            state["min"] = self._min.detach().cpu()
        if self._max is not None:
            state["max"] = self._max.detach().cpu()
        if self._max_abs is not None:
            state["max_abs"] = self._max_abs.detach().cpu()
        return state

def process_data(config: PreparationConfig, input_path: str, output_dir: str):
    # Expect input to be the HDF5 file from step 2
    h5_path = os.path.join(input_path, "activity_dataset.h5")
    if not os.path.exists(h5_path):
        # Fallback to checking if input_path is the h5 file itself
        if input_path.endswith(".h5") and os.path.exists(input_path):
            h5_path = input_path
        else:
             # Fallback to checking if input_path is a directory containing it
             possible = os.path.join(input_path, "activity_dataset.h5")
             if os.path.exists(possible):
                 h5_path = possible
             else:
                 raise FileNotFoundError(f"Could not find activity_dataset.h5 in {input_path}")

    print(f"Loading HDF5 dataset from: {h5_path}")
    # LazyActivityDataset expects a directory containing 'activity_dataset.h5'
    # We derived h5_path as the file path. We need to pass the directory.
    # If input_path was a file, we use its parent. If it was a dir, we use it.
    dataset_dir = os.path.dirname(h5_path)
    hdf5_dataset = LazyActivityDataset(dataset_dir)
    num_samples = len(hdf5_dataset)

    if config.max_samples is not None:
        num_samples = min(num_samples, config.max_samples)

    all_data = []
    all_labels = []

    # Extract features
    for i in tqdm(range(num_samples), desc="Extracting features"):
        sample = hdf5_dataset[i]
        label = int(sample["label"])

        u_tensor = sample["u"]
        spikes = sample["spikes"]

        ticks, neurons = u_tensor.shape
        if config.max_ticks is not None and ticks > config.max_ticks:
            ticks = config.max_ticks
            u_tensor = u_tensor[:ticks]
            # filter spikes
            mask = spikes[:, 0] < ticks
            spikes = spikes[mask]

        firing_tensor = torch.zeros(ticks, neurons, dtype=torch.float32)
        if len(spikes) > 0:
            spike_ticks, spike_neurons = spikes[:, 0], spikes[:, 1]
            firing_tensor[spike_ticks, spike_neurons] = 1.0

        feature_tensors = []
        for ft in config.feature_types:
            if ft == "firings":
                feature_tensors.append(firing_tensor)
            elif ft == "avg_S":
                feature_tensors.append(u_tensor.float())
            elif ft == "avg_t_ref":
                t_ref_vals = sample["t_ref"][0]
                feature_tensors.append(t_ref_vals.unsqueeze(0).expand(ticks, -1).float())
            else:
                raise ValueError(f"Unknown feature type: {ft}")

        if len(feature_tensors) == 1:
            time_series = feature_tensors[0]
        else:
            time_series = torch.cat(feature_tensors, dim=1)

        if time_series.numel() > 0:
            all_data.append(time_series)
            all_labels.append(label)

    if not all_data:
        print("No data extracted.")
        return

    # Shuffle and split
    print(f"Extracted {len(all_data)} samples. Splitting...")
    torch.manual_seed(42)
    np.random.seed(42)

    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    indices = torch.randperm(len(all_data))

    shuffled_data = [all_data[i] for i in indices]
    shuffled_labels = labels_tensor[indices]

    split_idx = int(len(shuffled_data) * config.train_split)
    train_data = shuffled_data[:split_idx]
    train_labels = shuffled_labels[:split_idx]
    test_data = shuffled_data[split_idx:]
    test_labels = shuffled_labels[split_idx:]

    # Scale
    scaler_path = None
    if config.scaler != "none":
        print(f"Fitting {config.scaler} scaler...")
        scaler = FeatureScaler(config.scaler, eps=config.scale_eps)
        scaler.fit(train_data)
        train_data = [scaler.transform(ts) for ts in train_data]
        test_data = [scaler.transform(ts) for ts in test_data]
        scaler_state = scaler.state_dict()
        scaler_path = os.path.join(output_dir, "scaler.pt")
        torch.save(scaler_state, scaler_path)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    torch.save(train_data, os.path.join(output_dir, "train_data.pt"))
    torch.save(train_labels, os.path.join(output_dir, "train_labels.pt"))
    torch.save(test_data, os.path.join(output_dir, "test_data.pt"))
    torch.save(test_labels, os.path.join(output_dir, "test_labels.pt"))

    # Metadata
    sample_dim = int(train_data[0].shape[1]) if len(train_data) > 0 else 0
    metadata = {
        "feature_types": config.feature_types,
        "num_features": len(config.feature_types),
        "total_feature_dim": sample_dim,
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "input_file": h5_path,
        "train_split": config.train_split,
        "scaler": config.scaler,
        "scale_eps": config.scale_eps,
    }
    if scaler_path:
        metadata["scaler_state_file"] = scaler_path

    with open(os.path.join(output_dir, "dataset_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved prepared data to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--input_dir", required=True, help="Directory containing activity_dataset.h5")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
        prep_cfg = PreparationConfig(**cfg_dict['preparation'])

    process_data(prep_cfg, args.input_dir, args.output_dir)
