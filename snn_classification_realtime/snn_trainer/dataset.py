"""Dataset and collate utilities for activity time-series."""

import gzip
import os
from typing import Any

import torch
from torch.utils.data import Dataset


def load_tensor_file(path: str, *, weights_only: bool = True) -> Any:
    """Load a .pt or .pt.gz file. Prefers .pt.gz if it exists for backward compatibility."""
    gz_path = path + ".gz" if not path.endswith(".gz") else path
    if os.path.exists(gz_path):
        with gzip.open(gz_path, "rb") as f:
            return torch.load(f, weights_only=weights_only)  # type: ignore[arg-type]
    return torch.load(path, weights_only=weights_only)


class ActivityDataset(Dataset):
    """Custom PyTorch Dataset for activity time-series data."""

    def __init__(self, data_path: str, labels_path: str) -> None:
        self.data = load_tensor_file(data_path)
        self.labels = load_tensor_file(labels_path)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function to handle variable length sequences by padding."""
    data, labels = zip(*batch)
    padded_data = torch.nn.utils.rnn.pad_sequence(
        data, batch_first=True, padding_value=0
    )
    stacked_labels = torch.stack(
        [torch.as_tensor(label) for label in labels], dim=0
    )
    return padded_data, stacked_labels
