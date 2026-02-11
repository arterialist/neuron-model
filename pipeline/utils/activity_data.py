"""
Activity data recording and loading utilities.

Provides HDF5-based recording and lazy loading for neural activity data.
Shared between build_activity_dataset.py and pipeline activity recorder.
"""

import os
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore

try:
    import torch
except ImportError:
    torch = None  # type: ignore


class HDF5TensorRecorder:
    """High-performance HDF5-based tensor recorder for neural activity data.

    Stores all samples in a single compressed HDF5 file for maximum scalability
    and performance. All neural activity data is stored in one file with
    extensible datasets, compression, and random access capabilities.
    """

    def __init__(self, output_dir: str, network: Any):
        """Initialize the recorder.

        Args:
            output_dir: Directory to save the HDF5 file
            network: Network object with neurons dict
        """
        if h5py is None:
            raise ImportError("h5py is required for binary format recording")

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 1. Map UUIDs to Dense Indices (0..N)
        # This allows us to use Matrix indices instead of Dictionary keys
        self.uuid_to_idx = {uid: i for i, uid in enumerate(network.neurons.keys())}
        self.idx_to_uuid = list(network.neurons.keys())
        self.num_neurons = len(self.idx_to_uuid)

        # 2. Extract layer structure from network
        # Determine layers by examining neuron metadata or structure
        self.layer_structure = self._extract_layer_structure(network)

        # Create single HDF5 file
        self.h5_path = os.path.join(output_dir, "activity_dataset.h5")
        self.h5_file = None
        self.current_sample_idx = 0

        # Per-sample buffers
        self.u_buf = None
        self.t_ref_buf = None
        self.fr_buf = None
        self.spikes: List[Tuple[int, int]] = []

        # Dataset references
        self.u_dataset = None
        self.t_ref_dataset = None
        self.fr_dataset = None
        self.spikes_dataset = None
        self.labels_dataset = None
        self.num_samples_dataset = None

    def _extract_layer_structure(self, network: Any) -> List[int]:
        """Extract layer structure from network topology."""
        # Try to determine layers from neuron metadata
        layers: Dict[int, List[Any]] = {}
        for neuron_id, neuron in network.neurons.items():
            meta = getattr(neuron, "metadata", None) or {}
            layer_idx = meta.get("layer", 0) if isinstance(meta, dict) else 0
            if layer_idx not in layers:
                layers[layer_idx] = []
            layers[layer_idx].append(neuron_id)

        if layers:
            # Sort layers and ensure neurons within each layer are in consistent order
            layer_structure = []
            for layer_idx in sorted(layers.keys()):
                layer_neurons = sorted(
                    layers[layer_idx], key=lambda x: self.uuid_to_idx[x]
                )
                layer_structure.append(len(layer_neurons))
            return layer_structure

        # Fallback: assume single layer if no layer metadata
        return [self.num_neurons]

    def _create_datasets(self, ticks: int) -> None:
        """Create extensible HDF5 datasets with compression and chunking"""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "w")

        # Create extensible datasets with chunking and compression
        # Datasets start at size 0 and auto-extend as needed

        # Dense arrays: (samples, ticks, neurons)
        chunk_shape_u = (1, ticks, self.num_neurons)
        max_shape_u = (None, ticks, self.num_neurons)

        self.u_dataset = self.h5_file.create_dataset(
            "u",
            shape=(0, ticks, self.num_neurons),
            maxshape=max_shape_u,
            dtype=np.float32,
            chunks=chunk_shape_u,
            compression="gzip",
            compression_opts=6,
        )

        self.t_ref_dataset = self.h5_file.create_dataset(
            "t_ref",
            shape=(0, ticks, self.num_neurons),
            maxshape=max_shape_u,
            dtype=np.float32,
            chunks=chunk_shape_u,
            compression="gzip",
            compression_opts=6,
        )

        self.fr_dataset = self.h5_file.create_dataset(
            "fr",
            shape=(0, ticks, self.num_neurons),
            maxshape=max_shape_u,
            dtype=np.float32,
            chunks=chunk_shape_u,
            compression="gzip",
            compression_opts=6,
        )

        # Sparse spikes: variable-length per sample
        self.spikes_dataset = self.h5_file.create_dataset(
            "spikes",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.special_dtype(vlen=np.dtype("int32")),
            chunks=(100,),
            compression="gzip",
        )

        # Labels: simple 1D array
        self.labels_dataset = self.h5_file.create_dataset(
            "labels",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(1000,),
            compression="gzip",
        )

        # Number of samples: track actual count
        self.num_samples_dataset = self.h5_file.create_dataset(
            "num_samples",
            shape=(1,),
            dtype=np.int32,
        )
        self.num_samples_dataset[0] = 0

        # Neuron IDs: store as variable-length strings for UUID compatibility
        neuron_ids_str = [str(uid) for uid in self.idx_to_uuid]
        self.h5_file.create_dataset(
            "neuron_ids",
            data=neuron_ids_str,
            dtype=h5py.special_dtype(vlen=str),
            compression="gzip",
        )

        # Layer structure: store as simple array
        self.h5_file.create_dataset(
            "layer_structure",
            data=np.array(self.layer_structure, dtype=np.int32),
            compression="gzip",
        )

    def init_buffer(self, ticks: int) -> None:
        """Initialize buffers for a new sample. Call before processing an image."""
        if self.h5_file is None:
            self._create_datasets(ticks)

        # Allocating buffers: [Ticks, Neurons]
        self.u_buf = np.zeros((ticks, self.num_neurons), dtype=np.float32)
        self.t_ref_buf = np.zeros((ticks, self.num_neurons), dtype=np.float32)
        self.fr_buf = np.zeros((ticks, self.num_neurons), dtype=np.float32)
        self.spikes = []  # Sparse list for spikes

    def capture_tick(self, tick_idx: int, neurons_dict: Dict[Any, Any]) -> None:
        """Capture neural state for a single tick.

        Args:
            tick_idx: Current tick index
            neurons_dict: Dictionary mapping neuron IDs to neuron objects
        """
        for uid, neuron in neurons_dict.items():
            idx = self.uuid_to_idx.get(uid)
            if idx is None:
                continue

            # Direct float assignment (No dictionary creation)
            self.u_buf[tick_idx, idx] = neuron.S
            self.t_ref_buf[tick_idx, idx] = neuron.t_ref
            self.fr_buf[tick_idx, idx] = neuron.F_avg

            if neuron.O > 0:
                self.spikes.append((tick_idx, idx))

    def save_sample(self, sample_idx: int, label: int) -> None:
        """Save the current sample to the HDF5 file.

        Args:
            sample_idx: Sample index in the dataset
            label: Class label for this sample
        """
        # Convert sparse spikes to array
        spike_arr = (
            np.array(self.spikes, dtype=np.int32).flatten()
            if self.spikes
            else np.zeros((0,), dtype=np.int32)
        )

        # Extend datasets if necessary
        current_size = self.u_dataset.shape[0]
        if sample_idx >= current_size:
            new_size = max(
                sample_idx + 1, current_size + 1000
            )  # Extend by at least 1000

            self.u_dataset.resize(
                (new_size, self.u_dataset.shape[1], self.u_dataset.shape[2])
            )
            self.t_ref_dataset.resize(
                (new_size, self.t_ref_dataset.shape[1], self.t_ref_dataset.shape[2])
            )
            self.fr_dataset.resize(
                (new_size, self.t_ref_dataset.shape[1], self.t_ref_dataset.shape[2])
            )
            self.spikes_dataset.resize((new_size,))
            self.labels_dataset.resize((new_size,))

        # Store the data
        self.u_dataset[sample_idx] = self.u_buf
        self.t_ref_dataset[sample_idx] = self.t_ref_buf
        self.fr_dataset[sample_idx] = self.fr_buf
        self.spikes_dataset[sample_idx] = spike_arr
        self.labels_dataset[sample_idx] = label

        self.current_sample_idx = sample_idx + 1
        # Update sample count
        self.num_samples_dataset[0] = self.current_sample_idx

    def save_sample_from_buffers(
        self,
        sample_idx: int,
        label: int,
        u_buf: np.ndarray,
        t_ref_buf: np.ndarray,
        fr_buf: np.ndarray,
        spike_arr: np.ndarray,
    ) -> None:
        """Save a sample from pre-computed buffers.

        Args:
            sample_idx: Sample index in the dataset
            label: Class label for this sample
            u_buf: Membrane potential buffer [ticks, neurons]
            t_ref_buf: Refractory period buffer [ticks, neurons]
            fr_buf: Firing rate buffer [ticks, neurons]
            spike_arr: Spikes array [num_spikes, 2]
        """
        # Ensure datasets exist
        if self.u_dataset is None:
            self._create_datasets(u_buf.shape[0])

        # Extend datasets if necessary
        current_size = self.u_dataset.shape[0]
        if sample_idx >= current_size:
            new_size = max(
                sample_idx + 1, current_size + 1000
            )  # Extend by at least 1000

            self.u_dataset.resize(
                (new_size, self.u_dataset.shape[1], self.u_dataset.shape[2])
            )
            self.t_ref_dataset.resize(
                (new_size, self.t_ref_dataset.shape[1], self.t_ref_dataset.shape[2])
            )
            self.fr_dataset.resize(
                (new_size, self.t_ref_dataset.shape[1], self.t_ref_dataset.shape[2])
            )
            self.spikes_dataset.resize((new_size,))
            self.labels_dataset.resize((new_size,))

        # Store the data
        self.u_dataset[sample_idx] = u_buf
        self.t_ref_dataset[sample_idx] = t_ref_buf
        self.fr_dataset[sample_idx] = fr_buf
        self.spikes_dataset[sample_idx] = spike_arr
        self.labels_dataset[sample_idx] = label

        if sample_idx >= self.current_sample_idx:
            self.current_sample_idx = sample_idx + 1
            # Update sample count
            self.num_samples_dataset[0] = self.current_sample_idx

    def close(self) -> None:
        """Close the HDF5 file, trimming datasets to actual size."""
        if self.h5_file is not None:
            # Trim datasets to actual number of samples to avoid wasted space
            # and potential issues with consumers that don't check num_samples
            actual_samples = self.current_sample_idx
            if actual_samples > 0 and self.u_dataset is not None:
                current_size = self.u_dataset.shape[0]
                if current_size > actual_samples:
                    # Resize all sample-indexed datasets to actual size
                    self.u_dataset.resize(
                        (
                            actual_samples,
                            self.u_dataset.shape[1],
                            self.u_dataset.shape[2],
                        )
                    )
                    self.t_ref_dataset.resize(
                        (
                            actual_samples,
                            self.t_ref_dataset.shape[1],
                            self.t_ref_dataset.shape[2],
                        )
                    )
                    self.fr_dataset.resize(
                        (
                            actual_samples,
                            self.fr_dataset.shape[1],
                            self.fr_dataset.shape[2],
                        )
                    )
                    self.spikes_dataset.resize((actual_samples,))
                    self.labels_dataset.resize((actual_samples,))

            self.h5_file.close()
            self.h5_file = None

    def __del__(self) -> None:
        """Ensure file is closed on deletion"""
        self.close()


class LazyActivityDataset:
    """Dataset class for reading neural activity data lazily from HDF5 files.

    Loads data on-demand from compressed HDF5 files for memory-efficient training
    and inference. Supports random access to any sample in the dataset.
    """

    def __init__(self, data_dir: str):
        """Initialize the lazy dataset.

        Args:
            data_dir: Directory containing activity_dataset.h5
        """
        if h5py is None:
            raise ImportError("h5py is required for binary format loading")

        self.data_dir = data_dir

        # Load HDF5 file
        h5_path = os.path.join(data_dir, "activity_dataset.h5")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 dataset file not found: {h5_path}")

        self.h5_file = h5py.File(h5_path, "r")

        # Read actual sample count
        if "num_samples" in self.h5_file:
            self.num_samples = int(self.h5_file["num_samples"][0])
        else:
            # Fallback for older files
            self.num_samples = self.h5_file["labels"].shape[0]

        # Load neuron IDs from HDF5 file
        if "neuron_ids" in self.h5_file:
            neuron_id_strings = list(self.h5_file["neuron_ids"])
            try:
                self.neuron_ids = [int(s) for s in neuron_id_strings]
            except ValueError:
                # If conversion to int fails, keep as strings (for UUID compatibility)
                self.neuron_ids = neuron_id_strings
        else:
            # Fallback to external metadata file for backward compatibility
            metadata_path = os.path.join(data_dir, "metadata.npz")
            if os.path.exists(metadata_path):
                metadata = np.load(metadata_path)
                self.neuron_ids = list(metadata["ids"])
            else:
                raise FileNotFoundError(
                    "Neuron metadata not found in HDF5 file or metadata.npz. "
                    "This dataset may have been created with an older version. "
                    "Please recreate the dataset with the current version."
                )

        # Load layer structure
        if "layer_structure" in self.h5_file:
            self.layer_structure = list(self.h5_file["layer_structure"])
        else:
            # Fallback: assume single layer
            self.layer_structure = [len(self.neuron_ids)]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample by index.

        Args:
            idx: Sample index

        Returns:
            Dictionary with u, t_ref, fr, spikes, label, neuron_ids
        """
        # Load from HDF5 file
        u = self.h5_file["u"][idx]  # Shape: [ticks, neurons]
        t_ref = self.h5_file["t_ref"][idx]
        fr = self.h5_file["fr"][idx]
        spikes_flat = self.h5_file["spikes"][idx]

        # Reshape flat spikes back to (N, 2) format
        if len(spikes_flat) > 0:
            spikes = spikes_flat.reshape(-1, 2)
        else:
            spikes = np.zeros((0, 2), dtype=np.int32)

        label = int(self.h5_file["labels"][idx])

        result: Dict[str, Any] = {
            "u": u,
            "t_ref": t_ref,
            "fr": fr,
            "spikes": spikes,
            "label": label,
            "neuron_ids": self.neuron_ids,
        }

        # Wrap in torch tensors if available
        if torch is not None:
            result["u"] = torch.from_numpy(result["u"])
            result["t_ref"] = torch.from_numpy(result["t_ref"])
            result["fr"] = torch.from_numpy(result["fr"])
            result["spikes"] = torch.from_numpy(result["spikes"])

        return result

    def close(self) -> None:
        """Close HDF5 file"""
        if hasattr(self, "h5_file") and self.h5_file:
            self.h5_file.close()
            self.h5_file = None

    def __del__(self) -> None:
        """Ensure file is closed on deletion"""
        self.close()


def is_binary_dataset(path: str) -> bool:
    """Check if a path contains binary format dataset.

    Args:
        path: Path to dataset file or directory

    Returns:
        True if binary format (HDF5), False otherwise
    """
    if os.path.isdir(path):
        return os.path.exists(os.path.join(path, "activity_dataset.h5"))
    return path.endswith(".h5")


def load_activity_dataset(
    path: str,
) -> Tuple[List[Dict[str, Any]], Optional[List[int]]]:
    """Load activity dataset from binary format.

    Args:
        path: Path to dataset file or directory

    Returns:
        Tuple of (records list, layer_structure or None)
    """
    # Enforce binary format
    if not is_binary_dataset(path):
        raise ValueError(f"Path does not appear to be a binary dataset: {path}")

    dataset = LazyActivityDataset(
        path if os.path.isdir(path) else os.path.dirname(path)
    )
    # Convert to records format for compatibility
    records = []
    for i in range(len(dataset)):
        sample = dataset[i]
        # Convert HDF5 format to record format
        records.append(
            {
                "label": sample["label"],
                "u": sample["u"],
                "t_ref": sample["t_ref"],
                "fr": sample["fr"],
                "spikes": sample["spikes"],
            }
        )
    layer_structure = dataset.layer_structure
    dataset.close()
    return records, layer_structure
