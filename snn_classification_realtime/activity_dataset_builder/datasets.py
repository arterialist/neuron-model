"""HDF5-based dataset classes for neural activity data."""

import os
from typing import Any

import numpy as np
import torch
import h5py


class LazyActivityDataset(torch.utils.data.Dataset):
    """Dataset class for reading neural activity data lazily from HDF5 files.

    Loads data on-demand from compressed HDF5 files for memory-efficient training
    and inference. Supports random access to any sample in the dataset.
    """

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir

        h5_path = os.path.join(data_dir, "activity_dataset.h5")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 dataset file not found: {h5_path}")

        self.h5_file = h5py.File(h5_path, "r")
        if "num_samples" in self.h5_file:
            self.num_samples = int(self.h5_file["num_samples"][0])
        else:
            self.num_samples = self.h5_file["labels"].shape[0]

        if "neuron_ids" in self.h5_file:
            neuron_id_strings = list(self.h5_file["neuron_ids"])
            try:
                self.neuron_ids = [int(s) for s in neuron_id_strings]
            except ValueError:
                self.neuron_ids = neuron_id_strings
        else:
            metadata_path = os.path.join(data_dir, "metadata.npz")
            if os.path.exists(metadata_path):
                metadata = np.load(metadata_path)
                self.neuron_ids = metadata["ids"]
            else:
                raise FileNotFoundError(
                    "Neuron metadata not found in HDF5 file or metadata.npz. "
                    "This dataset may have been created with an older version. "
                    "Please recreate the dataset with the current version."
                )

        if "layer_structure" in self.h5_file:
            self.layer_structure = list(self.h5_file["layer_structure"])
        else:
            self.layer_structure = [len(self.neuron_ids)]

        self.ablation = (
            self.h5_file.attrs.get("ablation", "none")
            if hasattr(self.h5_file, "attrs")
            else "none"
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        u = torch.from_numpy(self.h5_file["u"][idx])
        t_ref = torch.from_numpy(self.h5_file["t_ref"][idx])
        fr = torch.from_numpy(self.h5_file["fr"][idx])
        spikes_flat = self.h5_file["spikes"][idx]
        if len(spikes_flat) > 0:
            spikes = torch.from_numpy(spikes_flat.reshape(-1, 2))
        else:
            spikes = torch.zeros((0, 2), dtype=torch.int32)
        label = int(self.h5_file["labels"][idx])

        return {
            "u": u,
            "t_ref": t_ref,
            "fr": fr,
            "spikes": spikes,
            "label": label,
            "neuron_ids": self.neuron_ids,
        }

    def close(self) -> None:
        """Close HDF5 file."""
        if hasattr(self, "h5_file"):
            self.h5_file.close()

    def __del__(self) -> None:
        self.close()


class HDF5TensorRecorder:
    """High-performance HDF5-based tensor recorder for neural activity data.

    Stores all samples in a single compressed HDF5 file for maximum scalability
    and performance. All neural activity data is stored in one file with
    extensible datasets, compression, and random access capabilities.
    """

    def __init__(
        self,
        output_dir: str,
        network: Any,
        ablation_name: str | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.ablation_name = ablation_name or "none"
        os.makedirs(output_dir, exist_ok=True)

        self.uuid_to_idx = {uid: i for i, uid in enumerate(network.neurons.keys())}
        self.idx_to_uuid = list(network.neurons.keys())
        self.num_neurons = len(self.idx_to_uuid)
        self.layer_structure = self._extract_layer_structure(network)

        self.h5_path = os.path.join(output_dir, "activity_dataset.h5")
        self.h5_file = None
        self.current_sample_idx = 0

    def _extract_layer_structure(self, network: Any) -> list[int]:
        """Extract layer structure from network topology."""
        layers: dict[int, list[Any]] = {}
        for neuron_id, neuron in network.neurons.items():
            layer_idx = getattr(neuron, "metadata", {}).get("layer", 0)
            if layer_idx not in layers:
                layers[layer_idx] = []
            layers[layer_idx].append(neuron_id)

        if layers:
            layer_structure = []
            for layer_idx in sorted(layers.keys()):
                layer_neurons = sorted(
                    layers[layer_idx], key=lambda x: self.uuid_to_idx[x]
                )
                layer_structure.append(len(layer_neurons))
            return layer_structure

        return [self.num_neurons]

    def _create_datasets(self, ticks: int) -> None:
        """Create extensible HDF5 datasets with compression and chunking."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "w")

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

        self.spikes_dataset = self.h5_file.create_dataset(
            "spikes",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.special_dtype(vlen=np.dtype("int32")),
            chunks=(100,),
            compression="gzip",
        )

        self.labels_dataset = self.h5_file.create_dataset(
            "labels",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(1000,),
            compression="gzip",
        )

        self.num_samples_dataset = self.h5_file.create_dataset(
            "num_samples",
            shape=(1,),
            dtype=np.int32,
        )
        self.num_samples_dataset[0] = 0

        neuron_ids_str = [str(uid) for uid in self.idx_to_uuid]
        self.h5_file.create_dataset(
            "neuron_ids",
            data=neuron_ids_str,
            dtype=h5py.special_dtype(vlen=str),
            compression="gzip",
        )

        self.h5_file.create_dataset(
            "layer_structure",
            data=np.array(self.layer_structure, dtype=np.int32),
            compression="gzip",
        )

        self.h5_file.attrs["ablation"] = self.ablation_name

    def init_buffer(self, ticks: int) -> None:
        """Call this before starting a new image."""
        if self.h5_file is None:
            self._create_datasets(ticks)

        self.u_buf = np.zeros((ticks, self.num_neurons), dtype=np.float32)
        self.t_ref_buf = np.zeros((ticks, self.num_neurons), dtype=np.float32)
        self.fr_buf = np.zeros((ticks, self.num_neurons), dtype=np.float32)
        self.spikes: list[tuple[int, int]] = []

    def capture_tick(self, tick_idx: int, neurons_dict: dict[Any, Any]) -> None:
        """Fast capture loop."""
        for uid, neuron in neurons_dict.items():
            idx = self.uuid_to_idx[uid]
            self.u_buf[tick_idx, idx] = neuron.S
            self.t_ref_buf[tick_idx, idx] = neuron.t_ref
            self.fr_buf[tick_idx, idx] = neuron.F_avg
            if neuron.O > 0:
                self.spikes.append((tick_idx, idx))

    def save_sample(self, sample_idx: int, label: int) -> None:
        """Append sample to HDF5 file."""
        spike_arr = (
            np.array(self.spikes, dtype=np.int32).flatten()
            if self.spikes
            else np.zeros((0,), dtype=np.int32)
        )
        self._extend_and_store(sample_idx, label, self.u_buf, self.t_ref_buf, self.fr_buf, spike_arr)

    def save_sample_from_buffers(
        self,
        sample_idx: int,
        label: int,
        u_buf: np.ndarray,
        t_ref_buf: np.ndarray,
        fr_buf: np.ndarray,
        spike_arr: np.ndarray,
    ) -> None:
        """Append sample to HDF5 file using pre-computed buffers."""
        if self.h5_file is None:
            ticks = u_buf.shape[0] if u_buf is not None else 100
            self._create_datasets(ticks)

        self._extend_and_store(sample_idx, label, u_buf, t_ref_buf, fr_buf, spike_arr)

    def _extend_and_store(
        self,
        sample_idx: int,
        label: int,
        u_buf: np.ndarray,
        t_ref_buf: np.ndarray,
        fr_buf: np.ndarray,
        spike_arr: np.ndarray,
    ) -> None:
        """Extend datasets if needed and store the sample."""
        current_size = self.u_dataset.shape[0]
        if sample_idx >= current_size:
            new_size = max(sample_idx + 1, current_size + 1000)
            self.u_dataset.resize((new_size, self.u_dataset.shape[1], self.u_dataset.shape[2]))
            self.t_ref_dataset.resize((new_size, self.t_ref_dataset.shape[1], self.t_ref_dataset.shape[2]))
            self.fr_dataset.resize((new_size, self.fr_dataset.shape[1], self.fr_dataset.shape[2]))
            self.spikes_dataset.resize((new_size,))
            self.labels_dataset.resize((new_size,))

        self.u_dataset[sample_idx] = u_buf
        self.t_ref_dataset[sample_idx] = t_ref_buf
        self.fr_dataset[sample_idx] = fr_buf
        self.spikes_dataset[sample_idx] = spike_arr
        self.labels_dataset[sample_idx] = label

        self.current_sample_idx = sample_idx + 1
        self.num_samples_dataset[0] = self.current_sample_idx

    def close(self) -> None:
        """Close the HDF5 file."""
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def __del__(self) -> None:
        self.close()
