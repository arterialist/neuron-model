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
        build_meta: dict[str, Any] | None = None,
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
        self._build_meta = build_meta
        self.u_dataset = None
        self.t_ref_dataset = None
        self.fr_dataset = None
        self.spikes_dataset = None
        self.labels_dataset = None
        self.num_samples_dataset = None

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
        if self._build_meta:
            meta = self._build_meta
            self.h5_file.attrs["build_shuffle_seed"] = int(meta["shuffle_seed"])
            self.h5_file.attrs["build_ticks_per_image"] = int(meta["ticks_per_image"])
            self.h5_file.attrs["build_images_per_label"] = int(meta["images_per_label"])
            self.h5_file.attrs["build_dataset_name"] = str(meta["dataset_name"])
            self.h5_file.attrs["build_num_classes"] = int(meta["num_classes"])
            self.h5_file.attrs["build_network_path"] = str(meta["network_path"])
            self.h5_file.attrs["build_cifar10_color_norm"] = float(
                meta["cifar10_color_normalization_factor"]
            )
            self.h5_file.attrs["build_ablation"] = str(meta["ablation"])
            self.h5_file.attrs["build_tick_ms"] = int(meta["tick_ms"])
            self.h5_file.attrs["build_use_multiprocessing"] = int(
                bool(meta["use_multiprocessing"])
            )
            self.h5_file.attrs["build_fresh_run_per_label"] = int(
                bool(meta["fresh_run_per_label"])
            )
            self.h5_file.attrs["build_fresh_run_per_image"] = int(
                bool(meta["fresh_run_per_image"])
            )
            self.h5_file.attrs["build_export_network_states"] = int(
                bool(meta["export_network_states"])
            )
            self.h5_file.attrs["build_start_web_server"] = int(bool(meta["start_web_server"]))
            self.h5_file.attrs["build_dataset_base"] = str(meta["dataset_base"])

    def _validate_resume_attrs(self, identity: dict[str, Any]) -> list[str]:
        """Return human-readable errors if HDF5 attrs do not match ``identity``."""
        attrs = self.h5_file.attrs
        errs: list[str] = []
        checks: list[tuple[str, str, Any]] = [
            ("build_shuffle_seed", "shuffle_seed", int(identity["shuffle_seed"])),
            ("build_ticks_per_image", "ticks_per_image", int(identity["ticks_per_image"])),
            (
                "build_images_per_label",
                "images_per_label",
                int(identity["images_per_label"]),
            ),
            ("build_dataset_name", "dataset_name", str(identity["dataset_name"])),
            ("build_num_classes", "num_classes", int(identity["num_classes"])),
            (
                "build_network_path",
                "network_path",
                os.path.abspath(str(identity["network_path"])),
            ),
            ("build_ablation", "ablation", str(identity["ablation"])),
            (
                "build_cifar10_color_norm",
                "cifar10_color_normalization_factor",
                float(identity["cifar10_color_normalization_factor"]),
            ),
            ("build_tick_ms", "tick_ms", int(identity["tick_ms"])),
            (
                "build_use_multiprocessing",
                "use_multiprocessing",
                int(bool(identity["use_multiprocessing"])),
            ),
            (
                "build_fresh_run_per_label",
                "fresh_run_per_label",
                int(bool(identity["fresh_run_per_label"])),
            ),
            (
                "build_fresh_run_per_image",
                "fresh_run_per_image",
                int(bool(identity["fresh_run_per_image"])),
            ),
            (
                "build_export_network_states",
                "export_network_states",
                int(bool(identity["export_network_states"])),
            ),
            (
                "build_start_web_server",
                "start_web_server",
                int(bool(identity["start_web_server"])),
            ),
            ("build_dataset_base", "dataset_base", str(identity["dataset_base"])),
        ]
        for attr, field, expected in checks:
            if attr not in attrs:
                errs.append(f"HDF5 missing attribute {attr!r} (need current tool to resume)")
                continue
            got = attrs[attr]
            if isinstance(expected, float):
                ok = abs(float(got) - expected) <= 1e-9
            elif isinstance(expected, str):
                ok = str(got) == expected
            elif attr == "build_network_path":
                ok = os.path.abspath(str(got)) == expected
            else:
                ok = int(got) == int(expected)
            if not ok:
                errs.append(
                    f"{field}: HDF5 has {got!r}, this run has {expected!r}"
                )
        return errs

    def open_existing_for_resume(
        self,
        truncate_to: int,
        run_identity: dict[str, Any],
    ) -> None:
        """Open an existing HDF5 for append; optionally shrink the sample axis."""
        if not os.path.isfile(self.h5_path):
            raise FileNotFoundError(f"Cannot resume: missing {self.h5_path}")
        if self.h5_file is not None:
            raise RuntimeError("HDF5 file already open")

        ticks_per_image = int(run_identity["ticks_per_image"])

        self.h5_file = h5py.File(self.h5_path, "a")
        self.u_dataset = self.h5_file["u"]
        self.t_ref_dataset = self.h5_file["t_ref"]
        self.fr_dataset = self.h5_file["fr"]
        self.spikes_dataset = self.h5_file["spikes"]
        self.labels_dataset = self.h5_file["labels"]
        self.num_samples_dataset = self.h5_file["num_samples"]

        if self.u_dataset.shape[2] != self.num_neurons:
            raise ValueError(
                "Resume aborted: neuron count in HDF5 does not match current network "
                f"(file {self.u_dataset.shape[2]} vs network {self.num_neurons})."
            )
        if self.u_dataset.shape[1] != ticks_per_image:
            raise ValueError(
                "Resume aborted: ticks_per_image in HDF5 does not match current run "
                f"(file {self.u_dataset.shape[1]} vs {ticks_per_image})."
            )
        attr_errs = self._validate_resume_attrs(run_identity)
        if attr_errs:
            raise ValueError("Resume aborted: " + "; ".join(attr_errs))

        current_n = int(self.u_dataset.shape[0])
        if truncate_to < 0:
            raise ValueError("truncate_to must be >= 0")
        if truncate_to > current_n:
            raise ValueError(
                f"Manifest committed_global_samples ({truncate_to}) exceeds HDF5 rows ({current_n})."
            )
        if truncate_to < current_n:
            self._truncate_sample_dim(truncate_to)

        self.current_sample_idx = int(self.num_samples_dataset[0])

    def _truncate_sample_dim(self, n: int) -> None:
        """Shrink all sample-indexed datasets to the first n rows (discard partial tail)."""
        self.u_dataset.resize((n, self.u_dataset.shape[1], self.u_dataset.shape[2]))
        self.t_ref_dataset.resize((n, self.t_ref_dataset.shape[1], self.t_ref_dataset.shape[2]))
        self.fr_dataset.resize((n, self.fr_dataset.shape[1], self.fr_dataset.shape[2]))
        self.spikes_dataset.resize((n,))
        self.labels_dataset.resize((n,))
        self.num_samples_dataset[0] = n
        self.current_sample_idx = n

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
            if os.path.isfile(self.h5_path):
                raise RuntimeError(
                    "activity_dataset.h5 exists but recorder was not opened for resume; "
                    "call open_existing_for_resume() first."
                )
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
        prev = int(self.num_samples_dataset[0])
        self.num_samples_dataset[0] = max(prev, self.current_sample_idx)

    def close(self) -> None:
        """Close the HDF5 file."""
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def __del__(self) -> None:
        self.close()
