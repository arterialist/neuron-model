import os
import sys
import time
import json
import math
import random
import threading
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import h5py

try:
    import matplotlib

    matplotlib.use("Agg")  # Ensure headless rendering
    import matplotlib.pyplot as plt
except Exception as _plot_exc:  # noqa: F401
    plt = None

# Ensure local imports resolve
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path

from neuron.nn_core import NNCore
from neuron.network import NeuronNetwork
from neuron.network_config import NetworkConfig
from neuron.neuron import Neuron
from cli.web_viz.server import NeuralNetworkWebServer
from neuron import setup_neuron_logger


class LazyActivityDataset(torch.utils.data.Dataset):
    """Dataset class for reading neural activity data lazily from HDF5 files.

    Loads data on-demand from compressed HDF5 files for memory-efficient training
    and inference. Supports random access to any sample in the dataset.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir

        # Load HDF5 file
        h5_path = os.path.join(data_dir, "activity_dataset.h5")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 dataset file not found: {h5_path}")

        self.h5_file = h5py.File(h5_path, "r")
        # Read actual sample count
        if "num_samples" in self.h5_file:
            self.num_samples = int(self.h5_file["num_samples"][0])  # type: ignore
        else:
            # Fallback for older files
            self.num_samples = self.h5_file["labels"].shape[0]  # type: ignore

        # Load metadata from HDF5 file (preferred) or fallback to metadata.npz
        if "neuron_ids" in self.h5_file:
            # Convert string UUIDs back to integers (assuming they were stored as strings)
            neuron_id_strings = list(self.h5_file["neuron_ids"])  # type: ignore
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
                self.neuron_ids = metadata["ids"]
            else:
                raise FileNotFoundError(
                    "Neuron metadata not found in HDF5 file or metadata.npz. "
                    "This dataset may have been created with an older version. "
                    "Please recreate the dataset with the current version."
                )

        # Load layer structure
        if "layer_structure" in self.h5_file:
            self.layer_structure = list(self.h5_file["layer_structure"])  # type: ignore
        else:
            # Fallback: assume single layer
            self.layer_structure = [len(self.neuron_ids)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Load from HDF5 file
        u = torch.from_numpy(
            self.h5_file["u"][idx]  # type: ignore
        )  # Shape: [ticks, neurons]
        t_ref = torch.from_numpy(self.h5_file["t_ref"][idx])  # type: ignore
        fr = torch.from_numpy(self.h5_file["fr"][idx])  # type: ignore
        spikes_flat = self.h5_file["spikes"][idx]  # type: ignore
        # Reshape flat spikes back to (N, 2) format
        if len(spikes_flat) > 0:  # type: ignore
            spikes = torch.from_numpy(spikes_flat.reshape(-1, 2))  # type: ignore
        else:
            spikes = torch.zeros((0, 2), dtype=torch.int32)
        label = int(self.h5_file["labels"][idx])  # type: ignore

        return {
            "u": u,
            "t_ref": t_ref,
            "fr": fr,
            "spikes": spikes,
            "label": label,
            "neuron_ids": self.neuron_ids,
        }

    def close(self):
        """Close HDF5 file"""
        if hasattr(self, "h5_file"):
            self.h5_file.close()

    def __del__(self):
        """Ensure file is closed on deletion"""
        self.close()


class HDF5TensorRecorder:
    """High-performance HDF5-based tensor recorder for neural activity data.

    Stores all samples in a single compressed HDF5 file for maximum scalability
    and performance. All neural activity data is stored in one file with
    extensible datasets, compression, and random access capabilities.
    """

    def __init__(self, output_dir, network):
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

        # Metadata will be stored in HDF5 file itself

    def _extract_layer_structure(self, network):
        """Extract layer structure from network topology."""
        # Try to determine layers from neuron metadata
        layers = {}
        for neuron_id, neuron in network.neurons.items():
            layer_idx = getattr(neuron, "metadata", {}).get("layer", 0)
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

    def _create_datasets(self, ticks):
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

    def init_buffer(self, ticks):
        """Call this before starting a new image"""
        if self.h5_file is None:
            self._create_datasets(ticks)

        # Allocating 4 floats per neuron per tick (u, t_ref, firing_rate, fired)
        # Shape: [Ticks, Neurons]
        self.u_buf = np.zeros((ticks, self.num_neurons), dtype=np.float32)
        self.t_ref_buf = np.zeros((ticks, self.num_neurons), dtype=np.float32)
        self.fr_buf = np.zeros((ticks, self.num_neurons), dtype=np.float32)
        self.spikes = []  # Sparse list for spikes

    def capture_tick(self, tick_idx, neurons_dict):
        """Fast capture loop"""
        # OPTIMIZATION: If you have a global buffer in network, use it directly!
        # If not, use this fast loop using the pre-computed index map

        for uid, neuron in neurons_dict.items():
            idx = self.uuid_to_idx[uid]

            # Direct float assignment (No dictionary creation)
            self.u_buf[tick_idx, idx] = neuron.S
            self.t_ref_buf[tick_idx, idx] = neuron.t_ref
            self.fr_buf[tick_idx, idx] = neuron.F_avg

            if neuron.O > 0:
                self.spikes.append((tick_idx, idx))

    def save_sample(self, sample_idx, label):
        """Append sample to HDF5 file"""
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

    def close(self):
        """Close the HDF5 file"""
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def __del__(self):
        """Ensure file is closed on deletion"""
        self.close()


MNIST_IMAGE_SIZE = 28 * 28
MNIST_NUM_CLASSES = 10

# Dataset selection globals
selected_dataset = None
CURRENT_IMAGE_VECTOR_SIZE = 0
CURRENT_NUM_CLASSES = MNIST_NUM_CLASSES
CURRENT_DATASET_NAME = "mnist"


def prompt_str(message: str, default: str) -> str:
    resp = input(f"{message} [{default}]: ").strip()
    return resp if resp != "" else default


def prompt_int(message: str, default: int) -> int:
    resp = input(f"{message} [{default}]: ").strip()
    if resp == "":
        return int(default)
    try:
        return int(resp)
    except ValueError:
        print("Invalid integer, using default.")
        return int(default)


def prompt_float(message: str, default: float) -> float:
    resp = input(f"{message} [{default}]: ").strip()
    if resp == "":
        return float(default)
    try:
        return float(resp)
    except ValueError:
        print("Invalid number, using default.")
        return float(default)


def prompt_yes_no(message: str, default_no: bool = True) -> bool:
    default = "n" if default_no else "y"
    resp = input(f"{message} (y/n) [{default}]: ").strip().lower()
    if resp == "":
        return not default_no
    return resp == "y"


def load_mnist(train: bool) -> datasets.MNIST:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    last_err: Exception | None = None
    for root, dl in [("./data/mnist", False), ("./data", True)]:
        try:
            ds = datasets.MNIST(
                root=root, train=train, download=dl, transform=transform
            )
            return ds
        except Exception as e:
            last_err = e
            continue
    if last_err is not None:
        print(f"Failed to load MNIST from preferred roots: {last_err}")
        raise last_err
    raise RuntimeError("Failed to load MNIST dataset")


def select_and_load_dataset() -> None:
    """Select dataset (MNIST, CIFAR10, CIFAR100) and load training split with normalization to [-1,1] equivalents.

    Sets globals: selected_dataset, CURRENT_IMAGE_VECTOR_SIZE, CURRENT_NUM_CLASSES.
    """
    global \
        selected_dataset, \
        CURRENT_IMAGE_VECTOR_SIZE, \
        CURRENT_NUM_CLASSES, \
        CURRENT_DATASET_NAME
    print("Select dataset:")
    print("  1) MNIST")
    print("  2) CIFAR10")
    print("  3) CIFAR100")
    print("  4) USPS")
    print("  5) SVHN")
    print("  6) FashionMNIST")
    choice = input("Enter choice [1]: ").strip() or "1"

    root_candidates = [
        "./data",
        "./data/mnist",
        "./data/cifar",
        "./data/usps",
        "./data/svhn",
        "./data/fashionmnist",
    ]

    if choice == "1":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        last_err = None
        ds = None
        for root in root_candidates:
            try:
                ds = datasets.MNIST(
                    root=root, train=True, download=True, transform=transform
                )
                break
            except Exception as e:
                last_err = e
                continue
        if ds is None:
            raise RuntimeError(f"Failed to load MNIST: {last_err}")
        selected_dataset = ds
        img0, _ = selected_dataset[0]
        CURRENT_IMAGE_VECTOR_SIZE = int(img0.numel())
        CURRENT_NUM_CLASSES = 10
        CURRENT_DATASET_NAME = "mnist"
    elif choice == "2":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        last_err = None
        ds = None
        for root in root_candidates:
            try:
                ds = datasets.CIFAR10(
                    root=root, train=True, download=True, transform=transform
                )
                break
            except Exception as e:
                last_err = e
                continue
        if ds is None:
            raise RuntimeError(f"Failed to load CIFAR10: {last_err}")
        selected_dataset = ds
        img0, _ = selected_dataset[0]
        CURRENT_IMAGE_VECTOR_SIZE = int(img0.numel())
        CURRENT_NUM_CLASSES = 10
        CURRENT_DATASET_NAME = "cifar10"
    elif choice == "3":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        last_err = None
        ds = None
        for root in root_candidates:
            try:
                ds = datasets.CIFAR100(
                    root=root, train=True, download=True, transform=transform
                )
                break
            except Exception as e:
                last_err = e
                continue
        if ds is None:
            raise RuntimeError(f"Failed to load CIFAR100: {last_err}")
        selected_dataset = ds
        img0, _ = selected_dataset[0]
        CURRENT_IMAGE_VECTOR_SIZE = int(img0.numel())
        CURRENT_NUM_CLASSES = 100
        CURRENT_DATASET_NAME = "cifar100"
    elif choice == "4":
        # USPS (1x28x28)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        last_err = None
        ds = None
        for root in root_candidates:
            try:
                ds = datasets.USPS(
                    root=root, train=True, download=True, transform=transform
                )
                break
            except Exception as e:
                last_err = e
                continue
        if ds is None:
            raise RuntimeError(f"Failed to load USPS: {last_err}")
        selected_dataset = ds
        img0, _ = selected_dataset[0]
        CURRENT_IMAGE_VECTOR_SIZE = int(img0.numel())
        CURRENT_NUM_CLASSES = 10
        CURRENT_DATASET_NAME = "usps"
    elif choice == "5":
        # SVHN (3x32x32) — use train split for dataset building
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        last_err = None
        ds = None
        for root in root_candidates:
            try:
                ds = datasets.SVHN(
                    root=root, split="train", download=True, transform=transform
                )
                break
            except Exception as e:
                last_err = e
                continue
        if ds is None:
            raise RuntimeError(f"Failed to load SVHN: {last_err}")
        selected_dataset = ds
        img0, _ = selected_dataset[0]
        CURRENT_IMAGE_VECTOR_SIZE = int(img0.numel())
        CURRENT_NUM_CLASSES = 10
        CURRENT_DATASET_NAME = "svhn"
    elif choice == "6":
        # FashionMNIST (1x28x28)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        last_err = None
        ds = None
        for root in root_candidates:
            try:
                ds = datasets.FashionMNIST(
                    root=root, train=True, download=True, transform=transform
                )
                break
            except Exception as e:
                last_err = e
                continue
        if ds is None:
            raise RuntimeError(f"Failed to load FashionMNIST: {last_err}")
        selected_dataset = ds
        img0, _ = selected_dataset[0]
        CURRENT_IMAGE_VECTOR_SIZE = int(img0.numel())
        CURRENT_NUM_CLASSES = 10
        CURRENT_DATASET_NAME = "fashionmnist"
    else:
        raise ValueError("Invalid dataset choice.")


def pre_run_compatibility_check(
    network_sim: NeuronNetwork, input_layer_ids: list[int] | None
) -> tuple[NeuronNetwork, list[int]]:
    """Ensure dataset vector size fits network input capacity. Optionally prompt to change dataset or network.

    Returns possibly updated (network_sim, input_layer_ids).
    """
    global selected_dataset, CURRENT_IMAGE_VECTOR_SIZE, CURRENT_NUM_CLASSES

    def compute_input_capacity(
        sim: NeuronNetwork, input_ids: list[int] | None
    ) -> tuple[int, list[int], int]:
        if not input_ids:
            # Infer from metadata
            layers = infer_layers_from_metadata(sim)
            input_ids = (
                layers[0]
                if layers and layers[0]
                else list(sim.network.neurons.keys())[:100]
            )
        syn_counts = [
            len(sim.network.neurons[nid].postsynaptic_points) for nid in input_ids
        ]
        syn_per = min(syn_counts) if syn_counts else 0
        capacity = len(input_ids) * syn_per
        return capacity, input_ids, syn_per

    run_check = (
        input("Run dataset/network input compatibility check? (y/n) [y]: ")
        .strip()
        .lower()
        or "y"
    )
    if run_check == "n":
        # Ensure we have input ids
        if not input_layer_ids:
            layers = infer_layers_from_metadata(network_sim)
            input_layer_ids = (
                layers[0]
                if layers and layers[0]
                else list(network_sim.network.neurons.keys())[:100]
            )
        return network_sim, input_layer_ids

    while True:
        capacity, input_layer_ids_eff, syn_per = compute_input_capacity(
            network_sim, input_layer_ids
        )
        vec = CURRENT_IMAGE_VECTOR_SIZE
        if capacity >= vec and syn_per > 0 and len(input_layer_ids_eff) > 0:
            print(
                f"Compatibility OK: capacity={capacity} (inputs={len(input_layer_ids_eff)} × synapses={syn_per}) >= vector={vec}"
            )
            return network_sim, input_layer_ids_eff

        print(
            f"Incompatible: dataset vector={vec} > network capacity={capacity} (inputs={len(input_layer_ids_eff)} × synapses={syn_per})."
        )
        choice = (
            input(
                "Choose action: [d] change dataset, [n] change network, [i] ignore and proceed [n]: "
            )
            .strip()
            .lower()
            or "n"
        )
        if choice == "i":
            print("Proceeding with mismatch; many pixels may be dropped or aliased.")
            return network_sim, input_layer_ids_eff
        if choice == "d":
            try:
                select_and_load_dataset()
            except Exception as e:
                print(f"Failed to load dataset: {e}")
                continue
            # loop to recheck
            continue
        if choice == "n":
            sub = (
                input("[l] load network file, [b] rebuild network [l]: ")
                .strip()
                .lower()
                or "l"
            )
            if sub == "l":
                path = input("Enter network file path: ").strip()
                try:
                    network_sim = NetworkConfig.load_network_config(path)
                except Exception as e:
                    print(f"Failed to load network: {e}")
                    continue
            else:
                print(
                    "Rebuild not supported in this script; please load a compatible network."
                )
            input_layer_ids = None  # force recompute
            continue


def ceil_to_nearest_ten(value: int) -> int:
    return int(math.ceil(value / 10.0) * 10)


def compute_default_ticks_per_image(
    network_sim: NeuronNetwork, layers: List[List[int]]
) -> int:
    """Estimate propagation time based on network connectivity, similar to pattern_classification.compute_default_ticks_per_image."""
    net = network_sim.network
    out_deg: Dict[int, int] = {}
    in_deg: Dict[int, int] = {}
    for src, _src_term, dst, _dst_syn in net.connections:
        out_deg[src] = out_deg.get(src, 0) + 1
        in_deg[dst] = in_deg.get(dst, 0) + 1

    total = 0
    for layer in layers:
        max_in = max((in_deg.get(nid, 0) for nid in layer), default=0)
        max_out = max((out_deg.get(nid, 0) for nid in layer), default=0)
        total += max_in + max_out
    return max(ceil_to_nearest_ten(max(10, total)), 10)


def infer_layers_from_metadata(network_sim: NeuronNetwork) -> List[List[int]]:
    """Group neurons by their 'layer' metadata if present; otherwise fall back to a single layer list in ID order."""
    net = network_sim.network
    layer_to_neurons: Dict[int, List[int]] = {}
    for nid, neuron in net.neurons.items():
        layer_idx = (
            int(neuron.metadata.get("layer", 0)) if isinstance(neuron, Neuron) else 0
        )
        layer_to_neurons.setdefault(layer_idx, []).append(nid)
    if not layer_to_neurons:
        return [list(net.neurons.keys())]
    layers_indexed = [layer_to_neurons[k] for k in sorted(layer_to_neurons.keys())]
    return layers_indexed


def determine_input_mapping(
    network_sim: NeuronNetwork, layers: List[List[int]]
) -> Tuple[List[int], int]:
    """Return (input_layer_ids, input_synapses_per_neuron). Assumes first layer is inputs."""
    input_layer_ids = layers[0]
    if not input_layer_ids:
        raise ValueError("Input layer appears empty; cannot map images to signals.")
    first_input = network_sim.network.neurons[input_layer_ids[0]]
    input_synapses_per_neuron = max(1, len(first_input.postsynaptic_points))
    return input_layer_ids, input_synapses_per_neuron


def image_to_signals(
    image_tensor: torch.Tensor,
    input_layer_ids: List[int],
    input_synapses_per_neuron: int,
    network_sim: NeuronNetwork,
) -> List[Tuple[int, int, float]]:
    """Map an image to (neuron_id, synapse_id, strength) signals.

    Supports legacy dense input and CNN-style input (conv layer at index 0).
    Signals are normalized to [0, 1] range.
    """
    # Detect CNN-style input
    first_neuron = network_sim.network.neurons[input_layer_ids[0]]
    meta = getattr(first_neuron, "metadata", {}) or {}
    is_cnn_input = meta.get("layer_type") == "conv" and meta.get("layer", 0) == 0

    if not is_cnn_input:
        img_vec = image_tensor.view(-1).numpy().astype(np.float32)
        # Normalize from [-1, 1] to [0, 1]
        img_vec = (img_vec + 1.0) * 0.5
        num_input_neurons = len(input_layer_ids)
        signals: List[Tuple[int, int, float]] = []
        for pixel_index, pixel_value in enumerate(img_vec):
            target_neuron_index = pixel_index % num_input_neurons
            target_synapse_index = pixel_index // num_input_neurons
            target_synapse_index = min(
                target_synapse_index, input_synapses_per_neuron - 1
            )
            neuron_id = input_layer_ids[target_neuron_index]
            strength = float(pixel_value)
            signals.append((neuron_id, target_synapse_index, strength))
        return signals

    # CNN input: one neuron per kernel position; synapses map to receptive field pixels
    arr = image_tensor.detach().cpu().numpy().astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape for CNN input: {arr.shape}")

    signals: List[Tuple[int, int, float]] = []
    for neuron_id in input_layer_ids:
        neuron = network_sim.network.neurons[neuron_id]
        m = getattr(neuron, "metadata", {}) or {}
        k = int(m.get("kernel_size", 1))
        s = int(m.get("stride", 1))
        in_c = int(m.get("in_channels", arr.shape[0]))
        y_out = int(m.get("y", 0))
        x_out = int(m.get("x", 0))
        for c in range(in_c):
            for ky in range(k):
                for kx in range(k):
                    in_y = y_out * s + ky
                    in_x = x_out * s + kx
                    if in_y >= arr.shape[1] or in_x >= arr.shape[2]:
                        continue
                    syn_id = (c * k + ky) * k + kx
                    # Normalize from [-1, 1] to [0, 1]
                    strength = (float(arr[c, in_y, in_x]) + 1.0) * 0.5
                    signals.append((neuron_id, syn_id, strength))
    return signals


def collect_tick_snapshot(
    network_sim: NeuronNetwork,
    layers: List[List[int]],
    tick_index: int,
) -> Dict[str, Any]:
    """Collect per-layer arrays of neuron metrics for this tick."""
    net = network_sim.network
    layer_snapshots: List[Dict[str, Any]] = []
    for layer_idx, layer_ids in enumerate(layers):
        layer_S: List[float] = []
        layer_F_avg: List[float] = []
        layer_t_ref: List[float] = []
        layer_fire: List[int] = []
        for nid in layer_ids:
            neuron = net.neurons[nid]
            layer_S.append(float(neuron.S))
            layer_F_avg.append(float(neuron.F_avg))
            layer_t_ref.append(float(neuron.t_ref))
            layer_fire.append(1 if neuron.O > 0 else 0)
        layer_snapshots.append(
            {
                "layer_index": layer_idx,
                "neuron_ids": layer_ids,
                "S": layer_S,
                "F_avg": layer_F_avg,
                "t_ref": layer_t_ref,
                "fired": layer_fire,
            }
        )
    return {"tick": tick_index, "layers": layer_snapshots}


def compute_signal_grid(image_tensor: torch.Tensor) -> np.ndarray:
    """Compute a 2D grid of signal strengths matching the image's spatial size.

    Works for MNIST (1x28x28) and CIFAR (3x32x32) tensors normalized to [-1, 1].
    The grid represents per-pixel signal strengths used by image_to_signals:
    - Convert to [0,1] via (x + 1) * 0.5
    - Apply global max normalization across the entire image
    - For multi-channel images, average channels to a single grayscale plane
    """
    arr = image_tensor.detach().cpu().numpy().astype(np.float32)
    if arr.ndim == 2:
        # H x W → add channel dimension
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image tensor shape for signal grid: {arr.shape}")
    # Channels-first expected: C x H x W
    arr01 = (arr + 1.0) * 0.5
    max_val = float(np.max(arr01))
    if max_val > 0:
        arr01 = arr01 / max_val
    # Average across channels to get grayscale
    grid = np.mean(arr01, axis=0)
    return grid


def save_signal_plot(
    grid: np.ndarray, label: int, img_idx: int, out_dir: str = "plots"
) -> str:
    """Save a grayscale plot of the signal grid with the value annotated inside each pixel.
    Returns the saved file path. Requires matplotlib; otherwise no-op.
    """
    if plt is None:
        return ""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(grid, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])

    # Annotate each pixel with its value
    h, w = grid.shape
    for y in range(h):
        for x in range(w):
            val = float(grid[y, x])
            txt_color = "black" if val > 0.6 else "white"
            ax.text(
                x,
                y,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=5,
                color=txt_color,
            )

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"signals_label_{label}_idx_{img_idx}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    print("--- Build Activity Dataset from Network Simulation ---")

    # 1) Prompt for network file
    net_path = prompt_str(
        "Enter network file path", "networks/trained_20250914_171748.json"
    )
    if not os.path.isfile(net_path):
        print(f"Network file not found: {net_path}")
        return

    # 2) Load the network
    try:
        network_sim = NetworkConfig.load_network_config(net_path)
    except Exception as e:
        print(f"Failed to load network: {e}")
        return

    # Prepare core
    nn_core = NNCore()
    nn_core.neural_net = network_sim
    # Disable neuron logger noise
    try:
        setup_neuron_logger("CRITICAL")
    except Exception:
        pass
    try:
        nn_core.set_log_level("CRITICAL")
    except Exception:
        pass

    # Infer layers from metadata
    layers = infer_layers_from_metadata(network_sim)
    print(f"Detected {len(layers)} layers. Sizes: {[len(layer) for layer in layers]}")

    # 3) Prompt for web server start
    web_server = None
    if prompt_yes_no("Start web server?", default_no=True):
        print("Starting web visualization server on http://127.0.0.1:5555 ...")
        web_server = NeuralNetworkWebServer(nn_core, host="127.0.0.1", port=5555)
        t = threading.Thread(target=web_server.run, daemon=True)
        t.start()
        time.sleep(1.5)

    # 4) Load dataset (training split)
    # Dataset selection
    select_and_load_dataset()
    if selected_dataset is None:
        raise RuntimeError("Dataset failed to load")
    ds_train = selected_dataset

    # Preparation complete
    print("Preparation complete.")

    # Determine default ticks based on network structure
    default_ticks = compute_default_ticks_per_image(network_sim, layers)
    print(f"Suggested propagation ticks per image: {default_ticks}")
    ticks_per_image = prompt_int("Ticks to present each image", default_ticks)

    # Prompt for images per label
    images_per_label = prompt_int("Number of images to present per label", 5)

    # Prompt tick time in ms (0 means no delay)
    tick_ms = prompt_int("Tick time in milliseconds (0 = no delay)", 0)

    # Fresh run options: reload network per label and/or per image
    fresh_run_per_label = prompt_yes_no(
        "Fresh run per label? (reload network for each label)", default_no=True
    )

    fresh_run_per_image = prompt_yes_no(
        "Fresh run per image? (reload network for each image)", default_no=False
    )

    # If both are enabled, per-image takes precedence (more granular)
    if fresh_run_per_label and fresh_run_per_image:
        print(
            "Note: Both per-label and per-image fresh runs enabled. Per-image will take precedence."
        )
        fresh_run_per_label = (
            False  # Disable per-label since per-image is more granular
        )

    # Network state export option
    export_network_states = prompt_yes_no(
        "Export network state after each sample? (for synaptic analysis)",
        default_no=True,
    )

    # Dataset naming based on network file name
    network_base = os.path.splitext(os.path.basename(net_path))[0]
    dataset_base = prompt_str(
        "Dataset name base (files will be <base>_supervised/unsupervised_<ts>.json)",
        network_base,
    )

    # Determine input mapping
    input_layer_ids, input_synapses_per_neuron = determine_input_mapping(
        network_sim, layers
    )
    # Compatibility check: may update network and input ids
    network_sim, input_layer_ids = pre_run_compatibility_check(
        network_sim, input_layer_ids
    )
    nn_core.neural_net = network_sim
    # Recompute synapses per input after potential change
    input_synapses_per_neuron = max(
        1, len(network_sim.network.neurons[input_layer_ids[0]].postsynaptic_points)
    )

    # Build index of dataset by label
    label_to_indices: Dict[int, List[int]] = {i: [] for i in range(CURRENT_NUM_CLASSES)}
    for idx in range(len(ds_train)):
        _img, lbl = ds_train[idx]
        label_to_indices[int(lbl)].append(idx)

    # Choose output format
    use_binary_format = prompt_yes_no(
        "Use binary format?",
        default_no=False,
    )

    ts = int(time.time())
    # Initialize datasets
    if use_binary_format:
        # Binary tensor format
        dataset_dir = f"activity_datasets/{dataset_base}_{CURRENT_DATASET_NAME}_{ts}"
        recorder = HDF5TensorRecorder(dataset_dir, network_sim.network)
        records: List[Dict[str, Any]] = []  # Not used in binary mode
        print(f"Recording to single HDF5 file: {dataset_dir}/activity_dataset.h5")
    else:
        dataset_dir = f"activity_datasets/{dataset_base}_{CURRENT_DATASET_NAME}_{ts}"
        # Legacy JSON format
        records: List[Dict[str, Any]] = []
        recorder = None  # Not used in JSON mode
        print("Using legacy JSON format")

    plotted_labels: set[int] = set()

    # Prepare network state export directory (timestamp generated early for consistency)
    network_state_dir: Path | None = None
    if export_network_states:
        network_state_dir = (
            Path("network_state") / f"{dataset_base}_{CURRENT_DATASET_NAME}_{ts}"
        )
        network_state_dir.mkdir(parents=True, exist_ok=True)
        print(f"Network states will be exported to: {network_state_dir}")

    # Collection loop
    print("Starting collection loop ...")
    for label in tqdm(range(CURRENT_NUM_CLASSES), desc="Labels", leave=False):
        if fresh_run_per_label and not fresh_run_per_image:
            nn_core.set_log_level("CRITICAL")
            # Reload the network to its initial saved state
            network_sim = NetworkConfig.load_network_config(net_path)
            nn_core.neural_net = network_sim
            setup_neuron_logger("CRITICAL")

            # Recompute layers and input mapping to ensure consistency
            layers = infer_layers_from_metadata(network_sim)
            input_layer_ids, input_synapses_per_neuron = determine_input_mapping(
                network_sim, layers
            )

            # Reset simulation to start from tick 0
            network_sim.reset_simulation()

            # Reset nn_core tick counter to 0
            nn_core.state.current_tick = 0
            network_sim.current_tick = 0

        label_start = time.perf_counter()
        indices = label_to_indices.get(label, [])
        if not indices:
            continue
        chosen = random.sample(indices, min(images_per_label, len(indices)))
        for img_pos, img_idx in enumerate(
            tqdm(chosen, desc=f"Label {label} images", leave=False)
        ):
            # Fresh run per image: reload network for each image
            if fresh_run_per_image:
                network_sim = NetworkConfig.load_network_config(net_path)
                nn_core.neural_net = network_sim
                setup_neuron_logger("CRITICAL")

                # Recompute layers and input mapping to ensure consistency
                layers = infer_layers_from_metadata(network_sim)
                input_layer_ids, input_synapses_per_neuron = determine_input_mapping(
                    network_sim, layers
                )

                # Reset simulation to start from tick 0
                network_sim.reset_simulation()

                # Reset nn_core tick counter to 0
                nn_core.state.current_tick = 0
                network_sim.current_tick = 0

            image_start = time.perf_counter()
            img_tensor, actual_label = ds_train[img_idx]
            # Compose signals for this image
            signals = image_to_signals(
                img_tensor, input_layer_ids, input_synapses_per_neuron, network_sim
            )

            # For the first image per label: save grayscale plot with annotations
            if int(actual_label) not in plotted_labels:
                grid = compute_signal_grid(img_tensor)
                save_signal_plot(grid, int(actual_label), int(img_idx))
                plotted_labels.add(int(actual_label))

            # Per-neuron cumulative firing counters for this presentation
            cumulative_fires: Dict[int, int] = {
                nid: 0 for nid in network_sim.network.neurons.keys()
            }

            # Initialize recorder buffer for this image (binary mode only)
            if use_binary_format and recorder:
                recorder.init_buffer(ticks_per_image)

            # Present image for ticks_per_image ticks
            with tqdm(
                total=ticks_per_image,
                desc=f"Label {label} img {img_pos + 1}/{len(chosen)}",
                leave=False,
            ) as tick_bar:
                for local_tick in range(ticks_per_image):
                    # Send the same image-encoded signals each tick (as in interactive_mnist)
                    tick_exec_start = time.perf_counter()
                    nn_core.send_batch_signals(signals)
                    nn_core.do_tick()
                    tick_exec_ms = (time.perf_counter() - tick_exec_start) * 1000.0

                    # Update firing counters
                    for nid, neuron in network_sim.network.neurons.items():
                        if neuron.O > 0:
                            cumulative_fires[nid] += 1

                    if use_binary_format and recorder:
                        # FAST BINARY CAPTURE - No dictionary creation!
                        recorder.capture_tick(local_tick, network_sim.network.neurons)
                    else:
                        # LEGACY JSON MODE - Keep existing logic
                        # Snapshot per-layer metrics
                        snapshot = collect_tick_snapshot(
                            network_sim, layers, tick_index=network_sim.current_tick
                        )

                        # Attach cumulative firing counts mapped per layer order
                        # Convert global counters to aligned arrays per layer for compactness
                        cum_layer_counts: List[List[int]] = []
                        for layer_ids in layers:
                            cum_layer_counts.append(
                                [int(cumulative_fires[n]) for n in layer_ids]
                            )

                        record_base = {
                            "image_index": int(img_idx),
                            "label": int(actual_label),
                            "tick": snapshot["tick"],
                            "layers": snapshot["layers"],
                            "cumulative_fires": cum_layer_counts,
                        }

                        records.append(record_base)

                    # Update tqdm metrics
                    elapsed_label = time.perf_counter() - label_start
                    elapsed_image = time.perf_counter() - image_start
                    tick_bar.set_postfix(
                        {
                            "tick": int(network_sim.current_tick),
                            "ms/t": f"{tick_exec_ms:.2f}",
                            "label_s": f"{elapsed_label:.2f}",
                            "img_s": f"{elapsed_image:.2f}",
                        }
                    )
                    tick_bar.update(1)

                    if tick_ms > 0:
                        time.sleep(tick_ms / 1000.0)

            # Export network state after this sample (after all ticks complete)
            if export_network_states and network_state_dir is not None:
                label_dir = network_state_dir / f"label_{label}"
                label_dir.mkdir(parents=True, exist_ok=True)
                state_filename = label_dir / f"sample_{img_pos}_img{img_idx}.json"
                NetworkConfig.save_network_config(
                    network_sim,
                    state_filename,
                    metadata={
                        "sample_index": img_pos,
                        "image_index": int(img_idx),
                        "label": int(actual_label),
                        "ticks_simulated": ticks_per_image,
                        "final_tick": int(network_sim.current_tick),
                    },
                )

            # Save binary sample (if using binary format)
            if use_binary_format and recorder:
                # Create global sample index across all labels
                global_sample_idx = (
                    sum(
                        len(label_to_indices[label_idx][:images_per_label])
                        for label_idx in range(label)
                    )
                    + img_pos
                )
                recorder.save_sample(global_sample_idx, int(actual_label))

    # Output files
    if use_binary_format:
        dataset_dir = f"activity_datasets/{dataset_base}_{CURRENT_DATASET_NAME}_{ts}"
        sample_count = sum(
            min(images_per_label, len(indices)) for indices in label_to_indices.values()
        )
        print(
            f"HDF5 dataset complete -> {dataset_dir}/activity_dataset.h5 ({sample_count} compressed samples)"
        )
    else:
        sup_name = f"{dataset_base}_{CURRENT_DATASET_NAME}_{ts}.json"
        print(f"Writing JSON dataset -> {sup_name} ({len(records)} records)")
        with open(sup_name, "w") as f:
            json.dump({"records": records}, f)

    # Close HDF5 file if using binary format
    if use_binary_format and recorder:
        recorder.close()

    print("Done.")


if __name__ == "__main__":
    main()
