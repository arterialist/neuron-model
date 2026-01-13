import os
import sys
import time
import json
import math
import random
import threading
from typing import List, Dict, Any, Tuple
from multiprocessing import Pool, Manager
import multiprocessing as mp

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

    def save_sample_from_buffers(
        self, sample_idx, label, u_buf, t_ref_buf, fr_buf, spike_arr
    ):
        """Append sample to HDF5 file using pre-computed buffers"""
        # Ensure datasets are created
        if self.h5_file is None:
            # Infer ticks from buffer shape
            ticks = u_buf.shape[0] if u_buf is not None else 100
            self._create_datasets(ticks)

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
# Global flag: when True, indicates colored CIFAR-10 with 3 synapses per pixel
IS_COLORED_CIFAR10 = False
# Normalization factor for each color channel in colored CIFAR-10 (default 0.5 for [0,1] range)
CIFAR10_COLOR_NORMALIZATION_FACTOR = 0.5


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
    print("  3) CIFAR10 (color)")
    print("  4) CIFAR100")
    print("  5) USPS")
    print("  6) SVHN")
    print("  7) FashionMNIST")
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
        IS_COLORED_CIFAR10 = False
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
        IS_COLORED_CIFAR10 = False
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
                ds = datasets.CIFAR10(
                    root=root, train=True, download=True, transform=transform
                )
                break
            except Exception as e:
                last_err = e
                continue
        if ds is None:
            raise RuntimeError(f"Failed to load CIFAR10 (color): {last_err}")
        selected_dataset = ds
        img0, _ = selected_dataset[0]
        # For colored CIFAR-10, vector size is pixels * 3 (RGB channels)
        CURRENT_IMAGE_VECTOR_SIZE = int(img0.shape[1] * img0.shape[2] * 3)
        CURRENT_NUM_CLASSES = 10
        CURRENT_DATASET_NAME = "cifar10_color"
        IS_COLORED_CIFAR10 = True

        # Prompt for normalization factor
        global CIFAR10_COLOR_NORMALIZATION_FACTOR
        default_factor = 0.33  # Equivalent to [0, 0.33] range
        normalization_factor = prompt_float(
            f"Normalization factor for each color channel [0, X] (default {default_factor} for [0, 0.33] range): ",
            default_factor,
        )
        CIFAR10_COLOR_NORMALIZATION_FACTOR = (
            normalization_factor / 2.0
        )  # Convert upper bound to normalization factor
        print(
            f"Each color channel will be normalized to [0, {normalization_factor:.3f}] range"
        )
    elif choice == "4":
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
        IS_COLORED_CIFAR10 = False
    elif choice == "5":
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
        IS_COLORED_CIFAR10 = False
    elif choice == "6":
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
        IS_COLORED_CIFAR10 = False
    elif choice == "7":
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
        IS_COLORED_CIFAR10 = False  # noqa: F841
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

    Supports legacy dense input, colored CIFAR-10, and CNN-style input (conv layer at index 0).
    Signals are normalized to [0, 1] range.
    """
    # Detect CNN-style input
    first_neuron = network_sim.network.neurons[input_layer_ids[0]]
    meta = getattr(first_neuron, "metadata", {}) or {}
    is_cnn_input = meta.get("layer_type") == "conv" and meta.get("layer", 0) == 0

    if not is_cnn_input:
        if IS_COLORED_CIFAR10:
            arr = image_tensor.detach().cpu().numpy().astype(np.float32)
            if arr.ndim == 3 and arr.shape[0] == 3:  # CHW format
                h, w = arr.shape[1], arr.shape[2]
                signals = []

                # Check if this network uses separate neurons per color channel
                # by examining metadata of the first input neuron
                first_neuron = network_sim.network.neurons[input_layer_ids[0]]
                meta = getattr(first_neuron, "metadata", {}) or {}
                separate_neurons_per_color = "color_channel" in meta

                if separate_neurons_per_color:
                    # Architecture 2: One neuron per spatial kernel per RGB channel
                    # Each neuron handles one color channel for a subset of pixels
                    total_spatial_positions = h * w
                    neurons_per_color = len(input_layer_ids) // 3

                    for y in range(h):
                        for x in range(w):
                            pixel_index = y * w + x
                            for c in range(3):  # RGB channels
                                # Calculate which spatial neuron handles this pixel for this color
                                spatial_neuron_idx = pixel_index % neurons_per_color
                                # Calculate global neuron index: color_offset + spatial_idx
                                global_neuron_idx = (
                                    c * neurons_per_color + spatial_neuron_idx
                                )

                                if global_neuron_idx >= len(input_layer_ids):
                                    continue  # Skip if neuron index exceeds available neurons

                                # Each neuron handles one color channel for multiple pixels
                                pixels_per_neuron = (
                                    total_spatial_positions // neurons_per_color
                                )
                                synapse_index = pixel_index // neurons_per_color

                                neuron_id = input_layer_ids[global_neuron_idx]
                                # Normalize from [-1, 1] to [0, X] where X is the specified upper bound
                                pixel_value = arr[c, y, x]
                                strength = (
                                    float(pixel_value) + 1.0
                                ) * CIFAR10_COLOR_NORMALIZATION_FACTOR
                                signals.append((neuron_id, synapse_index, strength))
                else:
                    # Architecture 1: One neuron per spatial kernel, 3 synapses per RGB channel
                    for y in range(h):
                        for x in range(w):
                            for c in range(3):  # RGB channels
                                # Calculate which input neuron handles this pixel
                                pixel_index = y * w + x
                                target_neuron_index = pixel_index % len(input_layer_ids)
                                # Each neuron handles multiple pixels, each pixel has 3 synapses
                                pixels_per_neuron = (h * w) // len(input_layer_ids)
                                if (
                                    pixel_index // len(input_layer_ids)
                                    >= pixels_per_neuron
                                ):
                                    continue  # This pixel doesn't fit in the network
                                base_synapse_index = (
                                    pixel_index // len(input_layer_ids)
                                ) * 3
                                synapse_index = base_synapse_index + c
                                if synapse_index >= input_synapses_per_neuron:
                                    continue  # Skip if synapse index exceeds available synapses

                                neuron_id = input_layer_ids[target_neuron_index]
                                # Normalize from [-1, 1] to [0, X] where X is the specified upper bound
                                pixel_value = arr[c, y, x]
                                strength = (
                                    float(pixel_value) + 1.0
                                ) * CIFAR10_COLOR_NORMALIZATION_FACTOR
                                signals.append((neuron_id, synapse_index, strength))
                return signals
        else:
            # Legacy dense mapping
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


def process_single_image_worker(args):
    """Worker function for multiprocessing image processing.

    Returns: (img_idx, label, u_buf, t_ref_buf, fr_buf, spikes, records)
    """
    (
        img_idx,
        actual_label,
        ticks_per_image,
        net_path,
        input_layer_ids,
        input_synapses_per_neuron,
        layers,
        use_binary_format,
        tick_ms,
        ds_train,
        process_id,
        progress_queue,
    ) = args

    # Load network for this worker
    network_sim = NetworkConfig.load_network_config(net_path)
    nn_core = NNCore()
    nn_core.neural_net = network_sim
    setup_neuron_logger("CRITICAL")
    nn_core.set_log_level("CRITICAL")

    # Reset simulation
    network_sim.reset_simulation()
    nn_core.state.current_tick = 0
    network_sim.current_tick = 0

    image_tensor, _ = ds_train[img_idx]
    signals = image_to_signals(
        image_tensor, input_layer_ids, input_synapses_per_neuron, network_sim
    )

    # Initialize buffers
    if use_binary_format:
        num_neurons = len(network_sim.network.neurons)
        u_buf = np.zeros((ticks_per_image, num_neurons), dtype=np.float32)
        t_ref_buf = np.zeros((ticks_per_image, num_neurons), dtype=np.float32)
        fr_buf = np.zeros((ticks_per_image, num_neurons), dtype=np.float32)
        spikes = []
        # Create uuid_to_idx mapping once before the tick loop
        uuid_to_idx = {
            uid: i for i, uid in enumerate(network_sim.network.neurons.keys())
        }

    # Per-neuron cumulative firing counters
    cumulative_fires = {nid: 0 for nid in network_sim.network.neurons.keys()}

    records = []  # For JSON mode

    # Send initial progress update
    if progress_queue is not None:
        progress_queue.put(
            {
                "process_id": process_id,
                "current_tick": 0,
                "total_ticks": ticks_per_image,
                "img_idx": img_idx,
                "label": actual_label,
                "completed": False,
            }
        )

    # Present image for ticks_per_image ticks
    for local_tick in range(ticks_per_image):
        # Send signals and do tick
        nn_core.send_batch_signals(signals)
        nn_core.do_tick()

        # Update firing counters
        for nid, neuron in network_sim.network.neurons.items():
            if neuron.O > 0:
                cumulative_fires[nid] += 1

        if use_binary_format:
            # Direct capture using uuid_to_idx mapping (created once before loop)
            for uid, neuron in network_sim.network.neurons.items():
                idx = uuid_to_idx[uid]
                u_buf[local_tick, idx] = neuron.S
                t_ref_buf[local_tick, idx] = neuron.t_ref
                fr_buf[local_tick, idx] = neuron.F_avg
                if neuron.O > 0:
                    spikes.append((local_tick, idx))
        else:
            # JSON mode snapshot
            snapshot = collect_tick_snapshot(
                network_sim, layers, tick_index=network_sim.current_tick
            )
            cum_layer_counts = []
            for layer_ids in layers:
                cum_layer_counts.append([int(cumulative_fires[n]) for n in layer_ids])

            record_base = {
                "image_index": int(img_idx),
                "label": int(actual_label),
                "tick": snapshot["tick"],
                "layers": snapshot["layers"],
                "cumulative_fires": cum_layer_counts,
            }
            records.append(record_base)

        # Send progress update
        if (
            progress_queue is not None and (local_tick + 1) % 10 == 0
        ):  # Update every 10 ticks
            progress_queue.put(
                {
                    "process_id": process_id,
                    "current_tick": local_tick + 1,
                    "total_ticks": ticks_per_image,
                    "img_idx": img_idx,
                    "label": actual_label,
                    "completed": False,
                }
            )

        if tick_ms > 0:
            time.sleep(tick_ms / 1000.0)

    # Send completion update
    if progress_queue is not None:
        progress_queue.put(
            {
                "process_id": process_id,
                "current_tick": ticks_per_image,
                "total_ticks": ticks_per_image,
                "img_idx": img_idx,
                "label": actual_label,
                "completed": True,
            }
        )

    return (
        img_idx,
        actual_label,
        u_buf if use_binary_format else None,
        t_ref_buf if use_binary_format else None,
        fr_buf if use_binary_format else None,
        spikes if use_binary_format else None,
        records if not use_binary_format else None,
    )


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

    # Prompt for multiprocessing
    use_multiprocessing = prompt_yes_no(
        "Use multiprocessing for parallel processing?", default_no=False
    )

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

    # Collection loop - keep the original structure with labels and samples progress bars
    print("Starting collection loop ...")
    global_sample_counter = 0
    processed_results = []

    # Create main labels progress bar
    labels_pbar = tqdm(range(CURRENT_NUM_CLASSES), desc="Labels", leave=False)

    for label_idx in range(CURRENT_NUM_CLASSES):
        labels_pbar.update(1)
        labels_pbar.set_description(f"Label {label_idx}")

        indices = label_to_indices.get(label_idx, [])
        if not indices:
            continue

        chosen = random.sample(indices, min(images_per_label, len(indices)))

        # Create samples progress bar for this label
        samples_pbar = tqdm(
            total=len(chosen),
            desc=f"Label {label_idx} samples",
            position=1,
            leave=False,
        )

        # Prepare tasks for this label
        label_tasks = []
        for img_pos, img_idx in enumerate(chosen):
            task = (
                img_idx,
                label_idx,  # actual_label
                ticks_per_image,
                net_path,
                input_layer_ids,
                input_synapses_per_neuron,
                layers,
                use_binary_format,
                tick_ms,
                ds_train,
            )
            label_tasks.append((task, label_idx, img_pos, global_sample_counter))
            global_sample_counter += 1

        # Process tasks for this label
        if use_multiprocessing and len(label_tasks) > 1:
            # Use a reasonable number of processes
            num_processes = min(mp.cpu_count(), len(label_tasks))

            # Create progress queue for workers to send updates
            manager = Manager()
            progress_queue = manager.Queue()

            with Pool(processes=num_processes) as pool:
                # Create individual progress bars for each process
                process_bars = {}
                process_progress = {}

                # Submit tasks for this label
                results = []
                task_idx = 0
                for task_data in label_tasks:
                    task, label, img_pos, global_sample_idx = task_data
                    # Assign process ID and add to task arguments
                    process_id = task_idx % num_processes
                    updated_task = task + (process_id, progress_queue)
                    result = pool.apply_async(
                        process_single_image_worker, (updated_task,)
                    )
                    results.append(
                        (
                            result,
                            label,
                            img_pos,
                            global_sample_idx,
                            task[0],
                            process_id,
                        )
                    )  # task[0] is img_idx
                    task_idx += 1

                # Monitor progress while tasks are running
                label_processed_results = []
                completed_count = 0
                last_update = {proc_id: 0 for proc_id in range(num_processes)}

                while completed_count < len(results):
                    # Read any available progress updates from the queue
                    while not progress_queue.empty():
                        try:
                            progress_data = progress_queue.get_nowait()
                            process_id = progress_data["process_id"]
                            process_progress[process_id] = progress_data

                            # Create progress bar for new process
                            if process_id not in process_bars:
                                img_idx = progress_data["img_idx"]
                                label = progress_data["label"]
                                total_ticks = progress_data["total_ticks"]
                                process_bars[process_id] = tqdm(
                                    total=total_ticks,
                                    desc=f"P{process_id + 1} L{label} I{img_idx}",
                                    leave=False,
                                    unit="ticks",
                                )
                        except Exception:
                            break

                    # Update progress bars
                    for proc_id in list(process_bars.keys()):
                        if proc_id in process_progress:
                            progress_data = process_progress[proc_id]
                            current_tick = progress_data["current_tick"]
                            total_ticks = progress_data["total_ticks"]

                            # Update progress bar
                            progress_increment = current_tick - last_update[proc_id]
                            if progress_increment > 0:
                                process_bars[proc_id].update(progress_increment)
                                last_update[proc_id] = current_tick

                            # Close completed bars
                            if progress_data.get("completed", False):
                                process_bars[proc_id].close()
                                del process_bars[proc_id]
                                if proc_id in process_progress:
                                    del process_progress[proc_id]

                    # Check for completed tasks
                    for i, result_info in enumerate(results):
                        (
                            result_future,
                            label,
                            img_pos,
                            global_sample_idx,
                            img_idx,
                            process_id,
                        ) = result_info
                        if result_future.ready() and not hasattr(
                            result_future, "_collected"
                        ):
                            result = result_future.get()
                            label_processed_results.append(
                                (result, label, img_pos, global_sample_idx, img_idx)
                            )
                            result_future._collected = True
                            completed_count += 1
                            samples_pbar.update(1)

                            # Save data immediately for this completed task
                            (
                                result_img_idx,
                                actual_label,
                                u_buf,
                                t_ref_buf,
                                fr_buf,
                                spikes,
                                records_json,
                            ) = result

                            # Save binary data
                            if use_binary_format and recorder:
                                spike_arr = (
                                    np.array(spikes, dtype=np.int32).flatten()
                                    if spikes
                                    else np.zeros((0,), dtype=np.int32)
                                )
                                recorder.save_sample_from_buffers(
                                    global_sample_idx,
                                    int(actual_label),
                                    u_buf,
                                    t_ref_buf,
                                    fr_buf,
                                    spike_arr,
                                )

                            # Save JSON records
                            if not use_binary_format and records_json:
                                records.extend(records_json)

                            # Export network state (handled sequentially since it's file I/O)
                            if export_network_states and network_state_dir is not None:
                                # For multiprocessing, we can't easily save network state from workers
                                # So we do it sequentially here by reloading and simulating the specific image
                                label_dir = network_state_dir / f"label_{label}"
                                label_dir.mkdir(parents=True, exist_ok=True)
                                state_filename = (
                                    label_dir / f"sample_{img_pos}_img{img_idx}.json"
                                )

                                # Reload network and simulate this specific image
                                network_sim_state = NetworkConfig.load_network_config(
                                    net_path
                                )
                                nn_core_state = NNCore()
                                nn_core_state.neural_net = network_sim_state
                                setup_neuron_logger("CRITICAL")
                                nn_core_state.set_log_level("CRITICAL")

                                network_sim_state.reset_simulation()
                                nn_core_state.state.current_tick = 0
                                network_sim_state.current_tick = 0

                                img_tensor, _ = ds_train[img_idx]
                                signals = image_to_signals(
                                    img_tensor,
                                    input_layer_ids,
                                    input_synapses_per_neuron,
                                    network_sim_state,
                                )

                                # Simulate for the required ticks
                                for _ in range(ticks_per_image):
                                    nn_core_state.send_batch_signals(signals)
                                    nn_core_state.do_tick()

                                NetworkConfig.save_network_config(
                                    network_sim_state,
                                    state_filename,
                                    metadata={
                                        "sample_index": img_pos,
                                        "image_index": int(img_idx),
                                        "label": int(actual_label),
                                        "ticks_simulated": ticks_per_image,
                                        "final_tick": int(
                                            network_sim_state.current_tick
                                        ),
                                    },
                                )

                            # Clear progress for this process (ready for next task)
                            if process_id in process_progress:
                                del process_progress[process_id]
                            last_update[process_id] = 0

                    time.sleep(
                        0.05
                    )  # Small delay to avoid busy waiting and allow progress bar updates

                # Close any remaining progress bars
                for bar in process_bars.values():
                    bar.close()

                # Final status update
                tqdm.write(
                    f"Label {label_idx} processing complete - {len(label_processed_results)} samples processed"
                )

                processed_results.extend(label_processed_results)

        else:
            # Sequential processing for this label
            for task_data in label_tasks:
                task, label, img_pos, global_sample_idx = task_data
                # For sequential processing, pass None for shared progress dict
                updated_task = task + (None, None)
                result = process_single_image_worker(updated_task)

                # Save data immediately for this completed task (sequential)
                (
                    result_img_idx,
                    actual_label,
                    u_buf,
                    t_ref_buf,
                    fr_buf,
                    spikes,
                    records_json,
                ) = result

                # Save binary data
                if use_binary_format and recorder:
                    spike_arr = (
                        np.array(spikes, dtype=np.int32).flatten()
                        if spikes
                        else np.zeros((0,), dtype=np.int32)
                    )
                    recorder.save_sample_from_buffers(
                        global_sample_idx,
                        int(actual_label),
                        u_buf,
                        t_ref_buf,
                        fr_buf,
                        spike_arr,
                    )

                # Save JSON records
                if not use_binary_format and records_json:
                    records.extend(records_json)

                # Export network state (handled sequentially since it's file I/O)
                if export_network_states and network_state_dir is not None:
                    # For sequential processing, we can save network state from workers
                    # So we do it here by reloading and simulating the specific image
                    label_dir = network_state_dir / f"label_{label}"
                    label_dir.mkdir(parents=True, exist_ok=True)
                    state_filename = label_dir / f"sample_{img_pos}_img{img_idx}.json"

                    # Reload network and simulate this specific image
                    network_sim_state = NetworkConfig.load_network_config(net_path)
                    nn_core_state = NNCore()
                    nn_core_state.neural_net = network_sim_state
                    setup_neuron_logger("CRITICAL")
                    nn_core_state.set_log_level("CRITICAL")

                    network_sim_state.reset_simulation()
                    nn_core_state.state.current_tick = 0
                    network_sim_state.current_tick = 0

                    img_tensor, _ = ds_train[img_idx]
                    signals = image_to_signals(
                        img_tensor,
                        input_layer_ids,
                        input_synapses_per_neuron,
                        network_sim_state,
                    )

                    # Simulate for the required ticks
                    for _ in range(ticks_per_image):
                        nn_core_state.send_batch_signals(signals)
                        nn_core_state.do_tick()

                    NetworkConfig.save_network_config(
                        network_sim_state,
                        state_filename,
                        metadata={
                            "sample_index": img_pos,
                            "image_index": int(img_idx),
                            "label": int(actual_label),
                            "ticks_simulated": ticks_per_image,
                            "final_tick": int(network_sim_state.current_tick),
                        },
                    )

                processed_results.append(
                    (result, label, img_pos, global_sample_idx, task[0])
                )
                samples_pbar.update(1)

        samples_pbar.close()

    labels_pbar.close()

    # Handle signal plots (first image per label)
    plotted_labels: set[int] = set()
    for _, label, _, _, img_idx in processed_results:
        if label not in plotted_labels:
            img_tensor, _ = ds_train[img_idx]
            grid = compute_signal_grid(img_tensor)
            save_signal_plot(grid, label, int(img_idx))
            plotted_labels.add(label)

    # All data has already been saved progressively during processing
    print("All data saved progressively during processing")

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
