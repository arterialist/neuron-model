import os
import sys
import time
import json
import math
import random
import threading
from typing import List, Dict, Any, Tuple
import multiprocessing as mp
import yaml
import logging

import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import h5py

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Ensure local imports resolve
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuron.nn_core import NNCore
from neuron.network_config import NetworkConfig
from neuron.neuron import Neuron
from neuron import setup_neuron_logger
from pipeline.config import SimulationConfig

# Re-use the HDF5TensorRecorder class from the original script
class HDF5TensorRecorder:
    def __init__(self, output_dir, network):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.uuid_to_idx = {uid: i for i, uid in enumerate(network.neurons.keys())}
        self.idx_to_uuid = list(network.neurons.keys())
        self.num_neurons = len(self.idx_to_uuid)
        self.layer_structure = self._extract_layer_structure(network)
        self.h5_path = os.path.join(output_dir, "activity_dataset.h5")
        self.h5_file = None
        self.current_sample_idx = 0

    def _extract_layer_structure(self, network):
        layers = {}
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

    def _create_datasets(self, ticks):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "w")

        chunk_shape_u = (1, ticks, self.num_neurons)
        max_shape_u = (None, ticks, self.num_neurons)

        self.u_dataset = self.h5_file.create_dataset(
            "u", shape=(0, ticks, self.num_neurons), maxshape=max_shape_u, dtype=np.float32, chunks=chunk_shape_u, compression="gzip", compression_opts=6
        )
        self.t_ref_dataset = self.h5_file.create_dataset(
            "t_ref", shape=(0, ticks, self.num_neurons), maxshape=max_shape_u, dtype=np.float32, chunks=chunk_shape_u, compression="gzip", compression_opts=6
        )
        self.fr_dataset = self.h5_file.create_dataset(
            "fr", shape=(0, ticks, self.num_neurons), maxshape=max_shape_u, dtype=np.float32, chunks=chunk_shape_u, compression="gzip", compression_opts=6
        )
        self.spikes_dataset = self.h5_file.create_dataset(
            "spikes", shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=np.dtype("int32")), chunks=(100,), compression="gzip"
        )
        self.labels_dataset = self.h5_file.create_dataset(
            "labels", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(1000,), compression="gzip"
        )
        self.num_samples_dataset = self.h5_file.create_dataset("num_samples", shape=(1,), dtype=np.int32)
        self.num_samples_dataset[0] = 0
        neuron_ids_str = [str(uid) for uid in self.idx_to_uuid]
        self.h5_file.create_dataset("neuron_ids", data=neuron_ids_str, dtype=h5py.special_dtype(vlen=str), compression="gzip")
        self.h5_file.create_dataset("layer_structure", data=np.array(self.layer_structure, dtype=np.int32), compression="gzip")

    def save_sample_from_buffers(self, sample_idx, label, u_buf, t_ref_buf, fr_buf, spike_arr):
        if self.h5_file is None:
            ticks = u_buf.shape[0] if u_buf is not None else 100
            self._create_datasets(ticks)
        current_size = self.u_dataset.shape[0]
        if sample_idx >= current_size:
            new_size = max(sample_idx + 1, current_size + 1000)
            self.u_dataset.resize((new_size, self.u_dataset.shape[1], self.u_dataset.shape[2]))
            self.t_ref_dataset.resize((new_size, self.t_ref_dataset.shape[1], self.t_ref_dataset.shape[2]))
            self.fr_dataset.resize((new_size, self.t_ref_dataset.shape[1], self.t_ref_dataset.shape[2]))
            self.spikes_dataset.resize((new_size,))
            self.labels_dataset.resize((new_size,))

        self.u_dataset[sample_idx] = u_buf
        self.t_ref_dataset[sample_idx] = t_ref_buf
        self.fr_dataset[sample_idx] = fr_buf
        self.spikes_dataset[sample_idx] = spike_arr
        self.labels_dataset[sample_idx] = label
        self.current_sample_idx = sample_idx + 1
        self.num_samples_dataset[0] = self.current_sample_idx

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

# Helper functions adapted to be non-interactive
def load_dataset(dataset_name: str) -> Tuple[Any, int, int]:
    # Simplified loading logic based on name
    transform_gray = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    transform_color = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data_path = "./data"

    if dataset_name == "mnist":
        ds = datasets.MNIST(root=data_path, train=True, download=True, transform=transform_gray)
        return ds, 784, 10
    elif dataset_name == "cifar10":
        ds = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_color)
        return ds, 3072, 10
    # Add others as needed, defaulting to MNIST
    ds = datasets.MNIST(root=data_path, train=True, download=True, transform=transform_gray)
    return ds, 784, 10

def infer_layers_from_metadata(network_sim) -> List[List[int]]:
    net = network_sim.network
    layer_to_neurons = {}
    for nid, neuron in net.neurons.items():
        layer_idx = int(neuron.metadata.get("layer", 0))
        layer_to_neurons.setdefault(layer_idx, []).append(nid)
    if not layer_to_neurons:
        return [list(net.neurons.keys())]
    return [layer_to_neurons[k] for k in sorted(layer_to_neurons.keys())]

def determine_input_mapping(network_sim, layers) -> Tuple[List[int], int]:
    input_layer_ids = layers[0]
    first_input = network_sim.network.neurons[input_layer_ids[0]]
    input_synapses_per_neuron = max(1, len(first_input.postsynaptic_points))
    return input_layer_ids, input_synapses_per_neuron

def image_to_signals(image_tensor, input_layer_ids, input_synapses_per_neuron, network_sim):
    # Simplified signal conversion
    # Assuming dense or simple 1-1 mapping for now as per original script logic
    # The original script had complex logic for CNNs. We replicate the generic dense logic
    # and the CNN metadata logic.

    first_neuron = network_sim.network.neurons[input_layer_ids[0]]
    meta = getattr(first_neuron, "metadata", {}) or {}
    is_cnn_input = meta.get("layer_type") == "conv" and meta.get("layer", 0) == 0

    arr = image_tensor.detach().cpu().numpy().astype(np.float32)

    signals = []

    if is_cnn_input:
        if arr.ndim == 2: arr = arr[None, :, :]
        # CNN logic
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
                        if in_y >= arr.shape[1] or in_x >= arr.shape[2]: continue
                        syn_id = (c * k + ky) * k + kx
                        strength = (float(arr[c, in_y, in_x]) + 1.0) * 0.5
                        signals.append((neuron_id, syn_id, strength))
    else:
        # Dense logic
        img_vec = image_tensor.view(-1).numpy().astype(np.float32)
        img_vec = (img_vec + 1.0) * 0.5
        num_inputs = len(input_layer_ids)
        for px_idx, px_val in enumerate(img_vec):
            target_neuron_idx = px_idx % num_inputs
            target_syn_idx = px_idx // num_inputs
            if target_syn_idx < input_synapses_per_neuron:
                neuron_id = input_layer_ids[target_neuron_idx]
                signals.append((neuron_id, target_syn_idx, float(px_val)))

    return signals

def process_image(img_idx, label, ticks, net_path, input_ids, input_syns, layers, tick_ms, dataset, export_path=None):
    # Worker function logic, essentially just running the sim
    # Re-loading network per sample (Fresh Run per Image)
    network_sim = NetworkConfig.load_network_config(net_path)
    nn_core = NNCore()
    nn_core.neural_net = network_sim
    setup_neuron_logger("CRITICAL")
    nn_core.set_log_level("CRITICAL")
    network_sim.reset_simulation()

    img_tensor, _ = dataset[img_idx]
    signals = image_to_signals(img_tensor, input_ids, input_syns, network_sim)

    num_neurons = len(network_sim.network.neurons)
    uuid_to_idx = {uid: i for i, uid in enumerate(network_sim.network.neurons.keys())}

    u_buf = np.zeros((ticks, num_neurons), dtype=np.float32)
    t_ref_buf = np.zeros((ticks, num_neurons), dtype=np.float32)
    fr_buf = np.zeros((ticks, num_neurons), dtype=np.float32)
    spikes = []

    for t in range(ticks):
        nn_core.send_batch_signals(signals)
        nn_core.do_tick()

        for uid, neuron in network_sim.network.neurons.items():
            idx = uuid_to_idx[uid]
            u_buf[t, idx] = neuron.S
            t_ref_buf[t, idx] = neuron.t_ref
            fr_buf[t, idx] = neuron.F_avg
            if neuron.O > 0:
                spikes.append((t, idx))

    if export_path:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        NetworkConfig.save_network_config(
            network_sim,
            export_path,
            metadata={
                "image_index": int(img_idx),
                "label": int(label),
                "ticks_simulated": ticks
            }
        )

    return img_idx, label, u_buf, t_ref_buf, fr_buf, spikes

def run_recording(config: SimulationConfig, network_json_path: str, output_dir: str):
    logger = logging.getLogger("RecordActivity")

    # Load Network
    network_sim = NetworkConfig.load_network_config(network_json_path)
    layers = infer_layers_from_metadata(network_sim)
    input_ids, input_syns = determine_input_mapping(network_sim, layers)

    # Load Dataset (assuming MNIST for now based on config defaults or name)
    # Ideally config has dataset name
    ds_name = "mnist" # Default
    if "cifar10" in config.dataset_name_base.lower(): ds_name = "cifar10"

    dataset, vec_size, num_classes = load_dataset(ds_name)

    # Prepare recorder
    recorder = HDF5TensorRecorder(output_dir, network_sim.network)

    # Select samples
    label_to_indices = {i: [] for i in range(num_classes)}
    for idx in range(len(dataset)):
        _, lbl = dataset[idx]
        label_to_indices[int(lbl)].append(idx)

    tasks = []
    global_idx = 0
    for label_idx in range(num_classes):
        indices = label_to_indices.get(label_idx, [])
        chosen = random.sample(indices, min(config.images_per_label, len(indices)))
        for img_idx in chosen:
            tasks.append((img_idx, label_idx, global_global_idx := global_idx + 1)) # type: ignore
            global_idx += 1

    # Run simulation
    # Using simple sequential loop for reliability in this refactor, or simple Pool
    # Note: multiprocess pool with torch/hdf5 can be tricky, stick to sequential or careful spawning

    logger.info(f"Processing {len(tasks)} samples...")

    network_state_dir = os.path.join(output_dir, "network_state") if config.export_state else None

    for img_idx, label, g_idx in tqdm(tasks):
        export_path = None
        if network_state_dir:
            export_path = os.path.join(network_state_dir, f"sample_{g_idx}_img{img_idx}.json")

        # We pass net_path to reload it fresh every time (simulating 'fresh_run_per_image')
        # If fresh_run_per_image is False, we would need a different logic, but let's assume True for robustness
        res = process_image(img_idx, label, config.ticks_per_image, network_json_path, input_ids, input_syns, layers, config.tick_time, dataset, export_path=export_path)

        _, _, u, t, fr, s = res
        spike_arr = np.array(s, dtype=np.int32).flatten() if s else np.zeros((0,), dtype=np.int32)
        recorder.save_sample_from_buffers(g_idx - 1, label, u, t, fr, spike_arr)

    recorder.close()
    logger.info(f"Saved dataset to {recorder.h5_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--network_json", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
        sim_cfg = SimulationConfig(**cfg_dict['simulation'])

    run_recording(sim_cfg, args.network_json, args.output_dir)
