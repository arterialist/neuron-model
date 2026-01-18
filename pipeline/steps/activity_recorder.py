"""
Activity recorder step for the pipeline.

Records neural network activity datasets for classifier training.
Supports both binary (HDF5) and legacy JSON output formats.
"""

import json
import logging
import random
import sys
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neuron.nn_core import NNCore
from neuron.network_config import NetworkConfig as NeuronNetworkConfig
from neuron import setup_neuron_logger

from pipeline.config import ActivityRecordingConfig, DatasetType
from pipeline.steps.base import (
    PipelineStep,
    StepContext,
    StepResult,
    StepStatus,
    Artifact,
    StepRegistry,
    StepCancelledException,
)
from pipeline.utils.activity_data import HDF5TensorRecorder


def load_dataset_for_recording(
    dataset_type: DatasetType,
    train: bool = True,
) -> Tuple[Any, int, int, str, bool]:
    """Load a dataset for activity recording.

    Args:
        dataset_type: Type of dataset to load
        train: Whether to load training split (True) or test split (False)

    Returns:
        Tuple of (dataset, vector_size, num_classes, dataset_name, is_colored)
    """
    root_candidates = [
        "./data",
        "./data/mnist",
        "./data/cifar",
        "./data/usps",
        "./data/svhn",
        "./data/fashionmnist",
    ]

    if dataset_type == DatasetType.MNIST:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        for root in root_candidates:
            try:
                dataset = datasets.MNIST(
                    root=root, train=train, download=True, transform=transform
                )
                return dataset, 784, 10, "mnist", False
            except Exception:
                continue
        raise RuntimeError("Failed to load MNIST dataset")

    elif dataset_type == DatasetType.CIFAR10:
        # CIFAR10 grayscale-flattened (treats as single channel)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        for root in root_candidates:
            try:
                dataset = datasets.CIFAR10(
                    root=root, train=train, download=True, transform=transform
                )
                return dataset, 3072, 10, "cifar10", False
            except Exception:
                continue
        raise RuntimeError("Failed to load CIFAR10 dataset")

    elif dataset_type == DatasetType.CIFAR10_COLOR:
        # CIFAR10 color with RGB channels preserved
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        for root in root_candidates:
            try:
                dataset = datasets.CIFAR10(
                    root=root, train=train, download=True, transform=transform
                )
                return dataset, 3072, 10, "cifar10_color", True
            except Exception:
                continue
        raise RuntimeError("Failed to load CIFAR10 (color) dataset")

    elif dataset_type == DatasetType.CIFAR100:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        for root in root_candidates:
            try:
                dataset = datasets.CIFAR100(
                    root=root, train=train, download=True, transform=transform
                )
                return dataset, 3072, 100, "cifar100", False
            except Exception:
                continue
        raise RuntimeError("Failed to load CIFAR100 dataset")

    elif dataset_type == DatasetType.FASHION_MNIST:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        for root in root_candidates:
            try:
                dataset = datasets.FashionMNIST(
                    root=root, train=train, download=True, transform=transform
                )
                return dataset, 784, 10, "fashionmnist", False
            except Exception:
                continue
        raise RuntimeError("Failed to load Fashion-MNIST dataset")

    elif dataset_type == DatasetType.SVHN:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        for root in root_candidates:
            try:
                dataset = datasets.SVHN(
                    root=root,
                    split="train" if train else "test",
                    download=True,
                    transform=transform,
                )
                return dataset, 3072, 10, "svhn", False
            except Exception:
                continue
        raise RuntimeError("Failed to load SVHN dataset")

    elif dataset_type == DatasetType.USPS:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        for root in root_candidates:
            try:
                dataset = datasets.USPS(
                    root=root, train=train, download=True, transform=transform
                )
                # USPS images are 16x16
                return dataset, 256, 10, "usps", False
            except Exception:
                continue
        raise RuntimeError("Failed to load USPS dataset")

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def infer_layers_from_metadata(network_sim) -> List[List[int]]:
    """Groups neurons by their 'layer' metadata."""
    layer_map: Dict[int, List[int]] = {}
    for nid, neuron in network_sim.network.neurons.items():
        meta = getattr(neuron, "metadata", {}) or {}
        layer_idx = meta.get("layer", 0)
        layer_map.setdefault(layer_idx, []).append(nid)
    return [layer_map[k] for k in sorted(layer_map.keys())]


def determine_input_mapping(
    network_sim, layers: List[List[int]]
) -> Tuple[List[int], int]:
    """Determines input layer and synapses per neuron."""
    if not layers:
        raise ValueError("No layers")
    input_layer_ids = layers[0]
    sample_neuron = network_sim.network.neurons[input_layer_ids[0]]
    synapses_per = len(sample_neuron.postsynaptic_points)
    return input_layer_ids, max(1, synapses_per)


def image_to_signals(
    image_tensor: torch.Tensor,
    input_layer_ids: List[int],
    input_synapses_per_neuron: int,
    network_sim: Any,
    is_colored: bool = False,
    color_normalization: float = 0.5,
) -> List[Tuple[int, int, float]]:
    """Map an image to (neuron_id, synapse_id, strength) signals.

    Supports legacy dense input, colored CIFAR-10, and CNN-style input (conv layer at index 0).
    Signals are normalized to [0, 1] range.

    Args:
        image_tensor: Input image tensor (CHW or HW format)
        input_layer_ids: List of neuron IDs in the input layer
        input_synapses_per_neuron: Number of synapses per input neuron
        network_sim: Neural network simulation object
        is_colored: Whether this is a colored (RGB) image
        color_normalization: Normalization factor for color channels

    Returns:
        List of (neuron_id, synapse_id, strength) tuples
    """
    # Detect CNN-style input from first neuron's metadata
    first_neuron = network_sim.network.neurons[input_layer_ids[0]]
    meta = getattr(first_neuron, "metadata", {}) or {}
    is_cnn_input = meta.get("layer_type") == "conv" and meta.get("layer", 0) == 0

    if not is_cnn_input:
        if is_colored:
            # Colored image (e.g., CIFAR-10) handling
            arr = image_tensor.detach().cpu().numpy().astype(np.float32)
            if arr.ndim == 3 and arr.shape[0] == 3:  # CHW format
                h, w = arr.shape[1], arr.shape[2]
                signals: List[Tuple[int, int, float]] = []

                # Check if this network uses separate neurons per color channel
                separate_neurons_per_color = "color_channel" in meta

                if separate_neurons_per_color:
                    # Architecture 2: One neuron per spatial kernel per RGB channel
                    total_spatial_positions = h * w
                    neurons_per_color = len(input_layer_ids) // 3

                    for y in range(h):
                        for x in range(w):
                            pixel_index = y * w + x
                            for c in range(3):  # RGB channels
                                spatial_neuron_idx = pixel_index % neurons_per_color
                                global_neuron_idx = (
                                    c * neurons_per_color + spatial_neuron_idx
                                )

                                if global_neuron_idx >= len(input_layer_ids):
                                    continue

                                pixels_per_neuron = (
                                    total_spatial_positions // neurons_per_color
                                )
                                synapse_index = pixel_index // neurons_per_color

                                neuron_id = input_layer_ids[global_neuron_idx]
                                pixel_value = arr[c, y, x]
                                strength = (
                                    float(pixel_value) + 1.0
                                ) * color_normalization
                                signals.append((neuron_id, synapse_index, strength))
                else:
                    # Architecture 1: One neuron per spatial kernel, 3 synapses per RGB
                    for y in range(h):
                        for x in range(w):
                            for c in range(3):  # RGB channels
                                pixel_index = y * w + x
                                target_neuron_index = pixel_index % len(input_layer_ids)
                                pixels_per_neuron = (h * w) // len(input_layer_ids)
                                if (
                                    pixel_index // len(input_layer_ids)
                                    >= pixels_per_neuron
                                ):
                                    continue
                                base_synapse_index = (
                                    pixel_index // len(input_layer_ids)
                                ) * 3
                                synapse_index = base_synapse_index + c
                                if synapse_index >= input_synapses_per_neuron:
                                    continue

                                neuron_id = input_layer_ids[target_neuron_index]
                                pixel_value = arr[c, y, x]
                                strength = (
                                    float(pixel_value) + 1.0
                                ) * color_normalization
                                signals.append((neuron_id, synapse_index, strength))
                return signals

        # Legacy dense mapping for grayscale images
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
        arr = arr[None, :, :]  # Add channel dimension
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


def process_single_image(
    image_tensor,
    label,
    img_idx,
    ticks_per_image,
    nn_core,
    input_layer_ids,
    input_synapses_per_neuron,
    layers,
    export_network_states,
    states_dir,
    is_colored: bool = False,
    color_normalization: float = 0.5,
) -> List[Dict[str, Any]]:
    """Process a single image and return activity records (JSON format)."""
    network_sim = nn_core.neural_net
    network_sim.reset_simulation()

    signals = image_to_signals(
        image_tensor,
        input_layer_ids,
        input_synapses_per_neuron,
        network_sim,
        is_colored=is_colored,
        color_normalization=color_normalization,
    )

    records = []
    cumulative_fires = {nid: 0 for nid in network_sim.network.neurons.keys()}

    for tick in range(ticks_per_image):
        nn_core.send_batch_signals(signals)
        nn_core.do_tick()

        tick_layers = []
        for l_idx, l_ids in enumerate(layers):
            S, F, fire = [], [], []
            for nid in l_ids:
                n = network_sim.network.neurons[nid]
                S.append(float(n.S))
                F.append(float(n.F_avg))
                f = 1 if n.O > 0 else 0
                fire.append(f)
                if f:
                    cumulative_fires[nid] += 1
            tick_layers.append(
                {"layer_index": l_idx, "S": S, "F_avg": F, "fired": fire}
            )

        records.append(
            {
                "image_index": int(img_idx),
                "label": int(label),
                "tick": int(tick),
                "layers": tick_layers,
                "cumulative_fires": [
                    [int(cumulative_fires[n]) for n in l_ids] for l_ids in layers
                ],
            }
        )

    if export_network_states and states_dir:
        state_path = states_dir / f"state_img_{img_idx}_label_{label}.json"
        NeuronNetworkConfig.save_network_config(network_sim, str(state_path))

    return records


def process_single_image_binary(
    image_tensor: torch.Tensor,
    label: int,
    sample_idx: int,
    ticks_per_image: int,
    nn_core: NNCore,
    input_layer_ids: List[int],
    input_synapses_per_neuron: int,
    recorder: HDF5TensorRecorder,
    export_network_states: bool,
    states_dir: Optional[Path],
    is_colored: bool = False,
    color_normalization: float = 0.5,
) -> None:
    """Process a single image and record activity in binary format.

    Args:
        image_tensor: Input image tensor
        label: Class label
        sample_idx: Sample index in the dataset
        ticks_per_image: Number of simulation ticks per image
        nn_core: Neural network core
        input_layer_ids: IDs of neurons in the input layer
        input_synapses_per_neuron: Number of synapses per input neuron
        recorder: HDF5 tensor recorder
        export_network_states: Whether to export network states
        states_dir: Directory for network state exports
        is_colored: Whether the image is colored
        color_normalization: Normalization factor for color channels
    """
    network_sim = nn_core.neural_net
    network_sim.reset_simulation()

    signals = image_to_signals(
        image_tensor,
        input_layer_ids,
        input_synapses_per_neuron,
        network_sim,
        is_colored=is_colored,
        color_normalization=color_normalization,
    )

    # Initialize buffer for this sample
    recorder.init_buffer(ticks_per_image)

    # Run simulation and capture each tick
    for tick in range(ticks_per_image):
        nn_core.send_batch_signals(signals)
        nn_core.do_tick()
        recorder.capture_tick(tick, network_sim.network.neurons)

    # Save sample to HDF5
    recorder.save_sample(sample_idx, label)

    # Export network state if requested
    if export_network_states and states_dir:
        state_path = states_dir / f"state_sample_{sample_idx}_label_{label}.json"
        NeuronNetworkConfig.save_network_config(network_sim, str(state_path))


def _process_image_worker(args: Tuple) -> Tuple:
    """Worker function for multiprocessing image processing.

    Each worker loads its own network instance to avoid sharing state.
    Returns: (sample_idx, label, u_buf, t_ref_buf, fr_buf, spike_arr, records)
    """
    (
        sample_idx,
        img_idx,
        label,
        ticks_per_image,
        network_path,
        input_layer_ids,
        input_synapses_per_neuron,
        layers,
        use_binary_format,
        is_colored,
        color_normalization,
        dataset_type_value,  # string value of DatasetType enum
    ) = args

    # Load network for this worker process
    network_sim = NeuronNetworkConfig.load_network_config(str(network_path))
    nn_core = NNCore()
    nn_core.neural_net = network_sim

    # Suppress logging in worker
    try:
        setup_neuron_logger("CRITICAL")
    except Exception:
        pass

    # Reset simulation
    network_sim.reset_simulation()

    # Load dataset in worker (datasets are not picklable across processes)
    dataset_type = DatasetType(dataset_type_value)
    ds, _, _, _, _ = load_dataset_for_recording(dataset_type)

    image_tensor, _ = ds[img_idx]

    signals = image_to_signals(
        image_tensor,
        input_layer_ids,
        input_synapses_per_neuron,
        network_sim,
        is_colored=is_colored,
        color_normalization=color_normalization,
    )

    if use_binary_format:
        # Initialize buffers
        num_neurons = len(network_sim.network.neurons)
        uuid_to_idx = {
            uid: i for i, uid in enumerate(network_sim.network.neurons.keys())
        }

        u_buf = np.zeros((ticks_per_image, num_neurons), dtype=np.float32)
        t_ref_buf = np.zeros((ticks_per_image, num_neurons), dtype=np.float32)
        fr_buf = np.zeros((ticks_per_image, num_neurons), dtype=np.float32)
        spikes: List[Tuple[int, int]] = []

        # Run simulation
        for tick in range(ticks_per_image):
            nn_core.send_batch_signals(signals)
            nn_core.do_tick()

            # Capture state
            for uid, neuron in network_sim.network.neurons.items():
                idx = uuid_to_idx[uid]
                u_buf[tick, idx] = neuron.S
                t_ref_buf[tick, idx] = neuron.t_ref
                fr_buf[tick, idx] = neuron.F_avg
                if neuron.O > 0:
                    spikes.append((tick, idx))

        spike_arr = (
            np.array(spikes, dtype=np.int32).flatten()
            if spikes
            else np.zeros((0,), dtype=np.int32)
        )

        return (sample_idx, label, u_buf, t_ref_buf, fr_buf, spike_arr, None)

    else:
        # JSON mode
        records = []
        cumulative_fires = {nid: 0 for nid in network_sim.network.neurons.keys()}

        for tick in range(ticks_per_image):
            nn_core.send_batch_signals(signals)
            nn_core.do_tick()

            tick_layers = []
            for l_idx, l_ids in enumerate(layers):
                S, F, fire = [], [], []
                for nid in l_ids:
                    n = network_sim.network.neurons[nid]
                    S.append(float(n.S))
                    F.append(float(n.F_avg))
                    f = 1 if n.O > 0 else 0
                    fire.append(f)
                    if f:
                        cumulative_fires[nid] += 1
                tick_layers.append(
                    {"layer_index": l_idx, "S": S, "F_avg": F, "fired": fire}
                )

            records.append(
                {
                    "image_index": int(img_idx),
                    "label": int(label),
                    "tick": int(tick),
                    "layers": tick_layers,
                    "cumulative_fires": [
                        [int(cumulative_fires[n]) for n in l_ids] for l_ids in layers
                    ],
                }
            )

        return (sample_idx, label, None, None, None, None, records)


@StepRegistry.register
class ActivityRecorderStep(PipelineStep):
    @property
    def name(self) -> str:
        return "activity_recording"

    def run(self, context: StepContext) -> StepResult:
        start_time = datetime.now()
        config: ActivityRecordingConfig = context.config
        log = context.logger or logging.getLogger(__name__)
        logs = []

        try:
            network_artifact = context.get_artifact("network", "network.json")
            network_sim = NeuronNetworkConfig.load_network_config(
                str(network_artifact.path)
            )
            nn_core = NNCore()
            nn_core.neural_net = network_sim

            try:
                setup_neuron_logger("CRITICAL")
            except:
                pass

            layers = infer_layers_from_metadata(network_sim)
            input_ids, syn_per = determine_input_mapping(network_sim, layers)

            log.info(f"Loading dataset: {config.dataset}")
            logs.append(f"Loading dataset {config.dataset}...")
            ds_train, _, num_classes, dataset_name, is_colored = (
                load_dataset_for_recording(config.dataset)
            )
            # Get color normalization factor from config
            color_normalization = getattr(config, "cifar10_color_normalization", 0.5)
            log.info(
                f"Dataset loaded: {dataset_name} ({len(ds_train)} samples), colored={is_colored}"
            )
            logs.append(f"Loaded {dataset_name} with {len(ds_train)} samples")

            label_to_indices = {i: [] for i in range(num_classes)}
            for idx in range(len(ds_train)):
                _, lbl = ds_train[idx]
                label_to_indices[int(lbl)].append(idx)

            tasks = []
            for lbl in range(num_classes):
                indices = label_to_indices.get(lbl, [])
                if indices:
                    tasks.extend(
                        [
                            (idx, lbl)
                            for idx in random.sample(
                                indices, min(config.images_per_label, len(indices))
                            )
                        ]
                    )

            step_dir = context.output_dir / self.name
            step_dir.mkdir(parents=True, exist_ok=True)
            states_dir = None
            if config.export_network_states:
                states_dir = step_dir / "network_states"
                states_dir.mkdir(exist_ok=True)
                log.info(f"Network state export enabled: {states_dir}")
                logs.append("Network state export enabled")

            # Check multiprocessing config
            use_multiprocessing = getattr(config, "use_multiprocessing", True)
            num_workers = min(cpu_count(), len(tasks)) if use_multiprocessing else 1

            if use_multiprocessing and num_workers > 1:
                log.info(
                    f"Processing {len(tasks)} samples with {num_workers} workers..."
                )
                logs.append(
                    f"Recording activity for {len(tasks)} samples (parallel, {num_workers} workers)..."
                )
            else:
                log.info(f"Processing {len(tasks)} samples sequentially...")
                logs.append(f"Recording activity for {len(tasks)} samples...")

            # Determine output format
            use_binary = getattr(config, "binary_format", True)
            log.info(f"Output format: {'binary (HDF5)' if use_binary else 'JSON'}")
            logs.append(
                f"Using {'binary HDF5' if use_binary else 'legacy JSON'} format"
            )

            if use_binary:
                # Binary format: use HDF5TensorRecorder
                recorder = HDF5TensorRecorder(str(step_dir), network_sim.network)
                num_samples = 0

                if use_multiprocessing and num_workers > 1:
                    # Parallel processing with multiprocessing
                    network_path = network_artifact.path

                    # Prepare worker arguments
                    worker_args = [
                        (
                            i,  # sample_idx
                            img_idx,
                            lbl,
                            config.ticks_per_image,
                            str(network_path),
                            input_ids,
                            syn_per,
                            layers,
                            True,  # use_binary_format
                            is_colored,
                            color_normalization,
                            config.dataset.value,  # dataset_type_value
                        )
                        for i, (img_idx, lbl) in enumerate(tasks)
                    ]

                    # Process in parallel
                    with Pool(processes=num_workers) as pool:
                        for result in pool.imap_unordered(
                            _process_image_worker, worker_args
                        ):
                            # Check for cancellation periodically
                            context.check_control_signals()

                            (
                                sample_idx,
                                label,
                                u_buf,
                                t_ref_buf,
                                fr_buf,
                                spike_arr,
                                _,
                            ) = result

                            # Save to HDF5
                            recorder.save_sample_from_buffers(
                                sample_idx, label, u_buf, t_ref_buf, fr_buf, spike_arr
                            )
                            num_samples += 1

                            if num_samples % 10 == 0 or num_samples == len(tasks):
                                log.info(
                                    f"Progress: {num_samples}/{len(tasks)} samples processed"
                                )
                                logs.append(
                                    f"Processed {num_samples}/{len(tasks)} samples"
                                )
                else:
                    # Sequential processing
                    for i, (img_idx, lbl) in enumerate(
                        tqdm(
                            tasks,
                            desc="Activity Recording",
                            file=sys.stdout,
                            disable=True,
                        )
                    ):
                        # Check for cancellation/pause
                        context.check_control_signals()

                        if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                            log.info(
                                f"Progress: {i + 1}/{len(tasks)} samples processed"
                            )
                            logs.append(f"Processed {i + 1}/{len(tasks)} samples")

                        image_tensor, _ = ds_train[img_idx]
                        process_single_image_binary(
                            image_tensor,
                            lbl,
                            i,  # Use sequential sample index
                            config.ticks_per_image,
                            nn_core,
                            input_ids,
                            syn_per,
                            recorder,
                            config.export_network_states,
                            states_dir,
                            is_colored=is_colored,
                            color_normalization=color_normalization,
                        )
                        num_samples += 1

                # Close the recorder
                recorder.close()

                output_path = step_dir / "activity_dataset.h5"
                artifacts = [
                    Artifact(
                        name=output_path.name,
                        path=output_path,
                        artifact_type="dataset",
                        size_bytes=output_path.stat().st_size,
                        metadata={
                            "num_samples": num_samples,
                            "format": "hdf5",
                            "ticks_per_sample": config.ticks_per_image,
                        },
                    )
                ]
                result_metrics = {"num_samples": num_samples}
                logs.append(f"Saved binary dataset to {output_path}")

            else:
                # Legacy JSON format
                all_records: List[Dict[str, Any]] = []

                if use_multiprocessing and num_workers > 1:
                    # Parallel processing for JSON format
                    network_path = network_artifact.path

                    # Prepare worker arguments
                    worker_args = [
                        (
                            i,  # sample_idx
                            img_idx,
                            lbl,
                            config.ticks_per_image,
                            str(network_path),
                            input_ids,
                            syn_per,
                            layers,
                            False,  # use_binary_format
                            is_colored,
                            color_normalization,
                            config.dataset.value,  # dataset_type_value
                        )
                        for i, (img_idx, lbl) in enumerate(tasks)
                    ]

                    processed = 0
                    with Pool(processes=num_workers) as pool:
                        for result in pool.imap_unordered(
                            _process_image_worker, worker_args
                        ):
                            # Check for cancellation periodically
                            context.check_control_signals()

                            _, _, _, _, _, _, records = result
                            if records:
                                all_records.extend(records)
                            processed += 1

                            if processed % 10 == 0 or processed == len(tasks):
                                log.info(
                                    f"Progress: {processed}/{len(tasks)} samples processed"
                                )
                                logs.append(
                                    f"Processed {processed}/{len(tasks)} samples"
                                )
                else:
                    # Sequential processing
                    for i, (img_idx, lbl) in enumerate(
                        tqdm(
                            tasks,
                            desc="Activity Recording",
                            file=sys.stdout,
                            disable=True,
                        )
                    ):
                        # Check for cancellation/pause
                        context.check_control_signals()

                        if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                            log.info(
                                f"Progress: {i + 1}/{len(tasks)} samples processed"
                            )
                            logs.append(f"Processed {i + 1}/{len(tasks)} samples")

                        image_tensor, _ = ds_train[img_idx]
                        records = process_single_image(
                            image_tensor,
                            lbl,
                            img_idx,
                            config.ticks_per_image,
                            nn_core,
                            input_ids,
                            syn_per,
                            layers,
                            config.export_network_states,
                            states_dir,
                            is_colored=is_colored,
                            color_normalization=color_normalization,
                        )
                        all_records.extend(records)

                output_path = step_dir / "activity_dataset.json"
                with open(output_path, "w") as f:
                    json.dump({"records": all_records}, f)

                artifacts = [
                    Artifact(
                        name=output_path.name,
                        path=output_path,
                        artifact_type="dataset",
                        size_bytes=output_path.stat().st_size,
                        metadata={
                            "num_records": len(all_records),
                            "format": "json",
                        },
                    )
                ]
                result_metrics = {"num_records": len(all_records)}
                logs.append(f"Saved JSON dataset to {output_path}")

            # Register network states as artifact if they exist
            if states_dir and states_dir.exists():
                artifacts.append(
                    Artifact(
                        name="network_states",
                        path=states_dir,
                        artifact_type="directory",
                        size_bytes=sum(
                            f.stat().st_size for f in states_dir.glob("*.json")
                        ),
                        metadata={"num_states": len(list(states_dir.glob("*.json")))},
                    )
                )

            return StepResult(
                status=StepStatus.COMPLETED,
                artifacts=artifacts,
                metrics=result_metrics,
                start_time=start_time,
                end_time=datetime.now(),
                logs=logs,
            )

        except StepCancelledException:
            raise
        except Exception as e:
            import traceback

            return StepResult(
                status=StepStatus.FAILED,
                error_message=str(e),
                start_time=start_time,
                end_time=datetime.now(),
                logs=logs + [traceback.format_exc()],
            )
