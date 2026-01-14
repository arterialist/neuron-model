"""
Activity recorder step for the pipeline.

Records neural network activity datasets for classifier training.
"""

import json
import logging
import os
import random
import sys
import time
from datetime import datetime
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
)


def load_dataset_for_recording(
    dataset_type: DatasetType,
    train: bool = True,
) -> Tuple[Any, int, int, str]:
    """Load a dataset for activity recording."""
    root_candidates = ["./data", "./data/mnist"]

    if dataset_type == DatasetType.MNIST:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        for root in root_candidates:
            try:
                dataset = datasets.MNIST(
                    root=root, train=train, download=True, transform=transform
                )
                return dataset, 784, 10, "mnist"
            except:
                continue
        raise RuntimeError("Failed to load MNIST")
    else:
        # Simplified for demo
        raise ValueError(f"Unsupported dataset type for demo: {dataset_type}")


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
    """Convert an image tensor to a list of neural signals."""
    arr = image_tensor.detach().cpu().numpy().flatten()
    # Normalize to [0, 1] if not already
    if arr.min() < -0.1:  # Likely normalized to [-1, 1]
        arr = (arr + 1.0) * 0.5

    signals = []
    for neuron_idx, neuron_id in enumerate(input_layer_ids):
        # Each neuron in the input layer gets input_synapses_per_neuron pixels
        start_idx = neuron_idx * input_synapses_per_neuron
        end_idx = start_idx + input_synapses_per_neuron

        for synapse_id, pixel_idx in enumerate(
            range(start_idx, min(end_idx, len(arr)))
        ):
            val = float(arr[pixel_idx])
            signals.append((neuron_id, synapse_id, val))

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
) -> List[Dict[str, Any]]:
    """Process a single image and return activity records."""
    network_sim = nn_core.neural_net
    network_sim.reset_simulation()

    signals = image_to_signals(
        image_tensor, input_layer_ids, input_synapses_per_neuron, network_sim
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
            ds_train, _, num_classes, dataset_name = load_dataset_for_recording(
                config.dataset
            )
            log.info(f"Dataset loaded: {dataset_name} ({len(ds_train)} samples)")
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

            all_records = []
            log.info(f"Processing {len(tasks)} samples sequentially...")
            logs.append(f"Recording activity for {len(tasks)} samples...")

            for i, (img_idx, lbl) in enumerate(
                tqdm(tasks, desc="Activity Recording", file=sys.stdout, disable=True)
            ):
                if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                    log.info(f"Progress: {i + 1}/{len(tasks)} samples processed")
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
                    metadata={"num_records": len(all_records)},
                )
            ]

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
                metrics={"num_records": len(all_records)},
                start_time=start_time,
                end_time=datetime.now(),
                logs=logs + [f"Saved to {output_path}"],
            )

        except Exception as e:
            import traceback

            return StepResult(
                status=StepStatus.FAILED,
                error_message=str(e),
                start_time=start_time,
                end_time=datetime.now(),
                logs=logs + [traceback.format_exc()],
            )
