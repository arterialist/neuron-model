"""Interactive mode: per-image loop with user-friendly output."""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn

from snn_classification_realtime.realtime_classifier.features import (
    collect_features_consistently,
)
from snn_classification_realtime.realtime_classifier.scaling import (
    apply_scaling_to_snapshot,
)
from snn_classification_realtime.realtime_classifier.input_mapping import (
    image_to_signals,
)
from snn_classification_realtime.realtime_classifier.network_utils import (
    infer_layers_from_metadata,
    determine_input_mapping,
)
from snn_classification_realtime.core.config import DatasetConfig


def run_interactive(
    *,
    dataset_config: DatasetConfig,
    network_config_path: str,
    neuron_cls: type,
    nn_core: Any,
    snn_model: nn.Module,
    scaler_state: dict[str, Any],
    feature_types: list[str],
    device: torch.device,
    args: Any,
) -> None:
    """Run interactive mode with per-image loop and user output."""
    softmax = nn.Softmax(dim=1)
    activity_buffer: list[list[float]] = []
    dataset = dataset_config.dataset

    base_ticks_per_image = args.ticks_per_image
    base_window_size = args.window_size
    max_ticks_to_add = int(base_ticks_per_image * (args.max_thinking_multiplier - 1.0))

    try:
        for i in range(len(dataset)):
            image_tensor, actual_label = dataset[i]
            print(
                f"\n--- Presenting Image {i + 1}/{len(dataset)} (Label: {actual_label}) ---"
            )

            network_sim = _load_fresh_network(network_config_path, neuron_cls)
            nn_core.neural_net = network_sim
            layers = infer_layers_from_metadata(network_sim)
            input_layer_ids, synapses_per_neuron = determine_input_mapping(
                network_sim, layers
            )

            signals = image_to_signals(
                image_tensor,
                network_sim,
                input_layer_ids,
                synapses_per_neuron,
                dataset_config,
            )
            activity_buffer.clear()
            network_sim.reset_simulation()

            current_ticks_per_image = args.ticks_per_image
            current_window_size = args.window_size
            ticks_added = 0
            prediction_history: list[bool] = []
            snn_spike_buffer: list[torch.Tensor] = []

            mem1 = snn_model.lif1.init_leaky()
            mem2 = snn_model.lif2.init_leaky()
            mem3 = snn_model.lif3.init_leaky()
            mem4 = snn_model.lif4.init_leaky()

            max_ticks = base_ticks_per_image + max_ticks_to_add
            tick = 0

            while tick < current_ticks_per_image and tick < max_ticks:
                nn_core.send_batch_signals(signals)
                nn_core.do_tick()

                snapshot = collect_features_consistently(
                    network_sim, layers, feature_types
                )
                if scaler_state:
                    snapshot = apply_scaling_to_snapshot(snapshot, scaler_state)
                activity_buffer.append(snapshot)
                if len(activity_buffer) > current_window_size:
                    activity_buffer.pop(0)

                if len(activity_buffer) == current_window_size:
                    with torch.no_grad():
                        current_input = (
                            torch.tensor([activity_buffer[-1]], dtype=torch.float32)
                            .to(device)
                        )
                        spk2, mem1, mem2, mem3, mem4 = snn_model(
                            current_input, mem1, mem2, mem3, mem4
                        )
                        snn_spike_buffer.append(spk2)
                        if len(snn_spike_buffer) > args.window_size:
                            snn_spike_buffer.pop(0)

                    if len(snn_spike_buffer) >= 1:
                        spk_rec = torch.stack(snn_spike_buffer, dim=0)
                        spike_counts = spk_rec.sum(dim=0)
                        probabilities = softmax(spike_counts)
                        top_prob, top_class = probabilities.max(1)
                        current_prediction = top_class.item()
                        current_confidence = top_prob.item()

                        prediction_history.append(
                            current_prediction == actual_label
                        )
                        if len(prediction_history) > 5:
                            prediction_history.pop(0)

                        probs_list = [
                            f"{label}: {p:.2%}"
                            for label, p in enumerate(probabilities[0])
                        ]
                        mapped_probs_list = [
                            (label, p) for label, p in enumerate(probabilities[0])
                        ]

                        print(
                            f"Tick {tick + 1}/{current_ticks_per_image} | "
                            f"Prediction: {top_class.item()} ({top_prob.item():.2%}) | "
                            f"Certainties: {probs_list}"
                        )

                        if (
                            args.think_longer
                            and len(prediction_history) >= 5
                            and not all(prediction_history[-5:])
                            and tick == current_ticks_per_image - 1
                            and ticks_added < max_ticks_to_add
                        ):
                            ticks_added += 1
                            current_ticks_per_image = (
                                base_ticks_per_image + ticks_added
                            )
                            current_window_size = base_window_size + int(
                                ticks_added
                                * (base_window_size / base_ticks_per_image)
                            )
                            print(
                                f"  → Extending thinking time by 10 ticks to "
                                f"{current_ticks_per_image} total (+{ticks_added} ticks)"
                            )

                        if top_class.item() == actual_label:
                            print("  → Correct prediction made!")
                        print("✅" if top_class.item() == actual_label else "❌")

                        sorted_probabilities = sorted(
                            mapped_probs_list,
                            key=lambda x: x[1],
                            reverse=True,
                        )
                        if len(sorted_probabilities) > 1:
                            closest_match = sorted_probabilities[1][0]
                            if closest_match == actual_label:
                                print(
                                    "Next closest match: "
                                    f"{closest_match}: {sorted_probabilities[1][1]:.2%}"
                                )

                tick += 1
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        print("\nExiting.")


def _load_fresh_network(network_config_path: str, neuron_cls: type) -> Any:
    from neuron.network_config import NetworkConfig

    return NetworkConfig.load_network_config(
        network_config_path, neuron_class=neuron_cls
    )
