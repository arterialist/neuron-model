"""Multiprocessing worker for single-image simulation."""

import time
from typing import Any

import numpy as np

from neuron.nn_core import NNCore
from neuron.network_config import NetworkConfig
from neuron.ablation_registry import get_neuron_class_for_ablation
from neuron import setup_neuron_logger

from snn_classification_realtime.core.config import DatasetConfig
from snn_classification_realtime.core.input_mapping import image_to_signals
from snn_classification_realtime.activity_dataset_builder.signals import collect_tick_snapshot


def process_single_image_worker(args: tuple[Any, ...]) -> tuple[Any, ...]:
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
        dataset_config,
        ablation_name,
        process_id,
        progress_queue,
    ) = args

    neuron_cls = get_neuron_class_for_ablation(ablation_name)
    network_sim = NetworkConfig.load_network_config(net_path, neuron_class=neuron_cls)
    nn_core = NNCore()
    nn_core.neural_net = network_sim
    setup_neuron_logger("CRITICAL")
    nn_core.set_log_level("CRITICAL")

    network_sim.reset_simulation()
    nn_core.state.current_tick = 0
    network_sim.current_tick = 0

    image_tensor, _ = ds_train[img_idx]
    signals = image_to_signals(
        image_tensor,
        input_layer_ids,
        input_synapses_per_neuron,
        network_sim,
        dataset_config,
    )

    if use_binary_format:
        num_neurons = len(network_sim.network.neurons)
        u_buf = np.zeros((ticks_per_image, num_neurons), dtype=np.float32)
        t_ref_buf = np.zeros((ticks_per_image, num_neurons), dtype=np.float32)
        fr_buf = np.zeros((ticks_per_image, num_neurons), dtype=np.float32)
        spikes: list[tuple[int, int]] = []
    else:
        u_buf = t_ref_buf = fr_buf = None
        spikes = None

    cumulative_fires = {nid: 0 for nid in network_sim.network.neurons.keys()}
    records: list[dict[str, Any]] = []

    if progress_queue is not None:
        progress_queue.put({
            "process_id": process_id,
            "current_tick": 0,
            "total_ticks": ticks_per_image,
            "img_idx": img_idx,
            "label": actual_label,
            "completed": False,
        })

    uuid_to_idx = {
        uid: i for i, uid in enumerate(network_sim.network.neurons.keys())
    }

    for local_tick in range(ticks_per_image):
        nn_core.send_batch_signals(signals)
        nn_core.do_tick()

        for nid, neuron in network_sim.network.neurons.items():
            if neuron.O > 0:
                cumulative_fires[nid] += 1

        if use_binary_format and u_buf is not None:
            for uid, neuron in network_sim.network.neurons.items():
                idx = uuid_to_idx[uid]
                u_buf[local_tick, idx] = neuron.S
                t_ref_buf[local_tick, idx] = neuron.t_ref
                fr_buf[local_tick, idx] = neuron.F_avg
                if neuron.O > 0:
                    spikes.append((local_tick, idx))
        else:
            snapshot = collect_tick_snapshot(
                network_sim, layers, tick_index=network_sim.current_tick
            )
            cum_layer_counts = [
                [int(cumulative_fires[n]) for n in layer_ids]
                for layer_ids in layers
            ]
            records.append({
                "image_index": int(img_idx),
                "label": int(actual_label),
                "tick": snapshot["tick"],
                "layers": snapshot["layers"],
                "cumulative_fires": cum_layer_counts,
            })

        if progress_queue is not None and (local_tick + 1) % 10 == 0:
            progress_queue.put({
                "process_id": process_id,
                "current_tick": local_tick + 1,
                "total_ticks": ticks_per_image,
                "img_idx": img_idx,
                "label": actual_label,
                "completed": False,
            })

        if tick_ms > 0:
            time.sleep(tick_ms / 1000.0)

    if progress_queue is not None:
        progress_queue.put({
            "process_id": process_id,
            "current_tick": ticks_per_image,
            "total_ticks": ticks_per_image,
            "img_idx": img_idx,
            "label": actual_label,
            "completed": True,
        })

    return (
        img_idx,
        actual_label,
        u_buf if use_binary_format else None,
        t_ref_buf if use_binary_format else None,
        fr_buf if use_binary_format else None,
        spikes if use_binary_format else None,
        records if not use_binary_format else None,
    )
